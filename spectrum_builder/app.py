"""
Streamlit front-end for Isotope Spectrum Builder.

This app generates detector-aware gamma spectra by combining:
- Signal events from Geant4 simulations (EnergySmeared, keV)
- Background events from resampling a real background spectrum (Energy, keV)

The app is designed to be modular:
- Core logic lives in the `spectrum_builder/` package.
- This file implements UI and orchestration only.

Caching:
- Streamlit reruns your script on every interaction.
- We use st.cache_data / st.cache_resource to keep datasets in memory across reruns.

License: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import streamlit as st

from spectrum_builder.config import DETECTOR_CONFIG
from spectrum_builder.data import DetectorPools, load_detector_pools
from spectrum_builder.export import spectrum_to_csv_bytes, spectrum_to_png_bytes
from spectrum_builder.sampling import sample_energies
from spectrum_builder.spectrum import Normalization, build_spectrum_dataframe

# Attribution (also printed on exported PNG)
IMAGE_BUILDER_NAME = "Created by Dr K. Karafasoulis"
IMAGE_BUILDER_URL = "http://karafasoulis.eu"


st.set_page_config(
    page_title="Isotope Gamma Spectrum Builder (NaI / CZT / HPGe)",
    page_icon="📊",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Caching helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading detector datasets...")
def _load_detector_pools_cached(detector: str) -> DetectorPools:
    """
    Load one detector's datasets and cache the resulting DetectorPools.

    Why cache_data?
    - The returned object contains pandas DataFrames.
    - Streamlit can hash/cache these, and will reuse across reruns.
    """
    cfg = DETECTOR_CONFIG[detector]
    return load_detector_pools(
        detector,
        signal_path=Path(cfg["signal_path"]),
        background_path=Path(cfg["background_path"]),
        info=cfg.get("info", detector),
    )


@st.cache_data(show_spinner=False)
def load_all_available_detectors() -> Dict[str, DetectorPools]:
    """
    Load all detectors that can be loaded successfully.

    Important:
    - This function intentionally does not report missing/invalid data files.
    - Detectors that fail to load are simply not shown in the UI.
    """
    pools: Dict[str, DetectorPools] = {}
    for det in DETECTOR_CONFIG.keys():
        try:
            pools[det] = _load_detector_pools_cached(det)
        except Exception:
            # Silent: do not report file presence/absence or exceptions to user.
            continue
    return pools


# -----------------------------------------------------------------------------
# App UI
# -----------------------------------------------------------------------------
st.title("Isotope Gamma Spectrum Builder")

st.markdown(
    """
Signal events are generated from **Geant4 simulations** and  
background events are obtained by **resampling a real background spectrum**.

The app generates a **single combined spectrum** (signal + background per bin).
"""
)

detector_pools = load_all_available_detectors()
available_detectors = list(detector_pools.keys())

if not available_detectors:
    st.error("No detector data are currently available. Please contact the administrator.")
    st.stop()

# Sidebar controls
st.sidebar.header("Detector Selection")
detector_type = st.sidebar.radio("Detector", available_detectors, index=0)
pools = detector_pools[detector_type]
st.sidebar.info(pools.info)

st.sidebar.header("Isotope & Binning")
isotopes = sorted(pools.signal_by_isotope.keys())
selected_isotope = st.sidebar.selectbox("Isotope", isotopes)

n_bins = st.sidebar.selectbox("Number of bins", [1024, 2048, 4096], index=0)

st.sidebar.header("Energy Range (optional)")
use_custom_range = st.sidebar.checkbox("Use custom energy range (keV)", value=False)
custom_emin = custom_emax = None
if use_custom_range:
    custom_emin = st.sidebar.number_input("Lower energy (keV)", min_value=0.0, value=0.0, step=10.0)
    custom_emax = st.sidebar.number_input("Upper energy (keV)", min_value=0.1, value=3000.0, step=10.0)

st.sidebar.header("Event Sampling")
n_signal = st.sidebar.number_input("Number of signal events", min_value=0, value=5000, step=100)
n_background = st.sidebar.number_input("Number of background events", min_value=0, value=5000, step=100)

st.sidebar.header("Normalization")
norm_label = st.sidebar.radio(
    "Normalization type",
    ["Raw counts", "Counts per second (cps)", "Unit area (Σ = 1)"],
    index=0,
)

if norm_label == "Raw counts":
    normalization = Normalization.RAW
elif norm_label == "Counts per second (cps)":
    normalization = Normalization.CPS
else:
    normalization = Normalization.UNIT_AREA

acq_time = None
if normalization == Normalization.CPS:
    acq_time = st.sidebar.number_input(
        "Acquisition time (s)",
        min_value=0.1,
        value=60.0,
        step=1.0,
        help="Used to convert total counts to counts per second.",
    )

random_seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

# -----------------------------------------------------------------------------
# Build spectrum
# -----------------------------------------------------------------------------
st.header("Generated Spectrum")

sig_df = pools.signal_by_isotope.get(selected_isotope)
if sig_df is None or sig_df.empty:
    st.error(f"No signal events found for isotope '{selected_isotope}'.")
    st.stop()

# Sampling with replacement
signal_energies = sample_energies(sig_df, "EnergySmeared", int(n_signal), seed=int(random_seed))
background_energies = sample_energies(pools.background_df, "Energy", int(n_background), seed=int(random_seed) + 1)

try:
    df, emin, emax, y_label = build_spectrum_dataframe(
        signal_energies_keV=signal_energies,
        background_energies_keV=background_energies,
        n_bins=int(n_bins),
        e_min_keV=(float(custom_emin) if use_custom_range else None),
        e_max_keV=(float(custom_emax) if use_custom_range else None),
        normalization=normalization,
        acquisition_time_s=(float(acq_time) if acq_time is not None else None),
    )
except Exception as exc:
    st.error(str(exc))
    st.stop()

col_plot, col_stats = st.columns([3, 1])

with col_plot:
    st.subheader(f"Combined Spectrum – {selected_isotope} ({detector_type})")
    st.caption(f"Bins: {n_bins} | Energy range: {emin:.1f} – {emax:.1f} keV | Y: {y_label}")

    # Dynamic plot (fast, interactive feel)
    st.line_chart(df.set_index("Energy_keV")[["Total_norm"]])

with col_stats:
    st.subheader("Summary")
    st.write(f"**Detector:** {detector_type}")
    st.write(f"**Isotope:** {selected_isotope}")
    st.write(f"**Signal events:** {int(n_signal)} (sampling with replacement)")
    st.write(f"**Background events:** {int(n_background)} (sampling with replacement)")
    st.write(f"**Number of bins:** {int(n_bins)}")
    st.write(f"**Energy range:** {emin:.1f} – {emax:.1f} keV")
    st.write(f"**Normalization:** {norm_label}")
    if acq_time is not None:
        st.write(f"**Acquisition time:** {float(acq_time):.1f} s")
    st.markdown("---")
    st.write("**Total counts (raw):**", int(df["Total_counts"].sum()))
    st.write("**Max counts in a bin (raw):**", int(df["Total_counts"].max()))

st.markdown("---")
st.subheader("Binned Spectrum Data (preview)")
st.dataframe(df.head(20))

# -----------------------------------------------------------------------------
# Downloads
# -----------------------------------------------------------------------------
st.subheader("Download Spectrum")

st.download_button(
    label="Download spectrum CSV",
    data=spectrum_to_csv_bytes(df),
    file_name=f"spectrum_{selected_isotope}_{detector_type}.csv",
    mime="text/csv",
)

signature = f"{IMAGE_BUILDER_NAME} | Isotope: {selected_isotope} | {IMAGE_BUILDER_URL}"
png_bytes = spectrum_to_png_bytes(
    df,
    title=f"Combined Spectrum – {selected_isotope} ({detector_type})",
    x_label="Energy (keV)",
    y_label=y_label,
    signature=signature,
)

st.download_button(
    label="Download spectrum image (PNG)",
    data=png_bytes,
    file_name=f"spectrum_{selected_isotope}_{detector_type}.png",
    mime="image/png",
)

st.markdown("---")
st.markdown(
    f"""
Created by **Dr K. Karafasoulis**  
{IMAGE_BUILDER_URL}

Acknowledgements to Dr A. Kyriakis for the CZT data.  
http://ailab.inp.demokritos.gr/
"""
)
