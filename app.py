"""
Streamlit front-end for Isotope Spectrum Builder.

Core functionality is implemented in `spectrum_builder/` so the project is:
- modular
- reusable
- easier to test and maintain

License: MIT
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from spectrum_builder.config import DETECTOR_CONFIG
from spectrum_builder.data import load_detector_pools
from spectrum_builder.export import spectrum_to_csv_bytes, spectrum_to_png_bytes
from spectrum_builder.sampling import sample_energies
from spectrum_builder.spectrum import Normalization, build_spectrum_dataframe

IMAGE_BUILDER_NAME = "Created by Dr K. Karafasoulis"
IMAGE_BUILDER_URL = "http://karafasoulis.eu"

st.set_page_config(
    page_title="Isotope Gamma Spectrum Builder (NaI / CZT / HPGe)",
    page_icon="📊",
    layout="wide",
)

st.title("Isotope Gamma Spectrum Builder")

st.markdown(
    """
Signal events are generated from **Geant4 simulations** and  
background events are obtained by **resampling a real background spectrum**.

The application generates a **single combined spectrum** (signal + background per bin).
"""
)

# Load detector pools silently (do not report missing files)
detector_pools = {}
available_detectors = []

for det, cfg in DETECTOR_CONFIG.items():
    try:
        pools = load_detector_pools(
            det,
            signal_path=Path(cfg["signal_path"]),
            background_path=Path(cfg["background_path"]),
            info=cfg.get("info", det),
        )
        detector_pools[det] = pools
        available_detectors.append(det)
    except Exception:
        continue

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
normalization = (
    Normalization.RAW
    if norm_label == "Raw counts"
    else Normalization.CPS
    if norm_label == "Counts per second (cps)"
    else Normalization.UNIT_AREA
)

acq_time = None
if normalization == Normalization.CPS:
    acq_time = st.sidebar.number_input(
        "Acquisition time (s)",
        min_value=0.1,
        value=60.0,
        step=1.0,
    )

random_seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

# Build spectrum
st.header("Generated Spectrum")

sig_df = pools.signal_by_isotope.get(selected_isotope)
if sig_df is None or sig_df.empty:
    st.error(f"No signal events found for isotope '{selected_isotope}'.")
    st.stop()

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
    # Dynamic plot
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

# Downloads
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
    """
Created by **Dr K. Karafasoulis**  
http://karafasoulis.eu

Acknowledgements to Dr A. Kyriakis for the CZT data.  
http://ailab.inp.demokritos.gr/
"""
)
