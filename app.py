import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO
import matplotlib.pyplot as plt

# --------------------------------------------------------
# CONFIG: PATHS TO CSV FILES PER DETECTOR
# --------------------------------------------------------
# Adjust filenames/paths as needed
DETECTOR_CONFIG = {
    "NaI": {
        "signal_path": Path("data/smeared_energies_NaI.csv"),
        "background_path": Path("data/background_sample_NaI.csv"),
        "info": "NaI(Tl) detector: 3'' × 3'' scintillation crystal.",
    },
    "CZT": {
        "signal_path": Path("data/smeared_energies_CZT.csv"),
        "background_path": Path("data/background_sample_CZT.csv"),
        "info": "CZT detector: 0.5 cm³ volume.",
    },
    "HPGe": {
        "signal_path": Path("data/smeared_energies_HPGe.csv"),
        "background_path": Path("data/background_sample_HPGe.csv"),
        "info": "HPGe detector: high-resolution semiconductor detector.",
    },
}

# Signature info for the downloadable image
IMAGE_BUILDER_NAME = "Gamma Spectrum Builder "
IMAGE_BUILDER_URL = "https://spectrum-builder.streamlit.app"

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

For each detector (NaI, CZT, HPGe), the app:
- Samples a chosen number of signal and background events (with replacement),
- Builds a binned energy spectrum (in keV),
- Sums signal and background per bin to form a **single combined spectrum**,  
- Optionally normalizes the spectrum (raw counts, counts/s, or unit area).
"""
)

# --------------------------------------------------------
# CACHED LOADERS
# --------------------------------------------------------
@st.cache_data
def load_signal(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError("Signal data not found.")
    df = pd.read_csv(path)
    expected = {"Isotope", "Event", "EnergySmeared"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError("Signal data format is invalid.")
    return df


@st.cache_data
def load_background(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError("Background data not found.")
    df = pd.read_csv(path)
    expected = {"Isotope", "Event", "Energy"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError("Background data format is invalid.")
    return df


@st.cache_data
def map_isotopes(df: pd.DataFrame, isotope_col: str):
    """Build a dict: isotope_name -> subset DataFrame for fast access."""
    return {iso: sub_df for iso, sub_df in df.groupby(isotope_col)}


# --------------------------------------------------------
# LOAD DATA FOR EACH DETECTOR (SILENTLY SKIP FAILURES)
# --------------------------------------------------------
detector_data = {}
available_detectors = []

for det_name, cfg in DETECTOR_CONFIG.items():
    sig_path = cfg["signal_path"]
    bkg_path = cfg["background_path"]
    try:
        sig_df = load_signal(str(sig_path))
        bkg_df = load_background(str(bkg_path))
        sig_by_iso = map_isotopes(sig_df, "Isotope")
        detector_data[det_name] = {
            "signal_df": sig_df,
            "background_df": bkg_df,
            "signal_by_isotope": sig_by_iso,
            "info": cfg["info"],
        }
        available_detectors.append(det_name)
    except Exception:
        # If anything fails for this detector, it simply won't appear in the UI
        continue

if not available_detectors:
    st.error("No detector data are currently available. Please contact the administrator.")
    st.stop()

# --------------------------------------------------------
# SIDEBAR: DETECTOR SELECTION AND PARAMETERS
# --------------------------------------------------------
st.sidebar.header("Detector Selection")
detector_type = st.sidebar.radio(
    "Detector",
    available_detectors,
    index=0,
)

active_data = detector_data[detector_type]
st.sidebar.info(active_data["info"])

# Isotope, binning, events, normalization
st.sidebar.header("Isotope & Binning")

active_signal_by_iso = active_data["signal_by_isotope"]
active_background_df = active_data["background_df"]

available_isotopes = sorted(active_signal_by_iso.keys())
if not available_isotopes:
    st.error(f"No isotopes available for detector {detector_type}.")
    st.stop()

selected_isotope = st.sidebar.selectbox(
    "Isotope",
    available_isotopes,
)

n_bins = st.sidebar.selectbox(
    "Number of bins",
    [1024, 2048, 4096],
    index=0,
)

# ----- NEW: Optional custom energy range -----
st.sidebar.header("Energy Range (optional)")
use_custom_range = st.sidebar.checkbox(
    "Use custom energy range (keV)", value=False
)

custom_Emin = None
custom_Emax = None
if use_custom_range:
    custom_Emin = st.sidebar.number_input(
        "Lower energy (keV)",
        min_value=0.0,
        value=0.0,
        step=10.0,
    )
    custom_Emax = st.sidebar.number_input(
        "Upper energy (keV)",
        min_value=0.1,
        value=3000.0,
        step=10.0,
    )

st.sidebar.header("Event Sampling")

n_signal = st.sidebar.number_input(
    "Number of signal events",
    min_value=0,
    value=5000,
    step=100,
)
n_background = st.sidebar.number_input(
    "Number of background events",
    min_value=0,
    value=5000,
    step=100,
)

st.sidebar.header("Normalization")

normalization = st.sidebar.radio(
    "Normalization type",
    ["Raw counts", "Counts per second (cps)", "Unit area (Σ = 1)"],
    index=0,
)

acq_time = None
if normalization == "Counts per second (cps)":
    acq_time = st.sidebar.number_input(
        "Acquisition time (s)",
        min_value=0.1,
        value=60.0,
        step=1.0,
        help="Used to convert total counts to counts per second.",
    )

random_seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    value=42,
    step=1,
)

# --------------------------------------------------------
# BUILD SPECTRUM
# --------------------------------------------------------
st.header("Generated Spectrum")

signal_iso_df = active_signal_by_iso.get(selected_isotope, None)
if signal_iso_df is None or signal_iso_df.empty:
    st.error(f"No signal events found for isotope '{selected_isotope}' for detector {detector_type}.")
    st.stop()

# Signal sampling (WITH REPLACEMENT ALWAYS)
if n_signal > 0:
    signal_sample = signal_iso_df.sample(
        n=n_signal,
        replace=True,
        random_state=random_seed,
    )
else:
    signal_sample = signal_iso_df.iloc[0:0]

# Background sampling (WITH REPLACEMENT ALWAYS)
if n_background > 0:
    background_sample = active_background_df.sample(
        n=n_background,
        replace=True,
        random_state=random_seed + 1,
    )
else:
    background_sample = active_background_df.iloc[0:0]

# Determine automatic energy range from data
max_signal_E = signal_sample["EnergySmeared"].max() if len(signal_sample) > 0 else 0.0
max_bkg_E = background_sample["Energy"].max() if len(background_sample) > 0 else 0.0
data_max_energy = float(max(max_signal_E, max_bkg_E))

if data_max_energy <= 0:
    st.error("No valid energies found in sampled events.")
    st.stop()

# Add a small margin to the automatic upper limit
auto_max_energy = np.ceil(data_max_energy / 10.0) * 10.0

# Decide final energy range
if use_custom_range:
    if custom_Emax is None or custom_Emin is None:
        st.error("Please specify both lower and upper energy when using a custom range.")
        st.stop()
    if custom_Emax <= custom_Emin:
        st.error("Upper energy must be greater than lower energy.")
        st.stop()
    e_min = float(custom_Emin)
    e_max = float(custom_Emax)
else:
    e_min = 0.0
    e_max = auto_max_energy

# Histogram edges and centers
bins = np.linspace(e_min, e_max, n_bins + 1)

signal_counts, edges = np.histogram(
    signal_sample["EnergySmeared"], bins=bins
)
background_counts, _ = np.histogram(
    background_sample["Energy"], bins=bins
)

total_counts = signal_counts + background_counts
bin_centers = 0.5 * (edges[:-1] + edges[1:])

spectrum_df = pd.DataFrame(
    {
        "Energy_keV": bin_centers,
        "Signal_counts": signal_counts,
        "Background_counts": background_counts,
        "Total_counts": total_counts,
    }
)

# --------------------------------------------------------
# APPLY NORMALIZATION
# --------------------------------------------------------
spectrum_df["Total_norm"] = spectrum_df["Total_counts"].astype(float)
y_label = "Counts"

if normalization == "Counts per second (cps)":
    if acq_time is not None and acq_time > 0:
        spectrum_df["Total_norm"] = spectrum_df["Total_counts"] / acq_time
        y_label = "Counts/s"
    else:
        st.warning("Acquisition time must be > 0 s for cps normalization. Using raw counts instead.")
elif normalization == "Unit area (Σ = 1)":
    total_sum = spectrum_df["Total_counts"].sum()
    if total_sum > 0:
        spectrum_df["Total_norm"] = spectrum_df["Total_counts"] / total_sum
        y_label = "Normalized counts (Σ = 1)"
    else:
        st.warning("Total counts are zero; unit-area normalization not applied.")

# --------------------------------------------------------
# DISPLAY (dynamic chart) + PREPARE IMAGE FOR DOWNLOAD
# --------------------------------------------------------
col_plot, col_stats = st.columns([3, 1])

with col_plot:
    st.subheader(f"Combined Spectrum – {selected_isotope} ({detector_type})")
    st.caption(
        f"Bins: {n_bins} | Energy range: {e_min:.1f} – {e_max:.1f} keV | Y: {y_label}"
    )
    # Dynamic chart
    st.line_chart(
        spectrum_df.set_index("Energy_keV")[["Total_norm"]]
    )

with col_stats:
    st.subheader("Summary")
    st.write(f"**Detector:** {detector_type}")
    st.write(f"**Isotope:** {selected_isotope}")
    st.write(f"**Signal events (with replacement):** {n_signal}")
    st.write(f"**Background events (with replacement):** {n_background}")
    st.write(f"**Number of bins:** {n_bins}")
    st.write(f"**Energy range:** {e_min:.1f} – {e_max:.1f} keV")
    st.write(f"**Normalization:** {normalization}")
    if normalization == "Counts per second (cps)" and acq_time is not None:
        st.write(f"**Acquisition time:** {acq_time:.1f} s")
    st.markdown("---")
    st.write("**Total counts (all bins, raw):**", int(total_counts.sum()))
    st.write("**Max counts in a bin (raw):**", int(total_counts.max()))

st.markdown("---")
st.subheader("Binned Spectrum Data (preview)")
st.dataframe(spectrum_df.head(20))

# --------------------------------------------------------
# PREPARE MATPLOTLIB FIGURE *ONLY* FOR DOWNLOAD
# --------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(spectrum_df["Energy_keV"], spectrum_df["Total_norm"])
ax.set_xlabel("Energy (keV)")
ax.set_ylabel(y_label)
ax.set_title(f"Combined Spectrum – {selected_isotope} ({detector_type})")

# Signature including isotope and URL
signature_text = (
    f"{IMAGE_BUILDER_NAME} | Isotope: {selected_isotope} | {IMAGE_BUILDER_URL}"
)
ax.text(
    0.99,
    0.01,
    signature_text,
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=8,
)

fig.tight_layout()

# --------------------------------------------------------
# DOWNLOAD BUTTONS
# --------------------------------------------------------
st.subheader("Download Spectrum")

# CSV download
csv_bytes = spectrum_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download spectrum CSV",
    data=csv_bytes,
    file_name=f"spectrum_{selected_isotope}_{detector_type}.csv",
    mime="text/csv",
)

# Image download
img_buffer = BytesIO()
fig.savefig(img_buffer, format="png", dpi=150, bbox_inches="tight")
img_buffer.seek(0)

st.download_button(
    label="Download spectrum image (PNG)",
    data=img_buffer,
    file_name=f"spectrum_{selected_isotope}_{detector_type}.png",
    mime="image/png",
)

st.markdown("---")
st.markdown(
    """
Created by **Dr K. Karafasoulis**  
[http://karafasoulis.eu](http://karafasoulis.eu)

Acknowledgements to Dr A. Kyriakis for the CZT data.
[http://ailab.inp.demokritos.gr/](http://ailab.inp.demokritos.gr/)

"""
)
