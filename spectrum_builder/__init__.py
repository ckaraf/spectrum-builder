"""
Isotope Spectrum Builder - Core Package

This package contains reusable, testable building blocks for:
- Loading detector event pools (signal/background) from CSV
- Sampling events with replacement
- Building binned spectra with configurable energy range and normalization
- Exporting spectra as CSV and PNG images

The Streamlit web UI in `app.py` uses this package as its backend.

License: MIT
"""

from .config import DETECTOR_CONFIG
from .data import DetectorPools, load_detector_pools, available_detectors
from .spectrum import Normalization, build_spectrum_dataframe
from .export import spectrum_to_csv_bytes, spectrum_to_png_bytes

__all__ = [
    "DETECTOR_CONFIG",
    "DetectorPools",
    "load_detector_pools",
    "available_detectors",
    "Normalization",
    "build_spectrum_dataframe",
    "spectrum_to_csv_bytes",
    "spectrum_to_png_bytes",
]
