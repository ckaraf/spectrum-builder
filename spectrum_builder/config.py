"""
Configuration for detector datasets.

Edit file paths here if you reorganize your repository.

Important:
- The Streamlit app will *silently hide* detectors whose datasets cannot be loaded.
- The CLI batch script also skips missing detectors silently.

License: MIT
"""

from pathlib import Path

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
