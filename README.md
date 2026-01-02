# Isotope Spectrum Builder

Generate **detector-aware gamma-ray spectra** by combining:

- **Signal events from Geant4 simulations**
- **Background events from resampling a real measured background spectrum**

Live demo: https://spectrum-builder.streamlit.app

## Features

- Detector selection: **NaI(Tl)**, **CZT**, **HPGe** (shown only if datasets exist)
- Isotope selection (auto-populated from datasets)
- Binning: 1024 / 2048 / 4096
- Optional custom energy range \([E_{min}, E_{max}]\) in keV
- Sampling **with replacement** (supports more events than in the dataset)
- Normalization:
  - raw counts
  - counts per second (cps)
  - unit-area (Σ = 1)
- Exports:
  - CSV spectrum
  - PNG image with embedded attribution and isotope name

## Installation (local)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
