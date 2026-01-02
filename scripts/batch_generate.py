#!/usr/bin/env python3
"""
Isotope Spectrum Builder - Batch Generator (CLI)

This script generates multiple spectra in batch mode using a jobs CSV file.

Why it exists
-------------
- The Streamlit app is great for interactive use.
- For dataset creation (ML training/validation, benchmarking, parameter sweeps),
  batch generation via command line is more efficient and reproducible.

Jobs CSV format
---------------
Required columns:
  detector,isotope,n_bins,n_signal,n_background

Optional columns:
  e_min_keV,e_max_keV,normalization,acq_time_s,seed,out_prefix

Allowed normalization values:
  raw, cps, unit_area

Output
------
For each valid job row:
  - <out>/<out_prefix>.csv   (binned spectrum)
  - <out>/<out_prefix>.png   (static image with attribution)

How to run
------
python scripts/batch_generate.py --jobs jobs.csv --out out/ --verbose


Notes
-----
- Sampling is done WITH replacement (supports > pool size).
- Detectors/isotopes that cannot be resolved are skipped (optionally reported in --verbose).
- This script is intentionally robust: it tries to keep going even if some rows are invalid.

License: MIT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# -------------------------------------------------------------------------
# Make imports robust:
# Add the repository root to sys.path so `import spectrum_builder` works
# regardless of how/where the script is invoked.
# -------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Now we can import the package modules safely.
from spectrum_builder.config import DETECTOR_CONFIG  # noqa: E402
from spectrum_builder.data import load_detector_pools  # noqa: E402
from spectrum_builder.export import spectrum_to_csv_bytes, spectrum_to_png_bytes  # noqa: E402
from spectrum_builder.sampling import sample_energies  # noqa: E402
from spectrum_builder.spectrum import Normalization, build_spectrum_dataframe  # noqa: E402

# Attribution for exported images
IMAGE_BUILDER_NAME = "Created by Dr K. Karafasoulis"
IMAGE_BUILDER_URL = "http://karafasoulis.eu"


def _err(msg: str, code: int = 1) -> None:
    """Print an error message to stderr and exit with a non-zero code."""
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def parse_norm(value: str) -> Normalization:
    """Parse normalization string into Normalization enum."""
    v = (value or "raw").strip().lower()
    if v == "raw":
        return Normalization.RAW
    if v == "cps":
        return Normalization.CPS
    if v in ("unit_area", "unit-area", "unitarea"):
        return Normalization.UNIT_AREA
    raise ValueError(f"Unknown normalization '{value}' (use raw, cps, unit_area).")


def load_jobs_csv(path: Path) -> pd.DataFrame:
    """Load jobs CSV with friendly errors."""
    if not path.exists():
        _err(f"Jobs file not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        _err(f"Failed to read jobs file '{path}': {e}")

    if df.empty:
        _err(f"Jobs file '{path}' is empty.")

    return df


def ensure_out_dir(out_dir: Path) -> None:
    """Create output directory with friendly errors."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        _err(f"Cannot create output directory '{out_dir}': {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch generate gamma spectra.")
    ap.add_argument("--jobs", required=True, type=Path, help="CSV file with spectrum jobs.")
    ap.add_argument("--out", required=True, type=Path, help="Output directory.")
    ap.add_argument("--verbose", action="store_true", help="Print skip reasons and progress.")
    args = ap.parse_args()

    jobs = load_jobs_csv(args.jobs)
    ensure_out_dir(args.out)

    required_cols = {"detector", "isotope", "n_bins", "n_signal", "n_background"}
    missing = required_cols - set(jobs.columns)
    if missing:
        _err(f"Jobs CSV missing required columns: {sorted(missing)}")

    # ---------------------------------------------------------------------
    # Load detector pools ONCE (fast). Detectors that fail to load are
    # silently skipped unless --verbose is enabled.
    # ---------------------------------------------------------------------
    pools_by_detector = {}
    for det, cfg in DETECTOR_CONFIG.items():
        try:
            pools_by_detector[det] = load_detector_pools(
                det,
                signal_path=Path(cfg["signal_path"]),
                background_path=Path(cfg["background_path"]),
                info=cfg.get("info", det),
            )
        except Exception as e:
            if args.verbose:
                print(f"[INFO] Detector '{det}' not loaded (skipping): {e}")

    if args.verbose:
        print("[INFO] Loaded detectors:", list(pools_by_detector.keys()))

    if not pools_by_detector:
        _err("No detector datasets could be loaded (check data paths / files).")

    created = 0
    skipped = 0

    # ---------------------------------------------------------------------
    # Process each row as one spectrum job
    # ---------------------------------------------------------------------
    for idx, row in jobs.iterrows():
        try:
            det = str(row["detector"]).strip()
            iso = str(row["isotope"]).strip()

            if det not in pools_by_detector:
                skipped += 1
                if args.verbose:
                    print(f"[SKIP row {idx}] Detector '{det}' not available. Available: {list(pools_by_detector.keys())}")
                continue

            pools = pools_by_detector[det]

            sig_df = pools.signal_by_isotope.get(iso)
            if sig_df is None or sig_df.empty:
                skipped += 1
                if args.verbose:
                    example = list(pools.signal_by_isotope.keys())[:20]
                    print(f"[SKIP row {idx}] Isotope '{iso}' not found for detector '{det}'. Example isotopes: {example}")
                continue

            # Required numeric fields
            n_bins = int(row["n_bins"])
            n_signal = int(row["n_signal"])
            n_background = int(row["n_background"])

            if n_bins <= 0:
                raise ValueError("n_bins must be > 0")
            if n_signal < 0 or n_background < 0:
                raise ValueError("n_signal and n_background must be >= 0")

            # Optional fields
            e_min = None
            e_max = None
            if "e_min_keV" in jobs.columns and pd.notna(row.get("e_min_keV")):
                e_min = float(row["e_min_keV"])
            if "e_max_keV" in jobs.columns and pd.notna(row.get("e_max_keV")):
                e_max = float(row["e_max_keV"])

            norm = Normalization.RAW
            if "normalization" in jobs.columns and pd.notna(row.get("normalization")):
                norm = parse_norm(str(row["normalization"]))

            acq_time = None
            if "acq_time_s" in jobs.columns and pd.notna(row.get("acq_time_s")):
                acq_time = float(row["acq_time_s"])

            # Seed: default depends on row index (so different rows differ deterministically)
            seed = 42 + int(idx)
            if "seed" in jobs.columns and pd.notna(row.get("seed")):
                seed = int(row["seed"])

            # Output prefix
            out_prefix = f"{idx:04d}_{iso}_{det}"
            if "out_prefix" in jobs.columns and pd.notna(row.get("out_prefix")):
                out_prefix = str(row["out_prefix"]).strip()

            # -------------------------------------------------------------
            # Sample events WITH replacement from the pools
            # -------------------------------------------------------------
            sig_e = sample_energies(sig_df, "EnergySmeared", n_signal, seed=seed)
            bkg_e = sample_energies(pools.background_df, "Energy", n_background, seed=seed + 1)

            # Build combined spectrum
            df, emin, emax, y_label = build_spectrum_dataframe(
                signal_energies_keV=sig_e,
                background_energies_keV=bkg_e,
                n_bins=n_bins,
                e_min_keV=e_min,
                e_max_keV=e_max,
                normalization=norm,
                acquisition_time_s=acq_time,
            )

            # -------------------------------------------------------------
            # Write outputs
            # -------------------------------------------------------------
            csv_path = args.out / f"{out_prefix}.csv"
            png_path = args.out / f"{out_prefix}.png"

            csv_path.write_bytes(spectrum_to_csv_bytes(df))

            signature = f"{IMAGE_BUILDER_NAME} | Isotope: {iso} | {IMAGE_BUILDER_URL}"
            png_bytes = spectrum_to_png_bytes(
                df,
                title=f"Combined Spectrum – {iso} ({det})",
                x_label="Energy (keV)",
                y_label=y_label,
                signature=signature,
            )
            png_path.write_bytes(png_bytes)

            created += 1
            if args.verbose:
                print(f"[OK row {idx}] Wrote {csv_path.name}, {png_path.name} | range {emin:.1f}-{emax:.1f} keV | norm={norm}")

        except Exception as e:
            skipped += 1
            if args.verbose:
                print(f"[SKIP row {idx}] {e}")
            continue

    print(f"Batch generation completed. Created: {created}, Skipped: {skipped}. Output dir: {args.out}")


if __name__ == "__main__":
    main()
