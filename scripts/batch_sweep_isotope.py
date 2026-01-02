#!/usr/bin/env python3
"""
Batch sweep generator: many spectra of ONE isotope, with varying event statistics.

Creates N spectra for a fixed (detector, isotope).
For each spectrum:
- Choose n_signal uniformly from [min_signal, max_signal]
- Choose n_background uniformly from [min_background, max_background]
- Sample events WITH replacement from signal and background pools
- Histogram with common bin edges
- Combine signal+background per bin
- Apply normalization (raw / cps / unit_area)
- Save ALL spectra to ONE output CSV:
    one spectrum per line, with metadata columns + bin values

Optional:
- Write bin metadata (edges, centers, bin width) once to a separate CSV via --write-bins.
  The bins file begins with comment-based metadata lines (prefixed with '#').

Output spectra CSV format:
  Isotope,Detector,n_bins,e_min_keV,e_max_keV,n_signal,n_background,normalization,acq_time_s,seed,
  bin_0000,bin_0001,...,bin_{n_bins-1}

Bins CSV format (if --write-bins is used):
  # detector=NaI
  # isotope=Cs-137
  ...
  bin_index,e_low_keV,e_high_keV,e_center_keV,bin_width_keV

Example usage:

python scripts/batch_sweep_isotope.py \
  --detector NaI --isotope "Cs-137" --n-spectra 1000 \
  --n-bins 2048 \
  --min-signal 5000 --max-signal 50000 \
  --min-background 5000 --max-background 50000 \
  --normalization raw \
  --out out/cs137_sweep.csv \
  --write-bins out/cs137_bins.csv \
  --verbose


License: MIT
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# -------------------------------------------------------------------------
# Make imports robust:
# Add repository root to sys.path so `import spectrum_builder` works
# regardless of where this script is invoked from.
# -------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectrum_builder.config import DETECTOR_CONFIG  # noqa: E402
from spectrum_builder.data import load_detector_pools  # noqa: E402
from spectrum_builder.sampling import sample_energies  # noqa: E402
from spectrum_builder.spectrum import Normalization, build_spectrum_dataframe  # noqa: E402

IMAGE_BUILDER_NAME = "Created by Dr K. Karafasoulis"
IMAGE_BUILDER_URL = "http://karafasoulis.eu"


def _err(msg: str, code: int = 1) -> None:
    """Print error and exit with non-zero status."""
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def parse_norm(value: str) -> Normalization:
    """Parse normalization string into enum."""
    v = (value or "raw").strip().lower()
    if v == "raw":
        return Normalization.RAW
    if v == "cps":
        return Normalization.CPS
    if v in ("unit_area", "unit-area", "unitarea"):
        return Normalization.UNIT_AREA
    raise ValueError("normalization must be one of: raw, cps, unit_area")


def compute_bin_metadata(emin: float, emax: float, n_bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (e_low, e_high, centers, widths) arrays of length n_bins.
    """
    edges = np.linspace(float(emin), float(emax), int(n_bins) + 1)
    e_low = edges[:-1]
    e_high = edges[1:]
    centers = 0.5 * (e_low + e_high)
    widths = e_high - e_low
    return e_low, e_high, centers, widths


def write_bins_csv(
    path: Path,
    *,
    emin: float,
    emax: float,
    n_bins: int,
    detector: str,
    isotope: str,
    normalization: str,
    acq_time_s: Optional[float],
    seed: int,
) -> None:
    """
    Write bin edges + centers + widths once to a CSV file.

    The file begins with comment-based metadata lines (prefixed with '#'),
    followed by a standard CSV header.

    You can load it with pandas:
        pd.read_csv("bins.csv", comment="#")
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    e_low, e_high, centers, widths = compute_bin_metadata(emin, emax, n_bins)

    with path.open("w", newline="", encoding="utf-8") as f:
        # Human-readable metadata header
        f.write("# Isotope Spectrum Builder - bin metadata\n")
        f.write(f"# detector={detector}\n")
        f.write(f"# isotope={isotope}\n")
        f.write(f"# n_bins={int(n_bins)}\n")
        f.write(f"# e_min_keV={float(emin)}\n")
        f.write(f"# e_max_keV={float(emax)}\n")
        f.write(f"# normalization={normalization}\n")
        f.write(f"# acquisition_time_s={'' if acq_time_s is None else float(acq_time_s)}\n")
        f.write(f"# base_seed={int(seed)}\n")
        f.write(f"# builder={IMAGE_BUILDER_NAME}\n")
        f.write(f"# url={IMAGE_BUILDER_URL}\n")

        w = csv.writer(f)
        w.writerow(["bin_index", "e_low_keV", "e_high_keV", "e_center_keV", "bin_width_keV"])
        for i in range(int(n_bins)):
            w.writerow([i, float(e_low[i]), float(e_high[i]), float(centers[i]), float(widths[i])])


def determine_fixed_range(
    *,
    use_custom: bool,
    custom_emin: Optional[float],
    custom_emax: Optional[float],
    sig_df,
    bkg_df,
    n_bins: int,
    base_seed: int,
) -> Tuple[float, float]:
    """
    Determine a single (emin, emax) used for ALL spectra in the output file.

    Why fixed range?
    - You want one spectrum per line with consistent bin meanings across lines.
    - That requires identical energy bounds and binning for all spectra.

    Behavior:
    - If user provided both custom bounds, use them.
    - Otherwise, compute an automatic range from a small probe sample.
    """
    if use_custom:
        assert custom_emin is not None and custom_emax is not None
        return float(custom_emin), float(custom_emax)

    # Probe sampling to estimate energy extent for stable bin definitions
    probe_signal = sample_energies(sig_df, "EnergySmeared", 5000, seed=base_seed)
    probe_bkg = sample_energies(bkg_df, "Energy", 5000, seed=base_seed + 1)

    # One call to the engine with auto bounds returns the chosen emin/emax
    _df_probe, emin, emax, _ = build_spectrum_dataframe(
        signal_energies_keV=probe_signal,
        background_energies_keV=probe_bkg,
        n_bins=int(n_bins),
        e_min_keV=None,
        e_max_keV=None,
        normalization=Normalization.RAW,
        acquisition_time_s=None,
    )
    return float(emin), float(emax)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate multiple spectra for one isotope with random event counts in given ranges."
    )

    ap.add_argument("--detector", required=True, help="Detector name (NaI, CZT, HPGe).")
    ap.add_argument("--isotope", required=True, help="Isotope label EXACTLY as in the signal CSV (Isotope column).")
    ap.add_argument("--n-spectra", type=int, required=True, help="Number of spectra to generate.")
    ap.add_argument("--n-bins", type=int, default=2048, help="Number of bins (1024/2048/4096).")

    ap.add_argument("--min-signal", type=int, required=True, help="Minimum signal events per spectrum.")
    ap.add_argument("--max-signal", type=int, required=True, help="Maximum signal events per spectrum.")
    ap.add_argument("--min-background", type=int, required=True, help="Minimum background events per spectrum.")
    ap.add_argument("--max-background", type=int, required=True, help="Maximum background events per spectrum.")

    ap.add_argument("--normalization", default="raw", help="Normalization: raw, cps, unit_area.")
    ap.add_argument("--acq-time", type=float, default=None, help="Acquisition time (s) for cps normalization.")

    ap.add_argument("--e-min", type=float, default=None, help="Optional lower energy bound (keV).")
    ap.add_argument("--e-max", type=float, default=None, help="Optional upper energy bound (keV).")

    ap.add_argument("--seed", type=int, default=42, help="Base RNG seed for reproducibility.")
    ap.add_argument("--out", type=Path, required=True, help="Output CSV path for spectra (single file).")

    ap.add_argument(
        "--write-bins",
        type=Path,
        default=None,
        help="Optional output CSV path for bin metadata (edges/centers/widths).",
    )

    ap.add_argument("--verbose", action="store_true", help="Print progress info.")

    args = ap.parse_args()

    # ------------------------
    # Validate user arguments
    # ------------------------
    if args.n_spectra <= 0:
        _err("--n-spectra must be > 0")
    if args.n_bins <= 0:
        _err("--n-bins must be > 0")

    if args.min_signal < 0 or args.max_signal < 0 or args.min_background < 0 or args.max_background < 0:
        _err("min/max events must be >= 0")

    if args.max_signal < args.min_signal:
        _err("--max-signal must be >= --min-signal")
    if args.max_background < args.min_background:
        _err("--max-background must be >= --min-background")

    norm = parse_norm(args.normalization)

    if norm == Normalization.CPS and (args.acq_time is None or args.acq_time <= 0):
        _err("--acq-time must be provided and > 0 when normalization=cps")

    # Range flags
    use_custom_range = (args.e_min is not None) or (args.e_max is not None)
    if (args.e_min is None) ^ (args.e_max is None):
        _err("Provide BOTH --e-min and --e-max, or neither (for automatic range).")
    if args.e_min is not None and args.e_max is not None and args.e_max <= args.e_min:
        _err("--e-max must be > --e-min")

    det = args.detector.strip()
    if det not in DETECTOR_CONFIG:
        _err(f"Unknown detector '{det}'. Known detectors: {list(DETECTOR_CONFIG.keys())}")

    # ------------------------
    # Load detector datasets
    # ------------------------
    cfg = DETECTOR_CONFIG[det]
    try:
        pools = load_detector_pools(
            det,
            signal_path=Path(cfg["signal_path"]),
            background_path=Path(cfg["background_path"]),
            info=cfg.get("info", det),
        )
    except Exception:
        _err(f"Could not load datasets for detector '{det}'. Check data files/paths.")

    iso = args.isotope.strip()
    sig_df = pools.signal_by_isotope.get(iso)
    if sig_df is None or sig_df.empty:
        example = sorted(list(pools.signal_by_isotope.keys()))[:30]
        _err(f"Isotope '{iso}' not found for detector '{det}'. Example isotopes: {example}")

    # ------------------------
    # Determine fixed energy range for consistent bins across all spectra
    # ------------------------
    emin_fixed, emax_fixed = determine_fixed_range(
        use_custom=use_custom_range,
        custom_emin=args.e_min,
        custom_emax=args.e_max,
        sig_df=sig_df,
        bkg_df=pools.background_df,
        n_bins=args.n_bins,
        base_seed=args.seed,
    )

    # Optionally write bins metadata once
    if args.write_bins is not None:
        write_bins_csv(
            args.write_bins,
            emin=emin_fixed,
            emax=emax_fixed,
            n_bins=args.n_bins,
            detector=det,
            isotope=iso,
            normalization=norm.value,
            acq_time_s=args.acq_time,
            seed=args.seed,
        )
        if args.verbose:
            print(f"[INFO] Wrote bins metadata to: {args.write_bins}")

    # RNG controls event count selection
    rng = np.random.default_rng(args.seed)

    # Prepare output header (metadata + bins)
    meta_cols = [
        "Isotope",
        "Detector",
        "n_bins",
        "e_min_keV",
        "e_max_keV",
        "n_signal",
        "n_background",
        "normalization",
        "acq_time_s",
        "seed",
    ]
    bin_cols = [f"bin_{i:04d}" for i in range(args.n_bins)]
    header = meta_cols + bin_cols

    # Ensure output directory exists
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------
    # Write spectra (one per row)
    # ------------------------
    created = 0
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for k in range(args.n_spectra):
            # Randomly select event counts (inclusive)
            n_signal = int(rng.integers(args.min_signal, args.max_signal + 1))
            n_background = int(rng.integers(args.min_background, args.max_background + 1))

            # Deterministic per-spectrum seed
            seed_k = int(args.seed + k)

            # Sample energies with replacement
            sig_e = sample_energies(sig_df, "EnergySmeared", n_signal, seed=seed_k)
            bkg_e = sample_energies(pools.background_df, "Energy", n_background, seed=seed_k + 1)

            try:
                df, _emin_used, _emax_used, _y_label = build_spectrum_dataframe(
                    signal_energies_keV=sig_e,
                    background_energies_keV=bkg_e,
                    n_bins=int(args.n_bins),
                    e_min_keV=float(emin_fixed),
                    e_max_keV=float(emax_fixed),
                    normalization=norm,
                    acquisition_time_s=(float(args.acq_time) if args.acq_time is not None else None),
                )
            except Exception as e:
                if args.verbose:
                    print(f"[SKIP {k}] {e}")
                continue

            values = df["Total_norm"].to_numpy(dtype=float)
            if values.shape[0] != args.n_bins:
                if args.verbose:
                    print(f"[SKIP {k}] Unexpected bin count {values.shape[0]} != {args.n_bins}")
                continue

            row = [
                iso,
                det,
                int(args.n_bins),
                float(emin_fixed),
                float(emax_fixed),
                int(n_signal),
                int(n_background),
                norm.value,
                (float(args.acq_time) if args.acq_time is not None else ""),
                int(seed_k),
            ] + values.tolist()

            writer.writerow(row)
            created += 1

            if args.verbose and (created % max(1, args.n_spectra // 10) == 0):
                print(f"[INFO] Created {created}/{args.n_spectra}")

    print(f"Done. Wrote {created} spectra to: {args.out}")
    if args.write_bins is not None:
        print(f"Bins metadata: {args.write_bins}")


if __name__ == "__main__":
    main()

