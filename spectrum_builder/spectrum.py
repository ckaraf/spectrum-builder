"""
Spectrum construction engine.

Implements:
- Energy-range selection (auto or user-defined)
- Histogramming signal and background with identical bin edges
- Summation per bin to produce one combined spectrum
- Normalization options: raw / cps / unit-area

License: MIT
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd


class Normalization(str, Enum):
    RAW = "raw"
    CPS = "cps"
    UNIT_AREA = "unit_area"


def _auto_energy_range(signal_e: np.ndarray, bkg_e: np.ndarray) -> Tuple[float, float]:
    """Auto range based on sampled energies (rounded up to 10 keV)."""
    # Use max(initial=0.0) so empty arrays won't crash.
    max_e = float(max(signal_e.max(initial=0.0), bkg_e.max(initial=0.0)))
    if max_e <= 0:
        raise ValueError("No valid energies found in sampled events.")
    emax = float(np.ceil(max_e / 10.0) * 10.0)
    return 0.0, emax


def _histogram(energies: np.ndarray, *, n_bins: int, e_min: float, e_max: float):
    """Histogram energies; return counts, edges, centers."""
    edges = np.linspace(e_min, e_max, n_bins + 1)
    counts, edges = np.histogram(energies, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return counts, edges, centers


def _apply_normalization(
    total_counts: np.ndarray,
    *,
    normalization: Normalization,
    acquisition_time_s: Optional[float],
) -> Tuple[np.ndarray, str]:
    """Apply normalization and return (normalized_values, y_label)."""
    y = total_counts.astype(float)

    if normalization == Normalization.RAW:
        return y, "Counts"

    if normalization == Normalization.CPS:
        if acquisition_time_s is None or acquisition_time_s <= 0:
            raise ValueError("Acquisition time must be > 0 for CPS normalization.")
        return y / float(acquisition_time_s), "Counts/s"

    if normalization == Normalization.UNIT_AREA:
        s = float(y.sum())
        if s > 0:
            return y / s, "Normalized counts (Σ = 1)"
        # If all zeros, return as-is
        return y, "Normalized counts (Σ = 1)"

    # Defensive fallback
    return y, "Counts"


def build_spectrum_dataframe(
    *,
    signal_energies_keV: np.ndarray,
    background_energies_keV: np.ndarray,
    n_bins: int,
    e_min_keV: Optional[float] = None,
    e_max_keV: Optional[float] = None,
    normalization: Normalization = Normalization.RAW,
    acquisition_time_s: Optional[float] = None,
):
    """
    Build a combined spectrum (signal + background per bin).

    Parameters
    ----------
    signal_energies_keV:
        1D array of signal energies (keV), typically from Geant4 simulation.
    background_energies_keV:
        1D array of background energies (keV), resampled from real measurements.
    n_bins:
        Number of histogram bins (e.g., 1024/2048/4096).
    e_min_keV, e_max_keV:
        Energy bounds in keV. If either is None, an automatic range is used.
    normalization:
        RAW, CPS, or UNIT_AREA.
    acquisition_time_s:
        Only required for CPS normalization.

    Returns
    -------
    df, emin, emax, y_label
        df columns:
            Energy_keV, Signal_counts, Background_counts, Total_counts, Total_norm
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    # Range selection
    if e_min_keV is None or e_max_keV is None:
        emin, emax = _auto_energy_range(signal_energies_keV, background_energies_keV)
    else:
        emin = float(e_min_keV)
        emax = float(e_max_keV)
        if emax <= emin:
            raise ValueError("Upper energy must be greater than lower energy.")

    # Histogram both contributions with common bin edges
    sig_counts, edges, centers = _histogram(signal_energies_keV, n_bins=n_bins, e_min=emin, e_max=emax)
    bkg_counts, _, _ = _histogram(background_energies_keV, n_bins=n_bins, e_min=emin, e_max=emax)

    # Combine per bin -> single spectrum
    total_counts = sig_counts + bkg_counts

    # Normalize after combination
    total_norm, y_label = _apply_normalization(
        total_counts,
        normalization=normalization,
        acquisition_time_s=acquisition_time_s,
    )

    df = pd.DataFrame(
        {
            "Energy_keV": centers,
            "Signal_counts": sig_counts,
            "Background_counts": bkg_counts,
            "Total_counts": total_counts,
            "Total_norm": total_norm,
        }
    )
    return df, emin, emax, y_label
