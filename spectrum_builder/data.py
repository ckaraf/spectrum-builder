"""
Data loading utilities for detector event pools.

Expected CSV formats:

Signal CSV:
- Isotope, Event, EnergySmeared (keV)

Background CSV:
- Isotope, Event, Energy (keV)

Notes:
- "Event" is not used in histogramming but is kept for traceability.

License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

_SIGNAL_COLS = {"Isotope", "Event", "EnergySmeared"}
_BKG_COLS = {"Isotope", "Event", "Energy"}


@dataclass(frozen=True)
class DetectorPools:
    """Holds in-memory datasets for a single detector."""
    detector: str
    info: str
    signal_df: pd.DataFrame
    background_df: pd.DataFrame
    signal_by_isotope: Dict[str, pd.DataFrame]


def _validate_columns(df: pd.DataFrame, expected: set, kind: str) -> None:
    """Raise an error if required columns are missing."""
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{kind} CSV missing columns: {sorted(missing)}")


def load_signal_csv(path: Path) -> pd.DataFrame:
    """Load and validate the signal event pool CSV."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    _validate_columns(df, _SIGNAL_COLS, "Signal")
    return df


def load_background_csv(path: Path) -> pd.DataFrame:
    """Load and validate the background event pool CSV."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    _validate_columns(df, _BKG_COLS, "Background")
    return df


def index_signal_by_isotope(signal_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Pre-index signal events by isotope for fast sampling."""
    return {iso: sub_df for iso, sub_df in signal_df.groupby("Isotope")}


def load_detector_pools(
    detector: str,
    *,
    signal_path: Path,
    background_path: Path,
    info: str,
) -> DetectorPools:
    """Load one detector’s datasets and return a DetectorPools object."""
    sig_df = load_signal_csv(signal_path)
    bkg_df = load_background_csv(background_path)
    sig_by_iso = index_signal_by_isotope(sig_df)
    return DetectorPools(
        detector=detector,
        info=info,
        signal_df=sig_df,
        background_df=bkg_df,
        signal_by_isotope=sig_by_iso,
    )


def available_detectors(detector_config: Dict[str, dict]) -> List[str]:
    """
    Return detectors that can be loaded.

    Intentionally silent: does not report missing files.
    """
    ok: List[str] = []
    for det, cfg in detector_config.items():
        try:
            load_detector_pools(
                det,
                signal_path=Path(cfg["signal_path"]),
                background_path=Path(cfg["background_path"]),
                info=cfg.get("info", det),
            )
            ok.append(det)
        except Exception:
            continue
    return ok
