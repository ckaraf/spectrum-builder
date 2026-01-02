"""
Sampling utilities.

Sampling is performed WITH replacement so we can generate spectra with more
events than are present in the underlying CSV pools.

License: MIT
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def sample_energies(
    df: pd.DataFrame,
    energy_column: str,
    n: int,
    *,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample n energies from df[energy_column] WITH replacement.

    Returns a NumPy array of dtype float.
    """
    if n <= 0:
        return np.array([], dtype=float)

    # Pandas sampling gives a convenient reproducible API via random_state.
    sample = df.sample(n=n, replace=True, random_state=seed)
    return sample[energy_column].to_numpy(dtype=float)
