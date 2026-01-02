"""
Export helpers for spectra.

The Streamlit UI uses a dynamic plot for display.
This module provides a separate static PNG export for reporting.

License: MIT
"""

from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd


def spectrum_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize spectrum DataFrame to UTF-8 CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def spectrum_to_png_bytes(
    df: pd.DataFrame,
    *,
    title: str,
    x_label: str,
    y_label: str,
    signature: str,
    dpi: int = 150,
) -> bytes:
    """
    Render the spectrum to PNG bytes and include a signature text on the plot.

    signature example:
        "Created by Dr K. Karafasoulis | Isotope: Cs-137 | http://karafasoulis.eu"
    """
    fig, ax = plt.subplots()
    ax.plot(df["Energy_keV"], df["Total_norm"])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.text(
        0.99,
        0.01,
        signature,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
