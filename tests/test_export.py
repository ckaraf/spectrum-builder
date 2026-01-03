import numpy as np

from spectrum_builder.export import spectrum_to_csv_bytes, spectrum_to_png_bytes
from spectrum_builder.spectrum import build_spectrum_dataframe, Normalization


def test_export_csv_and_png_bytes():
    signal = np.array([100, 200, 300], dtype=float)
    background = np.array([50, 150], dtype=float)

    df, _, _, ylab = build_spectrum_dataframe(
        signal_energies_keV=signal,
        background_energies_keV=background,
        n_bins=16,
        e_min_keV=0,
        e_max_keV=500,
        normalization=Normalization.RAW,
        acquisition_time_s=None,
    )

    csv_bytes = spectrum_to_csv_bytes(df)
    assert isinstance(csv_bytes, (bytes, bytearray))
    assert b"Energy_keV" in csv_bytes  # header exists

    png_bytes = spectrum_to_png_bytes(
        df,
        title="Test",
        x_label="Energy (keV)",
        y_label=ylab,
        signature="Created by ... | Isotope: Test | http://example.com",
    )
    assert isinstance(png_bytes, (bytes, bytearray))
    assert png_bytes[:4] == b"\x89PNG"
