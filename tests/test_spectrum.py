import numpy as np
import pytest

from spectrum_builder.spectrum import build_spectrum_dataframe, Normalization


def test_counts_conservation_raw():
    # energies fully inside [0, 500] so no event should be lost
    signal = np.array([100, 200, 300, 400], dtype=float)
    background = np.array([50, 150, 250], dtype=float)

    df, emin, emax, y_label = build_spectrum_dataframe(
        signal_energies_keV=signal,
        background_energies_keV=background,
        n_bins=10,
        e_min_keV=0,
        e_max_keV=500,
        normalization=Normalization.RAW,
        acquisition_time_s=None,
    )

    assert df.shape[0] == 10
    assert emin == 0.0
    assert emax == 500.0
    assert y_label == "Counts"

    # required columns from your implementation
    for col in ["Energy_keV", "Signal_counts", "Background_counts", "Total_counts", "Total_norm"]:
        assert col in df.columns

    # Total per bin must equal sum of components
    assert np.all(df["Total_counts"].to_numpy() ==
                  df["Signal_counts"].to_numpy() + df["Background_counts"].to_numpy())

    # Conservation of total counts (since all events are in range)
    assert int(df["Signal_counts"].sum()) == len(signal)
    assert int(df["Background_counts"].sum()) == len(background)
    assert int(df["Total_counts"].sum()) == len(signal) + len(background)


def test_unit_area_normalization():
    rng = np.random.default_rng(0)
    signal = rng.uniform(0, 1000, 1000)
    background = rng.uniform(0, 1000, 1000)

    df, _, _, y_label = build_spectrum_dataframe(
        signal_energies_keV=signal,
        background_energies_keV=background,
        n_bins=128,
        e_min_keV=0,
        e_max_keV=1000,
        normalization=Normalization.UNIT_AREA,
        acquisition_time_s=None,
    )

    assert np.isclose(df["Total_norm"].sum(), 1.0, atol=1e-12)
    assert "Σ = 1" in y_label


def test_cps_requires_acquisition_time():
    signal = np.array([100, 200], dtype=float)
    background = np.array([50], dtype=float)

    with pytest.raises(ValueError):
        build_spectrum_dataframe(
            signal_energies_keV=signal,
            background_energies_keV=background,
            n_bins=10,
            e_min_keV=0,
            e_max_keV=500,
            normalization=Normalization.CPS,
            acquisition_time_s=None,
        )


def test_cps_scaling_correct():
    signal = np.array([100, 200, 300], dtype=float)
    background = np.array([50, 150], dtype=float)

    df_raw, *_ = build_spectrum_dataframe(
        signal_energies_keV=signal,
        background_energies_keV=background,
        n_bins=10,
        e_min_keV=0,
        e_max_keV=500,
        normalization=Normalization.RAW,
        acquisition_time_s=None,
    )

    df_cps, *_ = build_spectrum_dataframe(
        signal_energies_keV=signal,
        background_energies_keV=background,
        n_bins=10,
        e_min_keV=0,
        e_max_keV=500,
        normalization=Normalization.CPS,
        acquisition_time_s=10.0,
    )

    # CPS should be RAW / time for each bin
    assert np.allclose(df_cps["Total_norm"].to_numpy(), df_raw["Total_counts"].to_numpy() / 10.0)


def test_invalid_energy_range_raises():
    signal = np.array([100], dtype=float)
    background = np.array([50], dtype=float)

    with pytest.raises(ValueError):
        build_spectrum_dataframe(
            signal_energies_keV=signal,
            background_energies_keV=background,
            n_bins=10,
            e_min_keV=500,
            e_max_keV=500,
            normalization=Normalization.RAW,
            acquisition_time_s=None,
        )
