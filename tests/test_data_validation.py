import pandas as pd
import pytest

from spectrum_builder.data import load_signal_csv, load_background_csv


def test_signal_csv_missing_columns_raises(tmp_path):
    p = tmp_path / "bad_signal.csv"
    pd.DataFrame({"Isotope": ["Cs-137"], "EnergySmeared": [662.0]}).to_csv(p, index=False)  # missing Event
    with pytest.raises(ValueError):
        load_signal_csv(p)


def test_background_csv_missing_columns_raises(tmp_path):
    p = tmp_path / "bad_bkg.csv"
    pd.DataFrame({"Isotope": ["BKG"], "Energy": [100.0]}).to_csv(p, index=False)  # missing Event
    with pytest.raises(ValueError):
        load_background_csv(p)
