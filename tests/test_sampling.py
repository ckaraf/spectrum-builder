import pandas as pd

from spectrum_builder.sampling import sample_energies


def test_sampling_with_replacement_allows_more_than_pool():
    df = pd.DataFrame({"EnergySmeared": [100.0, 200.0, 300.0]})
    x = sample_energies(df, "EnergySmeared", 100, seed=1)
    assert len(x) == 100


def test_sampling_reproducible_with_seed():
    df = pd.DataFrame({"EnergySmeared": [100.0, 200.0, 300.0]})
    a = sample_energies(df, "EnergySmeared", 10, seed=123).tolist()
    b = sample_energies(df, "EnergySmeared", 10, seed=123).tolist()
    assert a == b
