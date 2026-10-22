import pandas as pd

from src.features import _longest_run, _season_window, add_history


def test_season_windows():
    s, e = _season_window("Maha", 2020)
    assert (s.month, s.year, e.month, e.year) == (10, 2020, 3, 2021)
    s, e = _season_window("Yala", 2021)
    assert (s.month, e.month) == (4, 9)


def test_longest_dry_spell():
    assert _longest_run([True, True, False, True, True, True]) == 3


def test_lags_only_look_backwards():
    df = pd.DataFrame({"district": ["A"] * 4, "season": ["Maha"] * 4, "season_year": [2001, 2002, 2003, 2004],
                       "yield_kg_ha": [3000.0, 3100.0, 3200.0, 3300.0], "sown_ha": [10.0] * 4})
    out = add_history(df).sort_values("season_year")
    assert out["prev_yield_same"].tolist()[1:] == [3000.0, 3100.0, 3200.0]
    assert pd.isna(out["prev_yield_same"].iloc[0])
