"""Black-box extensions for gap-aware EWMA."""
import numpy as np
import pandas as pd

from src.features.weekly_features import _gap_aware_ewma_last


def test_single_observation_returns_itself():
    grp = pd.DataFrame({"season": [2024], "v": [17.5]})
    assert _gap_aware_ewma_last(grp, ["v"], halflife=3.0)["v"] == 17.5


def test_constant_series_returns_constant():
    grp = pd.DataFrame({"season": [2023, 2023, 2024, 2024], "v": [9.0] * 4})
    out = _gap_aware_ewma_last(grp, ["v"], halflife=3.0)
    assert abs(out["v"] - 9.0) < 1e-9


def test_result_bounded_by_min_max():
    """EWMA is a convex combination — output can't leave the observed range."""
    rng = np.random.default_rng(1)
    vals = rng.uniform(0, 40, 12)
    grp = pd.DataFrame({"season": [2023] * 6 + [2024] * 6, "v": vals})
    out = _gap_aware_ewma_last(grp, ["v"], halflife=2.5)["v"]
    assert vals.min() - 1e-9 <= out <= vals.max() + 1e-9


def test_huge_halflife_approaches_mean():
    grp = pd.DataFrame({"season": [2024] * 4, "v": [10.0, 20.0, 30.0, 40.0]})
    out = _gap_aware_ewma_last(grp, ["v"], halflife=1e6)["v"]
    assert abs(out - 25.0) < 0.01


def test_tiny_halflife_approaches_last_value():
    grp = pd.DataFrame({"season": [2024] * 4, "v": [10.0, 20.0, 30.0, 40.0]})
    out = _gap_aware_ewma_last(grp, ["v"], halflife=0.01)["v"]
    assert abs(out - 40.0) < 0.01


def test_multiple_columns_computed_independently():
    grp = pd.DataFrame({
        "season": [2024] * 3,
        "a": [10.0, 10.0, 10.0],
        "b": [0.0, 15.0, 30.0],
    })
    out = _gap_aware_ewma_last(grp, ["a", "b"], halflife=3.0)
    assert abs(out["a"] - 10.0) < 1e-9
    assert 0.0 < out["b"] < 30.0


def test_recent_games_weighted_more():
    rising = pd.DataFrame({"season": [2024] * 4, "v": [0.0, 0.0, 30.0, 30.0]})
    falling = pd.DataFrame({"season": [2024] * 4, "v": [30.0, 30.0, 0.0, 0.0]})
    v_rise = _gap_aware_ewma_last(rising, ["v"], halflife=3.0)["v"]
    v_fall = _gap_aware_ewma_last(falling, ["v"], halflife=3.0)["v"]
    assert v_rise > 15.0 > v_fall


def test_longer_offseason_gap_decays_more():
    """The farther back the prior season, the less its games matter."""
    values = [30.0, 30.0, 10.0, 10.0]
    one_gap = pd.DataFrame({"season": [2023, 2023, 2024, 2024], "v": values})
    long_gap = pd.DataFrame({"season": [2020, 2020, 2024, 2024], "v": values})
    v_one = _gap_aware_ewma_last(one_gap, ["v"], halflife=3.0)["v"]
    v_long = _gap_aware_ewma_last(long_gap, ["v"], halflife=3.0)["v"]
    assert v_long <= v_one
    assert v_long >= 10.0
