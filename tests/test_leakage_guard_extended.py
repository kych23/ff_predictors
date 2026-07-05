"""Black-box extensions for the leakage guard helpers.

Contracts from notes/weekly-pipeline-design.md:
- prior_weeks: (season < target) OR (season == target AND week < target_week);
  supports custom season_col/week_col; standalone.
- assert_no_future_week: raises LeakageError on (season > target) OR
  (season == target AND week >= target_week).
- prior_seasons: strict '< target season'.
"""
import pandas as pd
import pytest

from src.features import leakage_guard as lg


def test_prior_seasons_strictly_excludes_target():
    df = pd.DataFrame({"season": [2020, 2021, 2022], "v": [1, 2, 3]})
    out = lg.prior_seasons(df, 2020)
    assert out.empty


def test_prior_seasons_keeps_all_older():
    df = pd.DataFrame({"season": [2015, 2018, 2021, 2024], "v": [1, 2, 3, 4]})
    out = lg.prior_seasons(df, 2025)
    assert len(out) == 4


def test_prior_seasons_does_not_mutate_input():
    df = pd.DataFrame({"season": [2020, 2021, 2022], "v": [1, 2, 3]})
    before = df.copy()
    lg.prior_seasons(df, 2021)
    pd.testing.assert_frame_equal(df, before)


def test_prior_weeks_boundary_week_excluded():
    df = pd.DataFrame({"season": [2024] * 3, "week": [1, 2, 3], "v": [1, 2, 3]})
    out = lg.prior_weeks(df, 2024, 3)
    assert set(out["week"]) == {1, 2}


def test_prior_weeks_future_season_excluded_entirely():
    df = pd.DataFrame({
        "season": [2023, 2024, 2025, 2025],
        "week": [17, 1, 1, 2],
        "v": [1, 2, 3, 4],
    })
    out = lg.prior_weeks(df, 2024, 2)
    assert set(zip(out["season"], out["week"])) == {(2023, 17), (2024, 1)}


def test_prior_weeks_custom_column_names():
    df = pd.DataFrame({"yr": [2023, 2024, 2024], "wk": [10, 1, 5], "v": [1, 2, 3]})
    out = lg.prior_weeks(df, 2024, 3, season_col="yr", week_col="wk")
    assert set(zip(out["yr"], out["wk"])) == {(2023, 10), (2024, 1)}


def test_prior_weeks_empty_frame():
    df = pd.DataFrame(columns=["season", "week", "v"])
    out = lg.prior_weeks(df, 2024, 5)
    assert out.empty


def test_assert_no_future_week_boundary_raises():
    # week == target_week counts as future (as-of-kickoff rule)
    df = pd.DataFrame({"season": [2024], "week": [3]})
    with pytest.raises(lg.LeakageError):
        lg.assert_no_future_week(df, 2024, 3)


def test_assert_no_future_week_future_season_raises():
    df = pd.DataFrame({"season": [2025], "week": [1]})
    with pytest.raises(lg.LeakageError):
        lg.assert_no_future_week(df, 2024, 10)


def test_assert_no_future_week_clean_passes():
    df = pd.DataFrame({"season": [2023, 2024], "week": [17, 2]})
    lg.assert_no_future_week(df, 2024, 3)  # must not raise


def test_assert_no_future_week_empty_passes():
    lg.assert_no_future_week(pd.DataFrame(columns=["season", "week"]), 2024, 1)


def test_assert_no_future_empty_passes():
    lg.assert_no_future(pd.DataFrame(columns=["season", "v"]), 2024)


def test_leakage_error_is_exception():
    assert issubclass(lg.LeakageError, Exception)


def test_guards_compose():
    """prior_weeks output always satisfies assert_no_future_week for any target."""
    df = pd.DataFrame({
        "season": [2022] * 5 + [2023] * 5 + [2024] * 5,
        "week": list(range(1, 6)) * 3,
        "v": range(15),
    })
    for wk in range(1, 6):
        clean = lg.prior_weeks(df, 2024, wk)
        lg.assert_no_future_week(clean, 2024, wk)
