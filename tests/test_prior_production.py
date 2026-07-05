"""Tests for the cross-season prior-production feature block."""
import numpy as np
import pandas as pd

from src.features.prior_production import build_prior_production


def _weekly_rows(pid: str, season: int, carries: float, n_weeks: int = 16):
    return [{
        "player_id": pid, "season": season, "week": w,
        "carries": carries, "rushing_yards": carries * 4.0, "rushing_tds": 0,
        "targets": 4, "receptions": 3, "receiving_yards": 30.0,
        "receiving_tds": 0, "receiving_air_yards": 40.0,
        "attempts": 0, "completions": 0, "passing_yards": 0, "passing_tds": 0,
        "passing_interceptions": 0, "sacks_suffered": 0,
        "target_share": 0.15, "air_yards_share": 0.12,
    } for w in range(1, n_weeks + 1)]


def test_empty_input_returns_empty():
    out = build_prior_production(pd.DataFrame(), 2024)
    assert out.empty


def test_single_season_rates():
    stats = pd.DataFrame(_weekly_rows("00-a", 2023, carries=15.0))
    out = build_prior_production(stats, 2024)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["prior_carries_per_game"] == 15.0
    assert row["prior_yards_per_carry"] == 4.0
    assert row["seasons_of_history"] == 1
    assert row["last_season"] == 2023


def test_ewma_weights_recent_season_more():
    stats = pd.DataFrame(
        _weekly_rows("00-a", 2022, carries=10.0)
        + _weekly_rows("00-a", 2023, carries=20.0)
    )
    out = build_prior_production(stats, 2024, halflife=1.0)
    val = out.iloc[0]["prior_carries_per_game"]
    # EWMA of [10, 20] with halflife 1 -> closer to 20 than the plain mean
    assert 15.0 < val < 20.0


def test_zero_opportunity_rates_are_nan_not_inf():
    rows = _weekly_rows("00-b", 2023, carries=0.0)
    for r in rows:
        r["targets"] = 0
        r["receptions"] = 0
    stats = pd.DataFrame(rows)
    out = build_prior_production(stats, 2024)
    row = out.iloc[0]
    assert pd.isna(row["prior_yards_per_carry"]) or np.isfinite(row["prior_yards_per_carry"])
    assert not np.isinf(pd.to_numeric(row.drop(["player_id"]), errors="coerce")).any()


def test_target_share_stays_a_share():
    """target_share must remain a 0-1 share, never a row count (regression)."""
    stats = pd.DataFrame(_weekly_rows("00-a", 2023, carries=15.0))
    out = build_prior_production(stats, 2024)
    assert 0.0 <= out.iloc[0]["prior_target_share"] <= 1.0


def test_target_share_nan_when_column_missing():
    """When the source lacks target_share, the feature is NaN — not games count."""
    stats = pd.DataFrame(_weekly_rows("00-a", 2023, carries=15.0)).drop(
        columns=["target_share", "air_yards_share"])
    out = build_prior_production(stats, 2024)
    assert pd.isna(out.iloc[0]["prior_target_share"])
    assert pd.isna(out.iloc[0]["prior_air_yards_share"])
