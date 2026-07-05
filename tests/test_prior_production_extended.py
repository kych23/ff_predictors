"""Black-box extensions for cross-season prior production features.

Leakage contract: only seasons strictly before the target season may
contribute ("prior" production; pre-season data only per CLAUDE.md).
"""
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


def test_target_season_rows_dropped_without_guard():
    """Defense in depth: even when the caller forgets leakage_guard, the
    builder itself must drop target-season rows."""
    stats = pd.DataFrame(
        _weekly_rows("00-a", 2023, carries=10.0)
        + _weekly_rows("00-a", 2024, carries=99.0)  # target-season poison
    )
    out = build_prior_production(stats, 2024)
    assert len(out) == 1
    assert out.iloc[0]["prior_carries_per_game"] == 10.0
    assert out.iloc[0]["last_season"] == 2023


def test_future_season_rows_dropped_without_guard():
    stats = pd.DataFrame(
        _weekly_rows("00-a", 2023, carries=10.0)
        + _weekly_rows("00-a", 2025, carries=99.0)  # future poison
    )
    out = build_prior_production(stats, 2024)
    assert len(out) == 1
    assert out.iloc[0]["prior_carries_per_game"] == 10.0


def test_only_target_season_rows_yields_empty():
    stats = pd.DataFrame(_weekly_rows("00-a", 2024, carries=99.0))
    out = build_prior_production(stats, 2024)
    assert out.empty


def test_target_season_rows_never_contribute_after_guard():
    """Contract: callers pre-filter via leakage_guard.prior_seasons (the
    assembler-level poison test enforces this). Composed with the guard,
    in-season rows can never reach the features."""
    from src.features.leakage_guard import prior_seasons

    stats = pd.DataFrame(
        _weekly_rows("00-a", 2023, carries=10.0)
        + _weekly_rows("00-a", 2024, carries=99.0)  # target-season poison
    )
    out = build_prior_production(prior_seasons(stats, 2024), 2024)
    assert len(out) == 1
    assert out.iloc[0]["prior_carries_per_game"] == 10.0


def test_future_season_rows_never_contribute_after_guard():
    from src.features.leakage_guard import prior_seasons

    stats = pd.DataFrame(
        _weekly_rows("00-a", 2023, carries=10.0)
        + _weekly_rows("00-a", 2025, carries=99.0)  # future relative to target
    )
    out = build_prior_production(prior_seasons(stats, 2024), 2024)
    assert len(out) == 1
    assert out.iloc[0]["prior_carries_per_game"] == 10.0
    assert out.iloc[0]["last_season"] == 2023


def test_two_players_two_rows():
    stats = pd.DataFrame(
        _weekly_rows("00-a", 2023, carries=15.0)
        + _weekly_rows("00-b", 2023, carries=5.0)
    )
    out = build_prior_production(stats, 2024).set_index("player_id")
    assert len(out) == 2
    assert out.loc["00-a", "prior_carries_per_game"] == 15.0
    assert out.loc["00-b", "prior_carries_per_game"] == 5.0


def test_seasons_of_history_counts_distinct_prior_seasons():
    stats = pd.DataFrame(
        _weekly_rows("00-a", 2021, carries=10.0)
        + _weekly_rows("00-a", 2022, carries=12.0)
        + _weekly_rows("00-a", 2023, carries=14.0)
    )
    out = build_prior_production(stats, 2024)
    assert out.iloc[0]["seasons_of_history"] == 3


def test_huge_halflife_approaches_plain_mean():
    stats = pd.DataFrame(
        _weekly_rows("00-a", 2022, carries=10.0)
        + _weekly_rows("00-a", 2023, carries=20.0)
    )
    out = build_prior_production(stats, 2024, halflife=1000.0)
    val = out.iloc[0]["prior_carries_per_game"]
    assert abs(val - 15.0) < 0.1


def test_ewma_bounded_by_observed_seasons():
    stats = pd.DataFrame(
        _weekly_rows("00-a", 2022, carries=10.0)
        + _weekly_rows("00-a", 2023, carries=20.0)
    )
    out = build_prior_production(stats, 2024, halflife=2.0)
    val = out.iloc[0]["prior_carries_per_game"]
    assert 10.0 <= val <= 20.0


def test_last_season_is_most_recent_prior():
    stats = pd.DataFrame(
        _weekly_rows("00-a", 2019, carries=10.0)
        + _weekly_rows("00-a", 2022, carries=12.0)
    )
    out = build_prior_production(stats, 2024)
    assert out.iloc[0]["last_season"] == 2022
