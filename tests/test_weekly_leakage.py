"""Poison test for the as-of-kickoff weekly leakage rule.

Mirrors test_leakage.py but at weekly grain: injects a SENTINEL value into
future-week data and asserts it never reaches any output feature.
"""

import pytest

# §9.4: this module is pinned to the v1 config shape. api/, web/ and the
# weekly start/sit surface are FROZEN for the build window, so v1 stays
# alive underneath them and these tests are deselected from the default
# run rather than migrated. Thaw is a §22.2 follow-up.
pytestmark = pytest.mark.v1_frozen
import json

import numpy as np
import pandas as pd
import pytest

from src.features import leakage_guard as lg
from src.features.weekly_features import (
    build_opponent_dvp,
    build_rolling_player_stats,
)
from src.features.weekly_assemble import assemble_weekly_features

SENTINEL = 99999.0


def _make_player_stats(seasons_weeks, poison_week=None, poison_season=None):
    rows = []
    for season, week in seasons_weeks:
        val = SENTINEL if (season == poison_season and week == poison_week) else 10.0
        rows.append({
            "player_id": "00-aaa", "season": season, "week": week,
            "season_type": "REG", "team": "SF", "position": "RB",
            "carries": val, "rushing_yards": val * 4, "rushing_tds": 1,
            "targets": 5, "receptions": 3, "receiving_yards": val,
            "receiving_tds": 0, "target_share": 0.15,
            "passing_yards": 0, "passing_tds": 0, "passing_interceptions": 0,
        })
    return pd.DataFrame(rows)


def _make_weekly_labels(seasons_weeks, poison_week=None, poison_season=None):
    rows = []
    for season, week in seasons_weeks:
        fp = SENTINEL if (season == poison_season and week == poison_week) else 12.0
        rows.append({
            "player_id": "00-aaa", "season": season, "week": week,
            "position": "RB", "team": "SF", "fantasy_points": fp,
        })
        rows.append({
            "player_id": "00-bbb", "season": season, "week": week,
            "position": "RB", "team": "ARI", "fantasy_points": 8.0,
        })
    return pd.DataFrame(rows)


def _make_schedules(seasons_weeks):
    rows = []
    for season, week in seasons_weeks:
        rows.append({
            "season": season, "week": week, "game_type": "REG",
            "home_team": "SF", "away_team": "ARI",
            "total_line": 48.0, "spread_line": -3.0,
        })
    return pd.DataFrame(rows)


def test_prior_weeks_filters_correctly():
    all_weeks = [(2023, w) for w in range(1, 18)] + [(2024, w) for w in range(1, 5)]
    df = pd.DataFrame([{"season": s, "week": w, "v": 1} for s, w in all_weeks])
    out = lg.prior_weeks(df, target_season=2024, target_week=3)
    assert set(out[out["season"] == 2024]["week"]) == {1, 2}
    assert len(out[out["season"] == 2023]) == 17


def test_prior_weeks_week1_edge():
    all_weeks = [(2023, w) for w in range(1, 18)] + [(2024, 1)]
    df = pd.DataFrame([{"season": s, "week": w, "v": 1} for s, w in all_weeks])
    out = lg.prior_weeks(df, target_season=2024, target_week=1)
    assert len(out[out["season"] == 2024]) == 0
    assert len(out[out["season"] == 2023]) == 17


def test_assert_no_future_week_raises():
    df = pd.DataFrame([
        {"season": 2024, "week": 1}, {"season": 2024, "week": 2},
        {"season": 2024, "week": 3}, {"season": 2024, "week": 4},
    ])
    with pytest.raises(lg.LeakageError):
        lg.assert_no_future_week(df, 2024, 3)
    clean = lg.prior_weeks(df, 2024, 3)
    lg.assert_no_future_week(clean, 2024, 3)


def test_rolling_stats_no_leakage():
    weeks = [(2024, w) for w in range(1, 5)]
    stats = _make_player_stats(weeks, poison_week=3, poison_season=2024)
    filtered = lg.prior_weeks(stats, 2024, 3)
    filtered = filtered[filtered["season_type"] == "REG"]
    result = build_rolling_player_stats(filtered, pd.DataFrame(), {})
    for col in ["rolling_fppg", "rolling_target_share", "rolling_carries_pg"]:
        if col in result.columns:
            vals = result[col].dropna()
            assert (vals < SENTINEL).all(), f"SENTINEL leaked into {col}"


def test_dvp_no_leakage():
    weeks = [(2024, w) for w in range(1, 5)]
    labels = _make_weekly_labels(weeks, poison_week=3, poison_season=2024)
    schedules = _make_schedules(weeks)
    filtered_labels = lg.prior_weeks(labels, 2024, 3)
    result = build_opponent_dvp(filtered_labels, schedules, 2024, 3)
    if not result.empty:
        assert (result["dvp_fpa"] < SENTINEL).all(), "SENTINEL leaked into dvp_fpa"


def test_assembler_end_to_end_no_leakage():
    weeks_prior = [(2023, w) for w in range(1, 5)]
    weeks_current = [(2024, w) for w in range(1, 5)]
    all_weeks = weeks_prior + weeks_current

    player_stats = _make_player_stats(all_weeks, poison_week=3, poison_season=2024)
    weekly_labels = _make_weekly_labels(all_weeks, poison_week=3, poison_season=2024)
    schedules = _make_schedules(all_weeks)

    rosters = pd.DataFrame([
        {"gsis_id": "00-aaa", "season": 2024, "team": "SF", "position": "RB"},
        {"gsis_id": "00-bbb", "season": 2024, "team": "ARI", "position": "RB"},
    ])

    frames = {
        "player_stats": player_stats,
        "snap_counts": pd.DataFrame(),
        "schedules": schedules,
        "weekly_labels": weekly_labels,
        "projections": pd.DataFrame(columns=["player_id", "season", "position", "season_p50"]),
        "rosters": rosters,
        "pfr_to_gsis": {},
    }

    result = assemble_weekly_features(2024, 3, frames)
    if not result.empty:
        blob = json.dumps([r for r in result["features"]])
        assert str(int(SENTINEL)) not in blob
        assert str(SENTINEL) not in blob
