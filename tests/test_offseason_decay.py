"""Tests for gap-aware EWMA: offseason boundaries decay prior-season signal."""
import numpy as np
import pandas as pd

from src.features.weekly_features import (
    _gap_aware_ewma_last,
    build_opponent_dvp,
    build_rolling_player_stats,
)


def _stats_rows(pid, season_weeks, carries_by_game):
    rows = []
    for (season, week), carries in zip(season_weeks, carries_by_game):
        rows.append({
            "player_id": pid, "season": season, "week": week,
            "carries": carries, "rushing_yards": carries * 4.0, "rushing_tds": 0,
            "targets": 4, "receptions": 3, "receiving_yards": 30.0,
            "receiving_tds": 0, "target_share": 0.15,
            "passing_yards": 0, "passing_tds": 0, "passing_interceptions": 0,
        })
    return pd.DataFrame(rows)


def test_no_gap_matches_plain_ewma_weights():
    """Within a single season the gap clock is the plain game counter."""
    grp = pd.DataFrame({"season": [2024] * 4, "v": [10.0, 20.0, 30.0, 40.0]})
    out = _gap_aware_ewma_last(grp, ["v"], halflife=3.0)
    dist = np.array([3.0, 2.0, 1.0, 0.0])
    w = 0.5 ** (dist / 3.0)
    expected = float((w * grp["v"].to_numpy()).sum() / w.sum())
    assert abs(out["v"] - expected) < 1e-9


def test_offseason_gap_downweights_prior_season():
    """Same game sequence, but crossing a season boundary pulls the estimate
    toward the current-season games."""
    values = [30.0, 30.0, 10.0, 10.0]
    same_season = pd.DataFrame({"season": [2024] * 4, "v": values})
    crosses_gap = pd.DataFrame({"season": [2023, 2023, 2024, 2024], "v": values})
    v_same = _gap_aware_ewma_last(same_season, ["v"], halflife=3.0)["v"]
    v_gap = _gap_aware_ewma_last(crosses_gap, ["v"], halflife=3.0)["v"]
    assert v_gap < v_same          # prior-season 30s matter less across the gap
    assert v_gap > 10.0            # but they still matter


def test_nan_values_ignored():
    grp = pd.DataFrame({"season": [2024] * 3, "v": [np.nan, 20.0, np.nan]})
    out = _gap_aware_ewma_last(grp, ["v"], halflife=3.0)
    assert out["v"] == 20.0


def test_all_nan_returns_nan():
    grp = pd.DataFrame({"season": [2024] * 2, "v": [np.nan, np.nan]})
    assert np.isnan(_gap_aware_ewma_last(grp, ["v"], halflife=3.0)["v"])


def test_rolling_stats_decay_across_seasons():
    """A back-loaded prior season influences Week 1 less than the same recent
    games would in-season."""
    late_prior = _stats_rows(
        "00-a",
        [(2023, 15), (2023, 16), (2023, 17), (2024, 1), (2024, 2)],
        [30, 30, 30, 10, 10])
    all_current = _stats_rows(
        "00-a",
        [(2024, 1), (2024, 2), (2024, 3), (2024, 4), (2024, 5)],
        [30, 30, 30, 10, 10])
    r_gap = build_rolling_player_stats(late_prior, pd.DataFrame(), {})
    r_flat = build_rolling_player_stats(all_current, pd.DataFrame(), {})
    assert r_gap.iloc[0]["rolling_carries_pg"] < r_flat.iloc[0]["rolling_carries_pg"]


def test_dvp_decays_across_seasons():
    """Defense FPA from last season is discounted at the start of a new one."""
    def labels(season_weeks, fps):
        rows = []
        for (season, week), fp in zip(season_weeks, fps):
            rows.append({"player_id": "00-x", "season": season, "week": week,
                         "position": "RB", "team": "SF", "fantasy_points": fp})
        return pd.DataFrame(rows)

    def sched(season_weeks):
        return pd.DataFrame([
            {"season": s, "week": w, "home_team": "SF", "away_team": "ARI"}
            for s, w in season_weeks
        ])

    sw_gap = [(2023, 16), (2023, 17), (2024, 1), (2024, 2)]
    sw_flat = [(2024, 1), (2024, 2), (2024, 3), (2024, 4)]
    fps = [30.0, 30.0, 10.0, 10.0]
    dvp_gap = build_opponent_dvp(labels(sw_gap, fps), sched(sw_gap), 2024, 3)
    dvp_flat = build_opponent_dvp(labels(sw_flat, fps), sched(sw_flat), 2024, 5)
    v_gap = dvp_gap[dvp_gap["position"] == "RB"]["dvp_fpa"].iloc[0]
    v_flat = dvp_flat[dvp_flat["position"] == "RB"]["dvp_fpa"].iloc[0]
    assert v_gap < v_flat
