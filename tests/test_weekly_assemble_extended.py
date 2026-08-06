"""Black-box extensions for the weekly assembler.

Contracts from notes/weekly-pipeline-design.md:
- players on teams not playing the target week (bye) get NO row
- Vegas features from the target week's schedule row are allowed (pre-kickoff)
- rolling features come from prior weeks only
"""

import pytest

# §9.4: this module is pinned to the v1 config shape. api/, web/ and the
# weekly start/sit surface are FROZEN for the build window, so v1 stays
# alive underneath them and these tests are deselected from the default
# run rather than migrated. Thaw is a §22.2 follow-up.
pytestmark = pytest.mark.v1_frozen
import pandas as pd

from src.features.weekly_assemble import assemble_weekly_features


def _player_stats(entries):
    rows = []
    for pid, team, season, week, carries in entries:
        rows.append({
            "player_id": pid, "season": season, "week": week,
            "season_type": "REG", "team": team, "position": "RB",
            "carries": carries, "rushing_yards": carries * 4.0,
            "rushing_tds": 0, "targets": 5, "receptions": 3,
            "receiving_yards": 30.0, "receiving_tds": 0, "target_share": 0.15,
            "passing_yards": 0, "passing_tds": 0, "passing_interceptions": 0,
        })
    return pd.DataFrame(rows)


def _labels(entries):
    rows = []
    for pid, team, season, week, fp in entries:
        rows.append({"player_id": pid, "season": season, "week": week,
                     "position": "RB", "team": team, "fantasy_points": fp})
    return pd.DataFrame(rows)


def _schedules(games):
    return pd.DataFrame([
        {"season": s, "week": w, "game_type": "REG",
         "home_team": home, "away_team": away,
         "total_line": 48.0, "spread_line": -3.0}
        for s, w, home, away in games
    ])


def _frames(week_games, stats_entries, label_entries, rosters):
    return {
        "player_stats": _player_stats(stats_entries),
        "snap_counts": pd.DataFrame(),
        "schedules": _schedules(week_games),
        "weekly_labels": _labels(label_entries),
        "projections": pd.DataFrame(
            columns=["player_id", "season", "position", "season_p50"]),
        "rosters": pd.DataFrame(rosters),
        "pfr_to_gsis": {},
    }


def test_bye_week_player_excluded():
    """SEA doesn't play in week 3 -> its player gets no feature row."""
    games = ([(2024, w, "SF", "ARI") for w in (1, 2, 3)]
             + [(2024, w, "SEA", "LA") for w in (1, 2)])  # SEA idle week 3
    stats = ([("00-aaa", "SF", 2024, w, 10.0) for w in (1, 2)]
             + [("00-ccc", "SEA", 2024, w, 12.0) for w in (1, 2)])
    labels = ([("00-aaa", "SF", 2024, w, 12.0) for w in (1, 2)]
              + [("00-ccc", "SEA", 2024, w, 14.0) for w in (1, 2)]
              + [("00-bbb", "ARI", 2024, w, 8.0) for w in (1, 2)])
    rosters = [
        {"gsis_id": "00-aaa", "season": 2024, "team": "SF", "position": "RB"},
        {"gsis_id": "00-bbb", "season": 2024, "team": "ARI", "position": "RB"},
        {"gsis_id": "00-ccc", "season": 2024, "team": "SEA", "position": "RB"},
    ]
    result = assemble_weekly_features(2024, 3, _frames(games, stats, labels, rosters))
    assert "00-ccc" not in set(result["player_id"])
    assert "00-aaa" in set(result["player_id"])


def test_output_keyed_by_target_season_week():
    games = [(2024, w, "SF", "ARI") for w in (1, 2, 3)]
    stats = [("00-aaa", "SF", 2024, w, 10.0) for w in (1, 2)]
    labels = ([("00-aaa", "SF", 2024, w, 12.0) for w in (1, 2)]
              + [("00-bbb", "ARI", 2024, w, 8.0) for w in (1, 2)])
    rosters = [
        {"gsis_id": "00-aaa", "season": 2024, "team": "SF", "position": "RB"},
        {"gsis_id": "00-bbb", "season": 2024, "team": "ARI", "position": "RB"},
    ]
    result = assemble_weekly_features(2024, 3, _frames(games, stats, labels, rosters))
    assert not result.empty
    assert (result["season"] == 2024).all()
    assert (result["week"] == 3).all()


def test_vegas_features_present_for_target_week():
    """total_line 48 must reach the packed features as the game total."""
    games = [(2024, w, "SF", "ARI") for w in (1, 2, 3)]
    stats = [("00-aaa", "SF", 2024, w, 10.0) for w in (1, 2)]
    labels = ([("00-aaa", "SF", 2024, w, 12.0) for w in (1, 2)]
              + [("00-bbb", "ARI", 2024, w, 8.0) for w in (1, 2)])
    rosters = [
        {"gsis_id": "00-aaa", "season": 2024, "team": "SF", "position": "RB"},
        {"gsis_id": "00-bbb", "season": 2024, "team": "ARI", "position": "RB"},
    ]
    result = assemble_weekly_features(2024, 3, _frames(games, stats, labels, rosters))
    row = result[result["player_id"] == "00-aaa"].iloc[0]
    feats = row["features"]
    assert feats["wk_game_total"] == 48.0
    # implied totals of the two teams sum to the game total
    other = result[result["player_id"] == "00-bbb"].iloc[0]["features"]
    assert abs(feats["wk_implied_pts"] + other["wk_implied_pts"] - 48.0) < 1e-9


def test_rolling_stats_reflect_prior_weeks_only():
    """Player averaged 10 carries in weeks 1-2; the rolling feature for week 3
    must equal that (constant series EWMA)."""
    games = [(2024, w, "SF", "ARI") for w in (1, 2, 3)]
    stats = [("00-aaa", "SF", 2024, w, 10.0) for w in (1, 2)]
    labels = ([("00-aaa", "SF", 2024, w, 12.0) for w in (1, 2)]
              + [("00-bbb", "ARI", 2024, w, 8.0) for w in (1, 2)])
    rosters = [
        {"gsis_id": "00-aaa", "season": 2024, "team": "SF", "position": "RB"},
        {"gsis_id": "00-bbb", "season": 2024, "team": "ARI", "position": "RB"},
    ]
    result = assemble_weekly_features(2024, 3, _frames(games, stats, labels, rosters))
    feats = result[result["player_id"] == "00-aaa"].iloc[0]["features"]
    assert abs(feats["rolling_carries_pg"] - 10.0) < 1e-6


def test_week1_universe_from_rosters():
    """No current-season labels exist before week 1 — the roster supplies the
    player universe."""
    games = [(2023, 17, "SF", "ARI"), (2024, 1, "SF", "ARI")]
    stats = [("00-aaa", "SF", 2023, 17, 10.0)]
    labels = [("00-aaa", "SF", 2023, 17, 12.0), ("00-bbb", "ARI", 2023, 17, 8.0)]
    rosters = [
        {"gsis_id": "00-aaa", "season": 2024, "team": "SF", "position": "RB"},
        {"gsis_id": "00-bbb", "season": 2024, "team": "ARI", "position": "RB"},
    ]
    result = assemble_weekly_features(2024, 1, _frames(games, stats, labels, rosters))
    assert {"00-aaa", "00-bbb"} <= set(result["player_id"])
