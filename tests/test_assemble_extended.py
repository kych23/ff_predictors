"""Black-box extensions for the season feature assembler.

Fixture mirrors tests/test_leakage.py but without poison: verifies feature
VALUES flow through (not just that leakage is blocked).
"""
import datetime as dt

import pandas as pd

from src.config import load_config
from src.features.assemble import assemble_season


def _frames(target_season: int):
    pid_a, pid_b = "00-aaa", "00-bbb"
    seasons_prior = [target_season - 2, target_season - 1]

    def weekly(pid, season, carries):
        return [{
            "player_id": pid, "season": season, "week": w, "team": "SF",
            "position": "RB", "carries": carries, "rushing_yards": carries * 4,
            "rushing_tds": 1, "targets": 2, "receptions": 2,
            "receiving_yards": carries, "receiving_tds": 0,
            "receiving_air_yards": carries, "attempts": 0, "completions": 0,
            "passing_yards": 0, "passing_tds": 0, "passing_interceptions": 0,
            "sacks_suffered": 0, "target_share": 0.1, "air_yards_share": 0.1,
        } for w in range(1, 9)]

    rows = []
    for s in seasons_prior:
        rows += weekly(pid_a, s, 10.0)
        rows += weekly(pid_b, s, 8.0)
    stats = pd.DataFrame(rows)

    players = pd.DataFrame([
        {"player_id": pid_a, "name": "A", "position": "RB",
         "birth_date": "1996-01-01", "rookie_year": target_season - 2,
         "draft_round": 1, "draft_pick": 5},
        {"player_id": pid_b, "name": "B", "position": "RB",
         "birth_date": "1997-01-01", "rookie_year": target_season - 2,
         "draft_round": 2, "draft_pick": 40},
    ])
    rosters = pd.DataFrame([
        {"gsis_id": pid, "season": s, "team": "SF", "position": "RB", "week": 1}
        for pid in (pid_a, pid_b)
        for s in (target_season, target_season - 1)
    ])
    depth = pd.DataFrame([
        {"gsis_id": pid_a, "season": target_season, "week": 1, "club_code": "SF",
         "position": "RB", "depth_team": 1},
        {"gsis_id": pid_b, "season": target_season, "week": 1, "club_code": "SF",
         "position": "RB", "depth_team": 2},
    ])
    return {
        "players": players,
        "id_map": pd.DataFrame(columns=["gsis_id", "pfr_id"]),
        "stats": stats, "snaps": pd.DataFrame(), "opp": pd.DataFrame(),
        "depth": depth, "rosters": rosters, "combine": pd.DataFrame(),
    }


def test_one_row_per_player():
    Y = 2022
    rows = assemble_season(Y, _frames(Y), cfg=load_config(),
                           as_of_date=dt.date(Y, 9, 1))
    assert len(rows) == 2
    assert set(rows["player_id"]) == {"00-aaa", "00-bbb"}


def test_prior_production_values_flow_through():
    """Two identical prior seasons of 10 (resp. 8) carries/game -> EWMA of a
    constant is that constant."""
    Y = 2022
    rows = assemble_season(Y, _frames(Y), cfg=load_config(),
                           as_of_date=dt.date(Y, 9, 1))
    feats = {r["player_id"]: r["features"] for _, r in rows.iterrows()}
    assert feats["00-aaa"]["prior_carries_per_game"] == 10.0
    assert feats["00-bbb"]["prior_carries_per_game"] == 8.0


def test_features_are_json_safe_dicts():
    import json
    Y = 2022
    rows = assemble_season(Y, _frames(Y), cfg=load_config(),
                           as_of_date=dt.date(Y, 9, 1))
    for f in rows["features"]:
        assert isinstance(f, dict)
        json.dumps(f)  # must not raise


def test_veterans_not_flagged_rookie():
    """rookie_year == target-2 -> third-year players, not rookies."""
    Y = 2022
    rows = assemble_season(Y, _frames(Y), cfg=load_config(),
                           as_of_date=dt.date(Y, 9, 1))
    for f in rows["features"]:
        if "is_rookie" in f:
            assert not f["is_rookie"]
