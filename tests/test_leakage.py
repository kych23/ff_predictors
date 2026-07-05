"""Leakage gate (DrafterSpec.md §4.5 — non-negotiable).

Runs the feature pipeline on synthetic fixtures with a future-season value set to a
detectable sentinel and asserts it never appears in any output feature. Verifies the
CODE PATH, not stored data.
"""
import datetime as dt
import json

import pandas as pd
import pytest

from src.config import load_config
from src.features import leakage_guard as lg
from src.features.assemble import assemble_season

SENTINEL = 999999.0


def _synthetic_frames(target_season: int):
    """Two players with prior + current(poisoned) seasons across all sources."""
    pid_a, pid_b = "00-aaa", "00-bbb"
    seasons_prior = [target_season - 2, target_season - 1]

    def weekly(pid, season, val):
        return [{
            "player_id": pid, "season": season, "week": w, "team": "SF",
            "position": "RB", "carries": val, "rushing_yards": val * 4,
            "rushing_tds": 1, "targets": 2, "receptions": 2, "receiving_yards": val,
            "receiving_tds": 0, "receiving_air_yards": val, "attempts": 0,
            "completions": 0, "passing_yards": 0, "passing_tds": 0,
            "passing_interceptions": 0, "sacks_suffered": 0, "target_share": 0.1,
            "air_yards_share": 0.1,
        } for w in range(1, 9)]

    rows = []
    for s in seasons_prior:
        rows += weekly(pid_a, s, 10.0)
        rows += weekly(pid_b, s, 8.0)
    # POISONED current-season rows — must be excluded by the leakage filter.
    rows += weekly(pid_a, target_season, SENTINEL)
    rows += weekly(pid_b, target_season, SENTINEL)
    stats = pd.DataFrame(rows)

    players = pd.DataFrame([
        {"player_id": pid_a, "name": "A", "position": "RB", "birth_date": "1996-01-01",
         "rookie_year": target_season - 2, "draft_round": 1, "draft_pick": 5},
        {"player_id": pid_b, "name": "B", "position": "RB", "birth_date": "1997-01-01",
         "rookie_year": target_season - 2, "draft_round": 2, "draft_pick": 40},
    ])
    rosters = pd.DataFrame([
        {"gsis_id": pid_a, "season": target_season, "team": "SF", "position": "RB", "week": 1},
        {"gsis_id": pid_b, "season": target_season, "team": "SF", "position": "RB", "week": 1},
        {"gsis_id": pid_a, "season": target_season - 1, "team": "SF", "position": "RB", "week": 1},
        {"gsis_id": pid_b, "season": target_season - 1, "team": "SF", "position": "RB", "week": 1},
    ])
    depth = pd.DataFrame([
        {"gsis_id": pid_a, "season": target_season, "week": 1, "club_code": "SF",
         "position": "RB", "depth_team": 1},
        {"gsis_id": pid_b, "season": target_season, "week": 1, "club_code": "SF",
         "position": "RB", "depth_team": 2},
    ])
    frames = {
        "players": players, "id_map": pd.DataFrame(columns=["gsis_id", "pfr_id"]),
        "stats": stats, "snaps": pd.DataFrame(), "opp": pd.DataFrame(),
        "depth": depth, "rosters": rosters, "combine": pd.DataFrame(),
    }
    return frames


def test_sentinel_never_reaches_output():
    Y = 2022
    frames = _synthetic_frames(Y)
    rows = assemble_season(Y, frames, cfg=load_config(), as_of_date=dt.date(Y, 9, 1))
    assert len(rows) == 2
    blob = json.dumps([r for r in rows["features"]])
    assert str(int(SENTINEL)) not in blob
    assert str(SENTINEL) not in blob
    # prior-only values should reflect 10/8 carries, never the sentinel
    for f in rows["features"]:
        assert f["prior_carries_per_game"] is None or f["prior_carries_per_game"] < 100


def test_prior_seasons_filter_strict():
    df = pd.DataFrame({"season": [2019, 2020, 2021, 2022], "v": [1, 2, 3, 4]})
    out = lg.prior_seasons(df, 2021)
    assert set(out["season"]) == {2019, 2020}


def test_assert_no_future_raises():
    df = pd.DataFrame({"season": [2020, 2021, 2022], "v": [1, 2, 3]})
    with pytest.raises(lg.LeakageError):
        lg.assert_no_future(df, 2021)
    # clean frame does not raise
    lg.assert_no_future(lg.prior_seasons(df, 2021), 2021)


def test_preseason_rows_week1_only():
    df = pd.DataFrame({"season": [2022, 2022, 2022], "week": [1, 2, 5], "v": [1, 2, 3]})
    out = lg.preseason_rows(df, 2022)
    assert list(out["week"]) == [1]


def test_adp_wall_no_imports():
    """src/projection/ and src/features/ must never import src/ingest/adp."""
    import ast
    from pathlib import Path

    for d in [Path("src/projection"), Path("src/features")]:
        for f in d.rglob("*.py"):
            tree = ast.parse(f.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert "ingest.adp" not in node.module, \
                        f"ADP wall violated: {f} imports {node.module}"
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert "ingest.adp" not in alias.name, \
                            f"ADP wall violated: {f} imports {alias.name}"
