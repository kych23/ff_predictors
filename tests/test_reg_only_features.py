"""Regular-season-only prior production (feature/label consistency).

Season labels are REG-only (filtered at ingest staging), so prior-production
features must exclude postseason games too — otherwise playoff-team players get
inflated game counts and skewed rate features. Poison-sentinel style: absurd
postseason stat lines must never move any output feature.
"""
import datetime as dt
import json

import pandas as pd

from src.config import load_config
from src.features.assemble import _reg_rows, assemble_season

SENTINEL = 777777.0


def _weekly(pid, season, weeks, val, season_type="REG"):
    return [{
        "player_id": pid, "season": season, "week": w, "team": "SF",
        "season_type": season_type,
        "position": "RB", "carries": val, "rushing_yards": val * 4,
        "rushing_tds": 1, "targets": 2, "receptions": 2, "receiving_yards": val,
        "receiving_tds": 0, "receiving_air_yards": val, "attempts": 0,
        "completions": 0, "passing_yards": 0, "passing_tds": 0,
        "passing_interceptions": 0, "sacks_suffered": 0, "target_share": 0.1,
        "air_yards_share": 0.1,
    } for w in weeks]


def test_postseason_rows_never_reach_season_features():
    Y = 2023
    pid = "00-aaa"
    rows = []
    for s in (Y - 2, Y - 1):
        rows += _weekly(pid, s, range(1, 12), 10.0, season_type="REG")
        # POISONED playoff rows in prior seasons — must be filtered out.
        rows += _weekly(pid, s, range(19, 22), SENTINEL, season_type="POST")
    stats = pd.DataFrame(rows)

    players = pd.DataFrame([
        {"player_id": pid, "name": "A", "position": "RB", "birth_date": "1996-01-01",
         "rookie_year": Y - 2, "draft_round": 1, "draft_pick": 5},
    ])
    rosters = pd.DataFrame([
        {"gsis_id": pid, "season": Y, "team": "SF", "position": "RB", "week": 1},
        {"gsis_id": pid, "season": Y - 1, "team": "SF", "position": "RB", "week": 1},
    ])
    frames = {
        "players": players, "id_map": pd.DataFrame(columns=["gsis_id", "pfr_id"]),
        "stats": stats, "snaps": pd.DataFrame(), "opp": pd.DataFrame(),
        "depth": pd.DataFrame(), "rosters": rosters, "combine": pd.DataFrame(),
    }
    out = assemble_season(Y, frames, cfg=load_config(), as_of_date=dt.date(Y, 9, 1))
    assert len(out) == 1
    blob = json.dumps(list(out["features"]))
    assert str(int(SENTINEL)) not in blob
    feats = out["features"].iloc[0]
    # 11 REG games at 10 carries each; playoff sentinel would blow this up
    assert feats["prior_carries_per_game"] < 100


def test_reg_rows_uses_type_column_when_present():
    df = pd.DataFrame({"season": [2022, 2022], "week": [1, 19],
                       "season_type": ["REG", "POST"], "v": [1, 2]})
    assert list(_reg_rows(df)["v"]) == [1]

    df2 = pd.DataFrame({"season": [2022, 2022], "week": [1, 20],
                        "game_type": ["REG", "POST"], "v": [1, 2]})
    assert list(_reg_rows(df2)["v"]) == [1]


def test_reg_rows_week_cutoff_fallback_is_era_aware():
    # no type column: 2021+ keeps weeks <= 18, earlier seasons <= 17
    df = pd.DataFrame({"season": [2022, 2022, 2019, 2019],
                       "week": [18, 19, 17, 18], "v": [1, 2, 3, 4]})
    assert list(_reg_rows(df)["v"]) == [1, 3]


def test_reg_rows_passthrough_without_columns():
    df = pd.DataFrame({"x": [1, 2]})
    assert _reg_rows(df) is df
    assert _reg_rows(pd.DataFrame()).empty
