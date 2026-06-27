"""Greedy lineup-fill replacement levels + flex allocation (DrafterSpec.md §4.8.1)."""
import pandas as pd

from src.config import load_config
from src.recommender.replacement import compute_replacement


def _board():
    rows = []
    # RB values strictly above WR above TE, so all 12 flex slots go to RBs.
    for i in range(40):
        rows.append({"player_id": f"RB{i}", "position": "RB", "value": 100 - i})
    for i in range(40):
        rows.append({"player_id": f"WR{i}", "position": "WR", "value": 50 - i})
    for i in range(20):
        rows.append({"player_id": f"TE{i}", "position": "TE", "value": 30 - i})
    for i in range(20):
        rows.append({"player_id": f"QB{i}", "position": "QB", "value": 40 - i})
    return pd.DataFrame(rows)


def test_replacement_levels_with_flex_to_rb():
    cfg = load_config()  # 12 teams: RB24/WR24/TE12 pure, FLEX12, QB12
    rep = compute_replacement(_board(), cfg=cfg)
    # RB: 24 pure + 12 flex = 36 starters -> first non-starter is the 37th RB (value 64)
    assert rep.get("RB") == 100 - 36
    # WR: flex exhausted by RBs -> first non-starter is the 25th WR (value 26)
    assert rep.get("WR") == 50 - 24
    # TE: pure 12 -> first non-starter is the 13th TE (value 18)
    assert rep.get("TE") == 30 - 12
    # QB: no flex -> the 13th QB (value 28)
    assert rep.get("QB") == 40 - 12


def test_starter_count():
    cfg = load_config()
    rep = compute_replacement(_board(), cfg=cfg)
    teams = cfg.teams
    expected = teams * (2 + 2 + 1) + teams * 1 + teams * 1  # pure skill + flex + QB
    assert len(rep.starters) == expected


def test_replacement_below_starters():
    cfg = load_config()
    board = _board()
    rep = compute_replacement(board, cfg=cfg)
    for pos in ["RB", "WR", "TE", "QB"]:
        starter_vals = board[(board["position"] == pos) &
                             (board["player_id"].isin(rep.starters))]["value"]
        assert rep.get(pos) <= starter_vals.min()
