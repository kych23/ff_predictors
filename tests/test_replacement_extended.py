"""Black-box extensions for greedy replacement-level computation.

League contract (from test_replacement.py): 12 teams, pure starters
RB24/WR24/TE12/QB12 plus 12 FLEX (RB/WR/TE eligible).
"""
import pandas as pd

from src.config import load_config
from src.recommender.replacement import compute_replacement


def _board(rb_top=100.0, wr_top=50.0, te_top=30.0, qb_top=40.0, step=1.0):
    rows = []
    for i in range(40):
        rows.append({"player_id": f"RB{i}", "position": "RB", "value": rb_top - i * step})
    for i in range(40):
        rows.append({"player_id": f"WR{i}", "position": "WR", "value": wr_top - i * step})
    for i in range(30):
        rows.append({"player_id": f"TE{i}", "position": "TE", "value": te_top - i * step})
    for i in range(20):
        rows.append({"player_id": f"QB{i}", "position": "QB", "value": qb_top - i * step})
    return pd.DataFrame(rows)


def test_flex_goes_to_wr_when_wr_dominates():
    cfg = load_config()
    board = _board(rb_top=50.0, wr_top=100.0)  # WRs strictly best
    rep = compute_replacement(board, cfg=cfg)
    # WR: 24 pure + 12 flex = 36 starters -> 37th WR
    assert rep.get("WR") == 100 - 36
    # RB: flex exhausted by WRs -> 25th RB
    assert rep.get("RB") == 50 - 24


def test_flex_goes_to_te_when_te_dominates():
    cfg = load_config()
    board = _board(rb_top=50.0, wr_top=45.0, te_top=100.0)
    rep = compute_replacement(board, cfg=cfg)
    # TE: 12 pure + 12 flex = 24 starters -> 25th TE
    assert rep.get("TE") == 100 - 24
    assert rep.get("RB") == 50 - 24
    assert rep.get("WR") == 45 - 24


def test_flex_split_between_interleaved_positions():
    """RB and WR values interleave above TE: flex fills strictly by value,
    so the 12 flex slots split 6/6."""
    cfg = load_config()
    rows = []
    for i in range(40):
        rows.append({"player_id": f"RB{i}", "position": "RB", "value": 100.0 - i})
        rows.append({"player_id": f"WR{i}", "position": "WR", "value": 99.5 - i})
    for i in range(20):
        rows.append({"player_id": f"TE{i}", "position": "TE", "value": 30.0 - i})
        rows.append({"player_id": f"QB{i}", "position": "QB", "value": 40.0 - i})
    rep = compute_replacement(pd.DataFrame(rows), cfg=cfg)
    # 24 pure each; next 12 by value alternate RB/WR -> 6 flex each
    assert rep.get("RB") == 100.0 - 30   # 31st RB
    assert rep.get("WR") == 99.5 - 30    # 31st WR


def test_qb_replacement_never_gets_flex():
    cfg = load_config()
    # QBs dominate the board, but QB is not flex-eligible
    board = _board(qb_top=500.0)
    rep = compute_replacement(board, cfg=cfg)
    assert rep.get("QB") == 500 - 12  # exactly 12 QB starters, 13th is replacement


def test_replacement_levels_positive_relationship_to_pool_depth():
    """Same board, but a deeper elite RB class raises the RB replacement bar."""
    cfg = load_config()
    shallow = compute_replacement(_board(rb_top=60.0), cfg=cfg)
    deep = compute_replacement(_board(rb_top=100.0), cfg=cfg)
    assert deep.get("RB") > shallow.get("RB")
