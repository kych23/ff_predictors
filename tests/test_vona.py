"""VONA take-now-vs-wait logic (DrafterSpec.md §4.8.1)."""
import pandas as pd

from src.config import load_config
from src.recommender.recommend import recommend
from src.recommender.replacement import ReplacementLevels
from src.recommender.roster_state import RosterState
from src.recommender.vona import marginal_value, score_board


def _replacement():
    return ReplacementLevels(levels={"RB": 5.0, "WR": 5.0, "TE": 5.0, "QB": 5.0})


def test_scarce_position_beats_deep_at_equal_value():
    cfg = load_config()
    state = RosterState(cfg=cfg, draft_position=1)  # empty roster, all slots open
    board = pd.DataFrame([
        {"player_id": "A", "position": "RB", "value": 20, "adp": 10, "adp_stdev": 3},
        {"player_id": "A2", "position": "RB", "value": 8, "adp": 45, "adp_stdev": 8},
        {"player_id": "B", "position": "WR", "value": 20, "adp": 100, "adp_stdev": 15},
        {"player_id": "B2", "position": "WR", "value": 7, "adp": 120, "adp_stdev": 15},
    ])
    scored = score_board(board, state, _replacement(), next_pick=30)
    a = scored[scored["player_id"] == "A"].iloc[0]
    b = scored[scored["player_id"] == "B"].iloc[0]
    # equal marginal value, but the scarce RB (won't survive) outscores the deep WR
    assert a["marginal"] == b["marginal"]
    assert a["vona_score"] > b["vona_score"]
    # deep WR carries a large wait term (he'll still be there next pick)
    assert b["wait_term"] > a["wait_term"]


def test_last_pick_wait_term_zero():
    cfg = load_config()
    state = RosterState(cfg=cfg, draft_position=1)
    board = pd.DataFrame([
        {"player_id": "A", "position": "RB", "value": 20, "adp": 100, "adp_stdev": 15},
    ])
    scored = score_board(board, state, _replacement(), next_pick=None)
    row = scored.iloc[0]
    assert row["wait_term"] == 0.0
    # score reduces to pure marginal value
    assert row["vona_score"] == row["marginal"] == 15.0


def test_no_double_count_fixed_replacement():
    # marginal_value uses ONLY the fixed replacement bar, independent of the board.
    cfg = load_config()
    state = RosterState(cfg=cfg, draft_position=1)
    rep = _replacement()
    assert marginal_value(20, "RB", rep, state) == 15.0
    # filling the slot does not change the replacement-based marginal of a new starter
    assert marginal_value(18, "WR", rep, state) == 13.0


def test_empty_board_returns_empty():
    cfg = load_config()
    state = RosterState(cfg=cfg, draft_position=1)
    board = pd.DataFrame(columns=["player_id", "position", "value", "adp", "adp_stdev"])
    scored = score_board(board, state, _replacement(), next_pick=30)
    assert scored.empty
    assert "vona_score" in scored.columns


def test_all_below_replacement_bench_floor_zero():
    """Players below replacement with no open starter slot contribute 0 marginal,
    never negative bench value."""
    cfg = load_config()
    state = RosterState(cfg=cfg, draft_position=1)
    for slot, n in cfg.roster.slots.items():
        state.slot_fill[slot] = n  # every slot filled -> bench-only
    assert marginal_value(2.0, "RB", _replacement(), state) == 0.0


def test_nan_adp_stdev_does_not_crash():
    import numpy as np
    cfg = load_config()
    state = RosterState(cfg=cfg, draft_position=1)
    board = pd.DataFrame([
        {"player_id": "A", "position": "RB", "value": 20, "adp": 10,
         "adp_stdev": np.nan},
        {"player_id": "B", "position": "RB", "value": 15, "adp": 40,
         "adp_stdev": np.nan},
    ])
    scored = score_board(board, state, _replacement(), next_pick=30)
    assert len(scored) == 2
    assert scored["vona_score"].notna().all()


def test_roster_completion_hard_constraint():
    cfg = load_config()
    state = RosterState(cfg=cfg, draft_position=1)
    # Fill everything except one mandatory QB; make remaining picks == unfilled mandatory.
    # Simulate by directly setting slot_fill to all full except QB, and roster length.
    for slot, n in cfg.roster.slots.items():
        state.slot_fill[slot] = n
    state.slot_fill["QB"] = 0  # QB still needed (1 slot)
    # remaining picks: make exactly equal to unfilled mandatory (=1)
    state.my_roster = [{"player_id": f"p{i}", "position": "RB"}
                       for i in range(cfg.roster.rounds - 1)]
    proj = pd.DataFrame([
        {"player_id": "rb", "position": "RB", "p10": 9, "p50": 10, "p90": 11,
         "adp": 50, "adp_stdev": 10},
        {"player_id": "qb", "position": "QB", "p10": 4, "p50": 5, "p90": 6,
         "adp": 60, "adp_stdev": 10},
    ])
    rep = _replacement()
    recs = recommend(proj, state, rep, cfg=cfg)
    # despite RB having higher value, only the needed QB may be recommended
    assert set(recs["position"]) == {"QB"}
    assert bool(recs.iloc[0]["forced_completion"]) is True
