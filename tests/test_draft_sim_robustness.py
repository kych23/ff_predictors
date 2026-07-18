"""Draft-sim robustness regressions (found by synthetic mock-draft stress runs).

Bug 1: a positional run (e.g. bots hoarding QBs) could empty a needed position's
pool before the endgame roster-completion trigger (remaining <= unfilled) fired,
leaving the recommender's roster with no legal starter. The supply-aware
scarcity trip in recommend() forces the at-risk position while players remain.

Bug 2: _bot_pick returned None once the with-ADP pool ran dry, silently leaving
bot rosters short; it must fall back to the best-projected undrafted player.
"""
import numpy as np
import pandas as pd

from src.benchmark.draft_sim import _bot_pick
from src.config import load_config
from src.recommender.recommend import recommend
from src.recommender.replacement import ReplacementLevels
from src.recommender.roster_state import RosterState

CFG = load_config()


def _replacement():
    return ReplacementLevels(levels={"QB": 14.0, "RB": 8.0, "WR": 8.0, "TE": 6.0})


def _mid_draft_state(n_my_picks: int, my_positions, n_total_drafted: int):
    """State with my roster filled per my_positions and the league at
    n_total_drafted picks overall."""
    state = RosterState(cfg=CFG, draft_position=6)
    for i, pos in enumerate(my_positions):
        state.record_pick(f"mine{i}", pos, mine=True)
    for i in range(n_total_drafted - len(my_positions)):
        state.drafted.add(f"opp{i}")
    return state


def _board(n_qb: int) -> pd.DataFrame:
    rows = []
    for i in range(n_qb):
        rows.append({"player_id": f"QB{i}", "position": "QB", "p10": 8, "p50": 12,
                     "p90": 16, "adp": 100 + i, "adp_stdev": 10})
    for i in range(30):
        rows.append({"player_id": f"WR{i}", "position": "WR", "p10": 9, "p50": 13,
                     "p90": 18, "adp": 60 + i, "adp_stdev": 8})
    return pd.DataFrame(rows)


def test_scarce_needed_position_is_forced_before_pool_empties():
    """QB still needed, 2 QBs left, ~11 opponent picks before my next turn:
    the pick must be forced to QB even though remaining > unfilled."""
    # my roster: 8 picks, no QB; league at pick 96 (round 9 of 15)
    state = _mid_draft_state(8, ["RB", "WR", "RB", "WR", "TE", "RB", "WR", "RB"], 96)
    assert state.remaining_picks() > state.total_unfilled_mandatory()
    recs = recommend(_board(n_qb=2), state, _replacement(), cfg=CFG)
    assert not recs.empty
    assert recs.iloc[0]["position"] == "QB"
    assert bool(recs.iloc[0]["forced_completion"]) is True


def test_ample_supply_not_forced():
    """Same state but 25 QBs available — no scarcity, normal VONA ordering."""
    state = _mid_draft_state(8, ["RB", "WR", "RB", "WR", "TE", "RB", "WR", "RB"], 96)
    recs = recommend(_board(n_qb=25), state, _replacement(), cfg=CFG)
    assert not recs.empty
    assert not recs["forced_completion"].any()


def test_extinct_needed_position_does_not_crash():
    """Needed position with zero available players: warn, keep recommending."""
    state = _mid_draft_state(8, ["RB", "WR", "RB", "WR", "TE", "RB", "WR", "RB"], 96)
    recs = recommend(_board(n_qb=0), state, _replacement(), cfg=CFG)
    assert not recs.empty
    assert (recs["position"] == "WR").all()


def test_bot_pick_falls_back_when_adp_pool_exhausted():
    board = pd.DataFrame([
        {"player_id": "a", "position": "WR", "adp": 1.0, "p50": 10.0},
        {"player_id": "b", "position": "RB", "adp": np.nan, "p50": 14.0},
        {"player_id": "c", "position": "WR", "adp": np.nan, "p50": 9.0},
    ])
    slot_fill = {s: 0 for s in CFG.roster.slots}
    # 'a' already drafted -> only NaN-ADP players left -> best p50 wins
    assert _bot_pick(board, {"a"}, slot_fill, CFG) == "b"
    # everyone drafted -> None
    assert _bot_pick(board, {"a", "b", "c"}, slot_fill, CFG) is None
