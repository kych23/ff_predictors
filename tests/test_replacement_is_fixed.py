"""Replacement level is STRUCTURAL scarcity and is fixed preseason (§3).

`replacement.py` opens by saying it is "computed ONCE from preseason
projections and never recomputed mid-draft (§3): dynamic replacement + VONA
would double-count intra-draft scarcity". `recommend.score` then passed
``available`` — the shrinking board — so the level fell as a position drained.
Measured on the shipped bundle, replacement[RB] slid 9.92 -> 7.68 -> 4.83 ->
3.84 across picks 0, 24, 96 and 120.

The violation survived because it mostly cancels. With an open starter slot::

    vona = (value - repl) - (E_best - repl) = value - E_best

so the level drops out and no ranking moves — verified identical at picks 25,
49 and 97 despite that slide. It stops cancelling once `marginal` or
`wait_term` hits its ``max(0, .)`` clamp, which is the late rounds: at pick
133, 51 of 124 players changed score and the ordering changed with them.

That is why this needs a test rather than a comment. The bug is invisible for
two thirds of the draft, which is exactly how it survived.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.core.config import load_league, load_strategy
from src.engine.decision import recommend as tier2
from src.engine.decision import replacement as replacement_mod
from src.engine.decision.roster_state import RosterState


@pytest.fixture(scope="module")
def league():
    return load_league()


@pytest.fixture(scope="module")
def strategy(league):
    return load_strategy(league)


def _board(n_per_pos: int = 40) -> pd.DataFrame:
    """A synthetic board with a steep RB cliff and flat WR depth."""
    rows = []
    for pos, top, decay in (("QB", 20.0, 0.35), ("RB", 19.0, 0.55),
                            ("WR", 18.0, 0.22), ("TE", 14.0, 0.40)):
        for i in range(n_per_pos):
            rows.append({"player_id": f"{pos}{i:02d}",
                         "player_name": f"{pos} Player {i}",
                         "position": pos, "value": top - decay * i,
                         "adp": 1 + len(rows) * 0.9, "adp_stdev": 6.0,
                         "bye_week": 0})
    return pd.DataFrame(rows)


def _drain(board: pd.DataFrame, position: str, count: int) -> set[str]:
    """Take the best `count` players at a position — the scarcity event."""
    pool = board[board["position"] == position].sort_values(
        "value", ascending=False)
    return set(pool["player_id"].head(count))


# --------------------------------------------- the level itself must not move
def test_draining_a_position_moves_the_dynamic_level(league):
    """Pins the mechanism the fix removes; if this stops holding, the rest of
    this file proves nothing."""
    board = _board()
    taken = _drain(board, "RB", 25)
    avail = board[~board["player_id"].isin(taken)]

    preseason = replacement_mod.compute_replacement(board, cfg=league)
    dynamic = replacement_mod.compute_replacement(avail, cfg=league)
    assert dynamic.get("RB") < preseason.get("RB") - 0.5


def test_score_uses_the_preseason_level_when_given_one(league, strategy):
    board = _board()
    taken = _drain(board, "RB", 25)
    avail = board[~board["player_id"].isin(taken)]
    rs = RosterState(cfg=league, draft_position=1, drafted=taken)

    result = tier2.score(avail, rs, league, strategy, current_pick=26,
                         preseason_board=board)
    expected = replacement_mod.compute_replacement(board, cfg=league)
    assert result.replacement.get("RB") == pytest.approx(expected.get("RB"))


def test_the_level_is_stable_across_the_whole_draft(league, strategy):
    """The invariant stated plainly: structural scarcity is a preseason fact,
    so it cannot depend on how far the draft has run."""
    board = _board()
    order = board.sort_values("adp")["player_id"].tolist()
    seen = []
    for n in (0, 30, 60, 90, 120):
        taken = set(order[:n])
        avail = board[~board["player_id"].isin(taken)]
        rs = RosterState(cfg=league, draft_position=1, drafted=taken)
        result = tier2.score(avail, rs, league, strategy, current_pick=n + 1,
                             preseason_board=board)
        seen.append(result.replacement.get("RB"))
    assert max(seen) - min(seen) == pytest.approx(0.0), (
        f"replacement[RB] drifted across the draft: {seen}")


# ------------------------------------------------- and it changes the ranking
def _real_board():
    from src.engine.decision.board import Board
    return Board.from_bundle().players.copy()


def _compare_at(pick: int, league, strategy):
    """Fixed vs dynamic replacement on the REAL board, ADP draft order."""
    board = _real_board()
    order = board.sort_values("adp")["player_id"].astype(str).tolist()
    taken = set(order[:pick - 1])
    avail = board[~board["player_id"].astype(str).isin(taken)]
    position = board.set_index(board["player_id"].astype(str))["position"]
    rs = RosterState(cfg=league, draft_position=1, drafted=taken,
                     my_roster=[{"player_id": p, "position": position[p]}
                                for p in order[:pick - 1:12]])
    fixed = tier2.score(avail, rs, league, strategy, current_pick=pick,
                        preseason_board=board).ranked
    dynamic = tier2.score(avail, rs, league, strategy,
                          current_pick=pick).ranked
    return fixed["player_id"].tolist(), dynamic["player_id"].tolist()


@pytest.mark.parametrize("pick", [25, 49, 97])
def test_early_and_mid_ordering_is_unchanged(pick, league, strategy):
    """Why this was safe to change two weeks out, and why it hid for so long:
    while both terms are unclamped the level cancels exactly."""
    fixed, dynamic = _compare_at(pick, league, strategy)
    assert fixed == dynamic


def test_late_round_ordering_differs(league, strategy):
    """Where the cancellation breaks. Measured on the shipped bundle: at pick
    133, 51 of 124 remaining players changed score and the order moved."""
    fixed, dynamic = _compare_at(133, league, strategy)
    assert fixed != dynamic, (
        "the clamp is no longer reached, so this cannot show the difference")


def test_the_live_cockpit_path_passes_the_preseason_board():
    """`build_tiers` owns both tier-2 call sites — the fallback recommendation
    and the tier-0 shortlist. A shortlist built on a drifting level feeds the
    wrong candidates into the expensive simulation."""
    import inspect

    from src.app.cockpit import build

    source = inspect.getsource(build.build_tiers)
    assert source.count("preseason_board=frame") == 2, (
        "both tier2.score call sites in build_tiers must pin the level")
