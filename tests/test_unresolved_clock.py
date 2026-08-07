"""An unresolved pick must still advance the clock (§19).

A name the identity spine cannot resolve is still a pick that happened: the
room moved on and the next seat is up. The session used to count only
`drafted`, so ONE unresolvable name left `pick_number`, `on_the_clock`,
`is_my_turn` and `next_my_pick` short by one for the **rest of the draft** —
and, because the recommendation path derives its own counters from the board,
the rollout resumed the snake at the wrong seat too.

A human typist hits this rarely. A live league feed hits it constantly, which
is why it had to be fixed before the web cockpit could use one.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.app.cockpit.session import Session, snake_seat
from src.engine.decision.roster_state import RosterState


@pytest.fixture
def session(tmp_path):
    return Session(my_seat=3, teams=12, rounds=15,
                   snapshot_id="sn2_test", path=tmp_path / "s.json")


# ------------------------------------------------------------- the clock
def test_an_unresolved_pick_advances_the_pick_number(session):
    session.record_pick("p1")
    assert session.state.pick_number == 2
    session.record_unresolved("Jauan Jennngz")
    assert session.state.pick_number == 3


def test_the_seat_on_the_clock_advances_too(session):
    """The regression: without this, every seat after the first unresolved
    name is wrong for the remainder of the draft."""
    for i in range(4):
        session.record_pick(f"p{i}")
    before = session.on_the_clock()
    session.record_unresolved("garbled name")
    assert session.on_the_clock() == snake_seat(6, 12)
    assert session.on_the_clock() != before


def test_an_unresolved_pick_counts_toward_my_turn(session):
    """seat 3 (0-indexed) is on the clock at pick 4. Whether the third pick
    was resolved or not, my turn arrives at the same absolute pick."""
    for i in range(3):
        session.record_pick(f"p{i}")
    assert session.state.pick_number == 4 and session.is_my_turn()

    fresh = Session(my_seat=3, teams=12, rounds=15, snapshot_id="sn2_test",
                    path=session.path.with_name("fresh.json"))
    fresh.record_pick("q0")
    fresh.record_pick("q1")
    fresh.record_unresolved("garbled")            # the third pick, unidentified
    assert fresh.state.pick_number == 4
    assert fresh.is_my_turn(), "an unresolved pick must still consume a slot"


def test_two_unresolved_names_advance_by_two(session):
    session.record_unresolved("a")
    session.record_unresolved("b")
    assert session.state.pick_number == 3
    assert len(session.state.unresolved) == 2


def test_undo_removes_an_unresolved_pick_and_rewinds_the_clock(session):
    session.record_pick("p1")
    session.record_unresolved("garbled")
    assert session.state.pick_number == 3
    session.undo()
    assert session.state.pick_number == 2
    assert session.state.unresolved == []


# ------------------------------------------------ no sentinel on the board
def test_no_placeholder_id_leaks_into_drafted(session):
    """A sentinel id would be rejected by `Board.take` and would pollute
    `available()`, `by_seat` and the ledger. The count is carried separately."""
    session.record_unresolved("garbled")
    assert session.state.drafted == []
    assert session.state.by_seat == {}


def test_the_unresolved_text_is_kept_for_the_operator(session):
    session.record_unresolved("Jauan Jennngz")
    assert session.state.unresolved == ["Jauan Jennngz"]


def test_replay_from_disk_preserves_the_advanced_clock(session, tmp_path):
    session.record_pick("p1")
    session.record_unresolved("garbled")
    reloaded = Session.load(tmp_path / "s.json")
    assert reloaded.state.pick_number == 3
    assert reloaded.state.unresolved == ["garbled"]


# ------------------------------------------------------- the other counters
def test_roster_state_counts_unresolved_picks():
    """`build_tiers` derives `position` from the BOARD, which can never hold an
    unresolved pick — so the count has to be handed to it explicitly."""
    from src.core.config import load_league

    cfg = load_league()
    rs = RosterState(cfg=cfg, draft_position=4, drafted={"a", "b"})
    assert rs.current_overall_pick() == 3
    rs_unres = RosterState(cfg=cfg, draft_position=4, drafted={"a", "b"},
                           unresolved_count=2)
    assert rs_unres.current_overall_pick() == 5


def test_rollout_pick_offset_shifts_the_snake():
    """Two rollouts from the same taken set but different offsets must start
    at different picks, or the offset is not wired in."""
    import pandas as pd

    from src.core.config import load_league, load_strategy
    from src.engine.sim.rollout import rollout

    cfg = load_league()
    strategy = load_strategy(cfg)
    n = cfg.teams * cfg.roster.rounds
    board = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(n)],
        "player_name": [f"P{i}" for i in range(n)],
        "position": (["RB", "WR", "QB", "TE"] * n)[:n],
        "adp": np.arange(1.0, n + 1.0),
        "value": np.linspace(20.0, 1.0, n),
        "team": ["AAA"] * n, "bye_week": [7] * n,
    })
    root = "ab" * 16
    a = rollout(board, cfg, strategy, my_seat=3, root=root, rep=0)
    b = rollout(board, cfg, strategy, my_seat=3, root=root, rep=0,
                pick_offset=3)
    # NOT a pick-count assertion: roster legality caps make the rollout
    # saturate well before teams*rounds, so both runs stop at the same total
    # and a count comparison would pass whether or not the offset is wired in.
    # What the offset actually changes is WHICH SEAT is on the clock, and
    # therefore who ends up with whom.
    assert [list(r) for r in a.rosters] != [list(r) for r in b.rosters], (
        "pick_offset must shift the snake; identical rosters mean it was "
        "ignored")
