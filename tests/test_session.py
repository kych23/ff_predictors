"""Event-sourced draft session (§19).

These tests are written for pick 47 with eleven people waiting: the operations
that must not fail are `undo` and recovery, and both reduce to replay. If
replay is right, everything else is a printout.
"""
from __future__ import annotations

import json

import pytest

from src.app.cockpit.session import (
    MY_PICK, PICK, DraftState, Event, Session, available, my_pick_numbers,
    snake_seat,
)
from src.core.errors import DataError


def _session(tmp_path=None, seat=3, teams=12, rounds=15):
    return Session(my_seat=seat, teams=teams, rounds=rounds,
                   snapshot_id="sn2_test",
                   path=(tmp_path / "draft.json") if tmp_path else None)


# ------------------------------------------------------------ snake order
def test_snake_order_reverses_every_round():
    assert [snake_seat(p, 12) for p in range(1, 13)] == list(range(12))
    assert [snake_seat(p, 12) for p in range(13, 25)] == list(range(11, -1, -1))
    assert snake_seat(25, 12) == 0


def test_seat_four_gets_the_expected_picks():
    """Seat 4 (0-indexed 3) in a 12-team snake: 4, 21, 28, 45, ..."""
    picks = my_pick_numbers(3, 12, 15)
    assert picks[:4] == [4, 21, 28, 45]
    assert len(picks) == 15


def test_every_pick_belongs_to_exactly_one_seat():
    owners = [my_pick_numbers(s, 12, 15) for s in range(12)]
    flat = sorted(p for seat in owners for p in seat)
    assert flat == list(range(1, 12 * 15 + 1))


def test_pick_number_is_one_indexed():
    with pytest.raises(ValueError, match="1-indexed"):
        snake_seat(0, 12)


def test_a_seat_outside_the_league_is_rejected():
    with pytest.raises(ValueError, match="outside a 12-team league"):
        Session(my_seat=12, teams=12, rounds=15)


# ---------------------------------------------------------------- replay
def test_state_is_a_pure_function_of_the_history():
    session = _session()
    for pid in ("a", "b", "c", "d"):
        session.record_pick(pid)
    replayed = session._replay()
    assert replayed.drafted == session.state.drafted
    assert replayed.my_roster == session.state.my_roster


def test_picks_are_attributed_to_the_seat_on_the_clock():
    session = _session(seat=3)
    for pid in ("a", "b", "c", "d"):
        session.record_pick(pid)
    # pick 4 is seat 3 — mine
    assert session.state.my_roster == ["d"]
    assert session.state.by_seat[0] == ["a"]
    assert session.state.by_seat[3] == ["d"]


def test_drafting_the_same_player_twice_is_refused():
    """A repeated name is a mis-entry, not a second pick. Allowing it
    desynchronises every downstream count."""
    session = _session()
    session.record_pick("a")
    with pytest.raises(DataError, match="already drafted"):
        session.record_pick("a")


def test_a_refused_command_leaves_the_log_clean():
    """If the rejected event stayed in the history, every later replay would
    raise and the session would be unrecoverable."""
    session = _session()
    session.record_pick("a")
    with pytest.raises(DataError):
        session.record_pick("a")
    assert len(session.events) == 1
    assert session._replay().drafted == ["a"]


# ------------------------------------------------------------------ undo
def test_undo_pops_one_event_and_replays():
    session = _session()
    session.record_pick("a")
    session.record_pick("b")
    session.undo()
    assert session.state.drafted == ["a"]
    assert session.state.pick_number == 2


def test_undo_reverses_a_zero_without_a_per_command_inverse():
    """An inverse for `zero` would have to remember whether the player was
    already zeroed. Nobody gets that right at pick 47; replay does."""
    session = _session()
    session.zero_player("a")
    session.zero_player("b")
    session.undo()
    assert session.state.zeroed == {"a"}


def test_undo_all_the_way_back_to_empty():
    session = _session()
    for pid in ("a", "b", "c"):
        session.record_pick(pid)
    for _ in range(3):
        session.undo()
    assert session.state.drafted == []
    with pytest.raises(DataError, match="nothing to undo"):
        session.undo()


def test_undo_restores_the_clock_and_the_roster():
    session = _session(seat=0)
    session.record_pick("mine")            # pick 1, seat 0
    assert session.state.my_roster == ["mine"]
    session.undo()
    assert session.state.my_roster == []
    assert session.is_my_turn()


# ------------------------------------------------------------- overrides
def test_a_zeroed_player_stays_on_the_board():
    """He is still draftable by someone else. Removing him would make the
    opponent model think he is gone."""
    session = _session()
    session.zero_player("a")
    assert "a" in available(["a", "b"], session.state)
    assert "a" in session.state.zeroed


def test_adp_override_is_recorded_and_replayed():
    session = _session()
    session.override_adp("a", 12.5)
    assert session.state.adp_overrides == {"a": 12.5}
    assert session._replay().adp_overrides == {"a": 12.5}


def test_an_unresolved_name_is_recorded_not_dropped():
    """The pick still happened. A board that thinks a player is available when
    he is gone will keep recommending him."""
    session = _session()
    session.record_unresolved("Jauan Jennngs")
    assert session.state.unresolved == ["Jauan Jennngs"]


# ------------------------------------------------------------ the clock
def test_clock_reports_who_is_up_and_how_far_away():
    session = _session(seat=3)
    assert session.on_the_clock() == 0
    assert session.picks_until_my_turn() == 3
    for pid in ("a", "b", "c"):
        session.record_pick(pid)
    assert session.is_my_turn()
    assert session.picks_until_my_turn() == 0
    assert "YOU" in session.describe_clock()


def test_the_draft_ends_after_every_pick():
    session = _session(seat=1, teams=2, rounds=2)
    for pid in "abcd":
        session.record_pick(pid)
    assert session.is_complete
    assert session.next_my_pick() is None
    assert "complete" in session.describe_clock()


# -------------------------------------------------------- crash recovery
def test_the_session_survives_a_restart(tmp_path):
    """Draft night is exactly when a terminal gets closed."""
    session = _session(tmp_path)
    for pid in ("a", "b", "c", "d"):
        session.record_pick(pid)
    session.zero_player("e")

    reloaded = Session.load(tmp_path / "draft.json")
    assert reloaded.state.drafted == ["a", "b", "c", "d"]
    assert reloaded.state.my_roster == ["d"]
    assert reloaded.state.zeroed == {"e"}
    assert reloaded.my_seat == 3 and reloaded.teams == 12


def test_the_history_is_written_after_every_command(tmp_path):
    session = _session(tmp_path)
    session.record_pick("a")
    written = json.loads((tmp_path / "draft.json").read_text())
    assert len(written["events"]) == 1
    session.record_pick("b")
    written = json.loads((tmp_path / "draft.json").read_text())
    assert len(written["events"]) == 2


def test_undo_is_persisted_too(tmp_path):
    session = _session(tmp_path)
    session.record_pick("a")
    session.record_pick("b")
    session.undo()
    assert len(Session.load(tmp_path / "draft.json").events) == 1


def test_the_write_is_atomic(tmp_path):
    """A crash mid-write must not leave a truncated history, which would make
    the one recovery path unusable."""
    import inspect
    source = inspect.getsource(Session.save)
    assert ".tmp" in source and "replace" in source


def test_resume_or_new_returns_the_saved_session(tmp_path):
    first = _session(tmp_path)
    first.record_pick("a")
    resumed = _session(tmp_path).resume_or_new()
    assert resumed.state.drafted == ["a"]


def test_a_fresh_path_starts_empty(tmp_path):
    assert _session(tmp_path).resume_or_new().state.drafted == []


def test_missing_session_file_is_named(tmp_path):
    with pytest.raises(DataError, match="no session at"):
        Session.load(tmp_path / "absent.json")


def test_events_round_trip_through_json():
    event = Event(kind=PICK, player_id="a", seat=2, raw_input="Gibbs",
                  at="2026-08-24T18:00:00+00:00")
    assert Event.from_dict(event.to_dict()) == event


def test_an_unknown_event_kind_replays_as_a_no_op():
    """Histories are additive: one written by a newer build must still replay
    under an older one, because recovery IS replay."""
    session = _session()
    session.events.append(Event(kind="something_new", player_id="x"))
    assert session._replay().drafted == []
