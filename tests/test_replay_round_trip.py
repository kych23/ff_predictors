"""A draft this app SAVED must be a draft this app can replay.

`replay.py` claims "the log format is exactly what ``Session.save`` writes, so
any real or rehearsed draft is replayable with no conversion step." It was not:
`PICK_KINDS` was spelled `("PICK", "MY_PICK")` while `Session` emits `"pick"`
and `"my_pick"`. Nothing raised — every event failed the membership test, so a
real saved draft replayed as an EMPTY queue and a draft that never advanced.

The checked-in fixture happened to be uppercase, so the suite stayed green
while the documented feature did not work at all. That is why this test
round-trips through `Session.save` instead of reading a fixture.
"""
from __future__ import annotations

import json

import pytest

from src.app.cockpit.session import MY_PICK, PICK, Session
from src.app.web.sources.replay import PICK_KINDS, ReplaySource


def _saved_session(tmp_path, picks=("Alpha", "Bravo", "Charlie")):
    path = tmp_path / "session.json"
    session = Session(my_seat=0, teams=12, rounds=15, snapshot_id="sn2_test",
                      path=path)
    for i, name in enumerate(picks):
        session.record_pick(f"p{i}", seat=i % 12, raw_input=name)
    session.save()
    return path


def test_the_kinds_come_from_the_session_module():
    """Not spelled again. The duplicate copy is exactly what drifted."""
    assert PICK_KINDS == (PICK, MY_PICK)
    assert PICK == "pick" and MY_PICK == "my_pick"


def test_a_saved_session_replays_every_pick(tmp_path):
    path = _saved_session(tmp_path)
    source = ReplaySource(path=path)
    source.start()
    assert source.remaining == 3, (
        "a real saved draft must replay; 0 here is the original bug")


def test_replay_preserves_order_and_names(tmp_path):
    path = _saved_session(tmp_path, picks=("Alpha", "Bravo", "Charlie"))
    source = ReplaySource(path=path)
    source.start()
    drained = [e for _ in range(3) for e in source.poll()]
    assert [e.raw_name for e in drained] == ["Alpha", "Bravo", "Charlie"]
    assert source.exhausted


def test_the_uppercase_fixture_still_replays(tmp_path):
    """Backward compatibility: the checked-in fixture predates the fix and is
    written uppercase. Both spellings must work, or fixing one breaks the
    other."""
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"events": [
        {"kind": "PICK", "player_id": "a", "raw_input": "Alpha", "seat": 0},
        {"kind": "MY_PICK", "player_id": "b", "raw_input": "Bravo", "seat": 1},
    ]}))
    source = ReplaySource(path=path)
    source.start()
    assert source.remaining == 2


def test_my_own_picks_replay_too(tmp_path):
    """`my_pick` is a separate kind. Dropping it would replay a draft in which
    my seat never picks — a legal-looking board with my roster missing."""
    path = tmp_path / "session.json"
    session = Session(my_seat=0, teams=12, rounds=15, snapshot_id="sn2_test",
                      path=path)
    session.record_pick("mine", seat=0, raw_input="Mine")
    session.record_pick("theirs", seat=1, raw_input="Theirs")
    session.save()

    kinds = [e["kind"] for e in json.loads(path.read_text())["events"]]
    assert MY_PICK in kinds, "fixture must exercise the my_pick branch"

    source = ReplaySource(path=path)
    source.start()
    assert source.remaining == 2


def test_non_pick_events_are_skipped(tmp_path):
    path = tmp_path / "session.json"
    path.write_text(json.dumps({"events": [
        {"kind": "pick", "player_id": "a", "raw_input": "Alpha", "seat": 0},
        {"kind": "zero", "player_id": "b", "seat": 1},
        {"kind": "adp_override", "player_id": "c", "seat": 2},
    ]}))
    source = ReplaySource(path=path)
    source.start()
    assert source.remaining == 1


def test_a_missing_log_is_a_clear_error(tmp_path):
    from src.core.errors import DataError

    with pytest.raises(DataError, match="no replay log"):
        ReplaySource(path=tmp_path / "nope.json").start()
