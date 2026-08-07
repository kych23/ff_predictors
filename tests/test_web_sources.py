"""Draft event sources (§11.5, §19).

The point of the protocol is that the cockpit works before Yahoo does. These
tests hold the two properties that makes true: every adapter satisfies the same
interface, and no adapter can take down a draft by raising on the poll path.
"""
from __future__ import annotations

import json

import pytest

from src.app.web.sources.base import (
    DraftEventSource,
    SourceStatus,
)
from src.app.web.sources.manual import ManualSource
from src.app.web.sources.replay import ReplaySource
from src.app.web.sources.yahoo import YahooSource
from src.core.errors import DataError


def _session_log(picks):
    return {"my_seat": 3, "teams": 12, "rounds": 15, "snapshot_id": "sn2_t",
            "events": [{"kind": k, "player_id": pid, "raw_input": raw,
                        "seat": seat, "at": "2026-08-24T18:00:00Z"}
                       for k, pid, raw, seat in picks]}


# ------------------------------------------------------ protocol conformance
@pytest.mark.parametrize("source", [
    ManualSource(), ReplaySource(), YahooSource(),
])
def test_every_adapter_satisfies_the_protocol(source):
    assert isinstance(source, DraftEventSource)
    assert isinstance(source.status, SourceStatus)


@pytest.mark.parametrize("source", [
    ManualSource(), YahooSource(),
])
def test_poll_never_raises_before_start(source):
    """A draft does not stop because a feed hiccuped."""
    assert source.poll() == []


# ------------------------------------------------------------------ manual
def test_manual_drains_exactly_once():
    src = ManualSource()
    src.start()
    src.submit("Jahmyr Gibbs")
    assert [e.raw_name for e in src.poll()] == ["Jahmyr Gibbs"]
    assert src.poll() == []


def test_manual_preserves_submission_order():
    src = ManualSource()
    src.start()
    for name in ("a", "b", "c"):
        src.submit(name)
    assert [e.raw_name for e in src.poll()] == ["a", "b", "c"]


def test_manual_passes_through_an_already_resolved_id():
    """The ambiguity flow: the operator picked a specific candidate."""
    src = ManualSource()
    src.start()
    src.submit("A. Jones", player_id="00-123")
    event = src.poll()[0]
    assert event.player_id == "00-123" and event.resolved


def test_manual_leaves_resolution_to_the_service():
    src = ManualSource()
    src.start()
    src.submit("  Bijan Robinson  ")
    event = src.poll()[0]
    assert event.raw_name == "Bijan Robinson"
    assert event.player_id is None and not event.resolved


# ------------------------------------------------------------------ replay
def test_replay_reproduces_the_log_in_order(tmp_path):
    log = tmp_path / "d.json"
    log.write_text(json.dumps(_session_log([
        ("PICK", "p1", "Ja'Marr Chase", 0),
        ("PICK", "p2", "Bijan Robinson", 1),
        ("MY_PICK", "p3", "Jahmyr Gibbs", 3),
    ])))
    src = ReplaySource(path=log)
    src.start()
    got = []
    while not src.exhausted:
        got.extend(src.poll())
    assert [e.raw_name for e in got] == [
        "Ja'Marr Chase", "Bijan Robinson", "Jahmyr Gibbs"]


def test_replay_emits_one_event_per_poll(tmp_path):
    """Returning the whole log at once would apply 180 picks between two
    recommendations and make the replay useless as a walkthrough."""
    log = tmp_path / "d.json"
    log.write_text(json.dumps(_session_log(
        [("PICK", f"p{i}", f"Player {i}", i % 12) for i in range(5)])))
    src = ReplaySource(path=log)
    src.start()
    assert len(src.poll()) == 1
    assert src.remaining == 4


def test_replay_does_not_pre_resolve_ids(tmp_path):
    """Replaying resolved ids would skip the identity cascade — the subsystem
    most likely to break under a live feed."""
    log = tmp_path / "d.json"
    log.write_text(json.dumps(_session_log([("PICK", "p1", "Chase", 0)])))
    src = ReplaySource(path=log)
    src.start()
    assert src.poll()[0].player_id is None


def test_replay_skips_non_pick_events(tmp_path):
    log = tmp_path / "d.json"
    log.write_text(json.dumps(_session_log([
        ("PICK", "p1", "Chase", 0),
        ("ZERO", "p1", "", None),
        ("UNRESOLVED", None, "garbled", None),
        ("PICK", "p2", "Bijan", 1),
    ])))
    src = ReplaySource(path=log)
    src.start()
    assert src.remaining == 2


def test_replay_falls_back_to_the_id_when_raw_input_was_dropped(tmp_path):
    """`Event.to_dict` omits empty fields, so most logs have no `raw_input`."""
    log = tmp_path / "d.json"
    log.write_text(json.dumps({"events": [
        {"kind": "PICK", "player_id": "00-0039139", "seat": 0}]}))
    src = ReplaySource(path=log)
    src.start()
    assert src.poll()[0].raw_name == "00-0039139"


def test_replay_reports_a_missing_log_clearly(tmp_path):
    src = ReplaySource(path=tmp_path / "nope.json")
    with pytest.raises(DataError, match="no replay log"):
        src.start()


def test_replay_reports_a_corrupt_log_clearly(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    src = ReplaySource(path=bad)
    with pytest.raises(DataError, match="not valid JSON"):
        src.start()


# ------------------------------------------------------------------- yahoo
def test_yahoo_reports_failure_rather_than_raising_without_credentials():
    """The shipped state. It must not take down a cockpit whose manual
    fallback works perfectly well."""
    src = YahooSource()
    src.start()
    assert src.status.state == "failed"
    assert "league_key" in src.status.detail
    assert src.poll() == []


def test_yahoo_reports_a_missing_token_specifically(tmp_path):
    src = YahooSource(league_key="nfl.l.12345",
                      token_path=tmp_path / "absent.json",
                      manager_map={"nfl.l.1.t.1": 0})
    src.start()
    assert src.status.state == "failed"
    assert "token" in src.status.detail


def test_yahoo_refuses_a_manager_id_seat_map(tmp_path):
    """`manager_map` is documented as team_key -> manager id, and no
    manager->seat table exists. Guessing would draft to the wrong team."""
    token = tmp_path / "t.json"
    token.write_text("{}")
    src = YahooSource(league_key="nfl.l.12345", token_path=token,
                      manager_map={"nfl.l.1.t.1": "kevin"})
    src.start()
    assert src.status.state == "failed"
    assert "seat" in src.status.detail


def test_yahoo_refuses_an_empty_seat_map(tmp_path):
    token = tmp_path / "t.json"
    token.write_text("{}")
    src = YahooSource(league_key="nfl.l.12345", token_path=token)
    src.start()
    assert src.status.state == "failed"
    assert "manager_map" in src.status.detail


def test_yahoo_walks_a_nested_payload():
    """Yahoo's JSON is deeply nested and numerically keyed; the parser must not
    depend on a shape nobody here has seen a real response from."""
    from src.app.web.sources.yahoo import _iter_draft_results

    payload = {"fantasy_content": {"league": [
        {}, {"draft_results": {
            "0": {"draft_result": {"pick": 1, "team_key": "t.1",
                                   "player_key": "nfl.p.100"}},
            "1": {"draft_result": {"pick": 2, "team_key": "t.2",
                                   "player_key": "nfl.p.200"}},
            "count": 2}}]}}
    picks = sorted(_iter_draft_results(payload), key=lambda p: p["pick"])
    assert [p["player_key"] for p in picks] == ["nfl.p.100", "nfl.p.200"]
