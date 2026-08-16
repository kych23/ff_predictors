"""Archive keeps a draft, delete removes it, and Replay lets you choose one.

Three separate promises, and each had a hole before this:

* **Delete** unlinked the session log and deliberately left the ledger alone,
  so a purged draft still had a full decision record behind it — a ledger
  describing a draft with no log.
* **Archive** wrote `session.<unix>.bak` files that nothing could ever list or
  read back. Archiving was a one-way trip.
* **Replay** always played one hardcoded fixture from `web.yaml`, no matter how
  many drafts had been archived.

The ledger's hash chain is why delete is careful rather than simple: entries
link to their predecessor, so only a contiguous suffix can be removed without
making the file look tampered with.
"""
from __future__ import annotations

import json

import pytest

from src.app.cockpit.ledger import DecisionLedger
from src.app.web import service as svc
from src.core.errors import DataError


# ------------------------------------------------------------- the ledger
def _ledger_with(tmp_path, sessions):
    ledger = DecisionLedger(tmp_path / "ledger.sqlite")
    for session_id in sessions:
        ledger.append(session_id, pick_no=1, tier=0,
                      recommendation={"leader": "x"},
                      snapshot_id="sn2_test", actual_pick="x")
    return ledger


def test_deleting_the_newest_draft_leaves_a_valid_chain(tmp_path):
    ledger = _ledger_with(tmp_path, ["a", "a", "b", "b", "b"])
    assert ledger.verify().ok
    assert ledger.delete_session("b") == 3
    assert ledger.verify().ok, "removing a suffix must not break the chain"
    assert len(ledger) == 2


def test_deleting_a_middle_draft_is_refused(tmp_path):
    """The refusal IS the feature. Everything after a removed entry would
    claim a prev_hash that no longer exists, and `verify()` cannot tell that
    apart from real tampering."""
    ledger = _ledger_with(tmp_path, ["a", "a", "b", "b"])
    with pytest.raises(DataError, match="hash chain"):
        ledger.delete_session("a")
    assert ledger.verify().ok
    assert len(ledger) == 4


def test_deleting_an_unknown_draft_is_a_no_op(tmp_path):
    ledger = _ledger_with(tmp_path, ["a"])
    assert ledger.delete_session("ghost") == 0
    assert len(ledger) == 1


def test_sessions_are_listed_newest_first(tmp_path):
    ledger = _ledger_with(tmp_path, ["old", "old", "new"])
    listed = ledger.sessions()
    assert [s["session_id"] for s in listed] == ["new", "old"]
    assert [s["entries"] for s in listed] == [1, 2]


def test_the_ledger_is_keyed_on_the_draft_not_the_bundle():
    """`ledger.append` was passed `snapshot_id`, so every draft run against one
    bundle shared an id and deleting one would have taken the others."""
    import inspect

    source = inspect.getsource(svc.CockpitService._record_ledger_entry) \
        if hasattr(svc.CockpitService, "_record_ledger_entry") \
        else inspect.getsource(svc.CockpitService)
    assert "self.ledger.append(\n                self.session.session_id," in source


# ------------------------------------------------------------ the archives
class _Cfg:
    def __init__(self, tmp_path):
        self._root = tmp_path
        self.session_path = "session.json"
        self.replay_path = "fixture.json"

    def resolved(self, value):
        return self._root / value


def _write_archive(cfg, stamp, *, picks, seat=0, session_id="s1"):
    path = cfg.resolved(f"session.{stamp}.bak")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "session_id": session_id, "started_at": f"2026-08-0{stamp % 9 + 1}T10:00:00+00:00",
        "my_seat": seat, "teams": 12, "rounds": 15, "snapshot_id": "sn2_test",
        "events": [{"kind": "pick", "player_id": f"p{i}", "raw_input": f"P{i}",
                    "seat": i % 12} for i in range(picks)],
    }))
    return path


def test_archives_are_listed_newest_first(tmp_path):
    cfg = _Cfg(tmp_path)
    for stamp in (1, 2, 3):
        path = _write_archive(cfg, stamp, picks=stamp)
        import os
        os.utime(path, (1_000 * stamp, 1_000 * stamp))
    listed = svc.list_archives(cfg)
    assert [a["picks"] for a in listed] == [3, 2, 1]


def test_an_archive_reports_what_the_picker_needs(tmp_path):
    cfg = _Cfg(tmp_path)
    _write_archive(cfg, 7, picks=5, seat=3)
    entry = svc.list_archives(cfg)[0]
    assert entry["picks"] == 5
    assert entry["seat"] == 3
    assert entry["started_at"] and entry["archived_at"]
    assert entry["readable"] is True


def test_a_corrupt_archive_is_listed_but_flagged(tmp_path):
    """It must still appear. A file you cannot read is exactly the one you
    want to see in the list so you can delete it."""
    cfg = _Cfg(tmp_path)
    cfg.resolved("session.9.bak").parent.mkdir(parents=True, exist_ok=True)
    cfg.resolved("session.9.bak").write_text("{ not json")
    entry = svc.list_archives(cfg)[0]
    assert entry["readable"] is False
    assert entry["picks"] == 0


def test_the_live_session_file_is_not_offered_as_an_archive(tmp_path):
    cfg = _Cfg(tmp_path)
    cfg.resolved("session.json").parent.mkdir(parents=True, exist_ok=True)
    cfg.resolved("session.json").write_text(json.dumps({"events": []}))
    _write_archive(cfg, 1, picks=2)
    assert [a["id"] for a in svc.list_archives(cfg)] == ["session.1.bak"]


# --------------------------------------------------- choosing what to replay
def test_an_archive_id_selects_that_archive(tmp_path):
    cfg = _Cfg(tmp_path)
    _write_archive(cfg, 1, picks=2)
    _write_archive(cfg, 2, picks=6)
    source = svc.make_source("replay", web_cfg=cfg, strategy=None,
                             archive_id="session.2.bak")
    source.start()
    assert source.remaining == 6


def test_no_archive_id_falls_back_to_the_fixture(tmp_path):
    cfg = _Cfg(tmp_path)
    cfg.resolved("fixture.json").parent.mkdir(parents=True, exist_ok=True)
    cfg.resolved("fixture.json").write_text(json.dumps({"events": [
        {"kind": "pick", "player_id": "a", "raw_input": "Alpha", "seat": 0}]}))
    source = svc.make_source("replay", web_cfg=cfg, strategy=None)
    source.start()
    assert source.remaining == 1


def test_an_archive_id_is_matched_against_the_listing_not_joined(tmp_path):
    """The id comes from a client and this endpoint opens whatever it is
    handed, so it is resolved by comparing against the directory listing.
    Joining it onto a path would make `../../etc/passwd` readable."""
    cfg = _Cfg(tmp_path)
    _write_archive(cfg, 1, picks=1)
    with pytest.raises(DataError, match="no archived draft"):
        svc.resolve_archive(cfg, "../../../etc/passwd")
    with pytest.raises(DataError, match="no archived draft"):
        svc.resolve_archive(cfg, "session.999.bak")


# ------------------------------------------- the draft's identity reaches the UI
def test_the_session_payload_carries_the_draft_identity():
    """The notes pad keys its localStorage on `session_id`. Falling back to
    `snapshot_id` would be wrong in the way that is hardest to notice: every
    draft run against one bundle shares it, so a new draft would open holding
    the previous draft's notes."""
    import inspect

    source = inspect.getsource(svc.CockpitService.session_payload)
    assert '"session_id": session.session_id' in source
    assert '"snapshot_id"' in source, "both are carried; they mean different things"


def test_session_id_is_stable_across_a_reload(tmp_path):
    """The pad has to survive a refresh, which means the key has to survive a
    reload of the log."""
    from src.app.cockpit.session import Session

    path = tmp_path / "session.json"
    first = Session(my_seat=0, teams=12, rounds=15, snapshot_id="sn2_t",
                    path=path)
    first.record_pick("a", seat=0, raw_input="Alpha")
    first.save()

    reloaded = Session.load(path)
    assert reloaded.session_id == first.session_id
    assert reloaded.session_id


def test_two_drafts_get_different_identities(tmp_path):
    from src.app.cockpit.session import Session

    a = Session(my_seat=0, teams=12, rounds=15, path=tmp_path / "a.json")
    b = Session(my_seat=0, teams=12, rounds=15, path=tmp_path / "b.json")
    assert a.session_id != b.session_id
