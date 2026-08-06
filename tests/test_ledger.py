"""Decision ledger (§19, §20.2).

The ledger exists so that in November I can see what the engine said in August,
before the season was known. A test suite that only checks appends and reads
would miss the point — the tests that matter are the ones that tamper.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from src.app.cockpit.ledger import (
    GENESIS_HASH, DecisionLedger, canonical_json, compute_hash,
)


@pytest.fixture
def ledger(tmp_path):
    with DecisionLedger(tmp_path / "draft.sqlite") as led:
        yield led


def _rec(leader="p1", tier=0):
    return {"leader": leader, "tier": tier,
            "indifference_set": [leader, "p2"], "dollars": {leader: 34.2}}


def _fill(ledger, n=4):
    for i in range(n):
        ledger.append("sess", pick_no=i + 1, tier=0,
                      recommendation=_rec(f"p{i}"),
                      actual_pick=f"p{i}", snapshot_id="sn2_abc")


# ------------------------------------------------------------- the chain
def test_genesis_links_to_sixty_four_zeros(ledger):
    entry = ledger.append("sess", 1, 0, _rec(), snapshot_id="sn2_abc")
    assert entry.prev_hash == GENESIS_HASH
    assert len(entry.entry_hash) == 64


def test_each_entry_links_to_the_previous(ledger):
    a = ledger.append("sess", 1, 0, _rec("p1"), snapshot_id="sn2_abc")
    b = ledger.append("sess", 2, 0, _rec("p2"), snapshot_id="sn2_abc")
    assert b.prev_hash == a.entry_hash


def test_a_clean_chain_verifies(ledger):
    _fill(ledger)
    result = ledger.verify()
    assert result.ok, result.detail
    assert result.broken_at is None


def test_editing_a_recommendation_is_detected(ledger):
    """The scenario the chain is FOR: after a bust, quietly rewriting what the
    engine recommended so it looks like it warned you."""
    _fill(ledger)
    conn = sqlite3.connect(ledger.path)
    conn.execute("UPDATE decision_ledger SET recommendation = ? WHERE id = 2",
                 (canonical_json(_rec("SOMEONE_ELSE")),))
    conn.commit()
    conn.close()

    result = ledger.verify()
    assert not result.ok
    assert result.broken_at == 2
    assert "edited after the fact" in result.detail


def test_editing_the_actual_pick_is_detected(ledger):
    _fill(ledger)
    conn = sqlite3.connect(ledger.path)
    conn.execute("UPDATE decision_ledger SET actual_pick = 'ghost' WHERE id = 3")
    conn.commit()
    conn.close()
    assert ledger.verify().broken_at == 3


def test_deleting_an_entry_breaks_the_link(ledger):
    _fill(ledger)
    conn = sqlite3.connect(ledger.path)
    conn.execute("DELETE FROM decision_ledger WHERE id = 2")
    conn.commit()
    conn.close()

    result = ledger.verify()
    assert not result.ok
    assert result.broken_at == 3
    assert "chain is at" in result.detail


def test_only_the_first_break_is_reported(ledger):
    """Every entry after a tampered one has a wrong prev_hash by construction.
    Listing them all buries the one that matters."""
    _fill(ledger, n=6)
    conn = sqlite3.connect(ledger.path)
    conn.execute("UPDATE decision_ledger SET tier = 9 WHERE id = 2")
    conn.commit()
    conn.close()
    assert ledger.verify().broken_at == 2


def test_hash_covers_every_declared_field():
    """If a field is in HASHED_FIELDS it must actually change the hash —
    otherwise the ledger advertises protection it does not provide."""
    from dataclasses import replace

    from src.app.cockpit.ledger import HASHED_FIELDS, LedgerEntry

    base = LedgerEntry(id=1, session_id="s", pick_no=1, tier=0,
                       recommendation=_rec(), actual_pick="p1",
                       snapshot_id="sn2_abc")
    reference = compute_hash(GENESIS_HASH, base)
    mutations = {"id": 2, "session_id": "other", "pick_no": 99, "tier": 3,
                 "recommendation": _rec("zzz"), "actual_pick": "zzz"}
    for field_name in HASHED_FIELDS:
        mutated = replace(base, **{field_name: mutations[field_name]})
        assert compute_hash(GENESIS_HASH, mutated) != reference, field_name


# ------------------------------------------------------ canonicalization
def test_key_order_does_not_change_the_hash():
    """json.dumps with default settings emits dict order, so two runs of the
    same data would hash differently and a clean chain would look tampered."""
    a = {"leader": "p1", "tier": 0}
    b = {"tier": 0, "leader": "p1"}
    assert canonical_json(a) == canonical_json(b)


def test_nan_is_rejected_rather_than_written_as_a_non_json_token():
    """`NaN` is not valid JSON. Writing it produces a file that reloads in
    Python and fails everywhere else, which is the worst kind of corruption:
    invisible until export."""
    with pytest.raises(ValueError):
        canonical_json({"p_best": float("nan")})


# ------------------------------------------------------------ semantics
def test_followed_is_derived_not_asserted(ledger):
    """'I followed the engine' and 'the engine said X, I picked Y' must not be
    able to disagree."""
    a = ledger.append("s", 1, 0, _rec("p1"), actual_pick="p1",
                      snapshot_id="sn2_abc")
    b = ledger.append("s", 2, 0, _rec("p1"), actual_pick="p9",
                      snapshot_id="sn2_abc")
    assert a.followed is True
    assert b.followed is False


def test_pending_picks_have_no_follow_verdict(ledger):
    entry = ledger.append("s", 1, 0, _rec(), snapshot_id="sn2_abc")
    assert entry.followed is None
    import math
    assert math.isnan(ledger.follow_rate())


def test_follow_rate_ignores_pending_picks(ledger):
    ledger.append("s", 1, 0, _rec("p1"), actual_pick="p1", snapshot_id="sn")
    ledger.append("s", 2, 0, _rec("p2"), actual_pick="p9", snapshot_id="sn")
    ledger.append("s", 3, 0, _rec("p3"), snapshot_id="sn")
    assert ledger.follow_rate() == pytest.approx(0.5)


def test_a_repeated_pick_number_is_a_correction_not_an_error(ledger):
    """`undo` then re-pick is the commonest draft-night correction. A UNIQUE
    constraint made it raise `sqlite3.IntegrityError` mid-draft — the chaos
    rehearsal crashed at pick 45 doing exactly that. An append-only chain
    records a correction by appending."""
    ledger.append("s", 1, 0, _rec("first"), actual_pick="first",
                  snapshot_id="sn")
    ledger.append("s", 1, 0, _rec("second"), actual_pick="second",
                  snapshot_id="sn")
    assert len(ledger) == 2
    assert ledger.verify().ok
    latest = ledger.latest_by_pick()[("s", 1)]
    assert latest.actual_pick == "second"
    assert len(ledger.superseded()) == 1


def test_follow_rate_counts_a_corrected_pick_once(ledger):
    """Otherwise an undo would weight that pick twice."""
    ledger.append("s", 1, 0, _rec("p1"), actual_pick="other", snapshot_id="sn")
    ledger.append("s", 1, 0, _rec("p1"), actual_pick="p1", snapshot_id="sn")
    assert ledger.follow_rate() == pytest.approx(1.0)


def test_the_ledger_reopens_and_keeps_chaining(tmp_path):
    """Draft night is exactly when a process gets restarted."""
    path = tmp_path / "draft.sqlite"
    with DecisionLedger(path) as led:
        _fill(led, n=3)
        tip = led.tip()
    with DecisionLedger(path) as reopened:
        entry = reopened.append("sess", 4, 1, _rec("p9"), snapshot_id="sn2_abc")
        assert entry.prev_hash == tip.entry_hash
        assert reopened.verify().ok
        assert len(reopened) == 4


def test_recommendation_round_trips_through_json(ledger):
    rec = _rec()
    ledger.append("s", 1, 0, rec, snapshot_id="sn")
    assert list(ledger)[0].recommendation == json.loads(canonical_json(rec))
