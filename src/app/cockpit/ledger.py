"""Hash-chained decision ledger (§19, §20.2). Local SQLite, no network.

Every recommendation and the pick actually made are appended to a chain:

    entry_hash = sha256(prev_hash ‖ canonical_json({id, session_id, pick_no,
                                                    tier, recommendation,
                                                    actual_pick}))

Genesis `prev_hash` is 64 zeros.

Why a chain and not a table. The ledger's job is answering, in November, "what
did the engine actually say at pick 4, before I knew how the season went?" A
plain table answers that only if nobody edited it, and the person most likely
to edit it is the person with the most reason to — after a bust, the temptation
to remember the engine as having warned you is real. A chain makes an edit
detectable rather than requiring discipline.

**JSON, not JSONB.** This is SQLite on a laptop during a draft, deliberately
outside Postgres: the recommendation path must not require a network. The
ledger is imported into the research DB afterwards by `scripts/import_ledger.py`.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.core.errors import DataError

GENESIS_HASH = "0" * 64

#: Fields covered by the hash, in this order. Adding a field here breaks every
#: existing chain, which is the correct behaviour — the hash is a statement
#: about exactly this content.
HASHED_FIELDS = ("id", "session_id", "pick_no", "tier", "recommendation",
                 "actual_pick")

SCHEMA = """
CREATE TABLE IF NOT EXISTS decision_ledger (
    id            INTEGER PRIMARY KEY,
    session_id    TEXT NOT NULL,
    pick_no       INTEGER NOT NULL,
    sim_run_id    TEXT,
    tier          INTEGER NOT NULL,
    recommendation TEXT NOT NULL,
    actual_pick   TEXT,
    followed      INTEGER,
    prev_hash     TEXT NOT NULL,
    entry_hash    TEXT NOT NULL,
    snapshot_id   TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
"""

#: NO uniqueness on (session_id, pick_no). A pick can legitimately be recorded
#: twice: `undo` followed by a re-pick is the single most common draft-night
#: correction, and an append-only chain records a correction by APPENDING, not
#: by rewriting. A UNIQUE constraint here made that sequence raise
#: `sqlite3.IntegrityError` mid-draft — found by the §22.1 chaos rehearsal,
#: which crashed at pick 45 doing exactly what a person does when someone
#: misreads a name. The chain keeps both entries; `latest_by_pick` decides
#: which one counts.


@dataclass(frozen=True)
class LedgerEntry:
    id: int
    session_id: str
    pick_no: int
    tier: int
    recommendation: dict[str, Any]
    actual_pick: str | None
    snapshot_id: str
    sim_run_id: str | None = None
    followed: bool | None = None
    prev_hash: str = GENESIS_HASH
    entry_hash: str = ""
    created_at: str = ""


def canonical_json(payload: dict[str, Any]) -> str:
    """Sorted keys, no whitespace, no NaN.

    Canonicalization is the whole hash. `json.dumps` with default settings
    emits dict order and platform float repr, so two runs of the same data can
    hash differently and the chain looks tampered when it is not.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=str)


def compute_hash(prev_hash: str, entry: LedgerEntry) -> str:
    payload = {f: getattr(entry, f) for f in HASHED_FIELDS}
    return hashlib.sha256(
        (prev_hash + canonical_json(payload)).encode("utf-8")).hexdigest()


class DecisionLedger:
    """Append-only chain over a local SQLite file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> DecisionLedger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------ writing
    def append(self, session_id: str, pick_no: int, tier: int,
               recommendation: dict[str, Any], *, snapshot_id: str,
               actual_pick: str | None = None, sim_run_id: str | None = None,
               created_at: str = "") -> LedgerEntry:
        """Append one decision. `followed` is derived, never passed in.

        Deriving it removes the only field a user could set inconsistently
        with the data beside it — "I followed the engine" and "the engine said
        X, I picked Y" must not be able to disagree.
        """
        prev = self.tip()
        prev_hash = prev.entry_hash if prev else GENESIS_HASH
        next_id = (prev.id + 1) if prev else 1
        leader = recommendation.get("leader")
        followed = None if actual_pick is None else (actual_pick == leader)

        entry = LedgerEntry(
            id=next_id, session_id=session_id, pick_no=pick_no, tier=tier,
            recommendation=recommendation, actual_pick=actual_pick,
            snapshot_id=snapshot_id, sim_run_id=sim_run_id, followed=followed,
            prev_hash=prev_hash, created_at=created_at,
        )
        entry_hash = compute_hash(prev_hash, entry)
        self._conn.execute(
            "INSERT INTO decision_ledger (id, session_id, pick_no, sim_run_id,"
            " tier, recommendation, actual_pick, followed, prev_hash,"
            " entry_hash, snapshot_id, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (entry.id, entry.session_id, entry.pick_no, entry.sim_run_id,
             entry.tier, canonical_json(entry.recommendation),
             entry.actual_pick,
             None if followed is None else int(followed),
             entry.prev_hash, entry_hash, entry.snapshot_id, entry.created_at),
        )
        self._conn.commit()
        return LedgerEntry(**{**asdict(entry), "entry_hash": entry_hash})

    # ------------------------------------------------------------ reading
    def _row_to_entry(self, row: sqlite3.Row) -> LedgerEntry:
        return LedgerEntry(
            id=row["id"], session_id=row["session_id"], pick_no=row["pick_no"],
            tier=row["tier"], recommendation=json.loads(row["recommendation"]),
            actual_pick=row["actual_pick"], snapshot_id=row["snapshot_id"],
            sim_run_id=row["sim_run_id"],
            followed=None if row["followed"] is None else bool(row["followed"]),
            prev_hash=row["prev_hash"], entry_hash=row["entry_hash"],
            created_at=row["created_at"],
        )

    def __iter__(self) -> Iterator[LedgerEntry]:
        for row in self._conn.execute(
                "SELECT * FROM decision_ledger ORDER BY id"):
            yield self._row_to_entry(row)

    def __len__(self) -> int:
        return int(self._conn.execute(
            "SELECT COUNT(*) FROM decision_ledger").fetchone()[0])

    def tip(self) -> LedgerEntry | None:
        row = self._conn.execute(
            "SELECT * FROM decision_ledger ORDER BY id DESC LIMIT 1").fetchone()
        return self._row_to_entry(row) if row else None

    def sessions(self) -> list[dict[str, Any]]:
        """One row per draft in the ledger, newest first."""
        rows = self._conn.execute(
            "SELECT session_id, COUNT(*) AS entries, MIN(created_at) AS first,"
            " MAX(created_at) AS last, MAX(id) AS max_id"
            " FROM decision_ledger GROUP BY session_id ORDER BY max_id DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, session_id: str) -> int:
        """Erase one draft's entries. Returns how many were removed.

        **Refuses anything but the tail of the chain**, and the refusal is the
        point. Every entry stores the hash of the one before it, so removing
        entries from the middle leaves everything after them claiming a
        `prev_hash` that no longer exists — `verify()` would report the file as
        tampered, which is indistinguishable from someone actually tampering
        with it. A delete that silently destroys the ledger's one guarantee is
        worse than a delete that refuses.

        Deleting the most recent draft IS safe: its entries are a contiguous
        suffix, so what remains is still a valid chain from genesis. That is
        the case the UI's delete button exercises, since you can only clear the
        draft you are in.
        """
        rows = self._conn.execute(
            "SELECT id FROM decision_ledger WHERE session_id = ? ORDER BY id",
            (session_id,)).fetchall()
        if not rows:
            return 0
        ids = [int(r["id"]) for r in rows]
        tail = [int(r["id"]) for r in self._conn.execute(
            "SELECT id FROM decision_ledger ORDER BY id DESC LIMIT ?",
            (len(ids),)).fetchall()]
        if sorted(tail) != ids:
            raise DataError(
                f"session {session_id} is not the newest in the ledger; "
                f"deleting it would break the hash chain for every entry "
                f"after it. Archive the draft instead."
            )
        self._conn.execute(
            "DELETE FROM decision_ledger WHERE session_id = ?", (session_id,))
        self._conn.commit()
        return len(ids)

    # ---------------------------------------------------------- verifying
    def verify(self) -> ChainVerification:
        """Recompute the whole chain. Reports the FIRST broken link.

        Reporting only the first is deliberate: every entry after a tampered
        one has a wrong `prev_hash` by construction, so listing them all buries
        the one that matters under noise.
        """
        expected_prev = GENESIS_HASH
        for entry in self:
            if entry.prev_hash != expected_prev:
                return ChainVerification(
                    False, entry.id,
                    f"entry {entry.id} claims prev_hash {entry.prev_hash[:12]}…"
                    f" but the chain is at {expected_prev[:12]}…")
            recomputed = compute_hash(entry.prev_hash, entry)
            if recomputed != entry.entry_hash:
                return ChainVerification(
                    False, entry.id,
                    f"entry {entry.id} content does not match its hash "
                    f"(stored {entry.entry_hash[:12]}…, recomputed "
                    f"{recomputed[:12]}…) — it was edited after the fact")
            expected_prev = entry.entry_hash
        return ChainVerification(True, None, f"{len(self)} entries verified")

    def latest_by_pick(self) -> dict[tuple[str, int], LedgerEntry]:
        """The entry that COUNTS for each pick — the last one appended.

        Earlier entries for the same pick are corrections that were superseded.
        They stay in the chain because deleting them would break it, and
        because "I undid this and took someone else" is exactly the kind of
        thing worth being able to see in November.
        """
        out: dict[tuple[str, int], LedgerEntry] = {}
        for entry in self:
            out[(entry.session_id, entry.pick_no)] = entry
        return out

    def superseded(self) -> list[LedgerEntry]:
        """Entries later corrected. Kept, never deleted.

        Keyed on the ledger's own row id, NOT Python's ``id()``. Both
        ``latest_by_pick`` and ``__iter__`` build fresh ``LedgerEntry`` objects
        from their own cursors, so the two sets never share object identity —
        whether ``id(e)`` happened to match came down to CPython reusing the
        address of a just-freed object, which made this return every entry or
        none depending on unrelated allocation. The row id is the stable
        identity the table already provides.
        """
        winners = {e.id for e in self.latest_by_pick().values()}
        return [e for e in self if e.id not in winners]

    def follow_rate(self) -> float:
        """Fraction of resolved picks where I took the engine's leader.

        Counts each pick ONCE, using its final entry — otherwise an undo would
        weight that pick twice.
        """
        rows = [e for e in self.latest_by_pick().values()
                if e.followed is not None]
        return sum(e.followed for e in rows) / len(rows) if rows else float("nan")


@dataclass(frozen=True)
class ChainVerification:
    ok: bool
    broken_at: int | None
    detail: str
    tampered_fields: list[str] = field(default_factory=list)
