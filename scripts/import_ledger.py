"""Import the draft-night decision ledger into the research DB (§20.2).

    venv/bin/python scripts/import_ledger.py [--ledger data/draft.sqlite]

The ledger is written to local SQLite during the draft because the
recommendation path must not require a network (§19). This moves it into the
research database afterwards, where it can be joined against outcomes.

**Verifies the chain before importing anything.** A broken chain is not
imported at all: the ledger's only value is that it is trustworthy, and
copying an edited chain into the research DB would launder exactly the
tampering the hashes exist to detect.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.cockpit.ledger import DecisionLedger  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", type=Path, default=Path("data/draft.sqlite"))
    ap.add_argument("--dry-run", action="store_true",
                    help="verify and summarize without writing")
    args = ap.parse_args()

    if not args.ledger.exists():
        print(f"no ledger at {args.ledger}")
        return 1

    with DecisionLedger(args.ledger) as ledger:
        entries = list(ledger)
        print(f"§20.2 LEDGER IMPORT — {args.ledger}")
        print(f"  {len(entries)} entries\n")

        result = ledger.verify()
        if not result.ok:
            print(f"  CHAIN BROKEN at entry {result.broken_at}")
            print(f"  {result.detail}")
            print("\n  REFUSING TO IMPORT. The ledger's only value is that it")
            print("  is trustworthy; copying an edited chain into the research")
            print("  DB would launder the tampering the hashes exist to catch.")
            return 1
        print(f"  chain verified: {result.detail}")

        by_tier: dict[int, int] = {}
        for entry in entries:
            by_tier[entry.tier] = by_tier.get(entry.tier, 0) + 1
        print("  picks by active tier: "
              + ", ".join(f"tier {t}: {n}" for t, n in sorted(by_tier.items())))
        rate = ledger.follow_rate()
        print(f"  follow rate: {rate:.1%}" if rate == rate
              else "  follow rate: n/a (no resolved picks)")

        if args.dry_run:
            print("\n  dry run — nothing written")
            return 0

        from sqlalchemy import Column, Integer, String, Text
        from sqlalchemy.orm import declarative_base

        from src.db.session import RESEARCH, get_engine, session_scope

        # LedgerBase is separate from the research Base on purpose (§20.2):
        # nothing else may create these tables, and the draft-night writer
        # never touches Postgres.
        LedgerBase = declarative_base()

        class DecisionLedgerRow(LedgerBase):
            __tablename__ = "decision_ledger"
            id = Column(Integer, primary_key=True)
            session_id = Column(String, primary_key=True)
            pick_no = Column(Integer, nullable=False)
            sim_run_id = Column(String, nullable=True)
            tier = Column(Integer, nullable=False)
            recommendation = Column(Text, nullable=False)
            actual_pick = Column(String, nullable=True)
            followed = Column(Integer, nullable=True)
            prev_hash = Column(String(64), nullable=False)
            entry_hash = Column(String(64), nullable=False)
            snapshot_id = Column(String, nullable=False)

        from src.app.cockpit.ledger import canonical_json

        engine = get_engine(RESEARCH)
        LedgerBase.metadata.create_all(bind=engine)
        with session_scope(RESEARCH) as session:
            for entry in entries:
                session.merge(DecisionLedgerRow(
                    id=entry.id, session_id=entry.session_id,
                    pick_no=entry.pick_no, sim_run_id=entry.sim_run_id,
                    tier=entry.tier,
                    recommendation=canonical_json(entry.recommendation),
                    actual_pick=entry.actual_pick,
                    followed=(None if entry.followed is None
                              else int(entry.followed)),
                    prev_hash=entry.prev_hash, entry_hash=entry.entry_hash,
                    snapshot_id=entry.snapshot_id,
                ))
        print(f"\n  imported {len(entries)} entries into the research DB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
