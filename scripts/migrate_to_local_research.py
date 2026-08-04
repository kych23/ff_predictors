#!/usr/bin/env python
"""One-time migration: copy the full Supabase (serving) DB into the local
research DB, table by table, via SQLAlchemy core (streamed + batched).

Used instead of pg_dump because the Supabase server (PG 17) is newer than the
local client (PG 15) and pg_dump refuses the mismatch. Non-destructive: reads
Supabase, writes local research. Idempotent per table (truncates the local
table first, then re-inserts).

  python scripts/migrate_to_local_research.py           # all tables
  python scripts/migrate_to_local_research.py --verify  # just compare row counts
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sqlalchemy import delete, func, insert, select

from src.db.init_db import init_db
from src.db.models import Base
from src.db.session import SERVING, RESEARCH, get_engine, session_scope

BATCH = 5000


def _tables():
    # Base.metadata includes product tables (draft_sessions) after init_db's import.
    return list(Base.metadata.sorted_tables)


def _count(session, tbl) -> int:
    return session.execute(select(func.count()).select_from(tbl)).scalar() or 0


def migrate() -> None:
    init_db()  # ensure local research has the full schema
    with session_scope(SERVING) as src, session_scope(RESEARCH) as dst:
        for tbl in _tables():
            total = _count(src, tbl)
            if total == 0:
                # still clear the local copy so a re-run stays idempotent
                dst.execute(delete(tbl)); dst.commit()
                print(f"{tbl.name:24s} 0 rows (skipped)")
                continue
            dst.execute(delete(tbl)); dst.commit()
            moved = 0
            result = src.execute(select(tbl),
                                 execution_options={"stream_results": True})
            batch = []
            for row in result:
                batch.append(dict(row._mapping))
                if len(batch) >= BATCH:
                    dst.execute(insert(tbl), batch); dst.commit()
                    moved += len(batch); batch = []
            if batch:
                dst.execute(insert(tbl), batch); dst.commit()
                moved += len(batch)
            print(f"{tbl.name:24s} {moved}/{total} rows")


def verify() -> int:
    try:
        import api.db_models  # noqa: F401 — register product tables
    except Exception:
        pass
    ok = True
    with session_scope(SERVING) as src, session_scope(RESEARCH) as dst:
        print(f"{'table':24s}{'supabase':>12}{'local':>12}{'':>4}")
        for tbl in _tables():
            s = _count(src, tbl)
            d = _count(dst, tbl)
            flag = "OK" if d >= s else "MISMATCH"
            if d < s:
                ok = False
            print(f"{tbl.name:24s}{s:>12}{d:>12}   {flag}")
    print("\nVERIFY:", "PASS — local is a superset" if ok else "FAIL — local missing rows")
    return 0 if ok else 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true", help="only compare row counts")
    args = p.parse_args()
    if args.verify:
        raise SystemExit(verify())
    migrate()
    raise SystemExit(verify())


if __name__ == "__main__":
    main()
