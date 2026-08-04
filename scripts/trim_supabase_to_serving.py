#!/usr/bin/env python
"""Trim the serving DB (Supabase) down to the serving subset to reclaim space.

Deletes research-only tables' data (kept in full in the local research DB) and
prunes weekly_features to the served seasons, then VACUUM FULLs each trimmed
table to return physical space. DESTRUCTIVE — run only after
scripts/migrate_to_local_research.py --verify reports PASS.

  python scripts/trim_supabase_to_serving.py --dry-run   # show what WOULD go
  python scripts/trim_supabase_to_serving.py --execute    # actually trim + vacuum
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from src.db.session import get_engine

# Fully cleared on serving (research-only; full copy lives in the research DB).
DELETE_TABLES = [
    "weekly_stats_raw",
    "weekly_labels",
    "season_features",
    "season_labels",
    "player_id_map",
]
# Kept on serving, but pruned to the served seasons only.
SERVED_SEASONS = (2024, 2026)  # demo (2024) + live draft/weekly (2026)


def _db_size(conn) -> str:
    return conn.execute(text("SELECT pg_size_pretty(pg_database_size(current_database()))")).scalar()


def _count(conn, sql: str) -> int:
    return conn.execute(text(sql)).scalar() or 0


def report(conn) -> None:
    print(f"current serving DB size: {_db_size(conn)}\n")
    print(f"{'table':22s}{'total rows':>12}{'to delete':>12}")
    for t in DELETE_TABLES:
        n = _count(conn, f"SELECT count(*) FROM {t}")
        print(f"{t:22s}{n:>12}{n:>12}   (drop all)")
    wf_total = _count(conn, "SELECT count(*) FROM weekly_features")
    kept = ", ".join(str(s) for s in SERVED_SEASONS)
    wf_del = _count(conn,
                    f"SELECT count(*) FROM weekly_features WHERE season NOT IN ({kept})")
    print(f"{'weekly_features':22s}{wf_total:>12}{wf_del:>12}   (keep {kept})")


def execute(conn) -> None:
    for t in DELETE_TABLES:
        conn.execute(text(f"DELETE FROM {t}"))
        print(f"cleared {t}")
    kept = ", ".join(str(s) for s in SERVED_SEASONS)
    conn.execute(text(f"DELETE FROM weekly_features WHERE season NOT IN ({kept})"))
    print(f"pruned weekly_features to seasons {kept}")
    for t in DELETE_TABLES + ["weekly_features"]:
        print(f"VACUUM FULL {t} …")
        conn.execute(text(f"VACUUM FULL {t}"))
    print(f"\nserving DB size after trim: {_db_size(conn)}")


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    args = p.parse_args()
    # AUTOCOMMIT so VACUUM FULL (cannot run in a transaction block) works.
    eng = get_engine("serving")
    with eng.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        if args.dry_run:
            report(conn)
            print("\n(dry-run — nothing deleted)")
        else:
            report(conn)
            execute(conn)


if __name__ == "__main__":
    main()
