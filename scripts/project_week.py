"""CLI: project an upcoming week and write forward weekly projections.

The live-serving path for "Who Should I Start?": builds as-of-kickoff features
for the target (season, week), trains the weekly model on all strictly-prior
labeled weeks, widens intervals with the OOF-fitted serving calibrator, and
writes projections under model_version ``weekly_fwd_v1.*``. start.py then
prefers these over OOF rows (and no longer falls back to season P50).

Must run AFTER build_weekly_data.py (weekly labels/features for training) and
train_projection.py (season P50 prior).

Usage:
  python scripts/project_week.py --season 2026 --week 1
  python scripts/project_week.py --season 2026 --week 1 --snapshot-id <id>
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.db.init_db import init_db
from src.db.session import resolve_snapshot
from src.projection.weekly_train import project_weekly_forward_and_write


def main() -> None:
    parser = argparse.ArgumentParser(description="Project an upcoming NFL week.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--snapshot-id", type=str, default=None,
                        help="Ingest snapshot to use (default: latest).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    init_db()
    sid = resolve_snapshot(args.snapshot_id)
    n = project_weekly_forward_and_write(args.season, args.week, snapshot_id=sid)
    print(f"forward weekly projections written: {n}")
    if n:
        print(f"next: python scripts/start.py --season {args.season} "
              f"--week {args.week} --roster <roster.yaml>")


if __name__ == "__main__":
    main()
