"""CLI: train the weekly projection model and write OOF projections.

Must run AFTER build_weekly_data.py (populates WeeklyLabel + WeeklyFeature).

Usage:
  python scripts/train_weekly_projection.py
  python scripts/train_weekly_projection.py --snapshot-id <id>
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.db.session import resolve_snapshot
from src.projection.weekly_train import train_weekly_and_write


def main() -> None:
    parser = argparse.ArgumentParser(description="Train weekly projection model.")
    parser.add_argument("--snapshot-id", type=str, default=None,
                        help="Ingest snapshot to use (default: latest).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sid = resolve_snapshot(args.snapshot_id)
    n = train_weekly_and_write(snapshot_id=sid)
    print(f"weekly projections written: {n}")


if __name__ == "__main__":
    main()
