#!/usr/bin/env python
"""Thin CLI over ``ingest.run_ingest`` (DrafterSpec.md §4.2).

Recreates the schema (idempotent) and pulls every raw source into the fresh DB.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.ingest.run_ingest import run_ingest


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Seed the DB from public sources (M0).")
    parser.add_argument("--start", type=int, default=cfg.backtest.start_season,
                        help="Start season (inclusive)")
    parser.add_argument("--end", type=int, default=cfg.backtest.end_season,
                        help="End season (inclusive)")
    parser.add_argument("--no-adp", action="store_true", help="Skip ADP ingestion")
    parser.add_argument("--notes", type=str, default="", help="Snapshot notes")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    snapshot_id = run_ingest(args.start, args.end, with_adp=not args.no_adp, notes=args.notes)
    print(f"Ingest complete. snapshot_id={snapshot_id}")


if __name__ == "__main__":
    main()
