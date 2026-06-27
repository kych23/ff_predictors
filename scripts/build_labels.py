#!/usr/bin/env python
"""CLI: build per-game season labels (DrafterSpec.md §4.4)."""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.labels.build_labels import build_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Build season_labels from weekly_stats_raw.")
    parser.add_argument("--snapshot-id", type=str, default=None,
                        help="Restrict to one ingest snapshot (recommended).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    n = build_labels(snapshot_id=args.snapshot_id)
    print(f"season_labels rows: {n}")


if __name__ == "__main__":
    main()
