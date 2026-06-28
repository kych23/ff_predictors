#!/usr/bin/env python
"""CLI: build preseason-safe season features for a season range (§4.5)."""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.db.session import resolve_snapshot
from src.features.assemble import build_features_range


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Build season_features (M2).")
    parser.add_argument("--start", type=int, default=cfg.backtest.start_season)
    parser.add_argument("--end", type=int, default=cfg.backtest.end_season)
    parser.add_argument("--snapshot-id", type=str, default=None,
                        help="Ingest snapshot to use (default: latest).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sid = resolve_snapshot(args.snapshot_id)
    n = build_features_range(args.start, args.end, snapshot_id=sid)
    print(f"season_features rows: {n}")


if __name__ == "__main__":
    main()
