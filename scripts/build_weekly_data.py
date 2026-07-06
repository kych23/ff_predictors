"""Build weekly labels + weekly features.

Must run AFTER seed_db.py (populates WeeklyStatsRaw) and train_projection.py
(populates Projection table for season_p50). Step 6 of run_pipeline.sh —
serves the weekly "Who Should I Start?" feature.

Usage:
  python scripts/build_weekly_data.py --start 2017 --end 2025
  python scripts/build_weekly_data.py --start 2017 --end 2025 --snapshot-id <id>
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.db.init_db import init_db
from src.db.session import resolve_snapshot
from src.features.weekly_assemble import build_weekly_features_range
from src.labels.build_labels import build_weekly_labels

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description="Build weekly labels + features")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--snapshot-id", type=str, default=None)
    args = ap.parse_args()

    init_db()
    sid = resolve_snapshot(args.snapshot_id)

    logger.info("building weekly labels...")
    n_labels = build_weekly_labels(snapshot_id=sid)
    logger.info("weekly_labels: %d rows", n_labels)

    logger.info("building weekly features for %d..%d...", args.start, args.end)
    n_feats = build_weekly_features_range(args.start, args.end,
                                          snapshot_id=sid)
    logger.info("weekly_features: %d rows", n_feats)
    logger.info("done.")


if __name__ == "__main__":
    main()
