#!/usr/bin/env python
"""Copy the serving subset from the research DB to the serving DB.

Run after a pipeline/training run so the API (Supabase serving DB) picks up
fresh projections/adp/players. Reads RESEARCH_DATABASE_URL, writes DATABASE_URL.

  python scripts/publish_serving.py                  # weekly_features: latest season
  python scripts/publish_serving.py --wf-season 2026 # pin weekly_features season
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.db.publish import publish_serving


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish serving subset research->serving.")
    parser.add_argument("--wf-season", type=int, default=None,
                        help="weekly_features season to publish (default: latest present)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    counts = publish_serving(weekly_feature_season=args.wf_season)
    print("published:", ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
