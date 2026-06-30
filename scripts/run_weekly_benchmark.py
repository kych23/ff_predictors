#!/usr/bin/env python
"""CLI: run the weekly lineup benchmark (points left on bench).

Must run AFTER train_weekly_projection.py (populates WeeklyProjection) and
build_weekly_data.py (populates WeeklyLabel).

Usage:
  python scripts/run_weekly_benchmark.py
  python scripts/run_weekly_benchmark.py --start 2017 --end 2025 --strategy balanced
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
from sqlalchemy import select

from src.benchmark.weekly_bench import run_weekly_benchmark
from src.config import load_config
from src.db.models import WeeklyLabel, WeeklyProjection
from src.db.session import SessionLocal

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Weekly lineup benchmark.")
    parser.add_argument("--start", type=int, default=cfg.backtest.start_season)
    parser.add_argument("--end", type=int, default=cfg.backtest.end_season)
    parser.add_argument("--strategy", type=str, default="balanced",
                        choices=["safe", "balanced", "upside"])
    args = parser.parse_args()

    session = SessionLocal()
    try:
        proj_rows = session.execute(select(WeeklyProjection)).scalars().all()
        projections = pd.DataFrame([
            {"player_id": r.player_id, "season": r.season, "week": r.week,
             "position": r.position, "p10": r.p10, "p50": r.p50, "p90": r.p90}
            for r in proj_rows
        ])

        label_rows = session.execute(select(WeeklyLabel)).scalars().all()
        labels = pd.DataFrame([
            {"player_id": r.player_id, "season": r.season, "week": r.week,
             "fantasy_points": r.fantasy_points}
            for r in label_rows
        ])
    finally:
        session.close()

    if projections.empty:
        print("No weekly projections found. Run train_weekly_projection.py first.")
        return
    if labels.empty:
        print("No weekly labels found. Run build_weekly_data.py first.")
        return

    seasons = list(range(args.start, args.end + 1))
    logger.info("running weekly benchmark for %d..%d (strategy=%s)",
                args.start, args.end, args.strategy)

    result = run_weekly_benchmark(projections, labels, seasons=seasons,
                                  strategy=args.strategy)

    if result.n_weeks == 0:
        print("No overlapping (season, week) pairs between projections and labels.")
        return

    print(f"\n===== Weekly Lineup Benchmark (strategy: {args.strategy}) =====")
    print(f"  Weeks evaluated: {result.n_weeks}")
    print(f"  Mean pts left on bench:   {result.mean_pts_left:.2f}")
    print(f"  Median pts left on bench: {result.median_pts_left:.2f}")
    print(f"  Optimal starter hit rate: {result.optimal_hit_rate:.1%}")
    print(f"  Target: <5.0 mean pts left on bench")
    target_met = result.mean_pts_left < 5.0
    print(f"  GATE {'PASS' if target_met else 'FAIL'}")

    # Per-season summary
    per_season = (
        result.per_week
        .groupby("season")
        .agg(
            weeks=("week", "count"),
            mean_pts_left=("pts_left_on_bench", "mean"),
            median_pts_left=("pts_left_on_bench", "median"),
            mean_rec_total=("recommended_total", "mean"),
            mean_opt_total=("optimal_total", "mean"),
        )
        .reset_index()
    )
    print("\n  Per-season breakdown:")
    print(per_season.to_string(index=False))


if __name__ == "__main__":
    main()
