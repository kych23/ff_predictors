"""Weekly lineup benchmark — points left on bench.

For each (season, week), given weekly projections and actual fantasy points:
  1. Build the ILP-optimal lineup from projected values (recommended)
  2. Build the ILP-optimal lineup from actual values (hindsight-optimal)
  3. Points left on bench = hindsight total - recommended actual total

The recommended lineup is chosen by projections but scored by actuals.
This measures how well the model picks the right starters, not just
whether it predicts point totals accurately.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.config import LeagueConfig, load_config
from src.lineup.optimizer import optimize_lineup

logger = logging.getLogger(__name__)


@dataclass
class WeeklyBenchResult:
    per_week: pd.DataFrame
    mean_pts_left: float
    median_pts_left: float
    optimal_hit_rate: float
    n_weeks: int


def bench_one_week(
    projections: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    cfg: Optional[LeagueConfig] = None,
    strategy: str = "balanced",
) -> dict:
    """Benchmark one week: recommended vs hindsight-optimal lineup.

    `projections`: player_id, position, p10, p50, p90
    `actuals`: player_id, fantasy_points (actual weekly score)

    Returns dict with recommended_total, optimal_total, pts_left_on_bench,
    n_starters, n_correct (starters matching hindsight-optimal).
    """
    cfg = cfg or load_config()

    merged = projections.merge(actuals[["player_id", "fantasy_points"]],
                               on="player_id", how="inner")
    if merged.empty:
        return {"recommended_total": 0, "optimal_total": 0,
                "pts_left_on_bench": 0, "n_starters": 0, "n_correct": 0}

    # Recommended lineup: optimize on projections
    rec_result = optimize_lineup(merged, cfg=cfg, strategy=strategy)
    rec_starter_ids = set(rec_result.starters["player_id"])

    # Score recommended starters by actuals
    rec_actual_total = float(
        merged[merged["player_id"].isin(rec_starter_ids)]["fantasy_points"].sum()
    )

    # Hindsight-optimal: optimize on actual fantasy_points
    hindsight = merged.copy()
    hindsight["p10"] = hindsight["fantasy_points"]
    hindsight["p50"] = hindsight["fantasy_points"]
    hindsight["p90"] = hindsight["fantasy_points"]
    opt_result = optimize_lineup(hindsight, cfg=cfg, strategy="balanced")
    opt_starter_ids = set(opt_result.starters["player_id"])
    opt_total = float(
        merged[merged["player_id"].isin(opt_starter_ids)]["fantasy_points"].sum()
    )

    n_correct = len(rec_starter_ids & opt_starter_ids)

    return {
        "recommended_total": rec_actual_total,
        "optimal_total": opt_total,
        "pts_left_on_bench": opt_total - rec_actual_total,
        "n_starters": len(rec_starter_ids),
        "n_correct": n_correct,
    }


def run_weekly_benchmark(
    weekly_projections: pd.DataFrame,
    weekly_labels: pd.DataFrame,
    *,
    seasons: Optional[Sequence[int]] = None,
    cfg: Optional[LeagueConfig] = None,
    strategy: str = "balanced",
) -> WeeklyBenchResult:
    """Run points-left-on-bench across all (season, week) pairs.

    `weekly_projections`: player_id, season, week, position, p10, p50, p90
    `weekly_labels`: player_id, season, week, fantasy_points
    """
    cfg = cfg or load_config()

    if seasons is not None:
        weekly_projections = weekly_projections[
            weekly_projections["season"].isin(seasons)].copy()
        weekly_labels = weekly_labels[
            weekly_labels["season"].isin(seasons)].copy()

    all_weeks = (
        weekly_projections[["season", "week"]]
        .drop_duplicates()
        .sort_values(["season", "week"])
    )

    rows = []
    for _, sw in all_weeks.iterrows():
        s, w = int(sw["season"]), int(sw["week"])
        proj_wk = weekly_projections[
            (weekly_projections["season"] == s) & (weekly_projections["week"] == w)
        ]
        act_wk = weekly_labels[
            (weekly_labels["season"] == s) & (weekly_labels["week"] == w)
        ]
        if proj_wk.empty or act_wk.empty:
            continue
        result = bench_one_week(proj_wk, act_wk, cfg=cfg, strategy=strategy)
        result["season"] = s
        result["week"] = w
        rows.append(result)

    if not rows:
        return WeeklyBenchResult(
            per_week=pd.DataFrame(), mean_pts_left=float("nan"),
            median_pts_left=float("nan"), optimal_hit_rate=float("nan"), n_weeks=0,
        )

    per_week = pd.DataFrame(rows)
    pts_left = per_week["pts_left_on_bench"]
    total_correct = per_week["n_correct"].sum()
    total_starters = per_week["n_starters"].sum()
    hit_rate = total_correct / total_starters if total_starters > 0 else 0.0

    return WeeklyBenchResult(
        per_week=per_week,
        mean_pts_left=float(pts_left.mean()),
        median_pts_left=float(pts_left.median()),
        optimal_hit_rate=hit_rate,
        n_weeks=len(per_week),
    )
