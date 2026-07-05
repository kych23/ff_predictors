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


ROSTER_DEPTH = {"QB": 2, "RB": 5, "WR": 5, "TE": 2}

# Evaluate early / mid / late draft slots and average, so the headline metric
# isn't conditioned on a single draft position.
BENCH_TEAM_SLOTS = (0, 5, 11)


def _simulate_roster(df: pd.DataFrame, n_teams: int = 12,
                     team_idx: int = 5) -> pd.DataFrame:
    """Simulate one team's roster via round-robin draft from the player pool.

    Ranks players per position by P50, assigns them to ``n_teams`` teams
    round-robin (snake-style), and returns the roster for ``team_idx``.
    Produces a realistic mid-pack roster instead of all-stars.
    Skips filtering when pool is too small for meaningful draft simulation.
    """
    total_depth = sum(ROSTER_DEPTH.values())
    if len(df) <= total_depth * 2:
        return df

    pieces = []
    for pos, depth in ROSTER_DEPTH.items():
        pos_pool = df[df["position"] == pos].sort_values("p50", ascending=False)
        total_slots = depth * n_teams
        pos_pool = pos_pool.head(total_slots).reset_index(drop=True)
        team_assignments = []
        for i in range(len(pos_pool)):
            rnd = i // n_teams
            pick_in_round = i % n_teams
            team = pick_in_round if rnd % 2 == 0 else (n_teams - 1 - pick_in_round)
            team_assignments.append(team)
        pos_pool["_team"] = team_assignments
        pieces.append(pos_pool[pos_pool["_team"] == team_idx].drop(columns=["_team"]))
    other = df[~df["position"].isin(ROSTER_DEPTH)]
    if not other.empty:
        pieces.append(other.head(2))
    return pd.concat(pieces, ignore_index=True) if pieces else df


def _bench_one_roster(roster: pd.DataFrame, cfg: LeagueConfig,
                      strategy: str) -> dict:
    """Recommended vs hindsight-optimal lineup for ONE simulated roster."""
    rec_result = optimize_lineup(roster, cfg=cfg, strategy=strategy)
    rec_starter_ids = set(rec_result.starters["player_id"])
    rec_actual_total = float(
        roster[roster["player_id"].isin(rec_starter_ids)]["fantasy_points"].sum()
    )

    # Hindsight-optimal: optimize on actual fantasy_points
    hindsight = roster.copy()
    hindsight["p10"] = hindsight["fantasy_points"]
    hindsight["p50"] = hindsight["fantasy_points"]
    hindsight["p90"] = hindsight["fantasy_points"]
    opt_result = optimize_lineup(hindsight, cfg=cfg, strategy="balanced")
    opt_starter_ids = set(opt_result.starters["player_id"])
    opt_total = float(
        roster[roster["player_id"].isin(opt_starter_ids)]["fantasy_points"].sum()
    )

    return {
        "recommended_total": rec_actual_total,
        "optimal_total": opt_total,
        "pts_left_on_bench": opt_total - rec_actual_total,
        "n_starters": len(rec_starter_ids),
        "n_correct": len(rec_starter_ids & opt_starter_ids),
    }


def bench_one_week(
    projections: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    cfg: Optional[LeagueConfig] = None,
    strategy: str = "balanced",
    team_slots: tuple = BENCH_TEAM_SLOTS,
) -> dict:
    """Benchmark one week: recommended vs hindsight-optimal lineup.

    `projections`: player_id, position, p10, p50, p90
    `actuals`: player_id, fantasy_points (actual weekly score)

    Simulates one roster per draft slot in ``team_slots`` and averages the
    totals (sums starter counts), so the metric reflects early/mid/late
    draft positions rather than being conditioned on a single slot.
    """
    cfg = cfg or load_config()

    merged = projections.merge(actuals[["player_id", "fantasy_points"]],
                               on="player_id", how="inner")
    if merged.empty:
        return {"recommended_total": 0, "optimal_total": 0,
                "pts_left_on_bench": 0, "n_starters": 0, "n_correct": 0}

    per_slot = [
        _bench_one_roster(_simulate_roster(merged, team_idx=slot), cfg, strategy)
        for slot in team_slots
    ]
    k = len(per_slot)
    return {
        "recommended_total": sum(r["recommended_total"] for r in per_slot) / k,
        "optimal_total": sum(r["optimal_total"] for r in per_slot) / k,
        "pts_left_on_bench": sum(r["pts_left_on_bench"] for r in per_slot) / k,
        "n_starters": sum(r["n_starters"] for r in per_slot),
        "n_correct": sum(r["n_correct"] for r in per_slot),
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
