"""Tier-1 ranking benchmark: projection engine vs ADP (DrafterSpec.md §4.7 / §4.7.1).

The honest, ADP-free test of "do we beat the market". For each RANKING_ELIGIBLE
season, ranks the same player set by (a) our engine p50 and (b) ADP, scores both by
NDCG@k against actual-season FPPG relevance, and runs the paired per-season gate
test (§4.7.1). NDCG@k is the PRIMARY gate metric; Spearman / top-N are diagnostics.

The engine itself never sees ADP (the wall); ADP enters only here, as the baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from src.projection.eval import ndcg_at_k, spearman, top_n_hit_rate

from .report import PairedTest, paired_test

DEFAULT_KS = (84, 36)  # startable universe (12*7) and early-round emphasis


@dataclass
class RankingResult:
    per_season: pd.DataFrame      # season, k, ndcg_engine, ndcg_adp, spearman_*, hit_*
    gate: dict                    # k -> PairedTest (engine vs ADP NDCG)

    def gate_passes(self, k: int = 84) -> bool:
        return k in self.gate and self.gate[k].significant


def _eval_season(df: pd.DataFrame, ks: Sequence[int]) -> List[dict]:
    """df: one season, columns p50, adp, fppg (actual). ADP lower = better."""
    rel = df["fppg"].to_numpy()
    engine_score = df["p50"].to_numpy()
    adp_score = -df["adp"].to_numpy()  # invert so higher = drafted earlier = "better"
    out = []
    for k in ks:
        kk = min(k, len(df))
        out.append({
            "k": k,
            "n": len(df),
            "ndcg_engine": ndcg_at_k(engine_score, rel, kk),
            "ndcg_adp": ndcg_at_k(adp_score, rel, kk),
            "spearman_engine": spearman(engine_score, rel),
            "spearman_adp": spearman(adp_score, rel),
            "hit_engine": top_n_hit_rate(engine_score, rel, kk),
            "hit_adp": top_n_hit_rate(adp_score, rel, kk),
        })
    return out


def run_ranking_benchmark(
    projections: pd.DataFrame,
    adp: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    eligible_seasons: Optional[Sequence[int]] = None,
    ks: Sequence[int] = DEFAULT_KS,
) -> RankingResult:
    """Run Tier-1 over the eligible seasons.

    projections: player_id, season, p50.  adp: player_id, season, adp.
    labels: player_id, season, fppg (actual).
    """
    proj = projections[["player_id", "season", "p50"]]
    mkt = adp[["player_id", "season", "adp"]]
    lbl = labels[["player_id", "season", "fppg"]]
    df = proj.merge(mkt, on=["player_id", "season"], how="inner") \
             .merge(lbl, on=["player_id", "season"], how="inner")
    if eligible_seasons is not None:
        df = df[df["season"].isin(set(eligible_seasons))]

    rows = []
    for season, grp in df.groupby("season"):
        for rec in _eval_season(grp, ks):
            rec["season"] = int(season)
            rows.append(rec)
    per_season = pd.DataFrame(rows)

    gate = {}
    for k in ks:
        sub = per_season[per_season["k"] == k].sort_values("season")
        if not sub.empty:
            gate[k] = paired_test(sub["ndcg_engine"], sub["ndcg_adp"])
    return RankingResult(per_season=per_season, gate=gate)
