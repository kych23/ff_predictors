from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .clean_data import clean_player_stats, split_df_by_position


def _add_priors(
    df: pd.DataFrame,
    group_key: Iterable[str],
    metrics: List[str],
    windows: List[int] = [3, 5],
    ewm_alpha: float = 0.7,
    priors_req : int = 1,
) -> pd.DataFrame:
    """Shift by one week, then rolling means and EWMA (exponentially weighted moving average). Returns df with new prior columns only"""
    if df.empty:
        return df.copy()

    work = df.sort_values(["player_id", "season", "week"]).copy()

    def _per_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(["season", "week"]).copy()
        for m in metrics:
            if m not in g.columns:
                continue
            prior = g[m].shift(1)
            for w in windows:
                g[f"{m}_avg_{w}"] = prior.rolling(w, min_periods=1).mean()
            g[f"{m}_ewm"] = prior.ewm(alpha=ewm_alpha, adjust=False).mean()

        g["gp_prior"] = g["week"].notna().astype(int).cumsum().shift(1).fillna(0).astype(int)
        return g

    work = work.groupby(list(group_key), group_keys=False).apply(_per_group)

    id_cols = [c for c in ["player_id", "season", "week", "team", "opponent_team", "position"] if c in work.columns]
    prior_cols = [c for c in work.columns if any(s in c for s in ("_avg_", "_ewm"))] + ["gp_prior"]
    out = work[id_cols + prior_cols].copy()

    priors_only = out[prior_cols]
    out = out.loc[~priors_only.isna().all(axis=1)].copy()
    
    # Require at least priors_req prior games to train on
    if "gp_prior" in out.columns:
        out = out.loc[out["gp_prior"] >= priors_req].copy()
    return out


def calc_position_priors(years: List[int]) -> Dict[str, pd.DataFrame]:
    """Compute prior features per position from cleaned stats """
    base = clean_player_stats(years)
    frames = split_df_by_position(base)

    qb_metrics = [
        "fantasy_points_ppr",
        "attempts",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "carries",
        "rushing_yards",
        "rushing_tds",
    ]
    qb_priors = _add_priors(frames["QB"], ["player_id"], qb_metrics)


    skill_metrics = [
        "fantasy_points_ppr",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "receiving_fumbles_lost",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "rushing_fumbles_lost",
    ]
    skill_priors = _add_priors(frames["SKILL"], ["player_id"], skill_metrics)

    k_metrics = [
        "fantasy_points_ppr",
        "fg_att",
        "fg_made_0_19", "fg_made_20_29", "fg_made_30_39", "fg_made_40_49", "fg_made_50_59", "fg_made_60_",
        "pat_made", "pat_att",
    ]
    k_priors = _add_priors(frames["K"], ["player_id"], k_metrics)

    return {"QB": qb_priors, "SKILL": skill_priors, "K": k_priors}