"""Role & change features — the core edge over public tools.

Sees role *as it is now*, not lagged production: Week-1-effective depth-chart rank
(per the as-of cutoff), team-change flag, same-position competition, and the
central **vacated-opportunity** signal.

Vacated opportunity (precise): for player ``p`` on team ``T`` in season ``Y``,
``vacated_X`` = Σ over players ``q`` who were on ``T`` in ``Y-1`` but are NOT on ``T``
in ``Y``, of ``q``'s ``Y-1`` ``X``, for ``X ∈ {targets, carries, air_yards}`` — both
absolute and as a share of ``T``'s ``Y-1`` team total. A player inheriting a large
vacated share is poised to break out regardless of his own prior role.

Prefers coarse, draft-day-stable role signals (projected starter / committee) over
fine ordering, to limit the late-July train/serve skew.


PORTED from src/features/role_change.py (DraftEngineDesign.md §9.2).
Edits: v2 config API (load_league), scoring moves to domain, and the
leakage guard is reached through platform.asof — the only platform
subpackage this layer may import (§9.0, §11.3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_VAC_COLS = {"targets": "targets", "carries": "carries", "air_yards": "receiving_air_yards"}


def _depth_features(depth_y: pd.DataFrame) -> pd.DataFrame:
    """Per-player Week-1 depth rank, starter flag, same-position competition."""
    if depth_y is None or depth_y.empty:
        return pd.DataFrame(columns=["player_id", "team", "position", "depth_chart_rank",
                                     "is_projected_starter", "same_position_competition",
                                     "has_depth_data"])
    d = depth_y.copy()
    d = d.drop(columns=[c for c in ["team", "player_id"] if c in d.columns], errors="ignore")
    d = d.rename(columns={"gsis_id": "player_id", "club_code": "team"})
    d["depth_chart_rank"] = pd.to_numeric(d.get("depth_team"), errors="coerce")
    # take the most authoritative (lowest week, lowest rank) row per player
    if "week" in d.columns:
        d = d.sort_values(["player_id", "week", "depth_chart_rank"])
    d = d.drop_duplicates(subset=["player_id"], keep="first")

    # same-position competition = teammates at same team+position ranked ahead
    comp_raw = depth_y.drop(
        columns=[c for c in ["team", "player_id"] if c in depth_y.columns], errors="ignore"
    )
    comp = (comp_raw.rename(columns={"gsis_id": "player_id", "club_code": "team"})
            .assign(rank=lambda x: pd.to_numeric(x.get("depth_team"), errors="coerce")))
    counts = []
    if "team" in comp.columns and "position" in comp.columns:
        for (_, _), grp in comp.groupby(["team", "position"]):
            ranks = grp[["player_id", "rank"]].dropna()
            for _, r in ranks.iterrows():
                ahead = int((ranks["rank"] < r["rank"]).sum())
                counts.append({"player_id": r["player_id"], "same_position_competition": ahead})
    comp_df = pd.DataFrame(counts).drop_duplicates("player_id") if counts else \
        pd.DataFrame(columns=["player_id", "same_position_competition"])

    keep = [c for c in ["player_id", "team", "position", "depth_chart_rank"] if c in d.columns]
    out = d[keep].copy()
    out["is_projected_starter"] = (out["depth_chart_rank"] == 1).astype("Int64")
    out = out.merge(comp_df, on="player_id", how="left")
    out["has_depth_data"] = 1
    return out


def _vacated(stats_prior_year: pd.DataFrame, rosters_y: pd.DataFrame,
             rosters_prev: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team vacated targets/carries/air_yards (absolute + share)."""
    empty_cols = ["team"] + [f"vacated_{x}" for x in _VAC_COLS] + \
                 [f"vacated_{x}_share" for x in _VAC_COLS]
    if (stats_prior_year is None or stats_prior_year.empty or
            rosters_y is None or rosters_y.empty or rosters_prev is None or rosters_prev.empty):
        return pd.DataFrame(columns=empty_cols)

    prev_team = rosters_prev[["gsis_id", "team"]].dropna().drop_duplicates("gsis_id")
    prev_team = prev_team.rename(columns={"team": "prev_team"})
    cur_team = rosters_y[["gsis_id", "team"]].dropna().drop_duplicates("gsis_id")
    cur_team = cur_team.rename(columns={"team": "cur_team"})
    membership = prev_team.merge(cur_team, on="gsis_id", how="left")
    membership["departed"] = membership["cur_team"] != membership["prev_team"]

    # prior-year production per player
    s = stats_prior_year.copy()
    for src in _VAC_COLS.values():
        if src not in s.columns:
            s[src] = 0.0
        s[src] = pd.to_numeric(s[src], errors="coerce").fillna(0.0)
    prod = s.groupby("player_id")[list(_VAC_COLS.values())].sum().reset_index()
    prod = prod.rename(columns={v: k for k, v in _VAC_COLS.items()})

    m = membership.merge(prod, left_on="gsis_id", right_on="player_id", how="left").fillna(
        {k: 0.0 for k in _VAC_COLS})

    # team Y-1 totals (denominator for shares)
    team_tot = m.groupby("prev_team")[list(_VAC_COLS)].sum().rename(
        columns={k: f"team_{k}" for k in _VAC_COLS}).reset_index()

    departed = m[m["departed"]]
    vac = departed.groupby("prev_team")[list(_VAC_COLS)].sum().rename(
        columns={k: f"vacated_{k}" for k in _VAC_COLS}).reset_index()
    vac = vac.merge(team_tot, on="prev_team", how="left")
    for k in _VAC_COLS:
        vac[f"vacated_{k}_share"] = vac[f"vacated_{k}"] / vac[f"team_{k}"].replace(0, np.nan)
    vac = vac.rename(columns={"prev_team": "team"})
    keep = ["team"] + [f"vacated_{k}" for k in _VAC_COLS] + [f"vacated_{k}_share" for k in _VAC_COLS]
    return vac[keep]


def build_role_change(
    depth_y: pd.DataFrame,
    rosters_y: pd.DataFrame,
    rosters_prev: pd.DataFrame,
    stats_prev: pd.DataFrame,
) -> pd.DataFrame:
    """One row per player for target season Y with role + change signals.

    Caller passes Week-1-effective depth (season Y), rosters for Y and Y-1, and
    weekly stats for Y-1 only (leakage-filtered upstream).
    """
    depth = _depth_features(depth_y)

    # team_changed from roster membership
    if rosters_y is not None and not rosters_y.empty:
        cur = rosters_y[["gsis_id", "team"]].dropna().drop_duplicates("gsis_id").rename(
            columns={"gsis_id": "player_id", "team": "team_y"})
    else:
        cur = pd.DataFrame(columns=["player_id", "team_y"])
    if rosters_prev is not None and not rosters_prev.empty:
        prev = rosters_prev[["gsis_id", "team"]].dropna().drop_duplicates("gsis_id").rename(
            columns={"gsis_id": "player_id", "team": "team_prev"})
    else:
        prev = pd.DataFrame(columns=["player_id", "team_prev"])

    base = cur.merge(prev, on="player_id", how="outer")
    base["team_changed"] = (
        base["team_y"].notna() & base["team_prev"].notna() &
        (base["team_y"] != base["team_prev"])
    ).astype("Int64")
    # rookies (no prev team) are effectively new-team
    base.loc[base["team_prev"].isna() & base["team_y"].notna(), "team_changed"] = 1

    out = base.merge(depth.drop(columns=["team"], errors="ignore"), on="player_id", how="outer")
    depth_team_map = (dict(zip(depth["player_id"], depth["team"], strict=False))
                      if not depth.empty and "team" in depth.columns else {})
    out["team"] = out["team_y"].fillna(out["player_id"].map(depth_team_map))

    vac = _vacated(stats_prev, rosters_y, rosters_prev)
    out = out.merge(vac, left_on="team_y", right_on="team", how="left", suffixes=("", "_v"))
    out["has_depth_data"] = pd.to_numeric(out["has_depth_data"], errors="coerce").fillna(0).astype("Int64")
    drop = [c for c in ["team_v", "team_prev"] if c in out.columns]
    return out.drop(columns=drop, errors="ignore")
