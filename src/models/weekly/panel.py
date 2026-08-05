"""Weekly player panel (§12.3).

One row per (season, week, player), scored through the league config. This is
the input to both the within-season variance model and the correlation matrix,
and it is the data-rich half of P2: roughly 78,000 skill-position player-weeks
against the ~1,000 usable player-seasons the projection mean is fitted on.

Reads exclusively through the ``AsOf`` handle, which is the only ``platform``
subpackage this layer may touch (§9.0).
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.core.config.league import LeagueConfig
from src.domain.scoring.engine import score_offense
from src.platform.asof.handle import AsOf

#: Slot identity is a ROLE, not a player: rank within (season, team, position).
#: That is what makes the correlation matrix stable across years even as the
#: individuals change.
DEFAULT_SLOT_DEPTH = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}


def build_weekly_panel(
    handle: AsOf, cfg: LeagueConfig, *, positions: Sequence[str] | None = None
) -> pd.DataFrame:
    """(season, week, player) with points scored through ``league.yaml``."""
    df = handle.player_stats()
    keep = list(positions or cfg.roster.modeled_positions)
    df = df[(df["season_type"] == "REG") & df["position"].isin(keep)].copy()
    df["points"] = score_offense(df, cfg.scoring.offense)
    return df[["season", "week", "team", "player_id", "player_display_name",
               "position", "points"]].reset_index(drop=True)


def player_season_summary(panel: pd.DataFrame, cfg: LeagueConfig) -> pd.DataFrame:
    """Per player-season mean and SD of weekly points.

    ``min_games_for_sigma`` is a label-quality floor: a three-game sample gives
    an SD estimate too noisy to regress on, and including it would bias the
    fitted relationship toward thin players.
    """
    agg = (panel.groupby(["season", "player_id", "position"], as_index=False)
                .agg(mu=("points", "mean"), sigma=("points", "std"),
                     games=("points", "size")))
    return agg[agg["games"] >= cfg.weekly.min_games_for_sigma].dropna(
        subset=["sigma"]).reset_index(drop=True)


def assign_slots(
    panel: pd.DataFrame, *, depth: dict[str, int] | None = None
) -> pd.DataFrame:
    """Label each player-season with its depth-chart slot (QB1, WR2, ...).

    Ranked by total points within (season, team, position), so the slot means
    "the team's Nth-best producer at that position that year" — the unit the
    correlation matrix is estimated over.
    """
    depth = depth or DEFAULT_SLOT_DEPTH
    totals = (panel.groupby(["season", "team", "player_id", "position"],
                            as_index=False)["points"].sum())
    totals["rank"] = (totals.groupby(["season", "team", "position"])["points"]
                            .rank(ascending=False, method="first"))
    totals = totals[
        totals.apply(lambda r: r["rank"] <= depth.get(r["position"], 0), axis=1)
    ].copy()
    totals["slot"] = totals["position"] + totals["rank"].astype(int).astype(str)
    return panel.merge(
        totals[["season", "team", "player_id", "position", "slot"]],
        on=["season", "team", "player_id", "position"], how="inner",
    )
