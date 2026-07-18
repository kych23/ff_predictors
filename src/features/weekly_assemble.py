"""Weekly feature assembly — the (player, season, week) assembly point.

Analogous to assemble.py for the season grain. Joins rolling player stats,
opponent DvP, Vegas context, and season P50 prior for each (season, week).
All backward-looking sources filtered through leakage_guard.prior_weeks().
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config import LeagueConfig, load_config
from src.db.models import Projection, WeeklyLabel
from src.db.session import session_scope
from src.db.upsert_data import upsert_weekly_features

from . import leakage_guard as lg
from .weekly_features import (
    build_opponent_dvp,
    build_rolling_player_stats,
    build_weekly_vegas,
)

logger = logging.getLogger(__name__)

_META_COLS = {"player_id", "season", "week", "position"}


from ._utils import json_safe as _json_safe


def assemble_weekly_features(target_season: int, target_week: int,
                             frames: Dict[str, pd.DataFrame], *,
                             cfg: Optional[LeagueConfig] = None,
                             snapshot_id: Optional[str] = None) -> pd.DataFrame:
    """Assemble all weekly feature blocks for one (season, week)."""
    cfg = cfg or load_config()
    schedules = frames["schedules"]
    player_stats = frames["player_stats"]
    snap_counts = frames.get("snap_counts", pd.DataFrame())
    weekly_labels = frames["weekly_labels"]
    projections = frames.get("projections", pd.DataFrame())
    rosters = frames.get("rosters", pd.DataFrame())
    pfr_to_gsis = frames.get("pfr_to_gsis", {})

    # 1. LEAKAGE FILTERING
    ps_filtered = lg.prior_weeks(player_stats, target_season, target_week)
    if "season_type" in ps_filtered.columns:
        ps_filtered = ps_filtered[ps_filtered["season_type"] == "REG"].copy()
    lg.assert_no_future_week(ps_filtered, target_season, target_week,
                             where="player_stats")

    sc_filtered = pd.DataFrame()
    if not snap_counts.empty:
        sc_filtered = lg.prior_weeks(snap_counts, target_season, target_week)
        if "game_type" in sc_filtered.columns:
            sc_filtered = sc_filtered[sc_filtered["game_type"] == "REG"].copy()

    wl_filtered = lg.prior_weeks(weekly_labels, target_season, target_week)
    lg.assert_no_future_week(wl_filtered, target_season, target_week,
                             where="weekly_labels")

    # 2. ROLLING PLAYER STATS
    rolling = build_rolling_player_stats(ps_filtered, sc_filtered, pfr_to_gsis)

    # 3. OPPONENT DvP
    dvp = build_opponent_dvp(wl_filtered, schedules, target_season, target_week)

    # 4. WEEKLY VEGAS (+ opponent resolution)
    vegas = build_weekly_vegas(schedules, target_season, target_week)
    if vegas.empty:
        return pd.DataFrame()

    # 5. PLAYER UNIVERSE
    teams_playing = set(vegas["team"].unique())

    if target_week == 1 or wl_filtered[
        pd.to_numeric(wl_filtered["season"], errors="coerce") == target_season
    ].empty:
        # Week 1 or no current-season labels yet: use rosters
        if not rosters.empty:
            ros = rosters.copy()
            ros["season"] = pd.to_numeric(ros.get("season"), errors="coerce")
            ros = ros[ros["season"] == target_season]
            if "gsis_id" in ros.columns:
                universe = ros[["gsis_id", "position", "team"]].rename(
                    columns={"gsis_id": "player_id"})
            elif "player_id" in ros.columns:
                universe = ros[["player_id", "position", "team"]]
            else:
                universe = pd.DataFrame(columns=["player_id", "position", "team"])
            universe["player_id"] = universe["player_id"].astype(str)
        else:
            universe = pd.DataFrame(columns=["player_id", "position", "team"])
    else:
        # Week 2+: most recent label appearance in current season
        cur_labels = wl_filtered[
            pd.to_numeric(wl_filtered["season"], errors="coerce") == target_season
        ].copy()
        cur_labels = cur_labels.sort_values("week")
        universe = cur_labels.drop_duplicates("player_id", keep="last")[
            ["player_id", "position", "team"]
        ].copy()

    universe["player_id"] = universe["player_id"].astype(str)
    universe = universe.drop_duplicates("player_id", keep="last")
    universe = universe[universe["team"].isin(teams_playing)].copy()
    if universe.empty:
        return pd.DataFrame()

    # Position fallback: projections -> player table
    if not projections.empty:
        proj_pos = projections[
            pd.to_numeric(projections["season"], errors="coerce") == target_season
        ][["player_id", "position"]].drop_duplicates("player_id")
        proj_pos = proj_pos.rename(columns={"position": "proj_position"})
        proj_pos["player_id"] = proj_pos["player_id"].astype(str)
        universe = universe.merge(proj_pos, on="player_id", how="left")
        universe["position"] = universe["position"].fillna(universe["proj_position"])
        universe = universe.drop(columns=["proj_position"])

    universe = universe.dropna(subset=["position"])

    df = universe.copy()
    df["season"] = target_season
    df["week"] = target_week

    # 6. SEASON P50 PRIOR
    if not projections.empty:
        p = projections[
            pd.to_numeric(projections["season"], errors="coerce") == target_season
        ].copy()
        p["player_id"] = p["player_id"].astype(str)
        if "season_p50" not in p.columns and "p50" in p.columns:
            p = p.rename(columns={"p50": "season_p50"})
        p = p[["player_id", "season_p50"]].drop_duplicates("player_id")
        df = df.merge(p, on="player_id", how="left")
    else:
        df["season_p50"] = np.nan

    # 7. OPPONENT JOIN
    df = df.merge(vegas[["team", "opponent", "wk_implied_pts", "wk_game_total", "wk_spread"]],
                  on="team", how="left")
    if not dvp.empty:
        dvp_r = dvp.rename(columns={"team": "opponent", "dvp_fpa": "opp_dvp_fpa"})
        df = df.merge(dvp_r, left_on=["opponent", "position"],
                      right_on=["opponent", "position"], how="left")
    else:
        df["opp_dvp_fpa"] = np.nan

    # JOIN ROLLING STATS
    if not rolling.empty:
        rolling["player_id"] = rolling["player_id"].astype(str)
        df = df.merge(rolling, on="player_id", how="left")
    else:
        for c in ["rolling_fppg", "rolling_target_share", "rolling_snap_share",
                   "rolling_carries_pg"]:
            df[c] = np.nan

    # 8. PACK FEATURES — dedup by player_id (merges can re-introduce dupes)
    df = df.drop_duplicates("player_id", keep="first")
    feat_cols = [c for c in df.columns if c not in _META_COLS
                 and c not in {"team", "opponent", "season_p50"}]
    feat_cols = list(set(feat_cols) | {"season_p50"})

    feature_dicts = [
        {c: (None if pd.isna(v) else _json_safe(v)) for c, v in rec.items()}
        for rec in df[feat_cols].to_dict(orient="records")
    ]
    return pd.DataFrame({
        "player_id": df["player_id"].values,
        "season": int(target_season),
        "week": int(target_week),
        "position": df["position"].values,
        "features": feature_dicts,
        "snapshot_id": snapshot_id or "latest",
    })


def _load_weekly_frames(load_seasons: list, session) -> Dict[str, pd.DataFrame]:
    """Load all source frames for weekly assembly (nflreadpy + DB tables)."""
    from sqlalchemy import select

    from src.ingest import sources
    from src.ingest.player_ids import build_id_map

    logger.info("loading nflreadpy sources for %d..%d", load_seasons[0], load_seasons[-1])
    player_stats = sources.load_player_stats(load_seasons)
    snap_counts = sources.load_snap_counts([s for s in load_seasons if s >= 2012])
    schedules = sources.load_schedules(load_seasons)
    rosters = sources.load_rosters(load_seasons)

    id_map = build_id_map()
    pfr_to_gsis = dict(zip(id_map["pfr_id"].astype("string"), id_map["gsis_id"].astype(str)))

    wl_rows = session.execute(select(WeeklyLabel)).scalars().all()
    weekly_labels = pd.DataFrame([
        {"player_id": r.player_id, "season": r.season, "week": r.week,
         "position": r.position, "team": r.team, "fantasy_points": r.fantasy_points}
        for r in wl_rows
    ]) if wl_rows else pd.DataFrame(
        columns=["player_id", "season", "week", "position", "team", "fantasy_points"])

    proj_rows = session.execute(select(Projection)).scalars().all()
    if proj_rows:
        pdf = pd.DataFrame([
            {"player_id": r.player_id, "season": r.season, "position": r.position,
             "p50": r.p50, "model_version": r.model_version}
            for r in proj_rows
        ])
        # pin the season prior to the current config's model; the lexicographic
        # fallback only fires when that version was never trained (stale table).
        current_mv = load_config().model_version
        if (pdf["model_version"] == current_mv).any():
            pdf = pdf[pdf["model_version"] == current_mv]
        else:
            logger.warning(
                "no projections for current model_version %s — falling back to "
                "latest by string sort (retrain to refresh season priors)", current_mv)
        pdf = pdf.sort_values("model_version").drop_duplicates(
            ["player_id", "season"], keep="last")
        projections = pdf.rename(columns={"p50": "season_p50"})[
            ["player_id", "season", "position", "season_p50"]]
    else:
        logger.warning("Projection table empty — season_p50 will be NaN")
        projections = pd.DataFrame(
            columns=["player_id", "season", "position", "season_p50"])

    return {
        "player_stats": player_stats,
        "snap_counts": snap_counts,
        "schedules": schedules,
        "rosters": rosters,
        "weekly_labels": weekly_labels,
        "projections": projections,
        "pfr_to_gsis": pfr_to_gsis,
    }


def build_weekly_features_range(start_season: int, end_season: int, *,
                                snapshot_id: Optional[str] = None) -> int:
    """Build weekly features for all (season, week) in range."""
    seasons = list(range(start_season, end_season + 1))
    # two prior seasons so players who missed a full year still have rolling stats
    load_seasons = list(range(start_season - 2, end_season + 1))

    with session_scope() as session:
        frames = _load_weekly_frames(load_seasons, session)
        weekly_labels = frames["weekly_labels"]

        total = 0
        for Y in seasons:
            yr_labels = weekly_labels[
                pd.to_numeric(weekly_labels["season"], errors="coerce") == Y
            ]
            if yr_labels.empty:
                logger.warning("no weekly_labels for season %d, skipping", Y)
                continue
            weeks = sorted(yr_labels["week"].unique())
            for w in weeks:
                rows = assemble_weekly_features(Y, int(w), frames, snapshot_id=snapshot_id)
                if not rows.empty:
                    n = upsert_weekly_features(rows, session)
                    total += n
            session.commit()
            logger.info("season %d: %d weeks processed", Y, len(weeks))

        logger.info("weekly_features total: %d rows", total)
        return total


def build_weekly_features_for_week(target_season: int, target_week: int, *,
                                   snapshot_id: Optional[str] = None) -> int:
    """Build (and upsert) weekly features for ONE (season, week).

    Forward-serving entry point: unlike the range builder, it does not require
    labels for the target week, so it works for an upcoming, not-yet-played
    week. The assembler itself is as-of-kickoff safe (leakage_guard filters,
    roster-based universe when the season has no labels yet, Vegas lines from
    the schedule)."""
    load_seasons = list(range(target_season - 2, target_season + 1))

    with session_scope() as session:
        frames = _load_weekly_frames(load_seasons, session)
        rows = assemble_weekly_features(target_season, target_week, frames,
                                        snapshot_id=snapshot_id)
        if rows.empty:
            logger.warning("no weekly feature rows for %d week %d (no games "
                           "scheduled, or empty player universe)",
                           target_season, target_week)
            return 0
        n = upsert_weekly_features(rows, session)
        session.commit()
        logger.info("weekly_features for %d week %d: %d rows",
                    target_season, target_week, n)
        return n
