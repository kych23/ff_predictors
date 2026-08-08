"""Feature assembly — the single (player, Y) assembly point

Joins prior-production and role-change blocks for season ``Y``; attaches
``age_at_season_start`` (explicit feature Age note), sets the row's position
**as of season Y** and a ``position_changed`` flag, enforces the as-of /
leakage rule at this boundary, and emits the ``season_features`` rows (feature set
as opaque JSONB).

# REBUILD: rookie/college-derived features (draft capital, combine, college
# production) were dropped here when src/features/rookies.py and
# src/ingest/college.py were deleted in the monte-carlo strip (CFBD descoped).
# `is_rookie` is still computed directly from rookie_year and is unaffected;
# has_college_stats/is_udfa/has_combine now always default to 0 via the
# existing flag-default loop below. Per the target architecture, rookie priors
# return via ADP + NFL draft capital instead of CFBD college stats.

All backward-looking sources are filtered through ``leakage_guard`` to seasons
strictly before Y before any aggregation; preseason-Y artifacts (Week-1 depth chart,
roster, age, draft capital) are allowed.


PORTED from src/features/assemble.py (DraftEngineDesign.md §9.2).
Edits: v2 config API (load_league), scoring moves to domain, and the
leakage guard is reached through platform.asof — the only platform
subpackage this layer may import (§9.0, §11.3).
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from src.core.config.league import LeagueConfig, load_league
from src.platform.asof import guards as lg

from ._utils import json_safe as _json_safe
from .college import build_college_features
from .draft_capital import build_draft_capital
from .prior_production import build_prior_production
from .role_change import build_role_change, normalize_depth_charts
from .team_context import CTX_COLS, build_team_context

logger = logging.getLogger(__name__)

# Non-feature bookkeeping columns excluded from the JSONB feature payload.
_META_COLS = {"player_id", "season", "position", "is_rookie", "as_of_date",
              "team", "team_y", "last_season"}


def _age_at(birth_date, as_of: dt.date) -> float | None:
    if birth_date is None or (isinstance(birth_date, float) and pd.isna(birth_date)):
        return None
    try:
        bd = pd.to_datetime(birth_date).date()
    except Exception:
        return None
    return round((as_of - bd).days / 365.25, 2)


def _season_length(season: int) -> int:
    return 17 if season >= 2021 else 16


def _reg_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Regular-season rows only — labels are REG-only (run_ingest), so prior
    production must be too, or playoff-team players get skewed rate features.

    Falls back to a week cutoff (18 weeks from 2021, 17 before) for sources
    without a season/game type column (e.g. ff_opportunity)."""
    if df is None or df.empty:
        return df
    if "season_type" in df.columns:
        return df[df["season_type"] == "REG"].copy()
    if "game_type" in df.columns:
        return df[df["game_type"] == "REG"].copy()
    if "week" in df.columns and "season" in df.columns:
        wk = pd.to_numeric(df["week"], errors="coerce")
        sn = pd.to_numeric(df["season"], errors="coerce")
        reg_weeks = np.where(sn >= 2021, 18, 17)
        return df[wk.isna() | (wk <= reg_weeks)].copy()
    return df


def assemble_season(
    target_season: int,
    frames: dict[str, pd.DataFrame],
    *,
    cfg: LeagueConfig | None = None,
    as_of_date: dt.date | None = None,
    pfr_to_gsis: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Assemble all feature blocks for one target season into season_features rows.

    ``frames`` keys (full, unfiltered): players, stats, snaps, opp, depth, rosters,
    combine, id_map. Leakage filtering happens here.
    """
    cfg = cfg or load_league()
    Y = target_season
    if as_of_date is None:
        as_of_date = dt.date(Y, 9, 1)  # safe pre-kickoff default if schedule absent

    players = frames["players"]

    # --- leakage-filter backward-looking sources to seasons < Y, REG only ---
    stats_prior = lg.prior_seasons(_reg_rows(frames.get("stats", pd.DataFrame())), Y)
    snaps_prior = lg.prior_seasons(_reg_rows(frames.get("snaps", pd.DataFrame())), Y)
    opp_prior = lg.prior_seasons(_reg_rows(frames.get("opp", pd.DataFrame())), Y)
    lg.assert_no_future(stats_prior, Y, where="prior_production stats")
    lg.assert_no_future(snaps_prior, Y, where="prior_production snaps")
    lg.assert_no_future(opp_prior, Y, where="prior_production opp")

    # --- preseason-Y artifacts ---
    # Normalize BEFORE the as-of guard. `preseason_rows` passes a frame
    # through untouched when it has no `season` column, so the current
    # nflverse schema would skip the leakage filter entirely.
    depth_all = normalize_depth_charts(frames.get("depth", pd.DataFrame()))
    depth_y = lg.preseason_rows(depth_all, Y)
    rosters_all = frames.get("rosters", pd.DataFrame())
    rosters_y = rosters_all[pd.to_numeric(rosters_all.get("season"), errors="coerce") == Y] \
        if not rosters_all.empty else rosters_all
    rosters_prev = rosters_all[pd.to_numeric(rosters_all.get("season"), errors="coerce") == Y - 1] \
        if not rosters_all.empty else rosters_all
    stats_prev = stats_prior[pd.to_numeric(stats_prior.get("season"), errors="coerce") == Y - 1] \
        if not stats_prior.empty else stats_prior

    # --- blocks ---
    prior = build_prior_production(
        stats_prior, Y, halflife=cfg.training.ewm_halflife_seasons,
        pfr_to_gsis=pfr_to_gsis, snaps_prior=snaps_prior, opp_prior=opp_prior,
        offense=cfg.scoring.offense,
    )
    role = build_role_change(depth_y, rosters_y, rosters_prev, stats_prev)

    # --- player universe for Y: rostered in Y, plus prior-production players ---
    # When current-year rosters are available, use them as the authoritative set —
    # free agents and cut players won't appear in rosters_y even if they produced in Y-1.
    universe = set()
    if not rosters_y.empty:
        universe |= set(rosters_y["gsis_id"].dropna().astype(str))
    else:
        # No roster data for this season — fall back to prior production + role signals.
        universe |= set(prior["player_id"].astype(str)) if not prior.empty else set()
        universe |= set(role["player_id"].dropna().astype(str)) if not role.empty else set()

    # rookies = rookie_year == Y (no prior NFL production)
    p = players.copy()
    p["player_id"] = p["player_id"].astype(str)
    rookie_ids = p.loc[pd.to_numeric(p.get("rookie_year"), errors="coerce") == Y, "player_id"].tolist()
    universe |= set(rookie_ids)
    universe &= set(p["player_id"])  # only known players

    base = pd.DataFrame({"player_id": sorted(universe)})
    base["season"] = Y

    # merge blocks
    df = base.merge(prior, on="player_id", how="left")
    df = df.merge(role, on="player_id", how="left")

    # is_rookie
    df["is_rookie"] = df["player_id"].isin(set(rookie_ids))

    # --- draft capital (preseason-Y artifact: the NFL draft is in April) ---
    # The only feature that separates one rookie from another. Everything
    # prior_* is null for them, so without this a first-round back and an
    # undrafted back score nearly the same.
    capital = build_draft_capital(frames.get("id_map", pd.DataFrame()),
                                  df["player_id"])
    df = df.merge(capital, on="player_id", how="left")

    # --- college production (preseason-Y: a player's college career is over
    # before he is drafted, so this cannot see the season being projected) ---
    # Draft capital says what a team BELIEVES about a rookie; this is what he
    # actually did. Deflated to a within-conference z-score, because raw totals
    # reward the weakest schedule — see models/features/college.py.
    college = build_college_features(frames.get("cfb_stats", pd.DataFrame()),
                                     players, df["player_id"], Y)
    df = df.merge(college, on="player_id", how="left")

    # position as-of Y: prefer roster position, else depth, else players table
    pos_map = {}
    if not rosters_y.empty:
        pos_map = dict(zip(rosters_y["gsis_id"].astype(str), rosters_y["position"], strict=False))
    pos_players = dict(zip(p["player_id"], p.get("position"), strict=False))
    df["position"] = df["player_id"].map(pos_map).fillna(df["player_id"].map(pos_players))

    # position_changed vs Y-1 roster
    if not rosters_prev.empty:
        prev_pos = dict(zip(rosters_prev["gsis_id"].astype(str), rosters_prev["position"], strict=False))
        df["position_changed"] = df.apply(
            lambda r: int(r["player_id"] in prev_pos and prev_pos[r["player_id"]] != r["position"]),
            axis=1,
        )
    else:
        df["position_changed"] = 0

    # --- market team scoring context (preseason-Y artifact: Week-1 betting line) ---
    # NOT routed through leakage_guard.prior_seasons — this is intentionally the
    # CURRENT season's preseason line (known pre-kickoff), attached by season-Y team.
    sched_all = frames.get("schedules", pd.DataFrame())
    sched_y = (sched_all[pd.to_numeric(sched_all.get("season"), errors="coerce") == Y]
               if sched_all is not None and not sched_all.empty else pd.DataFrame())
    team_ctx = build_team_context(sched_y)
    team_key = df["team_y"].fillna(df["team"]) if "team_y" in df.columns else df.get("team")
    if not team_ctx.empty and team_key is not None:
        ctx_idx = team_ctx.set_index("team")
        for col in CTX_COLS:
            df[col] = team_key.astype(str).map(ctx_idx[col])
    else:
        for col in CTX_COLS:
            df[col] = np.nan

    # age (explicit feature) + season_length + indicator defaults
    birth = dict(zip(p["player_id"], p.get("birth_date"), strict=False))
    df["age_at_season_start"] = df["player_id"].map(birth).map(lambda b: _age_at(b, as_of_date))
    df["season_length"] = _season_length(Y)
    # `has_college_stats` is no longer here: it is produced by
    # `build_college_features` and forcing it to 0 alongside the genuinely
    # unbuilt flags would zero a real feature.
    for flag in ["has_depth_data", "is_udfa", "has_combine"]:
        if flag not in df.columns:
            df[flag] = 0
        df[flag] = pd.to_numeric(df[flag], errors="coerce").fillna(0).astype(int)
    df["is_rookie"] = df["is_rookie"].astype(bool)
    df["as_of_date"] = as_of_date.isoformat()

    # restrict to modeled positions (K/DEF never modeled)
    df = df[df["position"].isin(cfg.roster.modeled_positions)].copy()

    # pack features into JSONB (vectorized: to_dict beats iterrows ~10x here)
    feat_cols = [c for c in df.columns if c not in _META_COLS]
    feature_dicts = [
        {c: (None if pd.isna(v) else _json_safe(v)) for c, v in rec.items()}
        for rec in df[feat_cols].to_dict(orient="records")
    ]
    return pd.DataFrame({
        "player_id": df["player_id"].values,
        "season": int(Y),
        "position": df["position"].values,
        "is_rookie": df["is_rookie"].astype(bool).values,
        "as_of_date": df["as_of_date"].values,
        "features": feature_dicts,
    })

