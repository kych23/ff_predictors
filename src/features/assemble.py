"""Feature assembly — the single (player, Y) assembly point

Joins prior-production, role-change, and rookie blocks for season ``Y``; attaches
``age_at_season_start`` (explicit feature, §4.5 Age note), sets the row's position
**as of season Y** and a ``position_changed`` flag (§4.0), enforces the as-of /
leakage rule at this boundary, and emits the ``season_features`` rows (feature set
as opaque JSONB).

All backward-looking sources are filtered through ``leakage_guard`` to seasons
strictly before Y before any aggregation; preseason-Y artifacts (Week-1 depth chart,
roster, age, draft capital) are allowed.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config import LeagueConfig, load_config
from src.db.session import SessionLocal
from src.db.upsert_data import upsert_season_features

from . import leakage_guard as lg
from .prior_production import build_prior_production
from .role_change import build_role_change
from .rookies import build_rookie_features

logger = logging.getLogger(__name__)

# Non-feature bookkeeping columns excluded from the JSONB feature payload.
_META_COLS = {"player_id", "season", "position", "is_rookie", "as_of_date",
              "team", "team_y", "last_season"}


def _age_at(birth_date, as_of: dt.date) -> Optional[float]:
    if birth_date is None or (isinstance(birth_date, float) and pd.isna(birth_date)):
        return None
    try:
        bd = pd.to_datetime(birth_date).date()
    except Exception:
        return None
    return round((as_of - bd).days / 365.25, 2)


def _season_length(season: int) -> int:
    return 17 if season >= 2021 else 16


def assemble_season(
    target_season: int,
    frames: Dict[str, pd.DataFrame],
    *,
    cfg: Optional[LeagueConfig] = None,
    as_of_date: Optional[dt.date] = None,
    pfr_to_gsis: Optional[Dict[str, str]] = None,
    college: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Assemble all feature blocks for one target season into season_features rows.

    ``frames`` keys (full, unfiltered): players, stats, snaps, opp, depth, rosters,
    combine, id_map. Leakage filtering happens here.
    """
    cfg = cfg or load_config()
    Y = target_season
    if as_of_date is None:
        as_of_date = dt.date(Y, 9, 1)  # safe pre-kickoff default if schedule absent

    players = frames["players"]
    id_map = frames.get("id_map", pd.DataFrame())

    # --- leakage-filter backward-looking sources to seasons < Y ---
    stats_prior = lg.prior_seasons(frames.get("stats", pd.DataFrame()), Y)
    snaps_prior = lg.prior_seasons(frames.get("snaps", pd.DataFrame()), Y)
    opp_prior = lg.prior_seasons(frames.get("opp", pd.DataFrame()), Y)
    lg.assert_no_future(stats_prior, Y, where="prior_production stats")
    lg.assert_no_future(snaps_prior, Y, where="prior_production snaps")
    lg.assert_no_future(opp_prior, Y, where="prior_production opp")

    # --- preseason-Y artifacts ---
    depth_all = frames.get("depth", pd.DataFrame())
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
    )
    role = build_role_change(depth_y, rosters_y, rosters_prev, stats_prev)

    # --- player universe for Y: rostered in Y, plus prior-production players ---
    universe = set()
    if not rosters_y.empty:
        universe |= set(rosters_y["gsis_id"].dropna().astype(str))
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

    rookies_block = build_rookie_features(rookie_ids, p, frames.get("combine", pd.DataFrame()),
                                          id_map if not id_map.empty else
                                          pd.DataFrame(columns=["gsis_id", "pfr_id"]),
                                          college)
    df = df.merge(rookies_block, on="player_id", how="left")

    # is_rookie
    df["is_rookie"] = df["player_id"].isin(set(rookie_ids))

    # position as-of Y: prefer roster position, else depth, else players table
    pos_map = {}
    if not rosters_y.empty:
        pos_map = dict(zip(rosters_y["gsis_id"].astype(str), rosters_y["position"]))
    pos_players = dict(zip(p["player_id"], p.get("position")))
    df["position"] = df["player_id"].map(pos_map).fillna(df["player_id"].map(pos_players))

    # position_changed vs Y-1 roster
    if not rosters_prev.empty:
        prev_pos = dict(zip(rosters_prev["gsis_id"].astype(str), rosters_prev["position"]))
        df["position_changed"] = df.apply(
            lambda r: int(r["player_id"] in prev_pos and prev_pos[r["player_id"]] != r["position"]),
            axis=1,
        )
    else:
        df["position_changed"] = 0

    # age (explicit feature) + season_length + indicator defaults
    birth = dict(zip(p["player_id"], p.get("birth_date")))
    df["age_at_season_start"] = df["player_id"].map(birth).map(lambda b: _age_at(b, as_of_date))
    df["season_length"] = _season_length(Y)
    for flag in ["has_depth_data", "has_college_stats", "is_udfa", "has_combine"]:
        if flag not in df.columns:
            df[flag] = 0
        df[flag] = pd.to_numeric(df[flag], errors="coerce").fillna(0).astype(int)
    df["is_rookie"] = df["is_rookie"].astype(bool)
    df["as_of_date"] = as_of_date.isoformat()

    # restrict to modeled positions (K/DEF never modeled, §4.2.1)
    df = df[df["position"].isin(cfg.roster.modeled_positions)].copy()

    # pack features into JSONB
    feat_cols = [c for c in df.columns if c not in _META_COLS]
    records = []
    for _, row in df.iterrows():
        feats = {c: (None if pd.isna(row[c]) else _json_safe(row[c])) for c in feat_cols}
        records.append({
            "player_id": row["player_id"],
            "season": int(Y),
            "position": row["position"],
            "is_rookie": bool(row["is_rookie"]),
            "as_of_date": row["as_of_date"],
            "features": feats,
        })
    return pd.DataFrame(records)


def _json_safe(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def build_features_range(
    start: int,
    end: int,
    *,
    snapshot_id: Optional[str] = None,
    config_path: Optional[str] = None,
) -> int:
    """Load sources, assemble features for [start, end], write season_features."""
    from src.ingest import sources
    from src.ingest.college import build_college_features
    from src.ingest.player_ids import build_college_bridge, build_id_map

    cfg = load_config(config_path)
    seasons = list(range(start, end + 1))
    load_seasons = list(range(min(seasons) - cfg.training.min_train_seasons - 1, max(seasons) + 1))

    logger.info("loading sources for %d..%d", load_seasons[0], load_seasons[-1])
    players = sources.load_players()
    players = players.rename(columns={"gsis_id": "player_id", "display_name": "name",
                                      "college_name": "college", "rookie_season": "rookie_year"})
    id_map = build_id_map()
    pfr_to_gsis = dict(zip(id_map["pfr_id"].astype("string"), id_map["gsis_id"].astype(str)))
    frames = {
        "players": players,
        "id_map": id_map,
        "stats": sources.load_player_stats(load_seasons),
        "snaps": sources.load_snap_counts([s for s in load_seasons if s >= 2012]),
        "opp": sources.load_ff_opportunity([s for s in load_seasons if s >= 2006]),
        "depth": sources.load_depth_charts(seasons),
        "rosters": sources.load_rosters(load_seasons),
        "combine": sources.load_combine(),
    }
    kickoffs = sources.week1_kickoff_dates(seasons)

    # College block: built ONCE for every rookie in the range, keyed by gsis_id, then
    # shared across seasons (assemble_season merges per-season rookies by player_id).
    ry = pd.to_numeric(players.get("rookie_year"), errors="coerce")
    rookie_seasons = {
        str(pid): int(y)
        for pid, y in zip(players["player_id"].astype(str), ry)
        if pd.notna(y) and start <= y <= end
    }
    birth_dates = dict(zip(players["player_id"].astype(str), players.get("birth_date", pd.Series(dtype=object))))
    college = build_college_features(rookie_seasons, build_college_bridge(), birth_dates=birth_dates)
    n_college = int(college["has_college_stats"].fillna(0).sum()) if not college.empty else 0
    logger.info("college features: %d rookies with real stats (of %d rookies)",
                n_college, len(rookie_seasons))

    session = SessionLocal()
    total = 0
    try:
        for Y in seasons:
            ko = kickoffs.get(Y)
            as_of = (ko - dt.timedelta(days=cfg.backtest.as_of_offset_days)).date() \
                if ko is not None else dt.date(Y, 9, 1)
            rows = assemble_season(Y, frames, cfg=cfg, as_of_date=as_of,
                                   pfr_to_gsis=pfr_to_gsis, college=college)
            if snapshot_id:
                rows["snapshot_id"] = snapshot_id
            n = upsert_season_features(rows, session)
            session.commit()
            total += n
            logger.info("season %d: %d feature rows (as_of=%s)", Y, n, as_of)
        return total
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
