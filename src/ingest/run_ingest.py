"""Full raw-data pull into the DB (DrafterSpec.md §4.2).

Replaces the old ``load_initial_data.py`` + the ingestion half of ``seed_db.py``.
Creates one ``ingest_snapshot`` per run; raw staging tables (``weekly_stats_raw``,
``adp``) are stamped with that ``snapshot_id`` so downstream training/benchmarking
pin a frozen snapshot and stay reproducible against nflverse corrections (§4.0).

Depth charts / snap counts / opportunity are read live (nflreadpy-cached) by the
feature builder; only the label-source box scores and ADP are persisted as
snapshot-stamped raw tables, matching the enumerated schema (§4.3).
"""
from __future__ import annotations

import logging
import uuid
from typing import List, Optional

import pandas as pd

from src.config import load_config
from src.db.init_db import init_db
from src.db.session import session_scope
from src.db.upsert_data import (
    upsert_ingest_snapshot,
    upsert_player_id_map,
    upsert_players,
    upsert_weekly_stats_raw,
)

from . import sources
from .player_ids import build_crosswalk, fantasypros_lookup, name_lookup, name_position_lookup

logger = logging.getLogger(__name__)

# Raw component columns scoring.py needs (§4.4.1) — never the precomputed total.
SCORING_RAW_COLS = [
    "passing_yards", "passing_tds", "passing_interceptions",
    "rushing_yards", "rushing_tds",
    "receptions", "receiving_yards", "receiving_tds",
    "passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions",
    "rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost",
    "special_teams_tds",
]


def _new_snapshot_id() -> str:
    return uuid.uuid4().hex[:16]


_INT32_MIN, _INT32_MAX = -2_147_483_648, 2_147_483_647


def _safe_int_col(series: "pd.Series") -> "pd.Series":
    """Coerce to numeric, null out anything outside PostgreSQL INTEGER range."""
    s = pd.to_numeric(series, errors="coerce")
    s[s.notna() & ~s.between(_INT32_MIN, _INT32_MAX)] = None
    return s


def _players_frame() -> pd.DataFrame:
    df = sources.load_players()
    out = pd.DataFrame({
        "player_id": df["gsis_id"].astype(str),
        "name": df.get("display_name"),
        "position": df.get("position"),
        "college": df.get("college_name"),
        "rookie_year": _safe_int_col(df.get("rookie_season")),
        "draft_year": _safe_int_col(df.get("draft_year")),
        "draft_round": _safe_int_col(df.get("draft_round")),
        "draft_pick": _safe_int_col(df.get("draft_pick")),
        "draft_team": df.get("draft_team"),
        "birth_date": df.get("birth_date").astype("string") if "birth_date" in df else None,
        "latest_team": df.get("latest_team"),
    })
    return out[out["player_id"].notna() & (out["player_id"] != "None")]


def _weekly_raw_frame(seasons: List[int], snapshot_id: str) -> pd.DataFrame:
    stats = sources.load_player_stats(seasons)
    if stats.empty:
        return stats
    # Regular season only for season-production labels.
    if "season_type" in stats.columns:
        stats = stats[stats["season_type"] == "REG"]
    present = [c for c in SCORING_RAW_COLS if c in stats.columns]
    base = stats[["player_id", "season", "week"]].copy()
    base["season_type"] = "REG"
    base["team"] = stats.get("team")
    base["position"] = stats.get("position")
    comp = stats[present].fillna(0.0)
    base["stats"] = comp.to_dict(orient="records")
    base["snapshot_id"] = snapshot_id
    base = base[base["player_id"].notna()]
    return base


def run_ingest(
    start: int,
    end: int,
    *,
    config_path: Optional[str] = None,
    with_adp: bool = True,
    notes: str = "",
) -> str:
    """Pull all raw sources for seasons [start, end] inclusive. Returns snapshot_id."""
    cfg = load_config(config_path)
    seasons = list(range(start, end + 1))
    snapshot_id = _new_snapshot_id()
    init_db()

    with session_scope() as session:
        # Snapshot row first so FK-style references are valid.
        upsert_ingest_snapshot(
            {"snapshot_id": snapshot_id, "nflreadpy_version": sources.nflreadpy_version(),
             "notes": notes, "coverage": None},
            session,
        )

        n_players = upsert_players(_players_frame(), session)
        logger.info("players upserted: %d", n_players)

        crosswalk = build_crosswalk()
        n_idmap = upsert_player_id_map(crosswalk.id_map, session)
        logger.info("id_map upserted: %d (college bridge match rate %.3f)",
                    n_idmap, crosswalk.match_rate)

        n_weekly = upsert_weekly_stats_raw(_weekly_raw_frame(seasons, snapshot_id), session)
        logger.info("weekly_stats_raw upserted: %d", n_weekly)
        session.commit()

        coverage = {}
        # REBUILD: ADP source is moving from Fantasy Football Calculator to Yahoo
        # (src/ingest/adp.py deleted in the monte-carlo strip). This block needs a
        # Yahoo-based build_adp_season equivalent; fantasypros_lookup/name_lookup/
        # name_position_lookup (src/ingest/player_ids.py) and upsert_adp/
        # upsert_adp_coverage (src/db/upsert_data.py) are still there and were
        # format-agnostic, so the new client should be able to reuse them.
        if with_adp:
            logger.warning(
                "ADP ingestion skipped: FFC client removed pending Yahoo replacement "
                "(see REBUILD note in src/ingest/run_ingest.py)")

        upsert_ingest_snapshot(
            {"snapshot_id": snapshot_id, "nflreadpy_version": sources.nflreadpy_version(),
             "notes": notes, "coverage": coverage or None},
            session,
        )
        session.commit()
        logger.info("ingest complete: snapshot_id=%s", snapshot_id)
        return snapshot_id
