"""The warm read-only frames every rankings request reads from.

Loaded **once**, in `lifespan`, never per request. Two reasons, and both are
load-bearing:

1. `rankings_csv.load` is not a read. It calls `blobs.put(...)` and mints a
   `ManifestEntry` — calling it per request would write a blob per page view.
   CLAUDE.md already records this repo learning the same lesson the expensive
   way, when `nflverse_covariates` fetched on every tier-0 recommendation and
   blew a 25 s allocator budget by 61.85 s.
2. Every route here is `async def` (enforced by
   `test_web_api.py::test_every_route_is_async`), so any parquet read left in a
   handler runs on the loop that serves the pick clock and the SSE stream.

Every frame degrades **independently** to an empty frame that still carries its
expected columns, so downstream code reads NaN rather than raising `KeyError`.
The one thing that never degrades is `board`: it is a function argument, not a
loaded artifact, so it cannot be what failed — and dropping it would make
`stale_player_ids` compute against an empty bundle and report every player on
the user's saved board as stale. That is why `empty()` takes it.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.constants import REPLACEMENT_FLOOR_QUANTILE

logger = logging.getLogger(__name__)

BOARD_COLS = ["player_id", "player_name", "position", "team", "bye_week",
              "value", "value_p10", "value_p90", "value_source", "coverage",
              "adp", "adp_stdev"]
TIER_COLS = ["player_id", "player_name", "position", "tier", "rank", "adp"]
RANK_COLS = ["player_name", "match_key", "position", "team", "consensus_rank",
             "rank_spread", "rank_sd", "n_platforms"]


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


@dataclass(frozen=True)
class RankingsData:
    """Everything the rankings surface reads, resolved once."""

    board: pd.DataFrame
    matrix: pd.DataFrame
    ranks: pd.DataFrame
    tiers: pd.DataFrame
    replacement: Mapping[str, float] = field(default_factory=dict)
    #: player_id -> match_key, precomputed. The rankings CSV carries no
    #: player_id at all, so the market join is by normalized name; doing it per
    #: request would mean normalizing 260 names per page view.
    match_keys: Mapping[str, str] = field(default_factory=dict)
    target_season: int = 0
    snapshots: Mapping[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------- building
    @classmethod
    def empty(cls, *, board: pd.DataFrame | None = None) -> RankingsData:
        players = _empty(BOARD_COLS) if board is None else board
        return cls(board=players, matrix=pd.DataFrame(),
                   ranks=_empty(RANK_COLS), tiers=_empty(TIER_COLS),
                   replacement=_replacement(players), match_keys={},
                   target_season=0, snapshots={})

    @classmethod
    def load(cls, cfg, strategy, web_cfg, players: pd.DataFrame,
             bundle_snapshot_id: str, *, root: Path) -> RankingsData:
        """Resolve every frame. Individual failures degrade; none raise.

        `root` is passed explicitly because `artifacts.ARTIFACT_ROOT` is the
        relative `Path("data/artifacts")` and `newest()` globs it directly — a
        server launched from anywhere but the repo root would silently find
        nothing and report it as "no tier artifact".
        """
        target_season, proj_snapshot = _target_season(root)
        matrix, matrix_snapshot = _load_matrix(root, target_season)
        tiers, tiers_snapshot = _load_tiers(root)
        ranks = _load_ranks(strategy, web_cfg)

        snapshots = {"bundle": bundle_snapshot_id, "matrix": matrix_snapshot,
                     "tiers": tiers_snapshot, "projections": proj_snapshot}
        # Logged rather than asserted equal: `newest` selects by mtime and the
        # bundle by path, so a retrain between train_projection_v2.py and
        # build_bundle.py can pair a bundle with another run's matrix. Making
        # that visible is honest; making it fatal would break the draft.
        logger.info("rankings data: season=%s snapshots=%s", target_season,
                    snapshots)

        return cls(board=players, matrix=matrix, ranks=ranks, tiers=tiers,
                   replacement=_replacement(players),
                   match_keys=_match_keys(players),
                   target_season=target_season, snapshots=snapshots)


def _replacement(players: pd.DataFrame) -> dict[str, float]:
    """Per-position replacement level, at the repo's own floor quantile.

    This is what makes an `overall` seed meaningful. Sorting the board by raw
    `value` puts 24 quarterbacks in the top 40 — per-game points are not
    comparable across positions when only one QB starts.
    """
    if players.empty or "value" not in players.columns:
        return {}
    grouped = players.groupby("position")["value"]
    return {str(pos): float(v)
            for pos, v in grouped.quantile(REPLACEMENT_FLOOR_QUANTILE).items()}


def _match_keys(players: pd.DataFrame) -> dict[str, str]:
    if players.empty or "player_name" not in players.columns:
        return {}
    from src.core.names import normalize_name

    keys = {str(pid): normalize_name(str(name))
            for pid, name in zip(players["player_id"], players["player_name"],
                                 strict=False)}
    # The collision that matters is two BUNDLE players folding to one name, not
    # two CSV rows: the market frame has no player_id to disambiguate with, so
    # both would silently claim the same ranks. Blank them and let the detail
    # report `matched: false`, which is true.
    seen: dict[str, int] = {}
    for key in keys.values():
        seen[key] = seen.get(key, 0) + 1
    return {pid: key for pid, key in keys.items() if seen[key] == 1}


def _target_season(root: Path) -> tuple[int, str]:
    """The season the bundle projects, from the projections manifest.

    NOT `matrix["season"].max()`: the newest matrix by mtime may be from a
    different run than the bundle, and inferring the season from it would then
    silently select the wrong rows.
    """
    try:
        from src.models.artifacts import newest

        meta = newest("projections_*.meta.json", root=root)
        if meta is not None:
            raw = json.loads(meta.read_text())
            season = int(raw.get("target_season", 0))
            if season:
                return season, str(raw.get("snapshot_id", ""))
    except Exception:                                      # noqa: BLE001
        logger.warning("could not read the projections manifest", exc_info=True)
    return 0, ""


def _load_matrix(root: Path, target_season: int) -> tuple[pd.DataFrame, str]:
    try:
        from src.models.artifacts import newest

        path = newest("training_matrix_*.parquet", root=root)
        if path is None:
            return pd.DataFrame(), ""
        snapshot = path.stem.replace("training_matrix_", "")

        # WITHOUT a season, serve NOTHING. The matrix holds every season from
        # 2012 — 5,961 rows against 907 for one season — and the index is not
        # unique across them, so an unfiltered frame makes `_matrix_row`'s
        # `.iloc[0]` pick the EARLIEST row. Ja'Marr Chase's card then reports
        # his 2021 rookie age of 21.5 as current, with no warning that it is
        # five years stale. A card that says "no data" is honest; a card
        # confidently showing the wrong season is not.
        if not target_season:
            logger.warning(
                "no target season resolved; serving no prior production "
                "rather than an arbitrary historical row")
            return pd.DataFrame(), snapshot

        frame = pd.read_parquet(path)
        if "season" not in frame.columns:
            return pd.DataFrame(), snapshot
        frame = frame[pd.to_numeric(frame["season"], errors="coerce")
                      == target_season].copy()
        frame["player_id"] = frame["player_id"].astype(str)
        if not frame["player_id"].is_unique:
            frame = frame.drop_duplicates("player_id", keep="last")
        return frame.set_index("player_id", drop=False), snapshot
    except Exception:                                      # noqa: BLE001
        logger.warning("training matrix unavailable", exc_info=True)
        return pd.DataFrame(), ""


def _load_tiers(root: Path) -> tuple[pd.DataFrame, str]:
    try:
        from src.models.artifacts import newest

        path = newest("tiers_*.parquet", root=root)
        if path is None:
            return _empty(TIER_COLS), ""
        frame = pd.read_parquet(path).copy()
        frame["player_id"] = frame["player_id"].astype(str)
        snapshot = path.stem.replace("tiers_", "")
        return frame, snapshot
    except Exception:                                      # noqa: BLE001
        logger.warning("tier artifact unavailable", exc_info=True)
        return _empty(TIER_COLS), ""


def _load_ranks(strategy, web_cfg) -> pd.DataFrame:
    """The market export. Resolved through `web_cfg`, because
    `strategy.adp.rankings_path` is a cwd-relative string."""
    path_raw = getattr(getattr(strategy, "adp", None), "rankings_path", "")
    if not path_raw:
        return _empty(RANK_COLS)
    try:
        from src.platform.sources import rankings_csv

        path = web_cfg.resolved(Path(str(path_raw)))
        if not path.exists():
            logger.warning("no rankings export at %s", path)
            return _empty(RANK_COLS)
        frame = rankings_csv.load(path).frame
        return frame.set_index("match_key", drop=False)
    except Exception:                                      # noqa: BLE001
        logger.warning("rankings export unavailable", exc_info=True)
        return _empty(RANK_COLS)


def vor(data: RankingsData, players: pd.DataFrame) -> pd.Series:
    """Value over replacement, the only cross-position-comparable number here."""
    if players.empty or "value" not in players.columns:
        return pd.Series(dtype=float)
    floor = players["position"].map(data.replacement)
    return players["value"] - floor.fillna(0.0).astype(float)


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))
