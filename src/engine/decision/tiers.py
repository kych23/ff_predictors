"""Tier-3 static board (§7.4) — the last rung of the degradation ladder.

Tier 3 fires when *any uncaught exception* escapes the tiers above it. Its only
job is to name a defensible player at a position the roster still needs, using
nothing but a parquet file. No simulator, no bundle loader, no model objects —
if any of those could fail here, tier 3 would share a failure mode with the
tiers it is supposed to catch, and the ladder would have three rungs instead of
four.

Tiers are 1-D k-means over `rate_p50` within position, k=8, relabeled in
descending order of cluster centre. **Not** Jenks: Jenks would add a dependency
the §24 list does not carry, and in one dimension k-means with a good
initialization is the same algorithm. `sklearn.cluster.KMeans` is already a
dependency of the projection model.

The tier number is the useful part on draft night. Two players in the same
tier are, as far as this board can tell, interchangeable — which is exactly the
call a drafter has to make in eight seconds with the room waiting.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.errors import DataError

#: k=8 per §7.4. Fewer than this and the top tier swallows the entire first
#: round; more and tiers stop meaning "interchangeable".
DEFAULT_K = 8

TIER_COLUMNS = ("player_id", "player_name", "position", "tier", "rank", "adp")


def assign_tiers(values: np.ndarray, k: int = DEFAULT_K,
                 seed: int = 0) -> np.ndarray:
    """1-D k-means labels, renumbered 1..k best-first.

    Raw KMeans labels are arbitrary integers, so relabeling by descending
    cluster centre is not cosmetic — without it "tier 1" means nothing.
    """
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    n = len(values)
    if n == 0:
        return np.empty(0, dtype=int)
    effective_k = min(k, n, len(np.unique(values)))
    if effective_k <= 1:
        return np.ones(n, dtype=int)

    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=effective_k, n_init=10, random_state=seed)
    labels = km.fit_predict(values)
    order = np.argsort(-km.cluster_centers_.ravel())
    remap = np.empty(effective_k, dtype=int)
    remap[order] = np.arange(1, effective_k + 1)
    return remap[labels]


def build_tier_table(board: pd.DataFrame, *, k: int = DEFAULT_K,
                     value_col: str = "value", seed: int = 0) -> pd.DataFrame:
    """One row per player: position tier and within-position rank."""
    missing = {"player_id", "position", value_col} - set(board.columns)
    if missing:
        raise DataError(f"board is missing {sorted(missing)}")

    has_adp = "adp" in board.columns
    frames = []
    for _position, group in board.groupby("position", sort=True):
        g = group.copy()
        g[value_col] = pd.to_numeric(g[value_col], errors="coerce").fillna(0.0)
        # Ties break on ADP. K and DST carry no projected value at all — they
        # are not modeled positions — so without this every kicker ties at 0.0
        # and the within-position order is whatever the groupby happened to
        # produce. ADP is the only signal left, and tier 3 is precisely the
        # situation where an arbitrary order becomes a real pick.
        if has_adp:
            g["_adp"] = pd.to_numeric(g["adp"], errors="coerce").fillna(9999.0)
            g = g.sort_values([value_col, "_adp"], ascending=[False, True])
            g = g.drop(columns=["_adp"])
        else:
            g = g.sort_values(value_col, ascending=False)
        g["tier"] = assign_tiers(g[value_col].to_numpy(), k=k, seed=seed)
        g["rank"] = np.arange(1, len(g) + 1)
        frames.append(g)

    out = pd.concat(frames, ignore_index=True)
    for col in TIER_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[list(TIER_COLUMNS)].sort_values(
        ["position", "rank"], ignore_index=True)


def write_tiers(table: pd.DataFrame, snapshot_id: str,
                root: Path = Path("data/artifacts")) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"tiers_{snapshot_id}.parquet"
    table.to_parquet(path, index=False)
    return path


def load_tiers(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataError(f"no tier artifact at {path}")
    return pd.read_parquet(path)


def tier3_recommendation(table: pd.DataFrame, needed_positions: list[str],
                         drafted: set[str]) -> pd.Series | None:
    """Best remaining player at a position the roster still needs.

    Returns None only when every needed position is exhausted, which the caller
    must render as "no recommendation" rather than as an empty pick — a silent
    empty is how a tier-3 fallback turns a bad night into a missed pick.
    """
    if not needed_positions:
        return None
    pool = table[table["position"].isin(needed_positions)
                 & ~table["player_id"].astype(str).isin(drafted)]
    if pool.empty:
        return None
    return pool.sort_values(["tier", "rank"]).iloc[0]
