"""nflverse pulls with blob write-through (§11.1, §11.2).

``nflreadpy`` does its own HTTP, so there is no response body to intercept.
Instead the *materialized frame* is serialized to parquet and content-addressed
— which is the property that actually matters: the digest names the data the
pipeline consumed, not the transport that delivered it.

nflverse applies stat corrections Monday-Wednesday, so a Thursday pull is the
cleanest available. A correction shows up as a new digest for the same logical
asset, which ``manifest.diff`` reports rather than silently absorbing.
"""
from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import pandas as pd

from src.platform.store import blobs
from src.platform.store.manifest import ManifestEntry, utc_now

SOURCE: Final = "nflverse"

#: logical asset -> nflreadpy loader name. Verified against nflreadpy 0.1.3.
ASSET_LOADERS: Final[dict[str, str]] = {
    "player_stats": "load_player_stats",
    "team_stats": "load_team_stats",
    "schedules": "load_schedules",
    "rosters": "load_rosters",
    "depth_charts": "load_depth_charts",
    "injuries": "load_injuries",
    "combine": "load_combine",
    "draft_picks": "load_draft_picks",
    "players": "load_players",
    "snap_counts": "load_snap_counts",
    "ff_opportunity": "load_ff_opportunity",
    "ff_playerids": "load_ff_playerids",
    # Play-by-play and on-field participation: red-zone opportunity,
    # routes run, team pace and defence allowed. Large pulls, so both
    # are fetched per season and reduced immediately.
    "pbp": "load_pbp",
    "participation": "load_participation",
}

#: Assets that accept a `seasons=` argument.
SEASONAL_ASSETS: Final = frozenset({
    "player_stats", "team_stats", "rosters", "depth_charts", "injuries",
    "draft_picks", "combine", "snap_counts", "ff_opportunity",
    "pbp", "participation",
})


@dataclass(frozen=True)
class Pull:
    frame: pd.DataFrame
    entry: ManifestEntry


def _to_pandas(obj) -> pd.DataFrame:
    """nflreadpy returns polars; normalize once, here."""
    return obj.to_pandas() if hasattr(obj, "to_pandas") else pd.DataFrame(obj)


def _frame_bytes(df: pd.DataFrame) -> bytes:
    """Deterministic parquet bytes for content addressing.

    Columns are sorted and the index dropped so that an incidental column
    reordering upstream does not masquerade as a data change.
    """
    buf = io.BytesIO()
    df[sorted(df.columns)].reset_index(drop=True).to_parquet(buf, index=False)
    return buf.getvalue()


def fetch(
    asset: str, *, seasons: Sequence[int] | None = None, blob_root=None,
    loader_module=None,
) -> Pull:
    if asset not in ASSET_LOADERS:
        raise KeyError(f"unknown nflverse asset {asset!r}; known: {sorted(ASSET_LOADERS)}")

    module = loader_module
    if module is None:  # pragma: no cover - exercised by the live-pull test
        import nflreadpy

        module = nflreadpy

    loader: Callable = getattr(module, ASSET_LOADERS[asset])
    if seasons is not None and asset in SEASONAL_ASSETS:
        raw = loader(seasons=list(seasons))
        params: dict[str, object] = {"seasons": list(seasons)}
    else:
        raw = loader()
        params = {}

    frame = _to_pandas(raw)
    digest = blobs.put(_frame_bytes(frame), root=blob_root)
    entry = ManifestEntry(
        source=SOURCE,
        logical_asset=asset,
        params_key=blobs.params_key(params),
        digest=digest,
        fetched_at=utc_now(),
        canonicalizer="identity",  # parquet bytes are already normalized
    )
    return Pull(frame=frame, entry=entry)


def load_from_blob(digest: str, *, blob_root=None) -> pd.DataFrame:
    """Rehydrate a frame from the store — no network, no nflreadpy."""
    return pd.read_parquet(io.BytesIO(blobs.get(digest, root=blob_root)))


def loaders_for_snapshot(
    entries: Sequence[ManifestEntry], *, blob_root=None
) -> dict[str, Callable[[], pd.DataFrame]]:
    """Build the ``AsOf`` loader map from a manifest — the offline path.

    This is what makes §21.1's gates replayable: given a snapshot id, every
    accessor reads from blobs and the network is never touched.
    """
    return {
        e.logical_asset: (lambda d=e.digest: load_from_blob(d, blob_root=blob_root))
        for e in entries
        if e.source == SOURCE
    }
