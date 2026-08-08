"""College Football Data pulls, content-addressed like every other source.

CFBD is the only source here behind an API key. `CFBD_API_KEY` is read from the
environment (a `.env` at the repo root is loaded if present), and a missing key
is NOT an error: the caller gets an empty frame and the college feature block
degrades to absent rather than the pipeline refusing to build. A draft board
that cannot be built is worse than one without college production.

**The id crosswalk is by NAME, and that is a deliberate concession.** nflverse
carries `cfb_player_id` as a slug (``joe-burrow-1``) while CFBD keys on a
numeric `playerId` (``4691138``) — different id spaces, so the obvious join
does not exist. Measured on the 2026 class: normalized name plus position
matches **264 of 313** rookie skill players, 84.3%, against 80.5% for the
ADP↔nflverse join this project already relies on. Unmatched players simply
carry no college features, which the model handles as missing data.
"""
from __future__ import annotations

import io
import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from src.platform.store import blobs
from src.platform.store.manifest import ManifestEntry, utc_now

SOURCE: Final = "cfbd"
BASE_URL: Final = "https://api.collegefootballdata.com"

#: Stat categories worth pulling. Defense and special teams are irrelevant to a
#: fantasy board and would triple the request count for nothing.
CATEGORIES: Final = ("receiving", "rushing", "passing")

#: The statTypes actually used, per category. CFBD returns one ROW PER STAT,
#: so this is also the pivot vocabulary.
KEEP_STATS: Final = {
    "receiving": ("REC", "YDS", "TD"),
    "rushing": ("CAR", "YDS", "TD"),
    "passing": ("YDS", "TD", "INT"),
}

_ENV_LOADED = False


@dataclass(frozen=True)
class Pull:
    frame: pd.DataFrame
    entry: ManifestEntry


def _load_env(root: Path | None = None) -> None:
    """Read a repo-root `.env` into the environment, once, without clobbering.

    No python-dotenv dependency for four lines, and `setdefault` so a real
    environment variable always beats the file.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    path = (root or Path(__file__).resolve().parents[3]) / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def api_key() -> str | None:
    _load_env()
    key = os.environ.get("CFBD_API_KEY", "").strip()
    return key or None


def _get(path: str, params: dict[str, Any], *, timeout: float = 60.0) -> list[dict]:
    key = api_key()
    if not key:
        return []
    query = "&".join(f"{k}={v}" for k, v in params.items())
    request = urllib.request.Request(
        f"{BASE_URL}{path}?{query}",
        headers={"Authorization": f"Bearer {key}",
                 "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _frame_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df[sorted(df.columns)].reset_index(drop=True).to_parquet(buf, index=False)
    return buf.getvalue()


def season_stats(seasons: Sequence[int], *, blob_root=None,
                 getter=_get) -> Pull:
    """Player-season production, one row per (player, season, category).

    CFBD returns one row PER STAT — ``{player, statType: "YDS", stat: "398"}``
    — so this pivots to columns. `conference` is carried through because it is
    what makes the production comparable: raw SEC yardage and raw MAC yardage
    are not the same quantity, and the feature layer standardizes within
    conference rather than pretending they are.
    """
    rows: list[dict] = []
    for season in seasons:
        for category in CATEGORIES:
            try:
                payload = getter("/stats/player/season",
                                 {"year": season, "seasonType": "regular",
                                  "category": category})
            except (urllib.error.URLError, TimeoutError, OSError):
                # A source that is down must not take the board with it.
                continue
            for row in payload:
                stat_type = str(row.get("statType", ""))
                if stat_type not in KEEP_STATS.get(category, ()):
                    continue
                rows.append({
                    "cfb_player_id": str(row.get("playerId", "")),
                    "player": str(row.get("player", "")),
                    "position": str(row.get("position") or ""),
                    "team": str(row.get("team") or ""),
                    "conference": str(row.get("conference") or ""),
                    "season": int(row.get("season", season)),
                    "column": f"{category}_{stat_type.lower()}",
                    "value": pd.to_numeric(row.get("stat"), errors="coerce"),
                })

    if rows:
        long = pd.DataFrame(rows)
        frame = (long.pivot_table(
            index=["cfb_player_id", "player", "position", "team",
                   "conference", "season"],
            columns="column", values="value", aggfunc="max")
            .reset_index())
        frame.columns.name = None
    else:
        frame = pd.DataFrame(columns=[
            "cfb_player_id", "player", "position", "team", "conference",
            "season"])

    digest = blobs.put(_frame_bytes(frame), root=blob_root)
    entry = ManifestEntry(
        source=SOURCE,
        logical_asset="player_season_stats",
        params_key=blobs.params_key({"seasons": list(seasons),
                                     "categories": list(CATEGORIES)}),
        digest=digest,
        fetched_at=utc_now(),
        canonicalizer="identity",
    )
    return Pull(frame=frame, entry=entry)
