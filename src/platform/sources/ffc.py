"""Fantasy Football Calculator ADP (§11.5.1).

**Primary ADP source**, not a fallback. Yahoo is blocked pending Fantasy Sports
API permission, and measurement on 2026-08-04 showed FFC covers what matters:

===========  ===============================================================
2021-2024    211 / 157 / 202 / 205 players, ``stdev`` populated on all
2025         ``status: "Error"``, 0 players — a genuine gap
2026 (live)  246 players, 4,460 drafts, 12-team PPR, 15 rounds
===========  ===============================================================

Draft night is therefore safe without Yahoo. The 2025 hole is what costs: the
conditional logit needs ADP contemporaneous with each draft to define the
choice set, so 2025 picks are unfittable and three of eleven managers lose
their only usable seasons.

Every response is written to the blob store, so refits and the reconciliation
gate replay offline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import httpx
import pandas as pd

from src.core.errors import DataError
from src.platform.store import blobs, canonicalize
from src.platform.store.manifest import ManifestEntry, utc_now

BASE_URL: Final = "https://fantasyfootballcalculator.com/api/v1/adp"
SOURCE: Final = "ffc"
USER_AGENT: Final = "ff_predictors/2.0 (personal draft assistant)"

#: Seasons FFC serves at all. 2025 returns status="Error" with zero players.
KNOWN_GAP_SEASONS: Final = frozenset({2025})


@dataclass(frozen=True)
class AdpPull:
    frame: pd.DataFrame
    entry: ManifestEntry
    n_players: int


def _canonical_params(fmt: str, teams: int, season: int | None) -> dict[str, object]:
    params: dict[str, object] = {"teams": teams}
    if season is not None:
        params["year"] = season
    return params


def fetch_adp(
    *, fmt: str = "ppr", teams: int = 12, season: int | None = None,
    client: httpx.Client | None = None, blob_root=None, timeout: float = 20.0,
) -> AdpPull:
    """Pull one ADP table and write it to the blob store.

    ``season=None`` asks FFC for the current window, which is what draft night
    uses. A season FFC does not serve raises rather than returning an empty
    frame — a silently empty ADP table would zero every survival curve.
    """
    params = _canonical_params(fmt, teams, season)
    owns_client = client is None
    client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=timeout)
    try:
        response = client.get(f"{BASE_URL}/{fmt}", params=params)
    finally:
        if owns_client:
            client.close()

    if response.status_code != 200:
        raise DataError(f"FFC returned HTTP {response.status_code} for {params}")

    canon = canonicalize.get(SOURCE)(response.content)
    digest = blobs.put(canon, root=blob_root)
    payload = response.json()
    players = payload.get("players") or []

    if not players:
        hint = (
            " (this season is a known FFC gap — see §11.5.1; the opponent model "
            "cannot fit picks from it)" if season in KNOWN_GAP_SEASONS else ""
        )
        raise DataError(
            f"FFC returned no players for {params} with status "
            f"{payload.get('status')!r}{hint}"
        )

    frame = pd.DataFrame(players)
    entry = ManifestEntry(
        source=SOURCE,
        logical_asset=f"adp/{fmt}",
        params_key=blobs.params_key(params),
        digest=digest,
        fetched_at=utc_now(),
        canonicalizer=SOURCE,
    )
    return AdpPull(frame=_normalize(frame), entry=entry, n_players=len(players))


#: FFC's position labels differ from league config. Measured 2026-08-04:
#: FFC ships DEF and PK where this league configures DST and K. Left
#: untranslated, every defense and kicker fails the identity join and lands in
#: the unresolved bucket for a reason that has nothing to do with identity.
POSITION_ALIASES: Final = {"DEF": "DST", "PK": "K"}

#: Columns the recommender consumes, in canonical order. Anything else FFC
#: sends is kept but sorted after these.
_LEADING_COLUMNS: Final = (
    "player_id", "player_name", "position", "team", "adp", "adp_stdev", "bye",
)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """FFC's payload -> the columns the recommender expects.

    ``stdev`` becomes ``adp_stdev`` because the log-normal survival curve is
    parameterized by (mean, sd) and a missing sd silently collapses it to a
    point mass.

    Column order is pinned. The canonicalizer sorts JSON keys before hashing,
    so a frame rehydrated from a blob would otherwise come back in a different
    order than the live pull and compare unequal despite identical data — which
    would make every offline-replay assertion either fail or be weakened to
    ignore ordering.
    """
    out = df.rename(columns={"name": "player_name", "stdev": "adp_stdev"})
    if "position" in out.columns:
        out["position"] = out["position"].replace(POSITION_ALIASES)
    for col in ("adp", "adp_stdev"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    leading = [c for c in _LEADING_COLUMNS if c in out.columns]
    trailing = sorted(c for c in out.columns if c not in leading)
    return out[leading + trailing].reset_index(drop=True)


def load_from_blob(digest: str, *, blob_root=None) -> pd.DataFrame:
    """Rehydrate a previously fetched pull — no network."""
    import json

    payload = json.loads(blobs.get(digest, root=blob_root))
    return _normalize(pd.DataFrame(payload.get("players") or []))
