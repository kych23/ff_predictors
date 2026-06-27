"""Fantasy Football Calculator ADP client.

Fetches full-PPR ADP by (teams, year) via the resilient ``ingest/http.py`` client,
resolves FFC names to ``gsis_id`` through the crosswalk, and records per-season
coverage so benchmark eligibility is derivable. Cache TTL = 24h for the
current season (FFC updates daily), immutable for past seasons. On draft day, if
FFC is unreachable the resilient client serves the last cached ADP with a loud
staleness warning rather than failing.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .http import ResilientClient
from .player_ids import normalize_name

logger = logging.getLogger(__name__)

FFC_URL = "https://fantasyfootballcalculator.com/api/v1/adp/{format}"
_CURRENT_TTL = 24 * 3600  # seconds


@dataclass
class AdpSeason:
    season: int
    format: str
    teams: int
    df: pd.DataFrame          # resolved ADP rows (-> adp table)
    n_players: int
    ranking_eligible: bool
    sim_eligible: bool


class FFCClient:
    def __init__(self) -> None:
        self._client = ResilientClient("ffc", rate_limit_per_min=60,
                                       cache_ttl_seconds=_CURRENT_TTL)

    def fetch(self, fmt: str, teams: int, year: Optional[int]) -> dict:
        params: Dict[str, object] = {"teams": teams}
        immutable = False
        if year is not None:
            params["year"] = year
            immutable = year < dt.date.today().year  # past seasons never change
        url = FFC_URL.format(format=fmt)
        return self._client.get_json(url, params=params, immutable=immutable)


def _resolve_stdev(row: dict) -> Optional[float]:
    """adp_stdev from the FFC ``stdev`` field; fallback (low-high)/4 (§4.2.2)."""
    stdev = row.get("stdev")
    if stdev not in (None, "", 0):
        try:
            return float(stdev)
        except (TypeError, ValueError):
            pass
    high, low = row.get("high"), row.get("low")
    try:
        if high is not None and low is not None:
            return abs(float(low) - float(high)) / 4.0
    except (TypeError, ValueError):
        pass
    return None


def build_adp_season(
    season: int,
    *,
    fmt: str,
    teams: int,
    fp_lookup: Dict[str, str],
    name_lookup: Dict[str, str],
    name_pos_lookup: Optional[Dict[tuple, str]] = None,
    adp_min_ranking: int = 0,
    adp_min_fullsim: int,
    snapshot_id: str,
    client: Optional[FFCClient] = None,
) -> AdpSeason:
    """Fetch one season's ADP and resolve players to gsis_id.

    Joins by ``fantasypros_id`` first, then normalized-name fallback (§4.2.2).
    """
    client = client or FFCClient()
    payload = client.fetch(fmt, teams, season)
    players = payload.get("players", []) if isinstance(payload, dict) else []

    rows: List[dict] = []
    for p in players:
        fp_id = str(p.get("player_id")) if p.get("player_id") is not None else None
        gsis = fp_lookup.get(fp_id) if fp_id else None
        if gsis is None and name_pos_lookup:
            pos = str(p.get("position") or "").upper()
            gsis = name_pos_lookup.get((normalize_name(p.get("name")), pos))
        if gsis is None:
            gsis = name_lookup.get(normalize_name(p.get("name")))
        if gsis is None:
            continue  # unresolved player excluded from the gsis-keyed table
        rows.append({
            "source": "ffc",
            "season": season,
            "format": fmt,
            "teams": teams,
            "player_id": gsis,
            "name": p.get("name"),
            "position": p.get("position"),
            "adp": float(p.get("adp")) if p.get("adp") is not None else None,
            "adp_stdev": _resolve_stdev(p),
            "n_drafts": p.get("times_drafted"),
            "snapshot_id": snapshot_id,
        })

    df = pd.DataFrame(rows).dropna(subset=["adp"]) if rows else pd.DataFrame()
    # Coverage measured against the RAW FFC list size, not just resolved rows,
    # so eligibility reflects true market depth (§4.0).
    n_players = len(players)
    return AdpSeason(
        season=season,
        format=fmt,
        teams=teams,
        df=df,
        n_players=n_players,
        ranking_eligible=n_players >= adp_min_ranking,
        sim_eligible=n_players >= adp_min_fullsim,
    )
