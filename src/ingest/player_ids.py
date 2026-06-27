"""Player-identity reconciliation + the college->NFL bridge

The single place player identity is resolved. gsis_id is the one true key; every other source ID maps to it via the ff_playerids crosswalk.

The college->NFL bridge has an explicit, ordered matching strategy:
  1. join ``load_draft_picks`` keys (gsis_id + cfb_player_id + pfr_player_id)
  2. else fuzzy-match on normalized name + college + class year
Normalization strips suffixes (Jr/Sr/II/III), punctuation, and applies a nickname map. Unmatched rookies are KEPT with college features NaN + ``has_college_stats=0`` (never drop, never zero-fill). The match rate is recorded and exposed.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from . import sources

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_NICKNAMES = {
    "mike": "michael", "chris": "christopher", "rob": "robert", "bob": "robert",
    "will": "william", "bill": "william", "joe": "joseph", "dave": "david",
    "matt": "matthew", "nate": "nathaniel", "tony": "anthony", "ben": "benjamin",
    "alex": "alexander", "josh": "joshua", "greg": "gregory", "ken": "kenneth",
    "steve": "steven", "jon": "jonathan", "dan": "daniel", "rich": "richard",
    "zach": "zachary", "cam": "cameron",
}


def normalize_name(name: Optional[str]) -> str:
    """Lowercase, strip accents/punctuation/suffixes; normalize the first-name nickname."""
    if not name or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"[^a-z ]", " ", s)
    parts = [p for p in s.split() if p and p not in _SUFFIXES]
    if not parts:
        return ""
    parts[0] = _NICKNAMES.get(parts[0], parts[0])
    return " ".join(parts)


@dataclass
class CrosswalkResult:
    id_map: pd.DataFrame          # gsis_id + external IDs (-> player_id_map table)
    bridge: pd.DataFrame          # gsis_id <-> cfb_player_id for rookies
    match_rate: float             # fraction of draftable rookies bridged to college id


def build_id_map() -> pd.DataFrame:
    """Build the gsis-keyed crosswalk DataFrame from ff_playerids."""
    df = sources.load_ff_playerids()
    df = df[df["gsis_id"].notna()].copy()
    out = pd.DataFrame({
        "gsis_id": df["gsis_id"].astype(str),
        "fantasypros_id": df.get("fantasypros_id"),
        "sleeper_id": df.get("sleeper_id"),
        "espn_id": df.get("espn_id"),
        "pfr_id": df.get("pfr_id"),
        "cfb_id": df.get("cfbref_id"),
        "merge_name": df.get("name").map(normalize_name) if "name" in df else None,
        "position": (df["position"].astype("string").str.upper()
                     if "position" in df.columns else pd.NA),
    })
    # years_exp is not in ff_playerids; derive nothing here (filled from rosters later).
    out["years_exp"] = pd.NA
    for c in ["fantasypros_id", "sleeper_id", "espn_id", "pfr_id", "cfb_id"]:
        if c in out:
            out[c] = out[c].astype("string")
    out = out.drop_duplicates(subset=["gsis_id"])
    return out


def build_college_bridge(draft_picks: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Map gsis_id -> cfb_player_id using draft_picks keys (strategy step 1).

    draft_picks carries both ``gsis_id`` and ``cfb_player_id`` directly, so this is
    the highest-confidence bridge. Fuzzy fallback (step 2) is applied in
    ``college.py`` when a rookie has no draft-pick row.
    """
    dp = draft_picks if draft_picks is not None else sources.load_draft_picks()
    cols = [c for c in ["gsis_id", "cfb_player_id", "pfr_player_id", "season", "college",
                        "pfr_player_name", "round", "pick"] if c in dp.columns]
    dp = dp[cols].copy()
    dp = dp[dp["gsis_id"].notna()]
    dp["norm_name"] = dp.get("pfr_player_name").map(normalize_name) if "pfr_player_name" in dp else ""
    dp = dp.rename(columns={"season": "draft_season"})
    return dp


def build_crosswalk() -> CrosswalkResult:
    id_map = build_id_map()
    bridge = build_college_bridge()
    bridged = bridge[bridge["cfb_player_id"].notna()] if "cfb_player_id" in bridge else bridge.iloc[0:0]
    match_rate = (len(bridged) / len(bridge)) if len(bridge) else 0.0
    return CrosswalkResult(id_map=id_map, bridge=bridge, match_rate=match_rate)


def fantasypros_lookup(id_map: pd.DataFrame) -> Dict[str, str]:
    """fantasypros_id -> gsis_id (primary ADP join key, §4.2.2)."""
    sub = id_map[id_map["fantasypros_id"].notna()]
    return dict(zip(sub["fantasypros_id"].astype(str), sub["gsis_id"].astype(str)))


def name_lookup(id_map: pd.DataFrame) -> Dict[str, str]:
    """normalized name -> gsis_id (ADP fallback join, §4.2.2). Ambiguous names dropped."""
    sub = id_map[id_map["merge_name"].astype(bool)]
    counts = sub["merge_name"].value_counts()
    unique_names = set(counts[counts == 1].index)
    sub = sub[sub["merge_name"].isin(unique_names)]
    return dict(zip(sub["merge_name"], sub["gsis_id"].astype(str)))


def name_position_lookup(id_map: pd.DataFrame) -> Dict[tuple, str]:
    """(normalized name, POSITION) -> gsis_id — the primary ADP name join.

    Disambiguates same-name players by position (e.g. Lamar Jackson QB vs the CB,
    Justin Jefferson WR vs the LB), which the name-only lookup drops as ambiguous.
    Only pairs that are STILL ambiguous (same name AND same position) are dropped.
    """
    if "position" not in id_map.columns:
        return {}
    sub = id_map[id_map["merge_name"].astype(bool) & id_map["position"].notna()].copy()
    sub["key"] = list(zip(sub["merge_name"], sub["position"].astype(str).str.upper()))
    counts = pd.Series([k for k in sub["key"]]).value_counts()
    unique = set(counts[counts == 1].index)
    sub = sub[sub["key"].isin(unique)]
    return dict(zip(sub["key"], sub["gsis_id"].astype(str)))
