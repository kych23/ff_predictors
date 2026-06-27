"""Rookie features: draft capital + combine + college (DrafterSpec.md §4.5.1).

Implements "no flat penalty" (§3): the model LEARNS the rookie mapping from draft
capital, athletic testing, and college production conditioned on position + landing
spot, instead of a blanket discount.

UDFAs (null draft capital) get a synthetic late draft slot (monotonic ordering)
PLUS an ``is_udfa`` indicator so the trees can isolate the undrafted class (some
UDFAs become stars). Missing college/combine -> NaN + indicator flag, never dropped
or zero-filled (§4.0).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# Synthetic draft slot for UDFAs: just past the last real pick (~262 in the 7-round era).
UDFA_SYNTHETIC_PICK = 300
UDFA_SYNTHETIC_ROUND = 8


def _height_to_inches(ht) -> Optional[float]:
    if ht is None or (isinstance(ht, float) and pd.isna(ht)):
        return None
    s = str(ht)
    if "-" in s:
        try:
            ft, inch = s.split("-")
            return int(ft) * 12 + float(inch)
        except (ValueError, TypeError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _combine_features(combine: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    """Athletic block keyed by gsis_id (join combine.pfr_id -> id_map.pfr_id)."""
    cols = ["gsis_id", "forty_time", "vertical", "broad_jump", "agility", "weight",
            "bmi", "speed_score", "has_combine"]
    if combine is None or combine.empty:
        return pd.DataFrame(columns=cols)
    c = combine.copy()
    pfr_to_gsis = dict(zip(id_map["pfr_id"].astype("string"), id_map["gsis_id"].astype(str)))
    c["gsis_id"] = c.get("pfr_id").astype("string").map(pfr_to_gsis)
    c = c[c["gsis_id"].notna()].copy()
    if c.empty:
        return pd.DataFrame(columns=cols)
    c["forty_time"] = pd.to_numeric(c.get("forty"), errors="coerce")
    c["vertical"] = pd.to_numeric(c.get("vertical"), errors="coerce")
    c["broad_jump"] = pd.to_numeric(c.get("broad_jump"), errors="coerce")
    cone = pd.to_numeric(c.get("cone"), errors="coerce")
    shuttle = pd.to_numeric(c.get("shuttle"), errors="coerce")
    c["agility"] = cone.fillna(shuttle)
    c["weight"] = pd.to_numeric(c.get("wt"), errors="coerce")
    inches = c.get("ht").map(_height_to_inches) if "ht" in c else np.nan
    c["bmi"] = 703.0 * c["weight"] / (pd.Series(inches, index=c.index) ** 2)
    # speed score = (weight * 200) / forty^4 (classic RB athleticism metric)
    c["speed_score"] = (c["weight"] * 200.0) / (c["forty_time"] ** 4)
    c["has_combine"] = 1
    return c.drop_duplicates("gsis_id")[cols]


def build_rookie_features(
    rookie_ids: list[str],
    players: pd.DataFrame,
    combine: pd.DataFrame,
    id_map: pd.DataFrame,
    college: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the rookie feature block for the given rookie gsis_ids.

    ``players`` supplies draft capital; ``college`` (optional, from
    ``ingest.college.build_college_features``) supplies the college block.
    """
    if not rookie_ids:
        return pd.DataFrame(columns=["player_id", "is_rookie"])

    p = players[players["player_id"].astype(str).isin(set(rookie_ids))].copy()
    out = pd.DataFrame({"player_id": p["player_id"].astype(str)})
    out["draft_round"] = pd.to_numeric(p.get("draft_round"), errors="coerce").values
    out["draft_pick"] = pd.to_numeric(p.get("draft_pick"), errors="coerce").values
    out["is_rookie"] = True

    # UDFA handling: no draft pick -> synthetic late slot + flag
    is_udfa = out["draft_pick"].isna() | (out["draft_pick"] == 0)
    out["is_udfa"] = is_udfa.astype("Int64")
    out.loc[is_udfa, "draft_pick"] = UDFA_SYNTHETIC_PICK
    out.loc[is_udfa & out["draft_round"].isna(), "draft_round"] = UDFA_SYNTHETIC_ROUND

    # combine block
    comb = _combine_features(combine, id_map)
    out = out.merge(comb, left_on="player_id", right_on="gsis_id", how="left").drop(
        columns=["gsis_id"], errors="ignore")
    out["has_combine"] = pd.to_numeric(out["has_combine"], errors="coerce").fillna(0).astype("Int64")

    # college block (NaN + has_college_stats=0 where the bridge failed, §4.0)
    if college is not None and not college.empty:
        out = out.merge(college, left_on="player_id", right_on="gsis_id", how="left").drop(
            columns=["gsis_id"], errors="ignore")
    out["has_college_stats"] = pd.to_numeric(
        out.get("has_college_stats", pd.Series(index=out.index)), errors="coerce"
    ).fillna(0).astype("Int64")
    return out
