"""College production from CFBD (DrafterSpec.md §4.2/§4.5.1).

CFBD = College Football Data, the source behind the R package ``cfbfastR``. The
Python ``cfbd`` client pins an incompatible ``pydantic`` v1, so we call the CFBD
REST API directly through the resilient ``ingest/http.py`` client (which still
honors the 60 req/min free-tier limit, retries, and disk cache). Requires
``CFBD_API_KEY`` in ``.env`` (CFBD is not anonymous).

College seasons are historical and static, so responses cache to disk and are
fetched once (``immutable=True``). Missing key / unmatched player -> the college
feature block is NaN with ``has_college_stats=0`` downstream (never drop, never
zero-fill, §4.0). This module produces a per-(gsis_id) college feature frame; the
NaN fallback is applied by ``features/rookies.py``.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional

import pandas as pd

from .http import ResilientClient
from .player_ids import normalize_name

logger = logging.getLogger(__name__)

CFBD_BASE = "https://api.collegefootballdata.com"

# Known abbreviation → full name so nflverse short forms match CFBD long forms.
_COLLEGE_ABBREV = {
    "usc": "southern california",
    "lsu": "louisiana state",
    "ucla": "california los angeles",
    "smu": "southern methodist",
    "tcu": "texas christian",
    "vmi": "virginia military",
    "unlv": "nevada las vegas",
    "utep": "texas el paso",
    "utsa": "texas san antonio",
    "fiu": "florida international",
    "fau": "florida atlantic",
    "ole miss": "mississippi",
}


def _normalize_college(name: Optional[str]) -> str:
    """Lowercase, strip accents/punctuation/filler; expand known abbreviations."""
    from .player_ids import ascii_fold
    s = ascii_fold(name)
    s = re.sub(r"\b(university|college|the|of|at|a&m)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return _COLLEGE_ABBREV.get(s, s)

# Conference -> competition level (Power 5 / Group of 5 / FCS), §4.5.1.
_P5 = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "Pac-10"}
_G5 = {"American Athletic", "Mountain West", "Sun Belt", "Mid-American",
       "Conference USA", "FBS Independents"}


def _competition_level(conference: Optional[str]) -> Optional[str]:
    if not conference or pd.isna(conference):
        return None
    if conference in _P5:
        return "P5"
    if conference in _G5:
        return "G5"
    return "FCS"


class CFBDClient:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("CFBD_API_KEY")
        self.enabled = bool(self.api_key)
        self._client = ResilientClient("cfbd", rate_limit_per_min=60)
        if not self.enabled:
            logger.warning(
                "CFBD_API_KEY not set — college features will be NaN "
                "(has_college_stats=0) per the missing-data rule (§4.0)."
            )

    def _get(self, path: str, params: Dict) -> list:
        if not self.enabled:
            return []
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            return self._client.get_json(
                CFBD_BASE + path, params=params, headers=headers, immutable=True
            )
        except Exception as exc:  # noqa: BLE001 - degrade to "no college data"
            logger.error("CFBD %s failed: %s", path, exc)
            return []

    def player_season_stats(self, year: int, category: str) -> pd.DataFrame:
        """/stats/player/season for one stat category (receiving/rushing/passing)."""
        data = self._get("/stats/player/season", {"year": year, "category": category})
        return pd.DataFrame(data)


def _pivot_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """CFBD returns long-format (player, statType, stat); pivot to wide per player.

    Keeps ``position`` in the index so the dominator can be position-aware (§4.5.1).
    """
    if df.empty:
        return df
    keep = [c for c in ["playerId", "player", "position", "team", "conference",
                        "category", "statType", "stat"] if c in df.columns]
    df = df[keep].copy()
    df["key"] = df["category"].astype(str) + "_" + df["statType"].astype(str)
    idx = ["playerId", "player", "position", "team", "conference"]
    idx = [c for c in idx if c in df.columns]
    wide = df.pivot_table(index=idx, columns="key", values="stat", aggfunc="first").reset_index()
    return wide


def build_college_features(
    rookie_seasons: Dict[str, int],
    bridge: pd.DataFrame,
    api_key: Optional[str] = None,
    birth_dates: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Build the college feature block keyed by gsis_id for the given rookies.

    Parameters
    ----------
    rookie_seasons : gsis_id -> NFL rookie season Y (college final season = Y-1).
    bridge : output of player_ids.build_college_bridge (gsis_id, cfb_player_id,
             norm_name, college, draft_season).

    Returns one row per resolvable gsis_id with the §4.5.1 college columns. Rookies
    with no match are simply absent (callers fill NaN + has_college_stats=0).
    """
    client = CFBDClient(api_key)
    if not client.enabled or not rookie_seasons:
        return pd.DataFrame(columns=["gsis_id", "has_college_stats"])

    # Final college season per rookie = NFL rookie season - 1.
    college_years = sorted({y - 1 for y in rookie_seasons.values() if y})
    cats = ["receiving", "rushing", "passing"]
    frames: List[pd.DataFrame] = []
    for year in college_years:
        for cat in cats:
            wide = _pivot_player_stats(client.player_season_stats(year, cat))
            if not wide.empty:
                wide["college_year"] = year
                frames.append(wide)
    if not frames:
        return pd.DataFrame(columns=["gsis_id", "has_college_stats"])

    college = pd.concat(frames, ignore_index=True)
    # Collapse a player's per-category pivot rows into one wide row per (player, year).
    grp_keys = [c for c in ["playerId", "player", "position", "college_year"]
                if c in college.columns]
    college = college.groupby(grp_keys, as_index=False).first()
    college["norm_name"] = college["player"].map(normalize_name)
    college["playerId"] = college["playerId"].astype(str)

    # Team-season totals (dominator denominators) derived from the FULL player pool —
    # no extra CFBD call, no fragile team-name match (§4.5.1 dominator).
    college = _attach_team_totals(college)

    # Bridge step 1: explicit cfb_player_id from draft_picks.
    br = bridge.copy()
    br["cfb_player_id"] = br.get("cfb_player_id").astype("string")
    matched = br.merge(
        college, left_on="cfb_player_id", right_on="playerId", how="inner"
    )
    # Bridge step 2: fuzzy name fallback for rookies with no cfb_player_id match.
    # Primary: (norm_name, norm_college) — highest precision; secondary: norm_name only
    # when the match is unique (one CFBD player per name).
    unmatched_gsis = set(rookie_seasons) - set(matched["gsis_id"])
    if unmatched_gsis:
        fb = br[br["gsis_id"].isin(unmatched_gsis)].copy()
        fb["norm_college"] = fb.get("college").map(_normalize_college)
        college_nc = college.copy()
        college_nc["norm_college"] = college_nc.get("team").map(_normalize_college)

        fuzzy = fb.merge(college_nc, on=["norm_name", "norm_college"],
                         how="inner", suffixes=("", "_c"))

        still_unmatched = unmatched_gsis - set(fuzzy.get("gsis_id", pd.Series()))
        if still_unmatched:
            fb2 = fb[fb["gsis_id"].isin(still_unmatched)]
            name_matches = fb2.merge(college_nc, on="norm_name",
                                     how="inner", suffixes=("", "_c"))
            # Only keep rookies where norm_name maps to exactly one CFBD player.
            unique_gsis = (
                name_matches.groupby("gsis_id")["playerId"].nunique()
                .pipe(lambda s: s[s == 1].index)
            )
            name_matches = name_matches[name_matches["gsis_id"].isin(unique_gsis)]
            fuzzy = pd.concat([fuzzy, name_matches], ignore_index=True)

        if not fuzzy.empty:
            matched = pd.concat([matched, fuzzy], ignore_index=True)

    # Keep each rookie's FINAL college season (= NFL rookie year - 1), not an
    # arbitrary earlier year that also got pulled for another rookie.
    matched["target_year"] = matched["gsis_id"].map(
        {g: (y - 1) for g, y in rookie_seasons.items() if y})
    final = matched[matched["college_year"] == matched["target_year"]]
    rest = matched[~matched["gsis_id"].isin(set(final["gsis_id"]))]
    if not rest.empty:
        rest = rest.sort_values("college_year").drop_duplicates("gsis_id", keep="last")
    matched = pd.concat([final, rest], ignore_index=True).drop_duplicates(subset=["gsis_id"])

    if birth_dates:
        matched["birth_date"] = matched["gsis_id"].map(birth_dates)

    return _derive_college_metrics(matched)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col), errors="coerce") if col in df else pd.Series(index=df.index, dtype=float)


# Yardage/TD columns summed to team-season totals for the dominator denominator.
_TEAM_TOTAL_COLS = ["receiving_YDS", "receiving_TD", "rushing_YDS", "rushing_TD"]


def _attach_team_totals(college: pd.DataFrame) -> pd.DataFrame:
    """Sum each (team, college_year)'s receiving/rushing yards+TDs over all players.

    The full CFBD player pool already gives every team's roster of stat-producers, so
    summing it IS the team-season total — no /stats/season team call, no team-name
    join fragility. These are the dominator denominators (§4.5.1).
    """
    if college.empty or "team" not in college.columns or "college_year" not in college.columns:
        for c in _TEAM_TOTAL_COLS:
            college[f"team_{c}"] = pd.NA
        return college
    work = college.copy()
    for c in _TEAM_TOTAL_COLS:
        work[c] = _num(work, c).fillna(0.0)
    totals = work.groupby(["team", "college_year"])[_TEAM_TOTAL_COLS].sum().reset_index()
    totals = totals.rename(columns={c: f"team_{c}" for c in _TEAM_TOTAL_COLS})
    return college.merge(totals, on=["team", "college_year"], how="left")


def _derive_college_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the §4.5.1 college features from wide CFBD stats.

    GP is absent from /stats/player/season (statTypes are LONG/REC/TD/YDS/YPR etc.),
    so per-game rates are impossible from this endpoint. v1 uses **season totals**
    plus per-opportunity ratios (YPR/YPC, completion%, YPA) — both leakage-safe and
    arguably stronger signals than per-game. ``college_dominator`` is the real
    team-share metric from derived team totals (§4.5.1).
    """
    if df.empty:
        return pd.DataFrame(columns=["gsis_id", "has_college_stats"])
    out = pd.DataFrame({"gsis_id": df["gsis_id"].astype(str)})
    rec_yds = _num(df, "receiving_YDS")
    rec = _num(df, "receiving_REC")
    rec_td = _num(df, "receiving_TD")
    rush_yds = _num(df, "rushing_YDS")
    rush_car = _num(df, "rushing_CAR")
    rush_td = _num(df, "rushing_TD")

    # Season totals (GP-free).
    out["college_receptions"] = rec
    out["college_rec_yards"] = rec_yds
    out["college_rec_td"] = rec_td
    out["college_rush_yards"] = rush_yds
    out["college_rush_carries"] = rush_car
    out["college_rush_td"] = rush_td
    # Per-opportunity efficiency (no GP needed) — prefer CFBD's own YPR/YPC, else derive.
    out["college_ypr"] = _num(df, "receiving_YPR").fillna(rec_yds / rec.where(rec > 0))
    out["college_ypc"] = _num(df, "rushing_YPC").fillna(rush_yds / rush_car.where(rush_car > 0))

    # Real dominator: position-aware share of team yards + TDs (§4.5.1). All numeric
    # (float) throughout — divide-by-zero guarded with .where, never replace->fillna.
    pos = df.get("position").astype("string").str.upper() if "position" in df else pd.Series(index=df.index, dtype="string")
    rb_mult = pos.eq("RB").astype(float)   # 1.0 for RB (add rushing), else 0.0
    # WR/TE: receiving share; RB: combined rush+rec share.
    player_yds = rec_yds.fillna(0.0) + rush_yds.fillna(0.0) * rb_mult
    player_td = rec_td.fillna(0.0) + rush_td.fillna(0.0) * rb_mult
    team_yds = _num(df, "team_receiving_YDS").fillna(0.0) + _num(df, "team_rushing_YDS").fillna(0.0) * rb_mult
    team_td = _num(df, "team_receiving_TD").fillna(0.0) + _num(df, "team_rushing_TD").fillna(0.0) * rb_mult
    yds_share = player_yds / team_yds.where(team_yds > 0)
    td_share = player_td / team_td.where(team_td > 0)
    dominator = 0.5 * yds_share + 0.5 * td_share
    # Dominator is a pass-catcher/RB metric — meaningless for QBs (their value is the
    # passing rates below), so leave it NaN rather than a noise value from a trick play.
    out["college_dominator"] = dominator.where(~pos.eq("QB"))

    # breakout_age = age during final college season (college_year − birth_year).
    # birth_date comes from nflverse players table, joined before this call.
    if "birth_date" in df.columns and "college_year" in df.columns:
        birth_year = pd.to_datetime(df["birth_date"], errors="coerce").dt.year
        out["breakout_age"] = pd.to_numeric(df["college_year"], errors="coerce") - birth_year
    else:
        out["breakout_age"] = pd.NA

    # college_target_share: CFBD /stats/player/season has no targets column (only REC).
    # Dropped — not stubbable without introducing a NaN-heavy feature. Dominator
    # captures the same signal for WR/TE more cleanly.

    out["competition_level"] = df.get("conference").map(_competition_level) if "conference" in df else None

    # QB college rates (per-opportunity, GP-free).
    out["college_completion_pct"] = _num(df, "passing_PCT")
    out["college_ypa"] = _num(df, "passing_YPA")
    pass_td = _num(df, "passing_TD")
    pass_att = _num(df, "passing_ATT").replace(0, pd.NA)
    out["college_td_rate"] = pass_td / pass_att

    # has_college_stats = 1 ONLY when real numeric production exists, not on a bare
    # name/id match with an empty payload (§4.0 — indicator must be honest).
    real = (out["college_rec_yards"].notna() | out["college_rush_yards"].notna()
            | _num(df, "passing_YDS").notna())
    out["has_college_stats"] = real.astype("Int64")
    return out
