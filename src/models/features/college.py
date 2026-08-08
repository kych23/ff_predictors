"""College production, deflated to something comparable across competition.

Rookies have no prior NFL production, so every `prior_*` feature is null for
them. Draft capital and depth chart tell you what a team thinks of a player and
where he sits; college production is the only evidence of what he has actually
done on a field. Without it the model separates rookies on intent alone.

**Raw college totals are not usable and the data says so loudly.** Sorted by
2025 receiving yards, the top four are Rhode Island (an FCS program), San José
State, UConn and North Texas — not one power-conference receiver. A thousand
yards against the CAA and a thousand against the SEC are different events, and
a model fed the raw number learns to prefer the weaker schedule.

So production enters as a **z-score within (season, position, conference)**.
That is the deflation: a receiver is measured against the other receivers who
faced comparable defences that year, not against the country. Conference TIER
is carried separately (power / group-of-five / FCS) so the model can still
learn the level gap that standardizing deliberately removes — being the best
receiver in the MAC is worth something, just not the same something as being
the best in the Big Ten.

Leakage: college seasons are strictly before a player's NFL rookie year, which
is before the target season by construction. Nothing here can see a snap of the
season being projected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.names import normalize_name

#: Emitted for every player. Null for anyone the name crosswalk missed and for
#: players with no college season on record.
COLLEGE_COLS = [
    "college_rec_yds_z", "college_rec_td_z", "college_rush_yds_z",
    "college_rush_td_z", "college_scrimmage_yds_z",
    "college_conference_tier", "college_final_season_age_proxy",
    "has_college_stats",
]

#: Power conferences. Standardizing within conference removes the level gap on
#: purpose; this puts a coarse version of it back as its own feature.
_POWER = {
    "SEC", "Big Ten", "ACC", "Big 12", "Pac-12", "Pac-10",
}
#: Everything at the FBS level that is not power. Anything else — FCS and
#: below — falls through to the lowest tier.
_GROUP_OF_FIVE = {
    "American Athletic", "Mountain West", "Mid-American", "Sun Belt",
    "Conference USA", "FBS Independents",
}

#: 3 = power, 2 = group of five, 1 = FCS and below. Ordinal on purpose: a tree
#: splitting on "at least G5" is a split worth having.
_TIER = {"power": 3, "g5": 2, "other": 1}

#: THE DEFLATOR, measured rather than chosen.
#:
#: Mean NFL ROOKIE-season points per game, by the tier of the player's last
#: college conference, over 507 name-matched players with a college season in
#: 2015-2024 and an NFL rookie row:
#:
#:     power           7.107 pts/g   (n=375)   ->  1.000
#:     group of five   5.806 pts/g   (n=107)   ->  0.817
#:     FCS and below   5.380 pts/g   (n= 25)   ->  0.757
#:
#: Production is multiplied by this BEFORE standardizing. Deflating first and
#: standardizing second is the whole point: a 1,300-yard Mountain West season
#: becomes ~1,062 effective yards and is then compared against everyone.
#:
#: An earlier version standardized WITHIN conference instead, which does the
#: opposite of what it sounds like — it measures how far a player is above his
#: own peers, so dominating a weak conference scores highest. It ranked SWAC
#: and NEC receivers above the entire Big Ten. The mistake is instructive
#: enough to leave written down.
TIER_DEFLATOR = {3: 1.000, 2: 0.817, 1: 0.757}

#: Minimum peers in a (season, position, conference) cell before a z-score
#: means anything. Below this the cell falls back to (season, position), which
#: is a weaker comparison but a real one.
MIN_PEERS = 8

_STAT_TO_FEATURE = {
    "receiving_yds": "college_rec_yds_z",
    "receiving_td": "college_rec_td_z",
    "rushing_yds": "college_rush_yds_z",
    "rushing_td": "college_rush_td_z",
    "scrimmage_yds": "college_scrimmage_yds_z",
}


def conference_tier(conference: str) -> int:
    name = (conference or "").strip()
    if name in _POWER:
        return _TIER["power"]
    if name in _GROUP_OF_FIVE:
        return _TIER["g5"]
    return _TIER["other"]


def _zscore(frame: pd.DataFrame, column: str, keys: list[str]) -> pd.Series:
    """Standardize within `keys`, leaving thin cells as NaN.

    Keys are (season, position): season because offensive levels drift, and
    position because a tight end's yardage is not a receiver's. Conference is
    deliberately NOT a key — see `TIER_DEFLATOR`.
    """
    grouped = frame.groupby(keys)[column]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    size = grouped.transform("size")
    z = (frame[column] - mean) / std.replace(0, np.nan)
    return z.where(size >= MIN_PEERS)


def standardize_production(stats: pd.DataFrame) -> pd.DataFrame:
    """Add a z-score per stat, within (season, position, conference).

    Falls back to (season, position) for cells with too few peers — a
    three-player conference cell produces a z-score that is mostly noise, and
    an honest wider comparison beats a precise meaningless one.
    """
    if stats is None or stats.empty:
        return pd.DataFrame(columns=["cfb_player_id", *_STAT_TO_FEATURE.values()])

    out = stats.copy()
    for column in ("receiving_yds", "receiving_td", "rushing_yds",
                   "rushing_td", "receiving_rec", "rushing_car"):
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    # Scrimmage yards: the position-neutral measure of how much offence ran
    # through this player, which is what transfers.
    out["scrimmage_yds"] = out["receiving_yds"] + out["rushing_yds"]

    out["college_conference_tier"] = out["conference"].map(conference_tier)
    deflator = out["college_conference_tier"].map(TIER_DEFLATOR).fillna(
        TIER_DEFLATOR[_TIER["other"]])

    for stat, feature in _STAT_TO_FEATURE.items():
        # Deflate FIRST, then standardize within (season, position) across the
        # whole country. Standardizing within conference would compare a player
        # only to the peers he already beat.
        out[f"_deflated_{stat}"] = out[stat] * deflator
        out[feature] = _zscore(out, f"_deflated_{stat}", ["season", "position"])
    return out


def build_college_features(cfb_stats: pd.DataFrame,
                           players: pd.DataFrame,
                           player_ids: pd.Series,
                           target_season: int) -> pd.DataFrame:
    """One row per player id: deflated college production, or nulls.

    ``players`` supplies the crosswalk (name, position, rookie_year). The join
    is on NORMALIZED NAME plus position because nflverse and CFBD do not share
    an id space — nflverse stores a slug (``joe-burrow-1``), CFBD a numeric id.
    Measured coverage on the 2026 class: 84.3%.

    Only college seasons STRICTLY BEFORE a player's NFL rookie year are
    eligible, and his most recent such season is the one used. That rule is
    what keeps the crosswalk honest as well as leak-free: without it a 2026
    rookie could match a same-named college player still on a roster today.
    """
    ids = pd.Series(list(player_ids), dtype=str).rename("player_id")
    empty = pd.DataFrame({"player_id": ids})
    for column in COLLEGE_COLS:
        empty[column] = 0 if column == "has_college_stats" else np.nan
    if cfb_stats is None or cfb_stats.empty or players is None or players.empty:
        return empty

    stats = standardize_production(cfb_stats)
    stats["_key"] = (stats["player"].map(normalize_name) + "|"
                     + stats["position"].astype(str))

    roster = players.copy()
    roster["player_id"] = roster["player_id"].astype(str)
    name_col = "name" if "name" in roster.columns else "display_name"
    roster["_key"] = (roster[name_col].map(normalize_name) + "|"
                      + roster["position"].astype(str))
    roster["rookie_year"] = pd.to_numeric(
        roster.get("rookie_year"), errors="coerce").fillna(target_season)
    roster = roster[roster["player_id"].isin(set(ids))]

    merged = roster[["player_id", "_key", "rookie_year"]].merge(
        stats, on="_key", how="inner")
    if merged.empty:
        return empty

    # Strictly before the NFL debut, and never at or after the target season.
    merged = merged[(merged["season"] < merged["rookie_year"])
                    & (merged["season"] < target_season)]
    if merged.empty:
        return empty

    # The FINAL college season is the one that describes the player who
    # entered the league; a sophomore year is a different athlete.
    merged = (merged.sort_values(["player_id", "season"])
                    .drop_duplicates("player_id", keep="last"))

    keep = ["player_id", *_STAT_TO_FEATURE.values(), "college_conference_tier"]
    out = empty.drop(columns=[c for c in COLLEGE_COLS if c != "has_college_stats"])
    out = out.drop(columns=["has_college_stats"]).merge(
        merged[keep], on="player_id", how="left")
    out["college_final_season_age_proxy"] = (
        target_season - merged.set_index("player_id")["season"]
        .reindex(out["player_id"]).to_numpy())
    out["has_college_stats"] = out["college_scrimmage_yds_z"].notna().astype(int)
    return out[["player_id", *COLLEGE_COLS]]
