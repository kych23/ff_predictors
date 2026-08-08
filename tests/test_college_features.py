"""College production, deflated for level of competition.

Rookies have no prior NFL production, so without this they are separated on
draft capital and depth chart alone — what a team BELIEVES about a player,
never what he has done.

**Deflation has a direction, and getting it backwards is easy.** Sorted by raw
2025 receiving yards the national leaders are Rhode Island (FCS), San José
State and UConn — raw college totals measure schedule as much as talent. The
first version of this module standardized WITHIN conference, which sounds like
a competition adjustment and is the opposite of one: it measures how far a
player is above his own peers, so dominating a weak conference scores highest.
It ranked SWAC and NEC receivers above the entire Big Ten.

The fix deflates first, by a MEASURED factor, then standardizes nationally.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.features.college import (
    MIN_PEERS,
    TIER_DEFLATOR,
    build_college_features,
    conference_tier,
    standardize_production,
)


# ------------------------------------------------------------- the tiers
@pytest.mark.parametrize(("conference", "tier"), [
    ("SEC", 3), ("Big Ten", 3), ("ACC", 3), ("Big 12", 3),
    ("Mountain West", 2), ("Mid-American", 2), ("American Athletic", 2),
    ("CAA", 1), ("SWAC", 1), ("Big Sky", 1), ("", 1), (None, 1),
])
def test_conference_tiers(conference, tier):
    assert conference_tier(conference) == tier


def test_the_deflator_is_ordered_and_measured():
    """Means of NFL rookie fppg by college tier over 507 matched players:
    power 7.107, group of five 5.806, FCS 5.380. Power is the numeraire."""
    assert TIER_DEFLATOR[3] == 1.0
    assert TIER_DEFLATOR[3] > TIER_DEFLATOR[2] > TIER_DEFLATOR[1] > 0.5


# ------------------------------------------------- the direction of the fix
def _two_conferences(n=40, power_yds=1000.0, weak_yds=1300.0):
    """A weak-conference player who out-produces every power player."""
    rng = np.random.default_rng(0)
    rows = []
    for conf, base in (("SEC", power_yds), ("SWAC", weak_yds)):
        for i in range(n):
            rows.append({
                "cfb_player_id": f"{conf}{i}", "player": f"{conf} P{i}",
                "position": "WR", "team": conf, "conference": conf,
                "season": 2025,
                "receiving_yds": base - i * 20 + rng.normal(0, 5),
                "receiving_td": 8.0, "receiving_rec": 60.0,
                "rushing_yds": 0.0, "rushing_td": 0.0, "rushing_car": 0.0,
            })
    return pd.DataFrame(rows)


def test_a_weak_conference_star_is_deflated_below_a_power_peer():
    """The headline. The SWAC leader out-gains the SEC leader on raw yards and
    must still not outrank him once competition is priced in."""
    out = standardize_production(_two_conferences())
    swac = out[out.conference == "SWAC"]["college_rec_yds_z"].max()
    sec = out[out.conference == "SEC"]["college_rec_yds_z"].max()
    assert swac < sec, "deflation is running backwards"


def test_deflation_preserves_order_within_a_conference():
    """It discounts a level, it does not reshuffle players who faced the same
    schedule."""
    stats = _two_conferences()
    out = standardize_production(stats)
    sec = out[out.conference == "SEC"].sort_values("receiving_yds",
                                                   ascending=False)
    assert sec["college_rec_yds_z"].is_monotonic_decreasing


def test_identical_production_orders_strictly_by_tier():
    """Hold production EXACTLY equal across three tiers; only the schedule
    differs. The resulting order is the deflator and nothing else.

    (An earlier version of this test gave the weak conference 300 more raw
    yards and then asserted its mean must be lower — which the deflator does
    not claim. It discounts a level; it does not erase a 300-yard gap.)
    """
    rows = []
    for conf in ("SEC", "Mountain West", "SWAC"):
        for i in range(MIN_PEERS + 4):
            rows.append({"cfb_player_id": f"{conf}{i}", "player": f"{conf}{i}",
                         "position": "WR", "team": conf, "conference": conf,
                         "season": 2025, "receiving_yds": 1000.0 - 20 * i,
                         "receiving_td": 8.0, "receiving_rec": 60.0,
                         "rushing_yds": 0.0, "rushing_td": 0.0,
                         "rushing_car": 0.0})
    out = standardize_production(pd.DataFrame(rows))
    by_tier = out.groupby("college_conference_tier")["college_rec_yds_z"].mean()
    assert by_tier.loc[3] > by_tier.loc[2] > by_tier.loc[1]


def test_an_equal_player_in_a_weaker_conference_scores_lower():
    """Identical raw production, different schedule — the whole point."""
    stats = _two_conferences(power_yds=1000.0, weak_yds=1000.0)
    out = standardize_production(stats)
    sec = out[out.conference == "SEC"]["college_rec_yds_z"].mean()
    swac = out[out.conference == "SWAC"]["college_rec_yds_z"].mean()
    assert swac < sec


# ------------------------------------------------------ standardizing keys
def test_thin_cells_do_not_produce_a_z_score():
    """A three-player cell yields a z-score made of noise."""
    rows = [{"cfb_player_id": f"x{i}", "player": f"P{i}", "position": "TE",
             "team": "T", "conference": "SEC", "season": 2025,
             "receiving_yds": 100.0 * i, "receiving_td": 1.0,
             "receiving_rec": 10.0, "rushing_yds": 0.0, "rushing_td": 0.0,
             "rushing_car": 0.0} for i in range(MIN_PEERS - 1)]
    out = standardize_production(pd.DataFrame(rows))
    assert out["college_rec_yds_z"].isna().all()


def test_positions_are_standardized_separately():
    """A tight end's yardage is not a receiver's, and a shared scale would
    make every TE look like a failed WR."""
    rows = []
    for pos, base in (("WR", 1000.0), ("TE", 400.0)):
        for i in range(MIN_PEERS + 4):
            rows.append({"cfb_player_id": f"{pos}{i}", "player": f"{pos}{i}",
                         "position": pos, "team": "T", "conference": "SEC",
                         "season": 2025, "receiving_yds": base - 10 * i,
                         "receiving_td": 5.0, "receiving_rec": 40.0,
                         "rushing_yds": 0.0, "rushing_td": 0.0,
                         "rushing_car": 0.0})
    out = standardize_production(pd.DataFrame(rows))
    for pos in ("WR", "TE"):
        z = out[out.position == pos]["college_rec_yds_z"]
        assert abs(z.mean()) < 0.2, f"{pos} should be centred on its own peers"


# ----------------------------------------------------------- the crosswalk
def _players():
    return pd.DataFrame({
        "player_id": ["a", "b", "c"],
        "name": ["Marvin Harrison Jr.", "Old Vet", "Unmatched Guy"],
        "position": ["WR", "WR", "WR"],
        "rookie_year": [2024, 2015, 2024],
    })


def _stats_for(name, season, conference="SEC", yds=1200.0):
    rows = [{"cfb_player_id": f"x{i}", "player": f"Filler {i}",
             "position": "WR", "team": "T", "conference": conference,
             "season": season, "receiving_yds": 300.0 + i, "receiving_td": 2.0,
             "receiving_rec": 20.0, "rushing_yds": 0.0, "rushing_td": 0.0,
             "rushing_car": 0.0} for i in range(MIN_PEERS + 2)]
    rows.append({"cfb_player_id": "target", "player": name, "position": "WR",
                 "team": "T", "conference": conference, "season": season,
                 "receiving_yds": yds, "receiving_td": 12.0,
                 "receiving_rec": 80.0, "rushing_yds": 0.0, "rushing_td": 0.0,
                 "rushing_car": 0.0})
    return pd.DataFrame(rows)


def test_the_suffix_fold_makes_the_crosswalk_work():
    """nflverse and CFBD do not share an id space, so the join is on
    normalized name — and 'Marvin Harrison Jr.' vs 'Marvin Harrison' is the
    mismatch class the normalizer exists for."""
    stats = _stats_for("Marvin Harrison", 2023)
    out = build_college_features(stats, _players(),
                                 pd.Series(["a", "b", "c"]), 2026)
    row = out[out.player_id == "a"].iloc[0]
    assert row["has_college_stats"] == 1
    assert row["college_rec_yds_z"] > 1.0


def test_a_college_season_at_or_after_the_nfl_debut_is_refused():
    """Leakage guard and crosswalk guard in one. A 2026 rookie must not match
    a same-named college player still playing today."""
    stats = _stats_for("Marvin Harrison", 2024)   # == rookie_year for 'a'
    out = build_college_features(stats, _players(), pd.Series(["a"]), 2026)
    assert out.iloc[0]["has_college_stats"] == 0


def test_an_unmatched_player_gets_nulls_not_zeros():
    """Absent is not zero. A zero college z-score claims an average season."""
    stats = _stats_for("Marvin Harrison", 2023)
    out = build_college_features(stats, _players(), pd.Series(["c"]), 2026)
    row = out.iloc[0]
    assert row["has_college_stats"] == 0
    assert pd.isna(row["college_rec_yds_z"])


def test_the_final_college_season_is_the_one_used():
    """A sophomore year describes a different athlete."""
    early = _stats_for("Marvin Harrison", 2021, yds=400.0)
    late = _stats_for("Marvin Harrison", 2023, yds=1400.0)
    out = build_college_features(pd.concat([early, late]), _players(),
                                 pd.Series(["a"]), 2026)
    assert out.iloc[0]["college_rec_yds_z"] > 1.0


def test_no_cfbd_data_degrades_to_nulls_rather_than_raising():
    """CFBD is the only source behind an API key. A board that cannot be built
    is worse than one without college features."""
    out = build_college_features(pd.DataFrame(), _players(),
                                 pd.Series(["a", "b"]), 2026)
    assert len(out) == 2
    assert (out["has_college_stats"] == 0).all()
