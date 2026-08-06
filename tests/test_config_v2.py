"""Config loader and validation tests (§10.5, §21.7)."""
from __future__ import annotations

import copy

import pytest
import yaml

from src.core.config.league import (
    DEFAULT_LEAGUE_PATH,
    _validate,
    compute_hash,
    load_league,
)
from src.core.config.strategy import DEFAULT_STRATEGY_PATH, load_strategy
from src.core.errors import ConfigError, PayoutImbalanceError, UnknownPrizeTypeError


@pytest.fixture(scope="module")
def league():
    return load_league()


@pytest.fixture()
def league_raw():
    return yaml.safe_load(DEFAULT_LEAGUE_PATH.read_text())


@pytest.fixture()
def strategy_raw():
    return yaml.safe_load(DEFAULT_STRATEGY_PATH.read_text())


def _load_strategy_from(tmp_path, raw, league):
    p = tmp_path / "strategy.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_strategy(league, str(p))


# --------------------------------------------------------------------- league
def test_league_loads_and_derives(league):
    assert league.teams == 12
    assert league.roster.rounds == 15
    assert league.model_version.startswith("draft_v2.")
    assert league.slot_plan.n_starters == 9
    assert league.slot_plan.laminar


def test_off_fumble_return_td_is_scored(league):
    """MEASURED derivable from player_stats.fumble_recovery_tds, so it is a
    scoring key rather than an exception (§10.2)."""
    assert league.scoring.offense["off_fumble_return_td"] == 6
    assert league.scoring_exceptions == ()


def test_fg_bands_cover_the_configured_domain(league_raw):
    _validate(league_raw)  # baseline passes
    league_raw["scoring"]["kicker"]["fg_bands"] = [[0, 39, 3], [45, 49, 4], [50, 99, 5]]
    with pytest.raises(ConfigError, match="gap"):
        _validate(league_raw)


def test_fg_bands_reject_overlap(league_raw):
    league_raw["scoring"]["kicker"]["fg_bands"] = [[0, 45, 3], [40, 49, 4], [50, 99, 5]]
    with pytest.raises(ConfigError, match="overlap"):
        _validate(league_raw)


def test_points_allowed_tiers_must_tile(league_raw):
    league_raw["scoring"]["dst"]["points_allowed_tiers"] = [[0, 0, 10], [1, 6, 7]]
    with pytest.raises(ConfigError, match="expected 999"):
        _validate(league_raw)


def test_bracket_must_resolve(league_raw):
    league_raw["schedule"]["playoff_teams"] = 7   # 7 - 2 = 5, not a power of two
    with pytest.raises(ConfigError, match="power of two"):
        _validate(league_raw)


def test_bracket_must_fit_in_playoff_weeks(league_raw):
    league_raw["schedule"]["playoff_weeks"] = [15, 16]  # needs 3 rounds
    with pytest.raises(ConfigError, match="playoff week"):
        _validate(league_raw)


def test_playoff_teams_cannot_exceed_teams(league_raw):
    league_raw["schedule"]["playoff_teams"] = 16
    with pytest.raises(ConfigError, match="exceeds teams"):
        _validate(league_raw)


def test_starter_slot_counts_must_be_positive(league_raw):
    league_raw["roster"]["slots"]["TE"] = 0
    with pytest.raises(ConfigError, match="must be >= 1"):
        _validate(league_raw)


def test_flex_eligibility_must_resolve(league_raw):
    league_raw["roster"]["flex_eligibility"]["FLEX"] = ["RB", "WR", "XX"]
    with pytest.raises(ConfigError, match="XX"):
        _validate(league_raw)


def test_calibration_buckets_match_the_real_calibrator(league):
    """Rule 7's other half: the canonical names in core must match what the
    calibrator actually produces. core cannot import the calibrator (layer
    rule), so the agreement is asserted here, where imports are unrestricted."""
    import pandas as pd

    from src.core.constants import CALIBRATION_BUCKETS
    from src.models.projection.calibrate import assign_bucket

    probes = [
        {"is_rookie": True},
        {"is_rookie": False, "seasons_of_history": 1},
        {"is_rookie": False, "seasons_of_history": 5, "team_changed": 1},
        {"is_rookie": False, "seasons_of_history": 5, "team_changed": 0},
    ]
    produced = {assign_bucket(pd.Series(p)) for p in probes}
    assert produced == set(CALIBRATION_BUCKETS)
    assert set(league.training.calibration_buckets) == produced


def test_wrong_bucket_names_are_rejected(league_raw):
    league_raw["training"]["calibration_buckets"] = ["rookie", "veteran"]
    with pytest.raises(ConfigError, match="does not match the canonical"):
        _validate(league_raw)


def test_scoring_exception_key_must_resolve(league_raw):
    league_raw["scoring_exceptions"] = [{"key": "offense.nonexistent"}]
    with pytest.raises(ConfigError, match="does not resolve"):
        _validate(league_raw)


def test_hash_ignores_comments_and_whitespace(league_raw):
    """The hash covers the parsed mapping, not the bytes, so a comment edit
    must not force a full pipeline re-run."""
    reordered = copy.deepcopy(league_raw)
    reordered["league"] = dict(reversed(list(reordered["league"].items())))
    assert compute_hash(league_raw) == compute_hash(reordered)


def test_hash_changes_on_a_scoring_change(league_raw):
    before = compute_hash(league_raw)
    league_raw["scoring"]["offense"]["rec"] = 0.5
    assert compute_hash(league_raw) != before


# ------------------------------------------------------------------- strategy
def test_strategy_loads(league):
    st = load_strategy(league)
    assert len(st.opponents.managers) == league.teams - 1
    assert st.decision_version.startswith("dec_v2.")
    assert st.adp.min_fullsim == league.teams * league.roster.rounds


def test_editing_strategy_does_not_change_model_version(league, tmp_path, strategy_raw):
    """The hash boundary: tuning a decision knob must leave models valid."""
    before_model = league.model_version
    strategy_raw["simulation"]["decision"]["indifference_zone_dollars"] = 5.0
    st = _load_strategy_from(tmp_path, strategy_raw, league)
    assert league.model_version == before_model
    assert st.simulation.decision["indifference_zone_dollars"] == 5.0


def test_manager_count_must_be_teams_minus_one(league, tmp_path, strategy_raw):
    strategy_raw["opponents"]["managers"].pop()
    with pytest.raises(ConfigError, match="teams - 1"):
        _load_strategy_from(tmp_path, strategy_raw, league)


def test_duplicate_manager_ids_rejected(league, tmp_path, strategy_raw):
    strategy_raw["opponents"]["managers"][1]["id"] = \
        strategy_raw["opponents"]["managers"][0]["id"]
    with pytest.raises(ConfigError, match="duplicate manager ids"):
        _load_strategy_from(tmp_path, strategy_raw, league)


def test_stale_league_hash_warns_rather_than_errors(league, tmp_path, strategy_raw):
    strategy_raw["requires_league_hash"] = "deadbeef1234"
    st = _load_strategy_from(tmp_path, strategy_raw, league)
    assert st.stale_strategy is True


# --------------------------------------------------------------------- payout
def _prizes(strategy_raw, prizes):
    strategy_raw["payout"]["prizes"] = prizes
    return strategy_raw


POT = 384.0

STRUCTURES = {
    "live_config": [
        {"id": "champ", "type": "final_rank", "rank": 1,
         "amount": {"dollars": 300}, "ties": "split"},
        {"id": "weekly", "type": "weekly_high_score", "weeks": {"from": 1, "to": 14},
         "amount": {"dollars": 6}, "ties": "split"},
    ],
    "winner_take_all": [
        {"id": "champ", "type": "final_rank", "rank": 1,
         "amount": {"pct_of_pot": 1.0}, "ties": "split"},
    ],
    "top_three": [
        {"id": "first", "type": "final_rank", "rank": 1,
         "amount": {"pct_of_pot": 0.60}, "ties": "split"},
        {"id": "second", "type": "final_rank", "rank": 2,
         "amount": {"pct_of_pot": 0.30}, "ties": "split"},
        {"id": "third", "type": "final_rank", "rank": 3,
         "amount": {"pct_of_pot": 0.10}, "ties": "split"},
    ],
    "champ_plus_points_title": [
        {"id": "champ", "type": "final_rank", "rank": 1,
         "amount": {"dollars": 300}, "ties": "split"},
        {"id": "points", "type": "season_points_rank", "rank": 1,
         "amount": {"dollars": 84}, "ties": "split"},
    ],
    "champ_weekly_points": [
        {"id": "champ", "type": "final_rank", "rank": 1,
         "amount": {"dollars": 250}, "ties": "split"},
        {"id": "weekly", "type": "weekly_high_score", "weeks": {"from": 1, "to": 14},
         "amount": {"dollars": 6}, "ties": "split"},
        {"id": "points", "type": "season_points_rank", "rank": 1,
         "amount": {"dollars": 50}, "ties": "split"},
    ],
}


@pytest.mark.parametrize("name", sorted(STRUCTURES))
def test_documented_payout_structures_conserve_the_pot(
    name, league, tmp_path, strategy_raw
):
    st = _load_strategy_from(tmp_path, _prizes(strategy_raw, STRUCTURES[name]), league)
    pot = st.payout.pot(league.teams)
    total = sum(
        p.amount.resolve(pot) * p.n_occurrences()
        for p in st.payout.prizes if not p.funded_external
    )
    assert total == pytest.approx(POT, abs=0.01)


def test_unbalanced_payout_is_refused(league, tmp_path, strategy_raw):
    prizes = [{"id": "champ", "type": "final_rank", "rank": 1,
               "amount": {"dollars": 300}, "ties": "split"}]
    with pytest.raises(PayoutImbalanceError) as exc:
        _load_strategy_from(tmp_path, _prizes(strategy_raw, prizes), league)
    assert exc.value.computed == 300.0
    assert exc.value.pot == POT


def test_unknown_prize_type_lists_registered_types(league, tmp_path, strategy_raw):
    prizes = [{"id": "x", "type": "most_touchdowns", "rank": 1,
               "amount": {"dollars": 384}, "ties": "split"}]
    with pytest.raises(UnknownPrizeTypeError, match="weekly_high_score"):
        _load_strategy_from(tmp_path, _prizes(strategy_raw, prizes), league)


def test_ties_all_rejected_in_a_conserved_prize(league, tmp_path, strategy_raw):
    prizes = list(STRUCTURES["live_config"])
    prizes[0] = {**prizes[0], "ties": "all"}
    with pytest.raises(ConfigError, match="ties: all"):
        _load_strategy_from(tmp_path, _prizes(strategy_raw, prizes), league)


def test_ties_all_permitted_when_externally_funded(league, tmp_path, strategy_raw):
    prizes = list(STRUCTURES["live_config"]) + [
        {"id": "sacko", "type": "final_rank", "rank": -1,
         "amount": {"dollars": -20}, "ties": "all", "funded": "external"},
    ]
    st = _load_strategy_from(tmp_path, _prizes(strategy_raw, prizes), league)
    sacko = next(p for p in st.payout.prizes if p.id == "sacko")
    assert sacko.funded_external and sacko.rank == -1


def test_negative_rank_within_bounds(league, tmp_path, strategy_raw):
    prizes = list(STRUCTURES["live_config"]) + [
        {"id": "sacko", "type": "final_rank", "rank": -13,
         "amount": {"dollars": -20}, "ties": "first", "funded": "external"},
    ]
    with pytest.raises(ConfigError, match="outside"):
        _load_strategy_from(tmp_path, _prizes(strategy_raw, prizes), league)


def test_weekly_prize_weeks_must_be_in_the_regular_season(
    league, tmp_path, strategy_raw
):
    prizes = list(STRUCTURES["live_config"])
    prizes[1] = {**prizes[1], "weeks": {"from": 1, "to": 17}}
    with pytest.raises(ConfigError, match="outside 1..14"):
        _load_strategy_from(tmp_path, _prizes(strategy_raw, prizes), league)


def test_amount_requires_exactly_one_form(league, tmp_path, strategy_raw):
    prizes = [{"id": "champ", "type": "final_rank", "rank": 1,
               "amount": {"dollars": 300, "pct_of_pot": 0.5}, "ties": "split"}]
    with pytest.raises(ConfigError, match="exactly one"):
        _load_strategy_from(tmp_path, _prizes(strategy_raw, prizes), league)


def test_weekly_occurrence_count_is_inclusive(league, tmp_path, strategy_raw):
    st = _load_strategy_from(tmp_path, strategy_raw, league)
    weekly = next(p for p in st.payout.prizes if p.type == "weekly_high_score")
    assert weekly.n_occurrences() == 14  # weeks 1..14 inclusive
