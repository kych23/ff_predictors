"""A playoff shape that cannot resolve must be REJECTED, not silently mangled.

`_validate` required ``playoff_teams - playoff_byes`` to be a power of two.
That constrains round 0 only. Byes rejoin in round 1, whose size is
``(pt - pb) / 2 + pb`` and can be odd — and `final_ranks` pairs with
``range(len(active) // 2)``, so the middle seed is neither eliminated nor
advanced. The rank array was `np.empty`, so that team kept uninitialized
memory and a prize keyed on a rank paid out against it.

Demonstrated with one character changed in this league's own config: 12 teams,
6 playoff teams, ``playoff_byes: 4``. 6 - 4 = 2 is a power of two, so the old
rule accepted it; round 1 had 5 active teams and a replication returned ranks
``[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]``. A search over every shape the old
rule admitted found 262 broken ones.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.core.config.league import bracket_round_sizes, load_league
from src.core.errors import ConfigError
from src.domain.schedule.bracket import final_ranks, regular_season_ranks

CONFIG = "config/league.yaml"


# ------------------------------------------------------- the round structure
def test_the_shipped_league_resolves():
    cfg = load_league().schedule
    sizes = bracket_round_sizes(cfg.playoff_teams, cfg.playoff_byes)
    assert sizes == [4, 4, 2], "6 teams / 2 byes: quarters, semis, final"
    assert len(sizes) * cfg.matchup_length_weeks <= len(cfg.playoff_weeks)


@pytest.mark.parametrize(("teams", "byes", "expected"), [
    (2, 0, [2]),
    (4, 0, [4, 2]),
    (8, 0, [8, 4, 2]),
    (6, 2, [4, 4, 2]),
    (12, 4, [8, 8, 4, 2]),
])
def test_valid_shapes_enumerate_their_rounds(teams, byes, expected):
    assert bracket_round_sizes(teams, byes) == expected


@pytest.mark.parametrize(("teams", "byes"), [
    (6, 4),     # round 1 has 5 — the one demonstrated above
    (4, 2),     # round 1 has 3
    (6, 1),     # round 0 has 5
    (5, 1),     # round 1 has 3
    (10, 2),    # round 2 has 3
])
def test_shapes_that_drop_a_seat_are_rejected(teams, byes):
    with pytest.raises(ConfigError, match="odd"):
        bracket_round_sizes(teams, byes)


@pytest.mark.parametrize(("teams", "byes"), [(4, 4), (4, 5), (0, 0), (4, -1)])
def test_degenerate_shapes_are_rejected(teams, byes):
    with pytest.raises(ConfigError):
        bracket_round_sizes(teams, byes)


def test_the_power_of_two_rule_alone_admits_broken_shapes():
    """Pins WHY the old rule was insufficient rather than merely that it was."""
    def old_rule(pt, pb):
        n = pt - pb
        return n > 0 and (n & (n - 1)) == 0

    admitted_and_broken = []
    for pt in range(2, 17):
        for pb in range(pt):
            if not old_rule(pt, pb):
                continue
            try:
                bracket_round_sizes(pt, pb)
            except ConfigError:
                admitted_and_broken.append((pt, pb))
    assert (6, 4) in admitted_and_broken
    assert len(admitted_and_broken) > 20


# ----------------------------------------------------------- config loading
def _league_with(tmp_path, **schedule):
    raw = copy.deepcopy(yaml.safe_load(Path(CONFIG).read_text()))
    raw["schedule"].update(schedule)
    path = tmp_path / "league.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_load_league_rejects_the_shape_that_corrupted_ranks(tmp_path):
    with pytest.raises(ConfigError, match="odd"):
        load_league(_league_with(tmp_path, playoff_byes=4))


def test_load_league_still_accepts_the_real_config(tmp_path):
    """The fix must not tighten the rule past what this league actually uses."""
    assert load_league(_league_with(tmp_path)).schedule.playoff_byes == 2


def test_a_bracket_too_deep_for_its_weeks_is_still_rejected(tmp_path):
    with pytest.raises(ConfigError, match="playoff week"):
        load_league(_league_with(tmp_path, playoff_teams=12, playoff_byes=4,
                                 playoff_weeks=[15, 16]))


# ------------------------------------------------- ranks are always complete
def _team_week(teams, weeks, reps, seed=0):
    return np.random.default_rng(seed).normal(
        110, 20, (teams, weeks, reps)).astype("float32")


def test_every_team_gets_a_rank_in_one_to_t():
    cfg = load_league().schedule
    tw = _team_week(12, 17, 200)
    ranks = final_ranks(tw, regular_season_ranks(tw, cfg)[0], cfg)
    assert ranks.min() >= 1 and ranks.max() <= 12


def test_ranks_are_a_permutation_in_every_replication():
    """A dropped seat shows up as a duplicated or missing placing, which is
    exactly what the uninitialized entry produced."""
    cfg = load_league().schedule
    tw = _team_week(12, 17, 300, seed=7)
    ranks = final_ranks(tw, regular_season_ranks(tw, cfg)[0], cfg)
    for r in range(ranks.shape[1]):
        assert sorted(ranks[:, r].tolist()) == list(range(1, 13))


def test_final_ranks_raises_rather_than_returning_garbage(monkeypatch):
    """Defence in depth: if a malformed shape ever reaches the bracket without
    passing through config validation, it must stop, not return memory."""
    cfg = load_league().schedule
    broken = copy.copy(cfg)
    object.__setattr__(broken, "playoff_byes", 4)
    tw = _team_week(12, 17, 5)
    with pytest.raises((ValueError, ConfigError)):
        final_ranks(tw, regular_season_ranks(tw, cfg)[0], broken)
