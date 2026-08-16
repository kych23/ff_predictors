"""Red zone, routes, pace and defence — what a season box score cannot say.

A carry on the two-yard line and a carry at midfield are the same row in a
season total and completely different fantasy events. Routes separate "on the
field and ignored" from "not on the field" — two opposite statements about a
player's role that a target count reads identically. Team pace is the size of
the pie every teammate divides.

All of which is a good story, and the measurement said no: wired in and
retrained, held-out Spearman moved +0.0015 across 2023-2025 and went the wrong
way in two of three seasons. The features are **built but not wired**; see
`scripts/train_projection_v2._pbp_aggregates` for the table and the two
conditions under which coming back would be worth it.

So these tests cover `pbp.py` as a working module — the reductions are correct
and stay correct — plus two at the bottom that pin the unwired state.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.core.config import load_league
from src.models.features import pbp


@pytest.fixture(scope="module")
def scoring():
    return load_league().scoring.offense


def _plays(rows):
    return pd.DataFrame(rows)


def _play(**kw):
    base = dict(season=2024, week=1, posteam="DET", defteam="CHI",
                play_type="run", yardline_100=50, receiver_player_id=None,
                rusher_player_id=None, pass_attempt=0, rush_attempt=1,
                yards_gained=4.0, touchdown=0, epa=0.1,
                game_seconds_remaining=1800, game_id="g1")
    base.update(kw)
    return base


# ------------------------------------------------------------- red zone
def test_only_plays_inside_the_twenty_count():
    frame = _plays([
        _play(yardline_100=5, rusher_player_id="a"),
        _play(yardline_100=19, rusher_player_id="a"),
        _play(yardline_100=21, rusher_player_id="a"),   # outside
        _play(yardline_100=60, rusher_player_id="a"),   # outside
    ])
    out = pbp.red_zone_opportunity(frame).set_index("player_id")
    assert out.loc["a", "rz_carries_per_game"] == pytest.approx(2.0)


def test_red_zone_share_reflects_the_team_not_just_the_count():
    """Ten carries on a team with forty is a different role from ten on a team
    with twelve, and only the share says which."""
    frame = _plays(
        [_play(yardline_100=5, rusher_player_id="a") for _ in range(3)]
        + [_play(yardline_100=5, rusher_player_id="b") for _ in range(9)])
    out = pbp.red_zone_opportunity(frame).set_index("player_id")
    assert out.loc["a", "rz_share_carries"] == pytest.approx(0.25)
    assert out.loc["b", "rz_share_carries"] == pytest.approx(0.75)


def test_targets_and_carries_are_counted_separately():
    frame = _plays([
        _play(yardline_100=8, play_type="pass", receiver_player_id="a"),
        _play(yardline_100=8, rusher_player_id="a"),
    ])
    out = pbp.red_zone_opportunity(frame).set_index("player_id")
    assert out.loc["a", "rz_targets_per_game"] == pytest.approx(1.0)
    assert out.loc["a", "rz_carries_per_game"] == pytest.approx(1.0)


def test_per_game_uses_the_players_own_appearances():
    """A player who missed half the season should not be divided by seventeen."""
    frame = _plays([
        _play(game_id="g1", yardline_100=5, rusher_player_id="a"),
        _play(game_id="g2", yardline_100=5, rusher_player_id="a"),
        _play(game_id="g3", yardline_100=5, rusher_player_id="b"),
    ])
    out = pbp.red_zone_opportunity(frame).set_index("player_id")
    assert out.loc["a", "rz_carries_per_game"] == pytest.approx(1.0)


def test_empty_play_by_play_yields_no_rows_not_a_crash():
    assert pbp.red_zone_opportunity(pd.DataFrame()).empty


# ------------------------------------------------------ team environment
def test_pace_and_pass_rate_describe_the_offence():
    frame = _plays(
        [_play(play_type="pass") for _ in range(6)]
        + [_play(play_type="run") for _ in range(4)])
    out = pbp.team_environment(frame).set_index("team")
    assert out.loc["DET", "team_plays_per_game"] == pytest.approx(10.0)
    assert out.loc["DET", "team_pass_rate"] == pytest.approx(0.6)


def test_red_zone_volume_is_a_team_attribute_too():
    frame = _plays([_play(yardline_100=10) for _ in range(3)]
                   + [_play(yardline_100=50) for _ in range(7)])
    out = pbp.team_environment(frame).set_index("team")
    assert out.loc["DET", "team_rz_plays_per_game"] == pytest.approx(3.0)


def test_kickoffs_and_punts_are_not_offensive_plays():
    frame = _plays([_play(play_type="run"), _play(play_type="punt"),
                    _play(play_type="kickoff")])
    out = pbp.team_environment(frame).set_index("team")
    assert out.loc["DET", "team_plays_per_game"] == pytest.approx(1.0)


# ------------------------------------------------------------- defence
def test_defence_is_split_by_what_it_gives_up_to_whom(scoring):
    """A defence stout against the run and porous against receivers is two
    different matchups depending on who you are starting — a single
    strength-of-schedule number averages that away."""
    frame = _plays(
        [_play(defteam="CHI", rusher_player_id="a", yards_gained=2.0)
         for _ in range(5)]
        + [_play(defteam="CHI", play_type="pass", receiver_player_id="b",
                 yards_gained=25.0) for _ in range(5)])
    out = pbp.defence_allowed(frame, scoring).set_index("team")
    assert out.loc["CHI", "def_fp_allowed_wr"] > out.loc["CHI", "def_fp_allowed_rb"]


def test_defence_is_scored_through_the_leagues_own_coefficients(scoring):
    frame = _plays([_play(defteam="CHI", rusher_player_id="a",
                          yards_gained=100.0, touchdown=1)])
    out = pbp.defence_allowed(frame, scoring).set_index("team")
    expected = 100.0 * scoring["rush_yd"] + 1 * scoring["rush_td"]
    assert out.loc["CHI", "def_fp_allowed_rb"] == pytest.approx(expected)


# -------------------------------------------------------------- routes
def test_routes_count_players_on_the_field_for_a_pass():
    frame = _plays([_play(play_type="pass", game_id="g1"),
                    _play(play_type="pass", game_id="g1")])
    part = pd.DataFrame({
        "nflverse_game_id": ["g1", "g1"],
        "offense_players": ["a;b;c", "a;b"],
    })
    out = pbp.routes_run(part, frame).set_index("player_id")
    assert out.loc["a", "routes_per_game"] == pytest.approx(2.0)
    assert out.loc["c", "routes_per_game"] == pytest.approx(1.0)


def test_missing_participation_is_not_an_error():
    """It only exists from 2018. Older seasons carry NaN, which is what a
    tree wants for 'unknown'."""
    assert pbp.routes_run(pd.DataFrame(), pd.DataFrame()).empty


# --------------------------------------- built, measured, NOT wired in
# These features were wired through `prior_production` and `assemble`, the
# model was retrained on them, and held-out Spearman moved +0.0015 on average
# across 2023-2025 — negative in two of the three seasons. The wiring was
# reverted; `scripts/train_projection_v2._pbp_aggregates` carries the numbers.
#
# The tests below pin the reverted state. They are not style checks: a feature
# measured at zero that quietly reappears costs ten minutes on every rebuild
# and, worse, launders a null result into an assumed improvement. Re-wiring is
# fine — it just has to be a decision someone makes with a new measurement,
# which means deleting these two tests on purpose.
def test_the_feature_assembly_does_not_read_play_by_play():
    import inspect

    from src.models.features import assemble, prior_production

    for module in (assemble, prior_production):
        source = inspect.getsource(module)
        assert "pbp" not in source, (
            f"{module.__name__} reads play-by-play again; the measured effect "
            "was zero, so re-wiring needs a fresh ablation, not a merge")


def test_the_training_pull_does_not_spend_ten_minutes_on_play_by_play():
    """`_pbp_aggregates` still exists and still works — it is simply not spread
    into `load_frames`. This pins that, because the cost is invisible: the
    build just gets slower."""
    import inspect

    import scripts.train_projection_v2 as train

    # Comments stripped: `load_frames` carries one naming the call precisely so
    # the way back is findable, and a substring check would trip over it.
    source = "\n".join(line.split("#")[0]
                       for line in inspect.getsource(train.load_frames).split("\n"))
    assert "_pbp_aggregates(" not in source
