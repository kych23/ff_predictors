"""Identity spine (§11.4).

The cascade's value is that a caller can tell an `exact_id` hit from a `fuzzy`
one. So these tests check the ORDER of the tiers and the reported provenance,
not just whether a lookup succeeded — a spine that resolves everything by fuzzy
match and reports "matched" is worse than one that resolves less and says so.
"""
from __future__ import annotations

import pandas as pd
import pytest
import yaml

from src.core.errors import ConfigError
from src.platform.identity.spine import (
    Spine,
    build_spine,
    load_overrides,
)


def _players():
    return pd.DataFrame([
        {"player_id": "00-001", "name": "Marvin Harrison Jr.", "position": "WR",
         "team": "ARI", "yahoo_id": "40001"},
        {"player_id": "00-002", "name": "Ja'Marr Chase", "position": "WR",
         "team": "CIN", "yahoo_id": "40002"},
        {"player_id": "00-003", "name": "Michael Pittman", "position": "WR",
         "team": "IND", "yahoo_id": None},
        {"player_id": "00-004", "name": "Michael Pittman", "position": "RB",
         "team": "LAC", "yahoo_id": None},
        {"player_id": "00-005", "name": "Rookie Guy", "position": "RB",
         "team": "NYJ", "yahoo_id": None, "is_rookie": True},
    ])


@pytest.fixture
def spine():
    return Spine(_players(), snapshot_id="sn2_test")


# --------------------------------------------------------- cascade order
def test_exact_id_wins_when_a_yahoo_id_is_supplied(spine):
    """Implemented but currently unreachable in production — Fantasy Sports
    API access is pending. Tested so that access is a config change."""
    row = spine.resolve("wrong name entirely", "WR", yahoo_id="40002")
    assert row.player_id == "00-002"
    assert row.tier == "exact_id"


def test_manual_outranks_deterministic():
    """A human already decided this one; the cascade must not second-guess it."""
    spine = Spine(_players(), snapshot_id="s",
                  overrides={"jamarr chase|WR": "00-001"})
    row = spine.resolve("Ja'Marr Chase", "WR")
    assert row.player_id == "00-001"
    assert row.tier == "manual"


def test_deterministic_folds_suffixes_and_punctuation(spine):
    for typed in ("Marvin Harrison Jr.", "marvin harrison", "MARVIN HARRISON III"):
        row = spine.resolve(typed, "WR")
        assert row.player_id == "00-001", typed
        assert row.tier == "deterministic"


def test_fuzzy_only_fires_after_the_deterministic_tier_misses(spine):
    row = spine.resolve("Jamar Chase", "WR")
    assert row.player_id == "00-002"
    assert row.tier == "fuzzy"
    assert row.score >= 92


def test_fuzzy_below_the_threshold_goes_to_the_confirm_card(spine):
    """92 is a real gate, not decoration. 'Jamar Chse' scores 90.9."""
    assert spine.resolve("Jamar Chse", "WR") is None


def test_a_bare_surname_does_not_resolve(spine):
    """The reason §11.4 names token_sort_ratio: rapidfuzz defaults to WRatio,
    which scores substrings on their own scale, and 'Chase' clears 92 against
    'Ja'Marr Chase' under it. A surname resolving to a first-round receiver is
    the wrong-join class the threshold exists to prevent."""
    assert spine.resolve("Chase", "WR") is None
    assert spine.resolve("Harrison", "WR") is None


def test_unresolvable_returns_none_for_the_confirm_card(spine):
    assert spine.resolve("Somebody Nobody", "WR") is None


# -------------------------------------------------------- ambiguity
def test_same_name_different_position_is_not_a_collision(spine):
    """Position is part of the key precisely so this resolves cleanly."""
    assert spine.resolve("Michael Pittman", "WR").player_id == "00-003"
    assert spine.resolve("Michael Pittman", "RB").player_id == "00-004"


def test_same_name_same_position_needs_a_team():
    rows = pd.DataFrame([
        {"player_id": "a", "name": "Josh Allen", "position": "QB", "team": "BUF"},
        {"player_id": "b", "name": "Josh Allen", "position": "QB", "team": "JAX"},
    ])
    spine = Spine(rows, snapshot_id="s")
    assert "josh allen|QB" in spine.ambiguous_keys
    assert spine.resolve("Josh Allen", "QB", team="JAX").player_id == "b"


def test_ambiguous_names_are_surfaced_not_silently_resolved():
    """A caller that ignores these will draft the wrong person."""
    rows = pd.DataFrame([
        {"player_id": "a", "name": "Josh Allen", "position": "QB", "team": "BUF"},
        {"player_id": "b", "name": "Josh Allen", "position": "QB", "team": "JAX"},
    ])
    spine = Spine(rows, snapshot_id="s")
    assert spine.resolve("Josh Allen", "QB") is None, (
        "without a team this is a coin flip and must go to the confirm card"
    )


def test_duplicate_rows_do_not_crash_construction():
    """A hard failure here would take the whole draft down over a data quirk."""
    rows = pd.concat([_players(), _players()], ignore_index=True)
    assert len(Spine(rows, snapshot_id="s")) == 10


def test_name_without_position_requires_global_uniqueness(spine):
    assert spine.resolve("Ja'Marr Chase").player_id == "00-002"
    assert spine.resolve("Michael Pittman") is None


# -------------------------------------------------------------- coverage
def test_rookies_are_declared_never_silently_zeroed(spine):
    """A rookie has no prior-season row by construction. Treating that as
    'unknown' routes him to replacement level; treating it as 0.0 makes him
    unstartable. The coverage tier sends him to the rookie calibration
    bucket instead."""
    row = spine.resolve("Rookie Guy", "RB")
    assert row.coverage == "rookie_declared"


def test_players_without_a_projection_are_flagged_not_dropped():
    rows = _players().assign(has_projection=[True, True, True, True, False])
    spine = Spine(rows, snapshot_id="s")
    assert spine.resolve("Rookie Guy", "RB").coverage == "no_projection"


def test_confidence_distinguishes_fuzzy_from_the_rest(spine):
    assert spine.resolve("Ja'Marr Chase", "WR").is_confident
    assert not spine.resolve("Jamar Chase", "WR").is_confident


# ------------------------------------------------------------- overrides
def test_overrides_normalize_their_keys(tmp_path):
    path = tmp_path / "ov.yaml"
    path.write_text(yaml.safe_dump(
        {"overrides": {"Marvin Harrison Jr.|wr": "00-001"}}))
    assert load_overrides(path) == {"marvin harrison|WR": "00-001"}


def test_override_without_a_position_is_rejected_by_name(tmp_path):
    path = tmp_path / "ov.yaml"
    path.write_text(yaml.safe_dump({"overrides": {"Marvin Harrison": "00-001"}}))
    with pytest.raises(ConfigError, match="Name\\|POSITION"):
        load_overrides(path)


def test_missing_override_file_is_empty_not_an_error(tmp_path):
    assert load_overrides(tmp_path / "absent.yaml") == {}


def test_shipped_overrides_file_parses():
    assert isinstance(load_overrides(), dict)


def test_override_pointing_at_an_unknown_id_falls_through():
    """A typo'd id must not shadow a resolution the cascade could have made."""
    spine = Spine(_players(), snapshot_id="s",
                  overrides={"jamarr chase|WR": "00-999"})
    row = spine.resolve("Ja'Marr Chase", "WR")
    assert row.player_id == "00-002"
    assert row.tier == "deterministic"


# ---------------------------------------------------------------- build
def test_build_spine_merges_yahoo_ids():
    yahoo = pd.DataFrame({"player_id": ["00-003"], "yahoo_id": ["40003"]})
    spine = build_spine(_players().drop(columns=["yahoo_id"]),
                        snapshot_id="s", yahoo_player_list=yahoo)
    assert spine.resolve("x", "WR", yahoo_id="40003").player_id == "00-003"


def test_build_spine_works_without_yahoo():
    """The whole point of the lower tiers: the draft is safe without Yahoo."""
    spine = build_spine(_players().drop(columns=["yahoo_id"]), snapshot_id="s")
    assert spine.resolve("Ja'Marr Chase", "WR").tier == "deterministic"


def test_missing_columns_are_named():
    with pytest.raises(ConfigError, match="position"):
        Spine(pd.DataFrame({"player_id": ["a"], "name": ["x"]}), snapshot_id="s")
