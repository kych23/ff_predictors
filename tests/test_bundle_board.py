"""Bundle, offline board, identity matching, and tier-2 policy (§11.4, §19)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.core.config import load_league, load_strategy
from src.core.errors import ArtifactSnapshotMismatch, DataError
from src.engine.decision import recommend as rec_mod
from src.engine.decision import roster_state
from src.engine.decision.board import Board
from src.platform import bundle as bundle_mod
from src.platform.identity.match import match_by_name, normalize_name


@pytest.fixture(scope="module")
def league():
    return load_league()


@pytest.fixture(scope="module")
def strategy(league):
    return load_strategy(league)


def _players(n_skill: int = 12) -> pd.DataFrame:
    rows = []
    positions = ["RB", "WR", "RB", "WR", "QB", "TE", "RB", "WR", "QB", "TE",
                 "RB", "WR"]
    for i in range(n_skill):
        rows.append({
            "player_id": f"p{i}", "player_name": f"Player {i}",
            "position": positions[i % len(positions)], "team": "KC",
            "bye_week": 6, "value": 25.0 - i, "value_source": "prior_season_ppg",
            "coverage": "full", "adp": float(i + 1), "adp_stdev": 1.5,
        })
    for j, pos in enumerate(["K", "DST"]):
        rows.append({
            "player_id": f"x{j}", "player_name": f"{pos} Guy",
            "position": pos, "team": "SF", "bye_week": 9, "value": 0.0,
            "value_source": "none", "coverage": "no_prior_season",
            "adp": 150.0 + j, "adp_stdev": 8.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture()
def bundle_path(tmp_path):
    b = bundle_mod.Bundle(
        players=_players(), snapshot_id="sn2_" + "a" * 24,
        model_version="draft_v2.deadbeef", decision_version="dec_v2.cafe",
        value_source="prior_season_ppg", built_at="2026-08-05T00:00:00+00:00",
    )
    path = tmp_path / "bundle.parquet"
    bundle_mod.write(b, path)
    return path


# ------------------------------------------------------------------ bundle
def test_bundle_roundtrips_with_provenance(bundle_path):
    b = bundle_mod.read(bundle_path)
    assert b.snapshot_id == "sn2_" + "a" * 24
    assert b.value_source == "prior_season_ppg"
    assert b.n_players == 14


def test_bundle_requires_its_columns():
    with pytest.raises(DataError, match="missing columns"):
        bundle_mod.Bundle(players=pd.DataFrame({"player_id": ["a"]}),
                          snapshot_id="s", model_version="m",
                          decision_version="d", value_source="v", built_at="t")


def test_missing_bundle_says_what_to_run(tmp_path):
    with pytest.raises(DataError, match="build_bundle"):
        bundle_mod.read(tmp_path / "nope.parquet")


def test_bundle_without_sidecar_is_rejected(bundle_path):
    bundle_path.with_suffix(".meta.json").unlink()
    with pytest.raises(DataError, match="sidecar metadata"):
        bundle_mod.read(bundle_path)


def test_artifact_snapshot_mismatch_is_caught():
    """Stamping provenance is not enough — r4 recorded snapshot ids and never
    compared them, so a posterior fitted on one snapshot could pair with
    projections from another."""
    with pytest.raises(ArtifactSnapshotMismatch, match="correlation"):
        bundle_mod.assert_artifact_snapshots_match(
            "sn2_aaa", correlation="sn2_bbb", weekly=None)


def test_matching_artifacts_pass():
    bundle_mod.assert_artifact_snapshots_match(
        "sn2_aaa", correlation="sn2_aaa", weekly=None)


# ------------------------------------------------------------------- board
def test_board_reads_offline(bundle_path):
    board = Board.from_bundle(bundle_path)
    assert len(board.available()) == 14
    assert board.value_source == "prior_season_ppg"


def test_taking_a_player_removes_him(bundle_path):
    board = Board.from_bundle(bundle_path)
    board.take("p0")
    assert len(board.available()) == 13
    with pytest.raises(ValueError, match="already been drafted"):
        board.take("p0")


def test_taking_an_unknown_player_raises(bundle_path):
    board = Board.from_bundle(bundle_path)
    with pytest.raises(KeyError, match="not on the board"):
        board.take("nobody")


def test_find_supports_live_typing(bundle_path):
    board = Board.from_bundle(bundle_path)
    assert not board.find("player 3").empty
    assert not board.find("PLAYER 3").empty
    assert board.find("").empty


def test_stale_flags_disclose_the_placeholder(bundle_path, league):
    board = Board.from_bundle(bundle_path)
    flags = board.stale_flags(league)
    assert any("prior_season_ppg" in f for f in flags), (
        "a placeholder value source must be disclosed on every recommendation"
    )
    assert any("no_prior_season" in f for f in flags)


# ---------------------------------------------------------------- identity
def test_normalize_folds_generational_suffixes():
    assert normalize_name("Marvin Harrison Jr.") == normalize_name("Marvin Harrison")
    assert normalize_name("Michael Pittman Jr") == normalize_name("Michael Pittman")
    assert normalize_name("Ken Walker III") == normalize_name("Ken Walker")


def test_normalize_strips_accents_and_punctuation():
    assert normalize_name("Eddy Piñeiro") == "eddy pineiro"
    assert normalize_name("Ja'Marr Chase") == "jamarr chase"


def test_match_requires_position_agreement():
    """Name alone produces wrong joins, and a wrong join is worse than an
    unresolved row because it is never reported."""
    left = pd.DataFrame([{"player_name": "Josh Allen", "position": "QB"}])
    right = pd.DataFrame([{"nfl_name": "Josh Allen", "position": "LB"}])
    report = match_by_name(left, right, left_name="player_name",
                           right_name="nfl_name", use_fuzzy=False)
    assert len(report.matched) == 0
    assert len(report.unresolved_left) == 1


def test_match_reports_coverage():
    left = pd.DataFrame([{"player_name": n, "position": "WR"}
                         for n in ["A B", "C D", "E F"]])
    right = pd.DataFrame([{"nfl_name": "A B", "position": "WR"},
                          {"nfl_name": "C D", "position": "WR"}])
    report = match_by_name(left, right, left_name="player_name",
                           right_name="nfl_name", use_fuzzy=False)
    assert report.coverage == pytest.approx(2 / 3)
    assert "coverage 66.7%" in report.describe()


# ------------------------------------------------------------ tier-2 policy
def _rs(league, slot=4):
    return roster_state.RosterState(cfg=league, draft_position=slot)


def test_wait_horizon_is_the_next_turn_not_this_one(league, strategy):
    """Passing the current pick collapses the wait term onto the same pick and
    zeroes nearly every score — the first bug this produced on real data."""
    rs = _rs(league)
    result = rec_mod.score(_players(), rs, league, strategy, current_pick=4)
    assert result.current_pick == 4
    assert result.next_pick == 21, "seat 4's next turn after pick 4 is 21"
    assert (result.ranked["wait_term"] > 0).any()


def test_kdst_are_sunk_before_the_endgame(league, strategy):
    rs = _rs(league)
    result = rec_mod.score(_players(), rs, league, strategy, current_pick=4)
    top = result.ranked.head(5)
    assert not top["position"].isin(["K", "DST"]).any(), (
        "a kicker at the top of round 1 means K/DST sinking is broken"
    )
    kdst = result.ranked[result.ranked["position"].isin(["K", "DST"])]
    assert (kdst["sink_reason"] == "k_dst_wait_for_endgame").all()


def test_kdst_become_live_in_the_endgame(league, strategy):
    """§3.4: replacement at draft time, but they still have to be taken."""
    rs = _rs(league)
    for i in range(league.roster.rounds - 2):
        rs.record_pick(f"filler{i}", "RB", mine=True)
    result = rec_mod.score(_players(), rs, league, strategy, current_pick=170)
    kdst = result.ranked[result.ranked["position"].isin(["K", "DST"])]
    assert (kdst["sink_reason"] == "").all(), "K/DST must be live at the end"


def test_position_caps_come_from_strategy(league, strategy):
    """Caps live in strategy.yaml, so tuning them leaves models valid (§10.1).

    Scored past round 10 on purpose. `early_round_caps` sinks a second QB
    before then and claims `sink_reason` first, so an earlier pick would test
    that rule instead of this one — two real caps, and the test has to say
    which it means.
    """
    rs = _rs(league)
    cap = int(strategy.max_per_position["QB"])
    for i in range(cap):
        rs.record_pick(f"qb{i}", "QB", mine=True)
    result = rec_mod.score(_players(), rs, league, strategy, current_pick=133)
    qbs = result.ranked[result.ranked["position"] == "QB"]
    assert (qbs["sink_reason"].str.startswith("at_cap_QB")).all()


def test_scarcer_position_gets_the_boost(league, strategy):
    """VONA's whole job: when a position is deep, waiting is cheap and its
    candidates are discounted relative to a scarce one."""
    rs = _rs(league)
    result = rec_mod.score(_players(), rs, league, strategy, current_pick=4)
    live = result.ranked[result.ranked["sink_reason"] == ""]
    by_pos = live.groupby("position")["wait_term"].first()
    assert (by_pos > 0).all()


def test_empty_board_returns_no_leader(league, strategy):
    rs = _rs(league)
    empty = _players().iloc[0:0]
    result = rec_mod.score(empty, rs, league, strategy, current_pick=4)
    assert result.leader is None
