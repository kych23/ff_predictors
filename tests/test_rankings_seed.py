"""Seeding a board, and the two ways an obvious design gets it wrong.

Both were caught by measurement on the live 260-player bundle, not by reasoning:

* sorted by raw per-game `value`, the top 40 overall is 24 QBs, and Daniel
  Jones (ADP 150.8) is 18th. Per-game points are not comparable across
  positions when one quarterback starts. Subtracting per-position replacement
  moves Spearman-vs-ADP from -0.575 to -0.826.
* the engine's tier artifact is per POSITION, `rank` restarting at 1 for each,
  so it cannot express an overall ordering at all.

These tests assert the RULE against the fixture's own numbers. Asserting the
bundle's measured output ("no QB in the top 40") against the 54-player fixture
would be red on day one — the fixture has different replacement levels.
"""
from __future__ import annotations

import pytest

from src.app.rankings.data import vor
from src.app.rankings.schema import (
    POSITION_SCOPES,
    TIER_COLORS,
    Board,
    RankingsError,
    Scope,
    Tier,
    empty_scopes,
    utc_now,
)
from src.app.rankings.seed import seed_scope, tier_size_for


def _flat(scope: Scope) -> list[str]:
    return [pid for tier in scope.tiers for pid in tier.player_ids]


def _board_with_overall(order: list[str]) -> Board:
    scopes = empty_scopes()
    scopes["overall"] = Scope(tiers=(Tier(id="t-1", label="Tier 1",
                                          color="t1",
                                          player_ids=tuple(order)),))
    now = utc_now()
    return Board(board_id="b", name="B", created_at=now, updated_at=now,
                 rev=1, seeded_from={}, scopes=scopes)


# ------------------------------------------------------------------- adp
def test_adp_orders_ascending(rankings_data):
    scope = seed_scope(rankings_data, "overall", "adp")
    ids = _flat(scope)
    adp = rankings_data.board.set_index("player_id")["adp"]
    ordered = [adp[pid] for pid in ids if adp[pid] == adp[pid]]
    assert ordered == sorted(ordered)


def test_a_null_adp_sorts_last_not_first(rankings_data):
    """Absent is not "pick 0". The fixture carries one null precisely here."""
    scope = seed_scope(rankings_data, "overall", "adp")
    ids = _flat(scope)
    adp = rankings_data.board.set_index("player_id")["adp"]
    null_ids = {pid for pid in ids if adp[pid] != adp[pid]}
    assert null_ids
    assert set(ids[-len(null_ids):]) == null_ids


# ------------------------------------------------------------------- vor
def test_engine_vor_subtracts_the_per_position_replacement(rankings_data):
    frame = rankings_data.board
    computed = vor(rankings_data, frame)
    for position, floor in rankings_data.replacement.items():
        rows = frame["position"] == position
        expected = frame.loc[rows, "value"] - floor
        assert (computed[rows] - expected).abs().max() < 1e-9


def test_engine_vor_orders_by_value_over_replacement(rankings_data):
    scope = seed_scope(rankings_data, "overall", "engine_vor")
    ids = _flat(scope)
    frame = rankings_data.board.set_index("player_id")
    scores = [frame.loc[pid, "value"]
              - rankings_data.replacement[frame.loc[pid, "position"]]
              for pid in ids]
    assert scores == sorted(scores, reverse=True)


def test_vor_and_raw_value_disagree_on_the_same_scope(rankings_data):
    """Compared on ONE scope, or the assertion is unfalsifiable.

    An earlier version of this test compared 54 overall ids against ~14 RB
    ids: those lists cannot be equal whatever `vor()` does, so it passed with
    `replacement` identically zero — the exact bug it claims to guard.
    """
    vor_order = _flat(seed_scope(rankings_data, "RB", "engine_vor"))
    raw_order = _flat(seed_scope(rankings_data, "RB", "engine_value"))
    assert sorted(vor_order) == sorted(raw_order), "different populations"
    # Within one position the two agree by construction (a constant offset),
    # so the real claim is cross-position: VOR must reorder the overall board.
    overall_vor = _flat(seed_scope(rankings_data, "overall", "engine_vor"))
    by_value = rankings_data.board.sort_values(
        ["value", "player_id"], ascending=[False, True])
    assert overall_vor != [str(p) for p in by_value["player_id"]], (
        "VOR produced the raw-value order; the replacement adjustment is inert")


# --------------------------------------------------- per-scope legality
@pytest.mark.parametrize("method", ["engine_value", "engine_tiers",
                                    "from_overall"])
def test_the_overall_scope_refuses_the_methods_it_cannot_express(
        rankings_data, method):
    with pytest.raises(RankingsError, match="cannot seed"):
        seed_scope(rankings_data, "overall", method)


def test_the_refusal_explains_why(rankings_data):
    """A bare 400 teaches nothing; the reason is the whole content."""
    with pytest.raises(RankingsError, match="24 quarterbacks"):
        seed_scope(rankings_data, "overall", "engine_value")
    with pytest.raises(RankingsError, match="per position"):
        seed_scope(rankings_data, "overall", "engine_tiers")


@pytest.mark.parametrize("scope", ["K", "DST"])
@pytest.mark.parametrize("method", ["engine_value", "engine_vor"])
def test_a_degenerate_position_refuses_a_projection_sort(rankings_data, scope,
                                                         method):
    """Every kicker shares one projected value, so the sort would be player_id
    order wearing a projection's name."""
    with pytest.raises(RankingsError, match="shares one projected value"):
        seed_scope(rankings_data, scope, method)


@pytest.mark.parametrize("scope", ["overall", *POSITION_SCOPES])
def test_adp_is_legal_everywhere(rankings_data, scope):
    assert seed_scope(rankings_data, scope, "adp").tiers


def test_an_unknown_method_is_rejected(rankings_data):
    with pytest.raises(RankingsError, match="unknown seed method"):
        seed_scope(rankings_data, "overall", "vibes")


# ---------------------------------------------------------- engine tiers
def test_engine_tiers_follows_the_artifacts_per_position_tiers(rankings_data):
    scope = seed_scope(rankings_data, "WR", "engine_tiers")
    artifact = rankings_data.tiers.set_index("player_id")
    for tier in scope.tiers:
        numbers = {artifact.loc[pid, "tier"] for pid in tier.player_ids}
        assert len(numbers) == 1, "a tier mixed two artifact tiers"


def test_engine_tiers_gives_kickers_a_single_tier(rankings_data):
    """All K are artifact tier 1. One tier is the correct answer, not a bug."""
    assert len(seed_scope(rankings_data, "K", "engine_tiers").tiers) == 1


def test_a_player_missing_from_the_artifact_lands_in_unranked(rankings_data):
    trimmed = rankings_data.tiers[rankings_data.tiers["position"] != "WR"]
    data = type(rankings_data)(**{**rankings_data.__dict__, "tiers": trimmed})
    scope = seed_scope(data, "WR", "engine_tiers")
    assert scope.tiers[-1].label == "Unranked"
    assert len(_flat(scope)) == int((data.board["position"] == "WR").sum())


def test_engine_tiers_without_an_artifact_says_so(rankings_data):
    from src.app.rankings.data import TIER_COLS, _empty

    data = type(rankings_data)(**{**rankings_data.__dict__,
                                  "tiers": _empty(TIER_COLS)})
    with pytest.raises(RankingsError, match="tier artifact"):
        seed_scope(data, "WR", "engine_tiers")


# ----------------------------------------------------------- from_overall
def test_from_overall_filters_the_overall_order_to_one_position(rankings_data):
    wrs = [str(p) for p in
           rankings_data.board.loc[rankings_data.board["position"] == "WR",
                                   "player_id"]]
    board = _board_with_overall(list(reversed(wrs)))
    scope = seed_scope(rankings_data, "WR", "from_overall", board=board)
    assert _flat(scope) == list(reversed(wrs))


def test_from_overall_appends_players_absent_from_overall(rankings_data):
    wrs = [str(p) for p in
           rankings_data.board.loc[rankings_data.board["position"] == "WR",
                                   "player_id"]]
    board = _board_with_overall(wrs[:2])
    scope = seed_scope(rankings_data, "WR", "from_overall", board=board)
    assert _flat(scope)[:2] == wrs[:2]
    assert set(_flat(scope)) == set(wrs), "a player was dropped"


def test_from_overall_on_an_empty_overall_says_seed_overall_first(
        rankings_data):
    board = _board_with_overall([])
    with pytest.raises(RankingsError, match="seed overall"):
        seed_scope(rankings_data, "RB", "from_overall", board=board)


# -------------------------------------------------------------- coverage
@pytest.mark.parametrize("scope", ["overall", "QB", "RB", "WR", "TE", "K",
                                   "DST"])
def test_a_seed_covers_every_eligible_player_exactly_once(rankings_data, scope):
    ids = _flat(seed_scope(rankings_data, scope, "adp"))
    frame = rankings_data.board
    expected = frame if scope == "overall" else frame[frame["position"] == scope]
    assert sorted(ids) == sorted(str(p) for p in expected["player_id"])


def test_seeding_is_deterministic(rankings_data):
    """Ties break on player_id, so a reseed cannot silently reshuffle a board.
    All 26 real defences share one value; without the tiebreak this drifts."""
    first = _flat(seed_scope(rankings_data, "DST", "adp"))
    second = _flat(seed_scope(rankings_data, "DST", "adp"))
    assert first == second


# ------------------------------------------------------------- tier size
def test_tier_size_defaults_differ_between_overall_and_positions():
    assert tier_size_for("overall", None) == 12
    assert tier_size_for("WR", None) == 6
    assert tier_size_for("overall", 4) == 4


def test_tier_size_chunks_the_output(rankings_data):
    scope = seed_scope(rankings_data, "overall", "adp", tier_size=5)
    assert all(len(t.player_ids) <= 5 for t in scope.tiers)


def test_tier_size_is_ignored_by_engine_tiers(rankings_data):
    """The artifact defines the boundaries; a size argument cannot re-cut them
    without destroying the thing the method is for."""
    a = seed_scope(rankings_data, "WR", "engine_tiers", tier_size=2)
    b = seed_scope(rankings_data, "WR", "engine_tiers", tier_size=50)
    assert [t.player_ids for t in a.tiers] == [t.player_ids for t in b.tiers]


def test_seeded_tiers_carry_labels_and_palette_colours(rankings_data):
    scope = seed_scope(rankings_data, "RB", "adp")
    assert scope.tiers[0].label == "Tier 1"
    assert all(t.color in TIER_COLORS for t in scope.tiers)


def test_seeding_an_empty_board_is_empty_not_an_error(rankings_data):
    from src.app.rankings.data import RankingsData

    assert seed_scope(RankingsData.empty(), "overall", "adp").tiers == ()
