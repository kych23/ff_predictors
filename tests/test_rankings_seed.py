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


def _board_with_tiers(tiers: list[tuple[str, str, list[str]]]) -> Board:
    scopes = empty_scopes()
    scopes["overall"] = Scope(tiers=tuple(
        Tier(id=tid, label=label, color=TIER_COLORS[i % len(TIER_COLORS)],
             player_ids=tuple(ids))
        for i, (tid, label, ids) in enumerate(tiers)))
    now = utc_now()
    return Board(board_id="b", name="B", created_at=now, updated_at=now,
                 rev=1, seeded_from={}, scopes=scopes)


def test_from_overall_carries_the_overall_tiers_across(rankings_data):
    """A position list seeded from overall must SHOW the overall grouping, not
    a re-chunk of its flattened order."""
    board_frame = rankings_data.board
    rbs = [str(p) for p in
           board_frame.loc[board_frame["position"] == "RB", "player_id"]]
    wrs = [str(p) for p in
           board_frame.loc[board_frame["position"] == "WR", "player_id"]]

    board = _board_with_tiers([
        ("t-1", "Tier 1", [rbs[0], wrs[0], rbs[1]]),
        ("t-2", "Tier 2", [wrs[1], wrs[2]]),          # no RB here at all
        ("t-3", "Tier 3", [rbs[2], wrs[3]]),
    ])
    scope = seed_scope(rankings_data, "RB", "from_overall", board=board)

    assert [t.label for t in scope.tiers][:2] == ["Tier 1", "Tier 3"], (
        "an overall tier with no RB must contribute no RB tier, and the "
        "numbering must not be resequenced — the gap is the information")
    assert scope.tiers[0].player_ids == (rbs[0], rbs[1])
    assert scope.tiers[1].player_ids == (rbs[2],)


def test_from_overall_keeps_each_tiers_colour(rankings_data):
    """Same group, same colour in both scopes, or the correspondence is
    invisible at a glance — which is the whole reason to align them."""
    frame = rankings_data.board
    rbs = [str(p) for p in frame.loc[frame["position"] == "RB", "player_id"]]
    board = _board_with_tiers([
        ("t-1", "Tier 1", [rbs[0]]),
        ("t-2", "Tier 2", [rbs[1]]),
    ])
    scope = seed_scope(rankings_data, "RB", "from_overall", board=board)
    overall = {t.label: t.color for t in board.scopes["overall"].tiers}
    for tier in scope.tiers:
        if tier.label in overall:
            assert tier.color == overall[tier.label]


def test_from_overall_does_not_chunk_by_a_fixed_size(rankings_data):
    """One big overall tier stays one tier, whatever `tier_size` says."""
    frame = rankings_data.board
    rbs = [str(p) for p in frame.loc[frame["position"] == "RB", "player_id"]]
    board = _board_with_tiers([("t-1", "Tier 1", rbs)])
    scope = seed_scope(rankings_data, "RB", "from_overall", board=board,
                       tier_size=2)
    assert len(scope.tiers) == 1
    assert len(scope.tiers[0].player_ids) == len(rbs)


def test_players_missing_from_overall_are_labelled_not_dropped(rankings_data):
    frame = rankings_data.board
    rbs = [str(p) for p in frame.loc[frame["position"] == "RB", "player_id"]]
    board = _board_with_tiers([("t-1", "Tier 1", rbs[:2])])
    scope = seed_scope(rankings_data, "RB", "from_overall", board=board)

    assert scope.tiers[-1].label == "Not in overall"
    flat = _flat(scope)
    assert sorted(flat) == sorted(rbs), "a back was dropped"


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
    """These are MAXIMUMS now, not fixed sizes — tiers are cut where the
    metric cliffs and the cap only splits a genuinely flat run."""
    assert tier_size_for("overall", None) == 24
    assert tier_size_for("WR", None) == 12
    assert tier_size_for("overall", 4) == 4


def test_tier_size_caps_the_output(rankings_data):
    scope = seed_scope(rankings_data, "overall", "adp", tier_size=5)
    assert all(len(t.player_ids) <= 5 for t in scope.tiers)


# --------------------------------------------------- natural tier breaks
def test_a_cliff_ends_a_tier():
    """Three tight values, a chasm, three more: two tiers, not one."""
    from src.app.rankings.seed import natural_breaks

    values = [10.0, 9.9, 9.8, 2.0, 1.9, 1.8]
    assert natural_breaks(values, max_size=99) == [0, 3]


def test_an_evenly_spaced_run_has_no_cliff_to_find():
    """No gap stands out, so nothing but the cap may split it — the players
    really are equally spaced."""
    from src.app.rankings.seed import natural_breaks

    values = [float(10 - i) for i in range(10)]
    assert natural_breaks(values, max_size=99) == [0]


def test_identical_values_are_one_tier():
    """All 21 kickers share one projection. There is no cliff in a flat line."""
    from src.app.rankings.seed import natural_breaks

    assert natural_breaks([5.0] * 12, max_size=99) == [0]


def test_the_cap_splits_a_flat_run():
    from src.app.rankings.seed import natural_breaks

    assert natural_breaks([5.0] * 10, max_size=4) == [0, 4, 8]


def test_the_gap_test_is_local_not_global():
    """The reason this is local: ADP gaps GROW with depth. A global threshold
    finds no cliff among the elite, where gaps are tenths, and slices only the
    tail — measured on the live board it produced a 29-player 'Tier 1'.

    Here the early gaps are ~0.1 with one 0.5 cliff, and the late gaps are all
    ~5. A global rule sees only the late region; a local one finds the cliff.
    """
    from src.app.rankings.seed import natural_breaks

    early = [0.0, 0.1, 0.2, 0.3, 0.8, 0.9, 1.0]      # a 0.5 jump at index 4
    late = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0]      # uniformly 5 apart
    starts = natural_breaks(early + late, max_size=99)
    assert 4 in starts, "the small early cliff was missed"


def test_a_tier_of_one_is_suppressed():
    from src.app.rankings.seed import natural_breaks

    # Cliffs after every single element; MIN_TIER stops one-player tiers.
    values = [100.0, 50.0, 25.0, 12.0, 6.0, 3.0]
    starts = natural_breaks(values, max_size=99)
    bounds = [*starts, len(values)]
    assert all(hi - lo >= 2 for lo, hi in zip(bounds, bounds[1:], strict=False))


def test_unpriced_players_get_their_own_tier(rankings_data):
    """A null ADP is "the market has no opinion", not a rank — it cannot
    participate in a distance test, so it trails in a labelled tier."""
    scope = seed_scope(rankings_data, "overall", "adp")
    assert scope.tiers[-1].label == "Unpriced"


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
