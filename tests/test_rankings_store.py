"""The board store: containment, atomicity, and conflicts that are detected.

These are the three properties that decide whether the user's hand-built
research survives. Everything else in the rankings feature can be rebuilt from
the bundle; a board cannot.
"""
from __future__ import annotations

import json

import pytest

from src.app.rankings.schema import (
    SCOPES,
    Board,
    RankingsError,
    Scope,
    Tier,
    empty_scopes,
    parse,
    slugify,
    to_json,
)
from src.app.rankings.store import RankingsStore, RevConflict


@pytest.fixture()
def store(tmp_path):
    return RankingsStore(tmp_path / "rankings")


def _scopes(**overrides) -> dict[str, Scope]:
    scopes = empty_scopes()
    for name, tiers in overrides.items():
        scopes[name] = Scope(tiers=tuple(tiers))
    return scopes


def _tier(tier_id="t-1", players=("a", "b"), color="t1") -> Tier:
    return Tier(id=tier_id, label="Tier 1", color=color,
                player_ids=tuple(players))


# --------------------------------------------------------- round tripping
def test_a_saved_board_round_trips(store):
    board = store.create("My 2026 Board", _scopes(overall=[_tier()]),
                         {"method": "adp"})
    again = store.load(board.board_id)
    assert again.board_id == "my-2026-board"
    assert again.scopes["overall"].tiers[0].player_ids == ("a", "b")
    assert again.rev == 1


def test_every_scope_is_present_on_disk(store):
    board = store.create("B", _scopes(overall=[_tier()]), {})
    raw = json.loads(store.path_for(board.board_id).read_text())
    assert set(raw["scopes"]) == set(SCOPES)


# ------------------------------------------------------------ containment
@pytest.mark.parametrize("hostile", [
    "../../etc/passwd", "a/b", ".hidden", "..", "", "A" * 65,
    "back\\slash", "with space", "Upper", "-leading", "nul\x00byte",
])
def test_a_board_id_cannot_escape_the_rankings_directory(store, hostile):
    with pytest.raises(RankingsError):
        store.path_for(hostile)


def test_a_legal_id_resolves_inside_the_root(store):
    assert store.path_for("my-board").parent == store.root.resolve()


# --------------------------------------------------------------- slugify
@pytest.mark.parametrize(("name", "slug"), [
    ("My 2026 Board", "my-2026-board"),
    ("  spaces  ", "spaces"),
    ("Ünïcodé Bôard", "unicode-board"),
    ("!!!weird!!!", "weird"),
    ("a" * 100, "a" * 64),
])
def test_slugify_cases(name, slug):
    assert slugify(name) == slug


def test_truncation_never_leaves_a_trailing_hyphen():
    """Truncate first, THEN strip: a cut landing mid-separator would otherwise
    leave `…board-`."""
    name = "x" * 63 + " tail"
    assert not slugify(name).endswith("-")


@pytest.mark.parametrize("name", ["", "   ", "!!!", "\x00"])
def test_a_name_with_no_usable_slug_is_refused(name):
    with pytest.raises(RankingsError):
        slugify(name)


def test_creating_a_colliding_slug_is_refused(store):
    store.create("My Board", empty_scopes(), {})
    with pytest.raises(RankingsError, match="already exists"):
        store.create("my board!", empty_scopes(), {})


# ------------------------------------------------------------- atomicity
def test_save_is_atomic_when_the_replace_fails(store, monkeypatch):
    board = store.create("B", _scopes(overall=[_tier()]), {})
    before = store.path_for(board.board_id).read_bytes()

    monkeypatch.setattr("src.app.rankings.store.os.replace",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    edited = Board(**{**board.__dict__,
                      "scopes": _scopes(overall=[_tier(players=("z",))])})
    with pytest.raises(OSError):
        store.save(edited, expected_rev=board.rev)

    assert store.path_for(board.board_id).read_bytes() == before


def test_corrupt_json_does_not_destroy_the_file(store):
    board = store.create("B", empty_scopes(), {})
    path = store.path_for(board.board_id)
    path.write_text("{ this is not json")

    with pytest.raises(json.JSONDecodeError):
        store.load(board.board_id)
    assert path.read_text() == "{ this is not json"


def test_an_unreadable_board_does_not_hide_the_others(store):
    store.create("Good", empty_scopes(), {})
    (store.root / "broken.json").write_text("{ nope")
    assert [b["board_id"] for b in store.list()] == ["good"]


# ------------------------------------------------------------- conflicts
def test_a_stale_rev_is_a_conflict_not_a_silent_overwrite(store):
    board = store.create("B", _scopes(overall=[_tier()]), {})
    first = store.save(board, expected_rev=1)
    assert first.rev == 2

    with pytest.raises(RevConflict) as caught:
        store.save(board, expected_rev=1)          # a second tab, stale
    assert caught.value.current.rev == 2


def test_the_conflict_carries_the_current_board(store):
    """The client cannot resolve a conflict it cannot see."""
    board = store.create("B", _scopes(overall=[_tier(players=("a",))]), {})
    store.save(Board(**{**board.__dict__,
                        "scopes": _scopes(overall=[_tier(players=("winner",))])}),
               expected_rev=1)
    with pytest.raises(RevConflict) as caught:
        store.save(board, expected_rev=1)
    assert caught.value.current.scopes["overall"].tiers[0].player_ids == \
        ("winner",)


def test_each_save_advances_the_rev(store):
    board = store.create("B", empty_scopes(), {})
    for expected in (1, 2, 3):
        board = store.save(board, expected_rev=expected)
    assert board.rev == 4


# ----------------------------------------------------------------- parse
def test_scopes_are_independent(store):
    """Reordering `overall` must not touch `RB` — the whole premise of keeping
    seven separate lists."""
    board = store.create(
        "B", _scopes(overall=[_tier(players=("a", "b"))],
                     RB=[_tier("t-9", players=("r1", "r2"))]), {})
    reordered = Board(**{**board.__dict__, "scopes": {
        **board.scopes,
        "overall": Scope(tiers=(_tier(players=("b", "a")),)),
    }})
    saved = store.save(reordered, expected_rev=1)
    assert saved.scopes["RB"].tiers[0].player_ids == ("r1", "r2")


def test_a_duplicate_player_in_one_scope_is_rejected():
    with pytest.raises(RankingsError, match="more than one tier"):
        parse({"board_id": "b", "name": "B", "scopes": {"overall": {"tiers": [
            {"id": "t-1", "player_ids": ["dup"]},
            {"id": "t-2", "player_ids": ["dup"]}]}}})


def test_the_same_player_may_sit_in_two_different_scopes():
    """He is one player in `overall` and one in `RB`; that is the design."""
    board = parse({"board_id": "b", "name": "B", "scopes": {
        "overall": {"tiers": [{"id": "t-1", "player_ids": ["x"]}]},
        "RB": {"tiers": [{"id": "t-1", "player_ids": ["x"]}]}}})
    assert board.scopes["overall"].tiers[0].player_ids == ("x",)


def test_a_duplicate_tier_id_in_one_scope_is_rejected():
    with pytest.raises(RankingsError, match="duplicate tier id"):
        parse({"board_id": "b", "name": "B", "scopes": {"overall": {"tiers": [
            {"id": "t-1", "player_ids": []},
            {"id": "t-1", "player_ids": []}]}}})


def test_a_missing_scope_is_filled_not_rejected():
    """Hand-editing a board is supported; forgetting six keys is not an error."""
    board = parse({"board_id": "b", "name": "B",
                   "scopes": {"overall": {"tiers": []}}})
    assert set(board.scopes) == set(SCOPES)
    assert board.scopes["DST"].tiers == ()


def test_an_unknown_scope_key_is_rejected():
    """A typo must not become a silently-ignored section of someone's board."""
    with pytest.raises(RankingsError, match="unknown scope"):
        parse({"board_id": "b", "name": "B", "scopes": {"FLEX": {"tiers": []}}})


def test_an_unknown_colour_is_rejected():
    with pytest.raises(RankingsError, match="unknown colour"):
        parse({"board_id": "b", "name": "B", "scopes": {"overall": {"tiers": [
            {"id": "t-1", "color": "chartreuse", "player_ids": []}]}}})


# ------------------------------------------------------------- migration
def test_a_newer_schema_version_is_refused_rather_than_guessed():
    with pytest.raises(RankingsError, match="schema_version"):
        parse({"schema_version": 99, "board_id": "b", "name": "B"})


def test_a_missing_schema_version_is_treated_as_v1():
    assert parse({"board_id": "b", "name": "B"}).schema_version == 1


# ---------------------------------------------------------------- bounds
def test_too_many_tiers_is_rejected():
    tiers = [{"id": f"t-{i}", "player_ids": []} for i in range(201)]
    with pytest.raises(RankingsError, match="exceeds"):
        parse({"board_id": "b", "name": "B",
               "scopes": {"overall": {"tiers": tiers}}})


def test_an_over_long_player_id_is_rejected():
    with pytest.raises(RankingsError, match="too long"):
        parse({"board_id": "b", "name": "B", "scopes": {"overall": {"tiers": [
            {"id": "t-1", "player_ids": ["x" * 65]}]}}})


# ------------------------------------------------------------------ list
def test_list_reports_player_counts_per_scope(store):
    store.create("B", _scopes(overall=[_tier(players=("a", "b", "c"))],
                              RB=[_tier("t-2", players=("r",))]), {})
    entry = store.list()[0]
    assert entry["counts"]["overall"] == 3
    assert entry["counts"]["RB"] == 1
    assert entry["counts"]["WR"] == 0


def test_listing_an_absent_directory_is_empty_not_an_error(store):
    assert store.list() == []


def test_delete_removes_the_file(store):
    board = store.create("B", empty_scopes(), {})
    store.delete(board.board_id)
    assert not store.path_for(board.board_id).exists()
    with pytest.raises(FileNotFoundError):
        store.load(board.board_id)


def test_to_json_round_trips_through_parse(store):
    board = store.create("B", _scopes(overall=[_tier()]), {"method": "adp"})
    assert parse(to_json(board)) == board
