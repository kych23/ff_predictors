"""Draft service contracts: session lifecycle, snake-aware mine-defaulting,
pick validation, undo-by-replay, and recommendation shape."""

import pytest

# §9.4: this module is pinned to the v1 config shape. api/, web/ and the
# weekly start/sit surface are FROZEN for the build window, so v1 stays
# alive underneath them and these tests are deselected from the default
# run rather than migrated. Thaw is a §22.2 follow-up.
pytestmark = pytest.mark.v1_frozen
import pytest

from api.draft_service import DraftNotFound, DraftService, InvalidPick
from src.config import load_config
from tests.test_api_replay import make_board


@pytest.fixture()
def svc(db_session):
    cfg = load_config()
    board = make_board()
    return DraftService(db=db_session, cfg=cfg, board_for=lambda season: board)


def test_create_and_state(svc):
    cfg = load_config()
    s = svc.create_session(season=2026, draft_position=1)
    st = svc.state(s.session_id)
    assert st["session_id"] == s.session_id
    assert st["teams"] == cfg.teams
    assert st["rounds"] == cfg.roster.rounds
    assert st["current_overall_pick"] == 1
    assert st["is_my_turn"] is True          # slot 1 owns pick 1
    assert st["picks"] == []
    assert st["my_roster"] == []


def test_create_rejects_bad_position(svc):
    cfg = load_config()
    with pytest.raises(InvalidPick):
        svc.create_session(season=2026, draft_position=cfg.teams + 1)


def test_state_unknown_session_raises(svc):
    with pytest.raises(DraftNotFound):
        svc.state("nope")


def test_pick_defaults_mine_from_snake(svc):
    s = svc.create_session(season=2026, draft_position=1)
    st = svc.record_pick(s.session_id, player_id="P0031")  # my turn -> mine
    assert st["picks"][0]["mine"] is True
    assert st["my_roster"][0]["player_id"] == "P0031"
    st = svc.record_pick(s.session_id, player_id="P0032")  # opponent turn
    assert st["picks"][1]["mine"] is False
    assert len(st["my_roster"]) == 1


def test_pick_rejects_unknown_and_duplicate(svc):
    s = svc.create_session(season=2026, draft_position=1)
    with pytest.raises(InvalidPick):
        svc.record_pick(s.session_id, player_id="GHOST")
    svc.record_pick(s.session_id, player_id="P0031")
    with pytest.raises(InvalidPick):
        svc.record_pick(s.session_id, player_id="P0031")


def test_skip_advances_without_player(svc):
    s = svc.create_session(season=2026, draft_position=2)
    st = svc.record_pick(s.session_id, skip=True)
    assert st["current_overall_pick"] == 2
    assert st["picks"][0]["skipped"] is True
    assert st["picks"][0]["player_id"] is None


def test_undo_pops_last_command(svc):
    s = svc.create_session(season=2026, draft_position=1)
    svc.record_pick(s.session_id, player_id="P0031")
    st = svc.record_pick(s.session_id, player_id="P0032")
    assert st["current_overall_pick"] == 3
    st = svc.undo(s.session_id)
    assert st["current_overall_pick"] == 2
    assert [p["player_id"] for p in st["picks"]] == ["P0031"]
    # undo on empty history is a no-op
    svc.undo(s.session_id)
    st = svc.undo(s.session_id)
    assert st["picks"] == []


def test_recommendations_shape_and_availability(svc):
    s = svc.create_session(season=2026, draft_position=1)
    svc.record_pick(s.session_id, player_id="P0031")
    recs = svc.recommendations(s.session_id, top_n=5)
    assert 0 < len(recs) <= 5
    drafted_ids = {"P0031"}
    for r in recs:
        assert r["player_id"] not in drafted_ids
        assert set(r) >= {"player_id", "name", "position", "vona_score", "value",
                          "p10", "p50", "p90", "draft_round", "target_quantile",
                          "forced_completion"}
    scores = [r["vona_score"] for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_bot_pick_records_opponent_by_adp(svc):
    s = svc.create_session(season=2026, draft_position=2)  # pick 1 is an opponent
    st = svc.bot_pick(s.session_id)
    assert len(st["picks"]) == 1
    assert st["picks"][0]["mine"] is False
    assert st["picks"][0]["player_id"] is not None
    assert st["my_roster"] == []
    assert st["current_overall_pick"] == 2


def test_bot_pick_never_duplicates(svc):
    s = svc.create_session(season=2026, draft_position=1)
    svc.record_pick(s.session_id, player_id="P0031")  # my pick
    st = svc.bot_pick(s.session_id)                     # bot takes best available by adp
    ids = [p["player_id"] for p in st["picks"]]
    assert len(ids) == len(set(ids))
    assert "P0031" not in ids[1:]
