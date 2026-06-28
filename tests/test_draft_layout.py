"""Snake pick layout (DrafterSpec.md §4.8.1)."""
from src.config import load_config
from src.recommender.roster_state import RosterState, my_pick_sequence, overall_pick


def test_snake_overall_pick_example():
    # position 6 of 12 -> picks 6, 19, 30, 43, ... (§4.8.1)
    seq = my_pick_sequence(6, 12, 4)
    assert seq == [6, 19, 30, 43]


def test_position_1_and_last():
    assert my_pick_sequence(1, 12, 3) == [1, 24, 25]
    assert my_pick_sequence(12, 12, 3) == [12, 13, 36]


def test_odd_even_rounds():
    assert overall_pick(1, 3, 12) == 3       # odd: straight
    assert overall_pick(2, 3, 12) == 22      # even: snake (12-3+1)=10 -> 12+10
    assert overall_pick(3, 3, 12) == 27


def test_next_pick_progression():
    cfg = load_config()
    state = RosterState(cfg=cfg, draft_position=6)
    # before any picks, next pick after pick 6 is 19
    assert state.next_my_pick(after_overall=6) == 19
    assert state.next_my_pick(after_overall=19) == 30


def test_position_bounds():
    cfg = load_config()
    # draft_position must be within 1..teams (validated at the CLI)
    seq = my_pick_sequence(cfg.teams, cfg.teams, cfg.roster.rounds)
    assert len(seq) == cfg.roster.rounds
    assert all(p >= 1 for p in seq)
