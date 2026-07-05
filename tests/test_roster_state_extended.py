"""Black-box extensions for snake-draft pick math."""
from src.config import load_config
from src.recommender.roster_state import RosterState, my_pick_sequence, overall_pick


def test_round1_pick_is_draft_position():
    for p in (1, 4, 7, 12):
        assert overall_pick(1, p, 12) == p


def test_round2_snake_closed_form():
    # even round: teams reverse -> overall = teams + (teams - position + 1)
    for p in (1, 5, 12):
        assert overall_pick(2, p, 12) == 12 + (12 - p + 1)


def test_round3_closed_form():
    for p in (1, 6, 12):
        assert overall_pick(3, p, 12) == 24 + p


def test_all_positions_partition_each_round():
    """Across 12 positions, each round's overall picks are exactly one block
    of 12 consecutive picks with no gaps or duplicates."""
    teams = 12
    for rnd in (1, 2, 3, 4, 5):
        picks = {overall_pick(rnd, p, teams) for p in range(1, teams + 1)}
        expected = set(range((rnd - 1) * teams + 1, rnd * teams + 1))
        assert picks == expected


def test_sequence_strictly_increasing():
    for p in (1, 3, 8, 12):
        seq = my_pick_sequence(p, 12, 15)
        assert len(seq) == 15
        assert all(a < b for a, b in zip(seq, seq[1:]))


def test_sequence_matches_overall_pick():
    seq = my_pick_sequence(4, 12, 6)
    assert seq == [overall_pick(r, 4, 12) for r in range(1, 7)]


def test_turn_gap_pattern():
    """Snake property: consecutive gaps for position p alternate between
    2*(teams-p)+1 and 2*p-1."""
    teams, p = 12, 3
    seq = my_pick_sequence(p, teams, 8)
    gaps = [b - a for a, b in zip(seq, seq[1:])]
    assert gaps[0] == 2 * (teams - p) + 1
    assert gaps[1] == 2 * p - 1
    assert gaps[2] == 2 * (teams - p) + 1


def test_next_my_pick_skips_own_pick():
    cfg = load_config()
    state = RosterState(cfg=cfg, draft_position=1)
    # position 1 in a snake: picks 1, 24, 25, 48, ...
    assert state.next_my_pick(after_overall=1) == 24
    assert state.next_my_pick(after_overall=24) == 25


def test_last_position_back_to_back_picks():
    teams = load_config().teams
    seq = my_pick_sequence(teams, teams, 4)
    # last position picks at the turn: consecutive overall picks
    assert seq[1] == seq[0] + 1
    assert seq[3] == seq[2] + 1
