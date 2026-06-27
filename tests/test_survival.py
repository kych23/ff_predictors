"""ADP survival distribution (DrafterSpec.md §4.8.1)."""
from src.recommender.survival import survival_probability


def test_monotonic_decreasing_in_pick_distance():
    # Farther-out picks -> lower survival for the same player.
    adp, sd = 30.0, 8.0
    s10 = survival_probability(adp, sd, 10)
    s30 = survival_probability(adp, sd, 30)
    s50 = survival_probability(adp, sd, 50)
    assert s10 > s30 > s50
    assert 0.0 <= s50 <= s10 <= 1.0


def test_elite_pick_floor():
    # An elite player (ADP ~2) almost never survives to a much later pick.
    s = survival_probability(2.0, 1.0, 20)
    assert s < 0.01


def test_high_survival_before_adp():
    # Well before his ADP, a player is very likely still available.
    s = survival_probability(80.0, 15.0, 30)
    assert s > 0.95


def test_late_flier_fat_right_tail():
    # Log-normal gives late fliers a fatter right tail than a symmetric normal:
    # a player with ADP 150 still has appreciable survival past pick 150.
    s = survival_probability(150.0, 40.0, 160)
    assert s > 0.30


def test_no_next_pick_returns_one():
    assert survival_probability(50.0, 10.0, None) == 1.0
