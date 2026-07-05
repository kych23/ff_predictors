"""Black-box extensions for the ADP survival distribution.

Contract (from test_survival.py docstrings): probability the player is still
on the board at a given future overall pick; log-normal around ADP.
"""
import pytest

from src.recommender.survival import survival_probability


@pytest.mark.parametrize("adp", [1.0, 10.0, 50.0, 150.0])
@pytest.mark.parametrize("sd", [1.0, 5.0, 20.0, 40.0])
@pytest.mark.parametrize("pick", [1, 5, 30, 100, 250])
def test_probability_bounds_grid(adp, sd, pick):
    s = survival_probability(adp, sd, pick)
    assert 0.0 <= s <= 1.0


@pytest.mark.parametrize("adp,sd", [(20.0, 5.0), (60.0, 12.0), (120.0, 30.0)])
def test_monotonic_decreasing_in_pick_fine_grid(adp, sd):
    picks = [1, 10, 25, 50, 75, 100, 150, 200]
    probs = [survival_probability(adp, sd, p) for p in picks]
    for earlier, later in zip(probs, probs[1:]):
        assert later <= earlier


def test_monotonic_increasing_in_adp():
    # at the same future pick, a later-ADP player is more likely to survive
    s_early = survival_probability(20.0, 10.0, 40)
    s_late = survival_probability(60.0, 10.0, 40)
    assert s_late > s_early


def test_first_pick_nearly_certain_survival():
    # nobody has picked before overall #1
    assert survival_probability(50.0, 10.0, 1) > 0.95


def test_way_past_adp_low_survival():
    assert survival_probability(10.0, 3.0, 120) < 0.05


def test_extreme_inputs_still_bounded():
    for s in (
        survival_probability(1.0, 0.5, 300),
        survival_probability(300.0, 100.0, 1),
        survival_probability(300.0, 100.0, 300),
    ):
        assert 0.0 <= s <= 1.0
