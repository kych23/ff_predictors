"""The confidence score has to mean something, or it is decoration.

It claims one thing: the chance this recommendation beats the next candidate.
The anchor is `p_best` — a real paired bootstrap over the CRN difference — and
every adjustment traces to a measurement rather than a feeling.

The design decision that matters is the DIRECTION of doubt. Uncertainty pulls
toward 50, never toward 0. A pick the engine cannot support is a coin flip, not
a mistake; driving it to 0 would assert the opposite recommendation, which the
evidence supports even less.
"""
from __future__ import annotations

import pytest

from src.engine.decision.confidence import NEUTRAL, score


def _clean(**kw):
    base = dict(tier=0, p_best=0.94, adp=30.0, current_pick=28,
                indifference_size=1, has_projection=True)
    base.update(kw)
    return score(**base)


# ------------------------------------------------------------ the anchor
def test_a_decisive_simulation_scores_near_p_best():
    assert _clean().score == 94


@pytest.mark.parametrize("p", [0.55, 0.70, 0.85, 0.99])
def test_the_score_tracks_p_best_when_nothing_is_wrong(p):
    assert _clean(p_best=p).score == round(100 * p)


def test_a_toss_up_simulation_reads_as_a_toss_up():
    assert _clean(p_best=0.50).score == 50


# --------------------------------------- doubt shrinks toward 50, not 0
@pytest.mark.parametrize("kw", [
    {"stopped_because": "deadline"},
    {"tier": 2, "p_best": None},
    {"tier": 3, "p_best": None},
    {"has_projection": False},
    {"indifference_size": 6},
    {"adp": 90.0},                       # a 62-pick reach
    {"stale_flags": ("no_projection",)},
])
def test_every_penalty_moves_toward_neutral_never_past_it(kw):
    """The property that keeps the number honest. A penalty that pushed below
    50 would be asserting the OTHER candidate is better, which none of these
    signals supports."""
    penalised = _clean(**kw).score
    assert penalised < _clean().score
    assert penalised >= NEUTRAL - 1


def test_penalties_compound():
    one = _clean(stopped_because="deadline").score
    two = _clean(stopped_because="deadline", indifference_size=4).score
    assert two < one


# ----------------------------------------------- the backtested reach rule
def test_taking_a_player_at_or_after_his_adp_is_never_penalised():
    """The engine's ordering edge is measured and real (+0.185 Spearman, all
    four backtested seasons). Only REACHING is unsupported."""
    assert _clean(adp=28.0, current_pick=28).score == 94
    assert _clean(adp=20.0, current_pick=28).score == 94


def test_a_large_reach_lands_at_a_coin_flip():
    """Backtested: the engine's most aggressive disagreements won 50 of 96
    head-to-head pairs, p=0.76. However sure the simulation is, that is 50."""
    assert _clean(adp=60.0, current_pick=28).score == pytest.approx(50, abs=1)


def test_reach_confidence_decays_monotonically():
    scores = [_clean(adp=28.0 + r, current_pick=28).score
              for r in (0, 5, 10, 15, 20, 30)]
    assert scores == sorted(scores, reverse=True)


def test_a_small_reach_costs_little():
    """Inside a round is the ordering edge being expressed, not a gamble."""
    assert _clean(adp=34.0, current_pick=28).score >= 90


# ------------------------------------------------------- missing evidence
def test_a_fallback_tier_cannot_look_confident():
    """Tier 2 is a heuristic and tier 3 is a static ADP list. Neither has a
    simulation to be confident about."""
    for tier in (2, 3):
        assert _clean(tier=tier, p_best=None).score <= 56


def test_a_truncated_run_is_heavily_discounted():
    """`p_best` on a deadline run is computed from the two initial draws, so
    the anchor itself is noise."""
    assert _clean(stopped_because="deadline").score <= 60


def test_a_kicker_is_not_a_confident_pick():
    """Every kicker shares one fitted value, so a dollar gap between two of
    them is simulation noise."""
    assert _clean(has_projection=False).score <= 65


def test_nan_p_best_does_not_produce_nan():
    assert score(tier=2, p_best=float("nan")).score == pytest.approx(52, abs=3)


# ------------------------------------------------------------- reporting
def test_every_penalty_states_its_reason():
    """A bare number is not auditable. If the score drops, it must say why."""
    result = _clean(stopped_because="deadline", indifference_size=3,
                    adp=45.0)
    assert len(result.drivers) >= 3
    assert any("clock" in d for d in result.drivers)
    assert any("indistinguishable" in d for d in result.drivers)
    assert any("market" in d for d in result.drivers)


def test_a_clean_recommendation_has_nothing_to_explain():
    assert _clean().drivers == []


@pytest.mark.parametrize(("value", "label"), [
    (95, "strong"), (78, "strong"), (65, "moderate"),
    (55, "slight"), (50, "coin flip"), (30, "coin flip"),
])
def test_labels_match_the_score(value, label):
    from src.engine.decision.confidence import Confidence

    assert Confidence(score=value).label == label


def test_the_score_is_always_in_range():
    for p in (0.0, 0.5, 1.0):
        for tier in (0, 2, 3):
            s = score(tier=tier, p_best=p, adp=200.0, current_pick=1,
                      indifference_size=8, has_projection=False)
            assert 0 <= s.score <= 100


def test_the_payload_carries_the_score_and_its_reasons():
    import inspect

    from src.app.web import schemas

    source = inspect.getsource(schemas.recommendation_payload)
    assert '"confidence"' in source
    assert "drivers" in source
