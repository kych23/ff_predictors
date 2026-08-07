"""Recommendation -> JSON payload (§17.3).

Two properties matter more than the rest:

* **The leader is a field, never a row position.** `ranked` sorts by raw
  `E_dollars` while the leader accounts for uncertainty, so the recommended
  player can legitimately appear second. A client that renders `candidates[0]`
  as "the pick" would show the wrong name — which is exactly what the terminal
  cockpit did.
* **No NaN escapes.** JSON has no NaN, and the ledger's `canonical_json` runs
  with `allow_nan=False` and *raises* on one, so a leaked NaN is a crashed
  ledger write rather than an ugly number.
"""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from src.app.web import schemas
from src.engine.decision.recommendation import Recommendation


def _payload(rec, board, **kw):
    defaults = dict(snapshot_id="sn2_test", generation=7, reps=512,
                    shortlist=6, budget_seconds=25.0)
    return schemas.recommendation_payload(rec, board.players,
                                          **{**defaults, **kw})


# ------------------------------------------------------------ the headline
def test_the_leader_is_named_even_when_it_does_not_sort_first(
        synthetic_board, fake_recommendation):
    """The observed live case: St. Brown at $44.81 on 2 draws rendered above
    the recommended Gibbs at $44.64 on 50."""
    payload = _payload(fake_recommendation(), synthetic_board)
    assert payload["leader"] == "rb-00"
    assert payload["candidates"][0]["player_id"] == "wr-00", (
        "fixture no longer reproduces the inversion")
    assert payload["candidates"][0]["e_dollars"] > payload["candidates"][1]["e_dollars"]


def test_the_leader_always_appears_in_candidates(synthetic_board,
                                                 fake_recommendation):
    payload = _payload(fake_recommendation(), synthetic_board)
    ids = [c["player_id"] for c in payload["candidates"]]
    assert payload["leader"] in ids


def test_a_leader_missing_from_dollars_is_still_carried(synthetic_board):
    """Defensive: a demoted tier can name a leader with no dollar estimate.
    Dropping it would leave the UI with a headline it cannot render."""
    rec = Recommendation(tier=2, leader="qb-01", leader_name="QB Player 1",
                         ranked=pd.DataFrame(), indifference_set=[])
    payload = _payload(rec, synthetic_board)
    assert payload["leader"] == "qb-01"
    assert payload["candidates"][0]["player_id"] == "qb-01"


def test_indifference_membership_is_marked_per_candidate(synthetic_board,
                                                         fake_recommendation):
    payload = _payload(fake_recommendation(), synthetic_board)
    assert all(c["in_indifference_set"] for c in payload["candidates"])


# -------------------------------------------------------------- NaN policy
def test_non_finite_floats_become_null(synthetic_board):
    rec = Recommendation(tier=2, leader="rb-00", leader_name="RB Player 0",
                         ranked=pd.DataFrame(), indifference_set=[],
                         p_best=float("nan"))
    payload = _payload(rec, synthetic_board)
    assert payload["p_best"] is None


def test_the_whole_payload_is_json_serialisable(synthetic_board,
                                                fake_recommendation):
    payload = _payload(fake_recommendation(), synthetic_board)
    text = json.dumps(payload)          # raises on NaN by default? no — check:
    assert "NaN" not in text, "a non-finite float reached the payload"


def test_the_ledger_projection_survives_canonical_json(synthetic_board,
                                                       fake_recommendation):
    """`canonical_json` uses allow_nan=False and RAISES, so this is a crash
    path, not a cosmetic one."""
    from src.app.cockpit.ledger import canonical_json

    payload = _payload(fake_recommendation(), synthetic_board)
    canonical_json(schemas.ledger_recommendation(payload))


def test_the_ledger_projection_drops_the_candidate_list(synthetic_board,
                                                        fake_recommendation):
    """Unbounded, and the most likely way a DataFrame smuggles itself into the
    hash via canonical_json's `default=str`."""
    payload = _payload(fake_recommendation(), synthetic_board)
    stored = schemas.ledger_recommendation(payload)
    assert "candidates" not in stored
    assert stored["leader"] == "rb-00"


def test_json_float_maps_every_non_finite_case():
    assert schemas._json_float(float("nan")) is None
    assert schemas._json_float(float("inf")) is None
    assert schemas._json_float(float("-inf")) is None
    assert schemas._json_float(None) is None
    assert schemas._json_float("nonsense") is None
    assert schemas._json_float(3) == 3.0


# ------------------------------------------------------------ tier mapping
def test_tier0_carries_dollars_and_per_candidate_draws(synthetic_board,
                                                       fake_recommendation):
    payload = _payload(fake_recommendation(), synthetic_board)
    leader = next(c for c in payload["candidates"] if c["player_id"] == "rb-00")
    assert leader["e_dollars"] == pytest.approx(44.64)
    assert leader["draws"] == 50
    assert leader["total_se"] == pytest.approx(math.hypot(1.42, 1.34), rel=1e-6)


def test_a_demoted_tier_reports_no_dollars_rather_than_zero(synthetic_board):
    """Zero is a value. None is the absence of one, and the UI must be able to
    tell those apart."""
    ranked = pd.DataFrame([{"player_id": "wr-01", "player_name": "WR Player 1",
                            "position": "WR", "adp": 14.5, "vona_score": 3.2}])
    rec = Recommendation(tier=2, leader="wr-01", leader_name="WR Player 1",
                         ranked=ranked, indifference_set=["wr-01"])
    payload = _payload(rec, synthetic_board)
    candidate = payload["candidates"][0]
    assert candidate["e_dollars"] is None
    assert candidate["vona_score"] == pytest.approx(3.2)


def test_board_metadata_is_joined_onto_candidates(synthetic_board,
                                                  fake_recommendation):
    payload = _payload(fake_recommendation(), synthetic_board)
    leader = next(c for c in payload["candidates"] if c["player_id"] == "rb-00")
    assert leader["position"] == "RB"
    assert leader["name"] == "RB Player 0"
    assert leader["adp"] is not None


def test_engine_settings_are_stamped_in(synthetic_board, fake_recommendation):
    """`reps` and `budget_seconds` change which candidate wins but cannot live
    in the hashed config, so the payload has to carry them or two runs could
    differ in leader under an identical decision_version."""
    payload = _payload(fake_recommendation(), synthetic_board,
                       reps=256, budget_seconds=12.0)
    assert payload["engine"] == {"reps": 256, "shortlist": 6,
                                 "budget_seconds": 12.0}


def test_narration_is_absent_from_the_recommendation_frame(
        synthetic_board, fake_recommendation):
    """It arrives as its own SSE event so a slow model never sits on the
    pick clock."""
    payload = _payload(fake_recommendation(), synthetic_board)
    assert payload["narration"] is None
