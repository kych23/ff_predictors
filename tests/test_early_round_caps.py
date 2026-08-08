"""A second tight end is legal all draft and wrong in round 3.

`max_per_position` caps what a roster may EVER hold. It cannot express "not
yet", so TE was capped at 2 with nothing stopping the engine taking the second
one in round 4. The objective prices a TE2 thinly — he has no starting slot of
his own and reaches the lineup only through FLEX, against every back and
receiver left — but thinly is not never, and a high-variance tight end can
still win a shortlist place off upside alone. A shortlist place is what buys
entry to the expensive simulation.

The rule is config, not code: `roster_preferences.early_round_caps`.
"""
from __future__ import annotations

import pytest

from src.core.config import load_league, load_strategy
from src.engine.decision import recommend as tier2
from src.engine.decision.board import Board
from src.engine.decision.roster_state import RosterState


@pytest.fixture(scope="module")
def league():
    return load_league()


@pytest.fixture(scope="module")
def strategy(league):
    return load_strategy(league)


@pytest.fixture(scope="module")
def board():
    return Board.from_bundle().players.copy()


def _score_at(round_no, league, strategy, board, held_positions):
    pick = (round_no - 1) * league.teams + 1
    order = board.sort_values("adp")["player_id"].astype(str).tolist()
    taken = set(order[:pick - 1])
    avail = board[~board["player_id"].astype(str).isin(taken)]
    rs = RosterState(
        cfg=league, draft_position=1, drafted=taken,
        my_roster=[{"player_id": f"held{i}", "position": p}
                   for i, p in enumerate(held_positions)])
    return tier2.score(avail, rs, league, strategy, current_pick=pick,
                       preseason_board=board).ranked


def test_the_rule_is_configured_not_hardcoded(strategy):
    rule = strategy.early_round_caps.get("TE")
    assert rule == {"max": 1, "through_round": 8}


@pytest.mark.parametrize("round_no", [1, 3, 5, 8])
def test_no_second_te_is_offered_early(round_no, league, strategy, board):
    ranked = _score_at(round_no, league, strategy, board, ["TE"])
    live = ranked[ranked["sink_reason"] == ""]
    assert not (live["position"] == "TE").any(), (
        f"a second TE reached the live board in round {round_no}")
    sunk = ranked[ranked["position"] == "TE"].iloc[0]
    assert sunk["sink_reason"] == "TE_max_1_through_round_8"


@pytest.mark.parametrize("round_no", [9, 12])
def test_a_second_te_is_allowed_once_the_cap_expires(round_no, league,
                                                     strategy, board):
    ranked = _score_at(round_no, league, strategy, board, ["TE"])
    live = ranked[ranked["sink_reason"] == ""]
    assert (live["position"] == "TE").any(), (
        f"round {round_no} is past the cap; a TE must be draftable again")


def test_the_first_te_is_never_blocked(league, strategy, board):
    """The cap is on DOUBLING UP. Holding none, a tight end must be offered —
    otherwise the rule would leave the mandatory TE slot unfillable."""
    ranked = _score_at(3, league, strategy, board, ["RB", "WR"])
    live = ranked[ranked["sink_reason"] == ""]
    assert (live["position"] == "TE").any()


def test_other_positions_are_untouched(league, strategy, board):
    """Only positions named in the config are capped. A second RB in round 3 is
    ordinary drafting and must stay on the board."""
    ranked = _score_at(3, league, strategy, board, ["RB"])
    live = ranked[ranked["sink_reason"] == ""]
    assert (live["position"] == "RB").any()


def test_my_simulated_future_self_obeys_the_same_cap(league, strategy):
    """`rollout` fills my later picks with a value heuristic. If it keeps
    doubling up where the live recommender refuses to, every candidate is
    valued against a roster the engine would never actually build."""
    import numpy as np
    import pandas as pd

    from src.engine.sim.rollout import _my_pick

    frame = pd.DataFrame({
        "player_id": ["te1", "rb1"], "position": ["TE", "RB"],
        "value": [99.0, 1.0], "adp": [10.0, 20.0],
    })
    caps = {"TE": {"max": 1, "through_round": 8}}

    early = _my_pick(frame, np.array([0, 1]), league, {"TE": 1}, pick_no=25,
                     early_round_caps=caps, teams=league.teams)
    assert early == 1, "round 3 must skip the TE even though he scores higher"

    late = _my_pick(frame, np.array([0, 1]), league, {"TE": 1}, pick_no=109,
                    early_round_caps=caps, teams=league.teams)
    assert late == 0, "past the cap the value heuristic decides again"


def test_a_cap_never_leaves_a_seat_unable_to_pick(league):
    """If the only legal players are capped, take one anyway. A roster that
    cannot make a pick is worse than one that doubles up."""
    import numpy as np
    import pandas as pd

    from src.engine.sim.rollout import _my_pick

    only_tes = pd.DataFrame({
        "player_id": ["te1", "te2"], "position": ["TE", "TE"],
        "value": [5.0, 9.0], "adp": [10.0, 20.0],
    })
    chosen = _my_pick(only_tes, np.array([0, 1]), league, {"TE": 1}, pick_no=25,
                      early_round_caps={"TE": {"max": 1, "through_round": 8}},
                      teams=league.teams)
    assert chosen == 1


def test_the_config_change_is_reflected_in_the_decision_version(strategy):
    """`early_round_caps` lives in strategy.yaml, so adding it re-hashes
    `decision_version` and reseeds every RNG stream — which is why the pinned
    hash in test_web_config.py moved with it. Models stay valid; only the
    decision layer is affected (§10.1)."""
    assert strategy.decision_version.startswith("dec_v2.")
    assert strategy.strategy_hash == "09e4cf71e67b"
