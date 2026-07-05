"""Black-box extensions for VONA scoring and marginal value."""
import numpy as np
import pandas as pd

from src.config import load_config
from src.recommender.recommend import recommend
from src.recommender.replacement import ReplacementLevels
from src.recommender.roster_state import RosterState
from src.recommender.vona import marginal_value, score_board


def _replacement():
    return ReplacementLevels(levels={"RB": 5.0, "WR": 5.0, "TE": 5.0, "QB": 5.0})


def _state():
    return RosterState(cfg=load_config(), draft_position=1)


def test_marginal_linear_in_value_above_replacement():
    state = _state()
    rep = _replacement()
    # marginal = value - replacement for an open starter slot
    assert marginal_value(25.0, "RB", rep, state) - marginal_value(20.0, "RB", rep, state) == 5.0


def test_marginal_at_replacement_is_zero():
    assert marginal_value(5.0, "RB", _replacement(), _state()) == 0.0


def test_score_board_returns_all_rows_with_scores():
    board = pd.DataFrame([
        {"player_id": f"p{i}", "position": pos, "value": 20.0 - i,
         "adp": 10.0 + 10 * i, "adp_stdev": 5.0}
        for i, pos in enumerate(["RB", "WR", "TE", "QB", "RB", "WR"])
    ])
    scored = score_board(board, _state(), _replacement(), next_pick=30)
    assert len(scored) == len(board)
    for col in ("vona_score", "marginal", "wait_term"):
        assert col in scored.columns
        assert scored[col].notna().all()


def test_wait_term_nonnegative():
    board = pd.DataFrame([
        {"player_id": "a", "position": "RB", "value": 20, "adp": 5, "adp_stdev": 2},
        {"player_id": "b", "position": "RB", "value": 15, "adp": 40, "adp_stdev": 10},
        {"player_id": "c", "position": "WR", "value": 18, "adp": 90, "adp_stdev": 20},
        {"player_id": "d", "position": "WR", "value": 6, "adp": 140, "adp_stdev": 25},
    ])
    scored = score_board(board, _state(), _replacement(), next_pick=25)
    assert (scored["wait_term"] >= 0).all()


def test_vona_never_exceeds_marginal():
    """vona = marginal - wait_term with wait_term >= 0, so vona <= marginal."""
    board = pd.DataFrame([
        {"player_id": "a", "position": "RB", "value": 20, "adp": 5, "adp_stdev": 2},
        {"player_id": "b", "position": "WR", "value": 18, "adp": 100, "adp_stdev": 20},
        {"player_id": "c", "position": "WR", "value": 10, "adp": 120, "adp_stdev": 20},
    ])
    scored = score_board(board, _state(), _replacement(), next_pick=25)
    assert (scored["vona_score"] <= scored["marginal"] + 1e-9).all()


def test_higher_value_higher_vona_same_market_profile():
    """Two players identical except projection value: better player never
    scores lower."""
    board = pd.DataFrame([
        {"player_id": "hi", "position": "RB", "value": 20, "adp": 30, "adp_stdev": 8},
        {"player_id": "lo", "position": "RB", "value": 12, "adp": 30, "adp_stdev": 8},
        {"player_id": "fill", "position": "RB", "value": 6, "adp": 90, "adp_stdev": 20},
    ])
    scored = score_board(board, _state(), _replacement(), next_pick=40)
    hi = scored[scored["player_id"] == "hi"].iloc[0]
    lo = scored[scored["player_id"] == "lo"].iloc[0]
    assert hi["vona_score"] >= lo["vona_score"]


def test_elite_scarcity_beats_weak_fallback():
    """Elite RB who will be gone by the next pick, with only a weak RB likely
    to remain: waiting costs nearly his full marginal, so his VONA stays close
    to marginal while the fallback's is heavily discounted."""
    board = pd.DataFrame([
        {"player_id": "elite", "position": "RB", "value": 20, "adp": 2,
         "adp_stdev": 1},
        {"player_id": "fallback", "position": "RB", "value": 8, "adp": 60,
         "adp_stdev": 10},
    ])
    scored = score_board(board, _state(), _replacement(), next_pick=50)
    elite = scored[scored["player_id"] == "elite"].iloc[0]
    fallback = scored[scored["player_id"] == "fallback"].iloc[0]
    assert elite["vona_score"] > fallback["vona_score"]
    # the best plausible alternative at pick 50 is weak, so the wait discount
    # cannot eat most of the elite player's marginal
    assert elite["wait_term"] < elite["marginal"] / 2


def test_recommend_early_draft_not_forced():
    cfg = load_config()
    state = RosterState(cfg=cfg, draft_position=1)  # nothing drafted yet
    proj = pd.DataFrame([
        {"player_id": "rb", "position": "RB", "p10": 15, "p50": 20, "p90": 25,
         "adp": 5, "adp_stdev": 2},
        {"player_id": "wr", "position": "WR", "p10": 12, "p50": 16, "p90": 22,
         "adp": 12, "adp_stdev": 4},
        {"player_id": "qb", "position": "QB", "p10": 10, "p50": 14, "p90": 18,
         "adp": 30, "adp_stdev": 8},
        {"player_id": "te", "position": "TE", "p10": 6, "p50": 9, "p90": 13,
         "adp": 50, "adp_stdev": 12},
    ])
    recs = recommend(proj, state, _replacement(), cfg=cfg)
    assert not recs.empty
    assert not recs["forced_completion"].any()
