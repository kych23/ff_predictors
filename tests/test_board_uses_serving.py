"""load_board must read the serving engine (Supabase), not the research default."""
from contextlib import contextmanager

import pandas as pd

import src.recommender.board as board
from src.config import load_config


def test_load_board_uses_serving_scope(monkeypatch):
    used = {"serving": False}

    @contextmanager
    def fake_serving():
        used["serving"] = True
        yield object()  # a dummy session; loaders are stubbed below

    monkeypatch.setattr(board, "serving_session_scope", fake_serving)
    monkeypatch.setattr(board, "load_projections_df",
                        lambda s, season: pd.DataFrame(
                            {"player_id": ["P1"], "position": ["RB"],
                             "p10": [1.0], "p50": [2.0], "p90": [3.0]}))
    monkeypatch.setattr(board, "load_adp_df",
                        lambda s, cfg, season: pd.DataFrame(
                            {"player_id": ["P1"], "adp": [5.0], "adp_stdev": [1.0]}))
    monkeypatch.setattr(board, "load_players_df",
                        lambda s: pd.DataFrame(
                            {"player_id": ["P1"], "name": ["A"], "team": ["SF"]}))
    monkeypatch.setattr(board, "compute_bye_weeks", lambda season: {})

    out = board.load_board(2026, load_config())
    assert used["serving"] is True
    assert list(out["player_id"]) == ["P1"]
