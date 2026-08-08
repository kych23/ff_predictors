"""Shared fixtures for the web-cockpit tests.

`data/` is gitignored, so a fresh clone has no `draft_night_bundle.parquet` and
no fitted artifacts. Backend tests therefore run against a synthetic `Board`
built here — small, deterministic, and complete enough to exercise every
position and the ADP ordering. The real bundle is used only by the manual
walkthrough and by `test_refactor_parity.py`, which skips without it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.engine.decision.board import Board

SNAPSHOT = "sn2_synthetic_test"

#: Enough at each position to fill twelve rosters' starters plus a bench.
_COUNTS = {"QB": 8, "RB": 14, "WR": 16, "TE": 8, "K": 4, "DST": 4}


@pytest.fixture
def synthetic_players() -> pd.DataFrame:
    rows = []
    adp = 1.0
    for pos, n in _COUNTS.items():
        for i in range(n):
            rows.append({
                "player_id": f"{pos.lower()}-{i:02d}",
                "player_name": f"{pos} Player {i}",
                "position": pos,
                "team": f"T{i % 8:02d}",
                "bye_week": 5 + (i % 6),
                "value": float(24 - i) if pos in ("RB", "WR") else float(16 - i),
                "value_source": "quantile_model",
                "coverage": "full",
                "adp": adp + i * 1.5,
                "adp_stdev": 4.0,
            })
        adp += 12.0
    frame = pd.DataFrame(rows).sort_values("adp").reset_index(drop=True)
    frame["value"] = frame["value"].clip(lower=1.0)
    return frame


@pytest.fixture
def synthetic_board(synthetic_players) -> Board:
    return Board(players=synthetic_players, snapshot_id=SNAPSHOT,
                 value_source="quantile_model", built_at="2026-08-06T00:00:00Z")


@pytest.fixture
def web_cfg(tmp_path):
    from src.core.config.web import EngineConfig, WebConfig

    return WebConfig(
        source="manual", auto_recommend=False,
        session_path=tmp_path / "session.json",
        ledger_path=tmp_path / "ledger.sqlite",
        engine=EngineConfig(reps=8, budget_seconds=2.0),
    )


@pytest.fixture
def service(synthetic_board, web_cfg):
    """A cockpit wired to the synthetic board, with no session started yet."""
    from src.app.web.service import CockpitService
    from src.core.config import load_league, load_strategy

    cfg = load_league()
    svc = CockpitService(cfg=cfg, strategy=load_strategy(cfg),
                         web_cfg=web_cfg, board=synthetic_board)
    yield svc
    svc.close()


@pytest.fixture
def started(service):
    service.start_session(seat=3, source="manual", resume=False)
    return service


@pytest.fixture
async def client(started):
    """ASGI transport, not Starlette's sync TestClient: it exercises the real
    async path rather than a sync bridge, which is the path production uses."""
    import httpx

    from src.app.web import api as api_mod

    api_mod.app.state.service = started
    transport = httpx.ASGITransport(app=api_mod.app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://cockpit") as c:
        yield c
    api_mod.app.state.service = None


def _fake_recommendation(leader="rb-00", tier=0):
    """Mirrors the observed live case where the recommended player does NOT
    sort first: a rival with a higher point estimate but far fewer draws."""
    from src.engine.decision.recommendation import Recommendation

    ranked = pd.DataFrame([
        {"player_id": "wr-00", "player_name": "WR Player 0", "position": "WR",
         "adp": 13.0, "E_dollars": 44.81, "aleatory_se": 2.96,
         "epistemic_se": 2.43, "draws": 2},
        {"player_id": "rb-00", "player_name": "RB Player 0", "position": "RB",
         "adp": 1.0, "E_dollars": 44.64, "aleatory_se": 1.42,
         "epistemic_se": 1.34, "draws": 50},
    ])
    return Recommendation(
        tier=tier, leader=leader, leader_name="RB Player 0", ranked=ranked,
        indifference_set=["rb-00", "wr-00"],
        dollars={"rb-00": 44.64, "wr-00": 44.81},
        aleatory_se={"rb-00": 1.42, "wr-00": 2.96},
        epistemic_se={"rb-00": 1.34, "wr-00": 2.43},
        p_best=0.61, draws_used=50, stopped_because="separated",
        separating_axis="champion", elapsed_s=9.6,
    )


@pytest.fixture
def fake_recommendation():
    return _fake_recommendation
