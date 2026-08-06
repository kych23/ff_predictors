"""Regressions for two production-only API bugs (found by live-DB smoke test,
invisible to the synthetic-board tests):

Bug 1: a board with a missing bye_week (NaN in a float64 column) crashed
GET /players — `df.where(pd.notna(df), None)` cannot store None in a float
column, so NaN reached pydantic's Optional[int] and raised ResponseValidation.

Bug 2: the draft_sessions product table was never registered on the shared
Base at init_db time (only conftest created it explicitly), so create_all
against Postgres silently omitted it.
"""

import pytest

# §9.4: this module is pinned to the v1 config shape. api/, web/ and the
# weekly start/sit surface are FROZEN for the build window, so v1 stays
# alive underneath them and these tests are deselected from the default
# run rather than migrated. Thaw is a §22.2 follow-up.
pytestmark = pytest.mark.v1_frozen
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.deps import get_board_for, get_db, get_snapshot_id
from api.main import create_app
from tests.test_api_replay import make_board


def _board_with_missing_bye():
    board = make_board()
    # a real board has free agents / players whose team has no schedule row yet
    board.loc[board.index[:5], "bye_week"] = np.nan
    board.loc[board.index[:3], "adp"] = np.nan
    return board


def test_players_endpoint_tolerates_nan_bye_week(db_session):
    app = create_app()
    board = _board_with_missing_bye()
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_board_for] = lambda: (lambda season: board)
    app.dependency_overrides[get_snapshot_id] = lambda: None
    client = TestClient(app)

    r = client.get("/players", params={"season": 2026})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == len(board)
    missing = [x for x in rows if x["bye_week"] is None]
    assert len(missing) == 5          # NaN serialized to null, not rejected
    assert all(x["adp"] is None for x in rows[:3])


def test_recommendations_tolerate_nan_columns(db_session):
    from api.draft_service import DraftService
    from src.config import load_config

    board = _board_with_missing_bye()
    svc = DraftService(db=db_session, cfg=load_config(),
                       board_for=lambda season: board)
    s = svc.create_session(season=2026, draft_position=1)
    recs = svc.recommendations(s.session_id, top_n=5)
    assert 0 < len(recs) <= 5
    # any NaN-derived field must be JSON null, never a float nan
    for r in recs:
        for v in r.values():
            assert not (isinstance(v, float) and np.isnan(v))


def test_init_db_registers_product_table_on_base():
    """draft_sessions must be on the shared Base metadata so create_all (which
    init_db runs against Postgres) provisions it — not only the SQLite conftest."""
    import api.db_models  # noqa: F401 — the import init_db performs
    from src.db.models import Base

    assert "draft_sessions" in Base.metadata.tables
