"""HTTP contract tests: status codes, response shapes, dependency overrides.
No real DB, no network — SQLite + synthetic board via dependency injection."""

import pytest

# §9.4: this module is pinned to the v1 config shape. api/, web/ and the
# weekly start/sit surface are FROZEN for the build window, so v1 stays
# alive underneath them and these tests are deselected from the default
# run rather than migrated. Thaw is a §22.2 follow-up.
pytestmark = pytest.mark.v1_frozen
import pytest
from fastapi.testclient import TestClient

from api.deps import get_board_for, get_db, get_snapshot_id
from api.main import create_app
from tests.test_api_replay import make_board


@pytest.fixture()
def client(db_session):
    app = create_app()
    board = make_board()
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_board_for] = lambda: (lambda season: board)
    app.dependency_overrides[get_snapshot_id] = lambda: None
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_players_listing(client):
    r = client.get("/players", params={"season": 2026})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 120
    assert {"player_id", "name", "position", "p10", "p50", "p90"} <= set(rows[0])


def test_session_lifecycle(client):
    r = client.post("/draft/sessions", json={"season": 2026, "draft_position": 1})
    assert r.status_code == 201
    sid = r.json()["session_id"]
    assert r.json()["is_my_turn"] is True

    r = client.post(f"/draft/sessions/{sid}/picks", json={"player_id": "P0031"})
    assert r.status_code == 200
    assert r.json()["picks"][0]["mine"] is True

    r = client.get(f"/draft/sessions/{sid}/recommendations", params={"top_n": 5})
    assert r.status_code == 200
    recs = r.json()
    assert 0 < len(recs) <= 5
    assert "P0031" not in [x["player_id"] for x in recs]

    r = client.post(f"/draft/sessions/{sid}/undo")
    assert r.status_code == 200
    assert r.json()["picks"] == []

    r = client.get(f"/draft/sessions/{sid}")
    assert r.status_code == 200
    assert r.json()["current_overall_pick"] == 1


def test_error_mapping(client):
    assert client.get("/draft/sessions/nope").status_code == 404
    r = client.post("/draft/sessions", json={"season": 2026, "draft_position": 999})
    assert r.status_code == 400
    r = client.post("/draft/sessions", json={"season": 2026, "draft_position": 1})
    sid = r.json()["session_id"]
    assert client.post(f"/draft/sessions/{sid}/picks",
                       json={"player_id": "GHOST"}).status_code == 400
    assert client.post(f"/draft/sessions/{sid}/picks", json={}).status_code == 400


def test_bot_pick_endpoint(client):
    r = client.post("/draft/sessions", json={"season": 2026, "draft_position": 2})
    sid = r.json()["session_id"]
    r = client.post(f"/draft/sessions/{sid}/bot-pick")
    assert r.status_code == 200
    assert r.json()["picks"][0]["mine"] is False
    assert r.json()["picks"][0]["player_id"] is not None


def test_cors_allows_frontend_origin(client):
    r = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"
