"""Board I/O must not run on the event loop.

Every route is `async def` — `test_web_api.py::test_every_route_is_async`
requires it. That test passes just as happily on an `async def` route that
calls `store.save()` inline, which would put a debounced save stream, a parquet
read and a JSON write on the same loop that serves the pick clock and the SSE
stream. `NotesPanel.tsx` refused the API for this exact reason.

So the async invariant alone does not protect anything here. This does: it
asserts the store call lands on a worker thread, by checking that no running
loop is visible from inside it.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from src.app.web import api as api_mod


@pytest.fixture()
async def board_client(rankings_store, rankings_data, service):
    api_mod.app.state.service = service
    api_mod.app.state.rankings = rankings_store
    api_mod.app.state.rankings_data = rankings_data
    transport = httpx.ASGITransport(app=api_mod.app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://cockpit") as client:
        yield client
    api_mod.app.state.service = None


def _assert_off_the_loop(label: str) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return                      # no loop here: we are on a worker thread
    raise AssertionError(
        f"{label} ran on the event loop; wrap it in asyncio.to_thread")


async def test_the_save_path_runs_off_the_event_loop(board_client,
                                                     rankings_store,
                                                     monkeypatch):
    original = type(rankings_store).save
    seen: list[str] = []

    def spy(self, board, *, expected_rev):
        _assert_off_the_loop("RankingsStore.save")
        seen.append(board.board_id)
        return original(self, board, expected_rev=expected_rev)

    monkeypatch.setattr(type(rankings_store), "save", spy)

    created = await board_client.post("/api/rankings",
                                      json={"name": "Offload", "seed_method": "adp"})
    board = created.json()
    response = await board_client.put(
        f"/api/rankings/{board['board_id']}",
        json={"expected_rev": board["rev"], "scopes": board["scopes"]})
    assert response.status_code == 200, response.text
    assert seen, "save was never called"


async def test_the_load_path_runs_off_the_event_loop(board_client,
                                                     rankings_store,
                                                     monkeypatch):
    original = type(rankings_store).load
    seen: list[str] = []

    def spy(self, board_id):
        _assert_off_the_loop("RankingsStore.load")
        seen.append(board_id)
        return original(self, board_id)

    monkeypatch.setattr(type(rankings_store), "load", spy)

    created = await board_client.post("/api/rankings", json={"name": "Read"})
    board_id = created.json()["board_id"]
    assert (await board_client.get(
        f"/api/rankings/{board_id}")).status_code == 200
    assert seen == [board_id]


async def test_the_catalogue_runs_off_the_event_loop(board_client,
                                                     monkeypatch):
    """260 rows of frame work per page load is the one that would be felt."""
    from src.app.rankings import detail as detail_mod

    original = detail_mod.catalogue
    seen: list[int] = []

    def spy(data):
        _assert_off_the_loop("catalogue")
        seen.append(1)
        return original(data)

    monkeypatch.setattr(api_mod_target(), "catalogue", spy)
    assert (await board_client.get(
        "/api/rankings/catalogue")).status_code == 200
    assert seen


def api_mod_target():
    """The route imports `catalogue` inside the handler, so patching the
    defining module is what the handler will actually see."""
    from src.app.rankings import detail as detail_mod

    return detail_mod
