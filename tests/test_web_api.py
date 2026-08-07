"""The HTTP surface (§19).

Driven through `httpx.ASGITransport`, which runs the real async path rather
than a sync bridge. These are assertions about behaviour, not smoke tests: the
things checked here are the ones that would cost a pick at 6pm on draft night —
seat indexing, duplicate handling, the ambiguity flow, and whether an
unresolvable name still advances the clock.
"""
from __future__ import annotations

import inspect

from src.app.web import api as api_mod


# ------------------------------------------------------- the async invariant
def test_every_route_is_async():
    """`service.py`'s single-writer rule and the thread-affine SQLite ledger
    both depend on this. Starlette runs a plain `def` route in a threadpool,
    which would break both silently."""
    sync = [r.path for r in api_mod.app.routes
            if hasattr(r, "endpoint") and not inspect.iscoroutinefunction(r.endpoint)]
    assert sync == [], f"sync routes would run off the event loop: {sync}"


# ------------------------------------------------------------------ session
async def test_session_reports_the_clock_and_seat(client):
    body = (await client.get("/api/session")).json()
    assert body["seat"] == 3
    assert body["pick_number"] == 1
    assert body["on_the_clock"] == 0
    assert body["teams"] == 12


async def test_seat_is_zero_indexed_end_to_end(client):
    """`build_tiers` is 1-indexed and `--seat` was 1-indexed; the API is not.
    A silent off-by-one here drafts for the wrong team."""
    body = (await client.get("/api/session")).json()
    assert body["seat"] == 3
    assert body["is_my_turn"] is False       # seat 3 is up at pick 4, not 1


async def test_a_seat_outside_the_league_is_rejected(client, service):
    resp = await client.post("/api/session",
                             json={"seat": 99, "source": "manual"})
    assert resp.status_code == 400


async def test_endpoints_404_without_a_session(service):
    import httpx

    api_mod.app.state.service = service      # started=False
    transport = httpx.ASGITransport(app=api_mod.app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://cockpit") as c:
        assert (await c.get("/api/session")).status_code == 404
        assert (await c.post("/api/picks",
                             json={"raw_name": "x"})).status_code == 404
    api_mod.app.state.service = None


# -------------------------------------------------------------------- picks
async def test_a_pick_is_recorded_and_advances_the_clock(client):
    resp = await client.post("/api/picks", json={"raw_name": "RB Player 0"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "recorded"
    assert body["pick_number"] == 2
    assert body["drafted_count"] == 1


async def test_a_duplicate_pick_is_409_not_400(client):
    """Both `Board.take` (ValueError) and `Session._replay` (DataError) can
    see this; the service normalises so the client gets one answer."""
    await client.post("/api/picks", json={"player_id": "rb-00"})
    resp = await client.post("/api/picks", json={"player_id": "rb-00"})
    assert resp.status_code == 409


async def test_an_unknown_player_id_is_404(client):
    resp = await client.post("/api/picks", json={"player_id": "nobody-99"})
    assert resp.status_code == 404


async def test_an_empty_pick_body_is_400(client):
    assert (await client.post("/api/picks", json={})).status_code == 400


async def test_an_ambiguous_name_returns_candidates_not_an_error(client):
    """A status code cannot carry the options, and the operator has to choose.

    Note the query is a PARTIAL: an exact name is resolved by the spine before
    the substring pass ever runs, which is the correct precedence."""
    resp = await client.post("/api/picks", json={"raw_name": "Player 1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ambiguous"
    assert len(body["candidates"]) > 1
    assert all({"player_id", "name", "position"} <= set(c)
               for c in body["candidates"])


async def test_an_exact_name_beats_the_substring_pass(client):
    """The spine runs first; otherwise every full name would be ambiguous
    against its own longer siblings ("RB Player 1" vs "RB Player 10")."""
    resp = await client.post("/api/picks", json={"raw_name": "RB Player 1"})
    assert resp.json()["status"] == "recorded"


async def test_the_operator_resolves_ambiguity_by_posting_an_id(client):
    body = (await client.post("/api/picks",
                              json={"raw_name": "Player 1"})).json()
    chosen = body["candidates"][0]["player_id"]
    resp = await client.post("/api/picks", json={"player_id": chosen})
    assert resp.status_code == 200 and resp.json()["status"] == "recorded"


async def test_an_unknown_name_is_reported_before_it_is_recorded(client):
    """Nothing is written until the operator confirms — otherwise a typo
    silently burns a pick."""
    resp = await client.post("/api/picks", json={"raw_name": "Zzzz Nobody"})
    assert resp.json()["status"] == "unresolved"
    assert (await client.get("/api/session")).json()["pick_number"] == 1


async def test_a_forced_unresolved_pick_still_advances_the_clock(client):
    """The regression this whole feature depends on: a live feed produces
    names nobody can resolve, and the draft still moved on."""
    resp = await client.post("/api/picks",
                             json={"raw_name": "Zzzz Nobody",
                                   "force_unresolved": True})
    body = resp.json()
    assert body["status"] == "unresolved"
    assert body["pick_number"] == 2
    assert body["unresolved"] == ["Zzzz Nobody"]
    assert body["drafted_count"] == 0


# --------------------------------------------------------------------- undo
async def test_undo_rewinds_by_replay(client):
    await client.post("/api/picks", json={"player_id": "rb-00"})
    await client.post("/api/picks", json={"player_id": "wr-00"})
    body = (await client.post("/api/undo")).json()
    assert body["drafted_count"] == 1
    assert body["pick_number"] == 2


async def test_undo_removes_an_unresolved_entry_too(client):
    await client.post("/api/picks", json={"raw_name": "Zzzz",
                                          "force_unresolved": True})
    body = (await client.post("/api/undo")).json()
    assert body["unresolved"] == []
    assert body["pick_number"] == 1


# -------------------------------------------------------------------- board
async def test_the_board_is_adp_ordered_and_excludes_drafted(client):
    before = (await client.get("/api/board?limit=5")).json()["players"]
    adps = [p["adp"] for p in before]
    assert adps == sorted(adps)
    top = before[0]["player_id"]
    await client.post("/api/picks", json={"player_id": top})
    after = (await client.get("/api/board?limit=5")).json()["players"]
    assert top not in {p["player_id"] for p in after}


async def test_board_limit_is_bounded(client):
    assert (await client.get("/api/board?limit=0")).status_code == 422
    assert (await client.get("/api/board?limit=5")).status_code == 200


# ----------------------------------------------------------- recommendation
async def test_recommendation_is_idle_before_any_run(client):
    body = (await client.get("/api/recommendation")).json()
    assert body["status"] == "idle"


async def test_a_second_run_while_one_is_in_flight_is_409(client, started):
    from src.app.web.service import RUNNING

    started.recommendation.status = RUNNING
    resp = await client.post("/api/recommendation")
    assert resp.status_code == 409


async def test_generation_advances_on_every_state_change(client, started):
    """The supersede check depends on this: a run whose generation no longer
    matches is discarded rather than shown against a board that moved."""
    before = started.generation
    await client.post("/api/picks", json={"player_id": "rb-00"})
    assert started.generation > before


# ------------------------------------------------------------------ session
async def test_a_second_session_without_resume_is_409(client):
    resp = await client.post("/api/session",
                             json={"seat": 3, "source": "manual"})
    assert resp.status_code == 409


async def test_resume_reopens_the_existing_session(client):
    await client.post("/api/picks", json={"player_id": "rb-00"})
    resp = await client.post(
        "/api/session", json={"seat": 3, "source": "manual", "resume": True})
    assert resp.status_code == 200
    assert resp.json()["drafted_count"] == 1


async def test_delete_archives_rather_than_destroys(client, started):
    await client.post("/api/picks", json={"player_id": "rb-00"})
    body = (await client.delete("/api/session")).json()
    assert body["status"] == "archived"
    from pathlib import Path
    assert Path(body["path"]).exists(), "the log must survive a reset"


async def test_the_source_chip_reports_the_real_adapter(client):
    """`ManualSource` defines `__len__`, so an EMPTY queue makes the adapter
    itself falsy. A truthiness check therefore reported source "none" for a
    perfectly healthy manual session — caught by driving a live server, not by
    any unit test."""
    body = (await client.get("/api/session")).json()
    assert body["source"]["name"] == "manual"
    assert body["source"]["state"] == "ok"


# ------------------------------------------- precommit-review regressions
async def test_a_seat_outside_the_league_is_refused_on_a_pick(client):
    """`rollout` indexes a fixed-length per-seat list, so an out-of-range seat
    raises IndexError mid-recommendation AND persists into `by_seat`, corrupting
    every later pick. `POST /api/session` validated this; `POST /api/picks`
    did not."""
    resp = await client.post("/api/picks",
                             json={"player_id": "rb-00", "seat": 99})
    assert resp.status_code == 409
    assert (await client.get("/api/session")).json()["drafted_count"] == 0


async def test_restarting_a_session_releases_the_previous_ledger(started):
    """`start_session` reassigned `source` and `ledger` without closing them, so
    every restart leaked a sqlite3 connection (and an OAuth client, for Yahoo)."""
    first = started.ledger
    started.start_session(seat=3, source="manual", resume=True)
    assert started.ledger is not first
    import sqlite3

    import pytest as _pytest
    with _pytest.raises(sqlite3.ProgrammingError):
        len(first)                      # closed


async def test_archiving_releases_the_source_and_ledger(started):
    started.archive_session()
    assert started.source is None and started.ledger is None


async def test_a_superseded_run_failure_does_not_poison_a_new_session(started):
    """The blocker: the success path checked `generation` before touching
    shared state; the FAILURE path did not, so an abandoned run that raised
    later stamped ERROR onto a brand-new session and broadcast rec_error for a
    recommendation that never failed."""
    from src.app.web import api as api_mod

    stale = started.generation
    started.generation += 5                      # a restart happened meanwhile

    def boom(_args):
        raise RuntimeError("stale run blew up")

    started.run_recommendation = boom
    await api_mod._run(started, {"seat": 4}, stale)
    assert started.recommendation.status != "error"
    assert started.recommendation.detail == ""


# --------------------------------------------------------- setup endpoint
async def test_league_endpoint_works_without_a_session(service):
    """The setup screen needs teams/rounds to render a seat picker, and
    `GET /api/session` 404s until a draft is started. Without this the app
    could not begin a draft at all."""
    import httpx

    api_mod.app.state.service = service          # no session started
    transport = httpx.ASGITransport(app=api_mod.app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://cockpit") as c:
        resp = await c.get("/api/league")
        assert resp.status_code == 200
        body = resp.json()
        assert body["teams"] == service.cfg.teams
        assert body["rounds"] == service.cfg.roster.rounds
        assert body["active"] is False
        assert "manual" in body["sources"]
    api_mod.app.state.service = None


async def test_league_reports_an_active_session(client):
    body = (await client.get("/api/league")).json()
    assert body["active"] is True
    assert body["session_exists"] is True


async def test_league_reports_a_resumable_session_on_disk(service):
    """`session_exists` is what makes "resume" an informed choice rather than a
    coin flip against a 409."""
    import httpx

    service.start_session(seat=3, source="manual", resume=False)
    service.session = None                       # simulate a fresh page load
    api_mod.app.state.service = service
    transport = httpx.ASGITransport(app=api_mod.app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://cockpit") as c:
        body = (await c.get("/api/league")).json()
        assert body["session_exists"] is True
        assert body["active"] is False
    api_mod.app.state.service = None


def test_the_repo_root_resolves_to_the_repo_root():
    """`parents[4]` climbed one level too far, so `web/dist` never existed from
    the app's point of view and the built frontend was silently never served.
    Nothing failed — the API kept answering — which is why this needs a test
    rather than a glance."""
    assert (api_mod._REPO_ROOT / "pyproject.toml").exists(), (
        f"_REPO_ROOT resolved to {api_mod._REPO_ROOT}, which is not the repo")
    assert api_mod.DIST == api_mod._REPO_ROOT / "web" / "dist"
