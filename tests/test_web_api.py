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


async def test_a_duplicate_pick_is_reported_as_already_recorded(client):
    """Both `Board.take` (ValueError) and `Session._replay` (DataError) can
    see this; the service normalises so the client gets one answer.

    That answer is now a 200, not a 409. The cockpit runs the Yahoo feed and
    manual entry as ONE mode — either may fill any pick — so the same pick
    arriving twice is the expected steady state, not a fault. The operator
    clicking a row the feed already recorded has done nothing wrong and the
    desired state already holds; a red banner there teaches distrust of a
    board that is correct. The current state rides along so a stale board
    corrects itself.

    A genuinely bad request is still an error: unknown player 404s, empty
    body 400s (below).
    """
    await client.post("/api/picks", json={"player_id": "rb-00"})
    resp = await client.post("/api/picks", json={"player_id": "rb-00"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "already_recorded"


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


async def test_bench_slots_always_render(client):
    """"Two bench spots left" is a real constraint in the last rounds, and a
    rail that only lists what you have cannot express it."""
    body = (await client.get("/api/session")).json()
    bench = [s for s in body["roster_slots"] if s["slot"] == "BENCH"]
    assert len(bench) == 6
    assert all(s["name"] is None for s in bench)


async def test_overflow_at_a_position_fills_flex_then_bench(client):
    """A third WR is a legal FLEX start, so it belongs in FLEX, not the bench.
    The fourth benches. This mirrors `RosterState._assign_slot` exactly rather
    than reimplementing it — two answers to "which slot" is the split-brain
    that produced the K/DST bug."""
    for pid in ("wr-00", "wr-01", "wr-02", "wr-03"):
        await client.post("/api/picks", json={"player_id": pid, "seat": 3})
    slots = (await client.get("/api/session")).json()["roster_slots"]
    filled = {s["slot"]: [x["name"] for x in slots
                          if x["slot"] == s["slot"] and x["name"]]
              for s in slots}
    assert len(filled["WR"]) == 2
    assert len(filled["FLEX"]) == 1
    assert len(filled["BENCH"]) == 1


async def test_bench_grows_past_its_configured_size_rather_than_dropping(client):
    """Never silently lose a drafted player: if more land on the bench than
    the league configures, show them all."""
    ids = [f"wr-{i:02d}" for i in range(10)]
    for pid in ids:
        await client.post("/api/picks", json={"player_id": pid, "seat": 3})
    slots = (await client.get("/api/session")).json()["roster_slots"]
    named = [s["name"] for s in slots if s["name"]]
    assert len(named) == len(ids), "every drafted player must appear somewhere"


async def test_a_second_recommendation_request_is_not_an_error(client, started):
    """Spam-clicking Recommend used to surface a raw 409 body on the draft
    screen. Asking for a recommendation while one is computing is the state the
    operator wanted, not a failure."""
    from src.app.web.service import RUNNING

    started.recommendation.status = RUNNING
    resp = await client.post("/api/recommendation")
    assert resp.status_code == 200
    assert resp.json()["status"] == RUNNING


async def test_a_run_announces_its_start(client, started, monkeypatch):
    """`auto_recommend` launches server-side, so a client that only hears about
    completion shows an idle button while the engine is busy."""
    from src.app.web import api as api_mod

    seen: list[str] = []
    monkeypatch.setattr(started, "publish",
                        lambda event, data: seen.append(event))
    monkeypatch.setattr(started, "run_recommendation",
                        lambda args: (_ for _ in ()).throw(RuntimeError("stop")))
    await api_mod._launch(started)
    assert "rec_started" in seen


# ---------------------------------------------------------- draft board
async def test_the_draft_grid_follows_the_snake(client):
    """Round 2 reverses. A grid that assumed straight order would attribute
    every pick after round 1 to the wrong team."""
    board = (await client.get("/api/board?limit=30")).json()["players"]
    for p in board[:14]:
        await client.post("/api/picks", json={"player_id": p["player_id"]})
    picks = (await client.get("/api/session")).json()["picks"]
    r1 = [p["seat"] for p in picks if p["round"] == 1]
    r2 = [p["seat"] for p in picks if p["round"] == 2]
    assert r1 == list(range(12))
    assert r2[0] == 11 and r2[1] == 10


async def test_an_unresolved_pick_still_occupies_a_grid_cell(client):
    """Built from the EVENT LOG, not `drafted` — an unresolved pick consumes a
    slot without adding an id, and deriving the grid from `drafted` would shift
    every later pick one column left."""
    await client.post("/api/picks", json={"player_id": "rb-00"})
    await client.post("/api/picks", json={"raw_name": "Zzzz Nobody",
                                          "force_unresolved": True})
    await client.post("/api/picks", json={"player_id": "wr-00"})
    picks = (await client.get("/api/session")).json()["picks"]
    assert [p["pick_number"] for p in picks] == [1, 2, 3]
    assert picks[1]["resolved"] is False
    assert picks[1]["name"] == "Zzzz Nobody"
    assert picks[2]["seat"] == 2, "the third pick must not slide left"


async def test_my_own_picks_are_flagged_for_the_grid(client):
    await client.post("/api/picks", json={"player_id": "rb-00", "seat": 3})
    picks = (await client.get("/api/session")).json()["picks"]
    assert picks[0]["is_mine"] is True


async def test_team_names_default_to_generic_placeholders(client):
    """Obvious placeholders until the real draft order is known — a wrong name
    that looks right is worse than one that looks like a placeholder."""
    body = (await client.get("/api/session")).json()
    assert body["team_names"] == [f"Team {i + 1}" for i in range(12)]


async def test_configured_team_names_are_padded_to_the_league_size(started):
    """A short list must not shrink the board."""
    started.web_cfg = started.web_cfg.__class__(
        **{**started.web_cfg.__dict__, "team_names": ("Griffin", "Echo")})
    names = started.team_names()
    assert names[:2] == ["Griffin", "Echo"]
    assert len(names) == 12 and names[2] == "Team 3"


# ------------------------------------------------- archive vs delete
async def test_archive_keeps_the_log_recoverable(client, started):
    await client.post("/api/picks", json={"player_id": "rb-00"})
    body = (await client.delete("/api/session")).json()
    assert body["status"] == "archived"
    from pathlib import Path
    assert Path(body["path"]).exists()


async def test_purge_removes_the_log(client, started):
    """Irreversible, which is why it is a separate request rather than what the
    obvious button does."""

    await client.post("/api/picks", json={"player_id": "rb-00"})
    live = started.web_cfg.resolved(started.web_cfg.session_path)
    assert live.exists()
    body = (await client.delete("/api/session?purge=true")).json()
    assert body["status"] == "deleted"
    assert body["path"] is None
    assert not live.exists()
    assert not list(live.parent.glob("*.bak")), "purge must not leave a copy"


async def test_neither_archive_nor_purge_touches_the_ledger(client, started):
    """The ledger is the record of what the engine recommended. Clearing a
    screen must not destroy it."""
    from src.app.cockpit.ledger import DecisionLedger

    ledger_path = started.web_cfg.resolved(started.web_cfg.ledger_path)
    await client.post("/api/picks", json={"player_id": "rb-00"})
    await client.delete("/api/session?purge=true")
    assert ledger_path.exists()
    led = DecisionLedger(ledger_path)
    try:
        assert led.verify().ok
    finally:
        led.close()


async def test_clearing_frees_the_seat_for_a_new_draft(client):
    await client.delete("/api/session?purge=true")
    resp = await client.post("/api/session",
                             json={"seat": 5, "source": "manual"})
    assert resp.status_code == 200
    assert resp.json()["seat"] == 5
