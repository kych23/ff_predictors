"""FastAPI surface for the draft cockpit (§19).

    venv/bin/uvicorn src.app.web.api:app --port 8000

**Every route is `async def`, and that is a correctness requirement.** Starlette
runs a plain `def` route in a threadpool; that would break the single-writer
rule `service.py` depends on and would use the thread-affine SQLite ledger
connection from the wrong thread. `test_web_api.py` introspects `app.routes` and
fails if a sync route ever appears.

**This deliberately relaxes the offline invariant** that `CLAUDE.md` states for
the terminal cockpit. Worth being precise about what changed: the engine core is
unchanged, the transport is new and networked, and the artifact/covariate load
was *already* networked (`bundle_build.nflverse_covariates` runs on every
tier-0 recommendation). Losing the network degrades this to manual entry, not
to nothing — `POST /api/picks` works whatever the configured source is.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.app.web import schemas
from src.app.web import service as svc
from src.app.web.resolve import AMBIGUOUS
from src.app.web.service import (
    ERROR,
    READY,
    RUNNING,
    CockpitService,
    RunInFlight,
    SessionExists,
)
from src.app.web.sources.base import DraftEvent
from src.core.errors import ConfigError, DataError

logger = logging.getLogger(__name__)

KEEPALIVE_SECONDS = 15.0
# src/app/web/api.py -> parents[3] is the repo root. parents[4] climbs one
# level too far, which silently disabled the static mount: the API kept working
# so nothing failed, and the built frontend was simply never served.
_REPO_ROOT = Path(__file__).resolve().parents[3]
DIST = _REPO_ROOT / "web" / "dist"


class BoardCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    seed_method: str | None = None
    tier_size: int | None = Field(None, ge=2, le=50)
    scopes: dict | None = None


class BoardPutRequest(BaseModel):
    expected_rev: int = Field(ge=1)
    scopes: dict
    name: str | None = Field(None, min_length=1, max_length=80)


class SeedRequest(BaseModel):
    scope: str
    method: str
    expected_rev: int = Field(ge=1)
    tier_size: int | None = Field(None, ge=2, le=50)


class SessionRequest(BaseModel):
    seat: int = Field(..., ge=0, description="0-indexed draft seat")
    source: str | None = None
    resume: bool = False
    archive_id: str | None = Field(
        None, description="with source=replay, which archived draft to replay "
                          "(an id from GET /api/replays). Omitted replays the "
                          "configured fixture.")


class PickRequest(BaseModel):
    raw_name: str | None = None
    player_id: str | None = None
    seat: int | None = Field(None, ge=0)
    force_unresolved: bool = False


def build_service() -> CockpitService:
    from src.core.config import load_league, load_strategy
    from src.core.config.web import load_web
    from src.engine.decision.board import Board

    cfg = load_league()
    return CockpitService(cfg=cfg, strategy=load_strategy(cfg),
                          web_cfg=load_web(), board=Board.from_bundle())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Everything is constructed here, never at import — so the ledger's
    connection and the poller task belong to the running loop."""
    try:
        app.state.service = build_service()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{exc}. Build the bundle first: "
            f"venv/bin/python scripts/build_bundle.py --season 2026") from exc
    # "My Board" — research state, no session required, never on the pick
    # clock. Loaded here so the parquet reads and the one manifest write in
    # `rankings_csv.load` happen at startup rather than per request.
    from src.app.rankings.data import RankingsData
    from src.app.rankings.store import RankingsStore

    service = app.state.service
    app.state.rankings = RankingsStore(
        service.web_cfg.resolved(service.web_cfg.rankings_dir))
    try:
        app.state.rankings_data = RankingsData.load(
            service.cfg, service.strategy, service.web_cfg,
            service.board.players, service.board.snapshot_id,
            root=service.web_cfg.resolved(Path("data/artifacts")))
    except Exception:                                      # noqa: BLE001
        # A malformed rankings export or a truncated parquet must never take
        # the cockpit down on draft night. `board=` is REQUIRED: it is an
        # argument rather than a loaded artifact, so it cannot be what failed,
        # and dropping it would report every player on a saved board as stale.
        logger.exception("rankings data unavailable; serving degraded")
        app.state.rankings_data = RankingsData.empty(
            board=service.board.players)

    app.state.rankings.render = lambda board: schemas.board_payload(
        board, app.state.rankings_data)

    app.state.poller = asyncio.create_task(_poll_loop(app))
    try:
        yield
    finally:
        app.state.poller.cancel()
        try:
            await app.state.poller
        except asyncio.CancelledError:
            pass
        app.state.service.close()


app = FastAPI(title="FantasyForecast draft cockpit", lifespan=lifespan)


def _service(request_app: FastAPI | None = None) -> CockpitService:
    service = getattr(app.state, "service", None)
    if service is None:                              # pragma: no cover
        raise HTTPException(503, "service not started")
    return service


def _require_session(service: CockpitService):
    if service.session is None:
        raise HTTPException(404, "no active session")
    return service.session


@app.get("/api/league")
async def league() -> dict:
    """What the setup screen needs BEFORE a session exists.

    `GET /api/session` 404s until one is started, so without this the client
    cannot render a seat picker — it has no idea how many seats there are. It
    also reports whether a session file is already on disk, which is what makes
    "resume" an informed choice rather than a coin flip against a 409.
    """
    service = _service()
    path = service.web_cfg.resolved(service.web_cfg.session_path)
    from src.core.config.web import SOURCES

    return {
        "teams": service.cfg.teams,
        "rounds": service.cfg.roster.rounds,
        "snapshot_id": service.board.snapshot_id,
        "players": int(len(service.players)),
        "sources": list(SOURCES),
        "default_source": service.web_cfg.source,
        "session_exists": path.exists(),
        "active": service.session is not None,
    }


# ------------------------------------------------------------------ session
@app.post("/api/session")
async def create_session(body: SessionRequest) -> dict:
    service = _service()
    if body.seat >= service.cfg.teams:
        raise HTTPException(
            400, f"seat must be 0..{service.cfg.teams - 1} (0-indexed)")
    source = body.source or service.web_cfg.source
    try:
        service.start_session(seat=body.seat, source=source,
                              resume=body.resume,
                              archive_id=body.archive_id)
    except SessionExists as exc:
        raise HTTPException(409, str(exc)) from exc
    except (DataError, ConfigError) as exc:
        raise HTTPException(400, str(exc)) from exc
    payload = service.session_payload()
    service.publish("state", payload)
    return payload


@app.get("/api/session")
async def get_session() -> dict:
    service = _service()
    _require_session(service)
    return service.session_payload()


@app.delete("/api/session")
async def delete_session(
    purge: bool = Query(False,
                        description="delete the log instead of archiving it"),
) -> dict:
    """Clear the active draft.

    Archiving is the default because it is recoverable. `purge=true` is not,
    which is why it has to be asked for explicitly rather than being what the
    obvious button does. The ledger survives both — it is the record of what
    the engine said, and clearing a screen must not destroy it.
    """
    service = _service()
    archived = service.archive_session(purge=purge)
    return {"status": "deleted" if purge else "archived",
            "path": str(archived) if archived else None}


@app.get("/api/replays")
async def list_replays() -> dict:
    """Archived drafts, newest first — what the Replay picker offers.

    Includes the checked-in fixture as a fallback entry so a fresh install has
    something to replay before any real draft has been archived.
    """
    service = _service()
    archives = svc.list_archives(service.web_cfg)
    fixture = service.web_cfg.replay_path
    return {"replays": archives,
            "fixture": str(fixture) if fixture else None,
            "count": len(archives)}


# -------------------------------------------------------------------- picks
@app.post("/api/picks")
async def record_pick(body: PickRequest) -> Any:
    """Ambiguity is a 200, not an error: the operator has to choose, and a
    status code cannot carry the options."""
    service = _service()
    _require_session(service)
    if not body.raw_name and not body.player_id:
        raise HTTPException(400, "one of raw_name or player_id is required")

    if body.player_id and body.player_id not in set(
            service.players["player_id"].astype(str)):
        raise HTTPException(404, f"{body.player_id} is not on the board")

    if body.raw_name and not body.player_id and not body.force_unresolved:
        resolution = service.resolve(body.raw_name)
        if resolution.status == AMBIGUOUS:
            return {"status": AMBIGUOUS,
                    "candidates": [c.to_dict() for c in resolution.candidates]}
        if resolution.status == "unresolved":
            return {"status": "unresolved", "raw_name": body.raw_name}

    event = DraftEvent(raw_name=body.raw_name or body.player_id or "",
                       seat=body.seat, player_id=body.player_id,
                       source="manual")
    try:
        outcome = service.apply_event(event)
    except DataError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    payload = service.session_payload()
    service.publish("state", payload)
    await _maybe_autorecommend(service)
    return {"status": outcome, **payload}


@app.post("/api/undo")
async def undo() -> dict:
    service = _service()
    _require_session(service)
    try:
        service.undo()
    except DataError as exc:
        raise HTTPException(409, str(exc)) from exc
    payload = service.session_payload()
    service.publish("state", payload)
    return payload


@app.get("/api/board")
async def get_board(limit: int = Query(50, ge=1, le=500)) -> dict:
    return _service().board_payload(limit)


# ----------------------------------------------------------- recommendation
@app.post("/api/recommendation")
async def start_recommendation() -> dict:
    service = _service()
    _require_session(service)
    try:
        await _launch(service)
    except RunInFlight:
        # Not an error condition from the operator's point of view: they asked
        # for a recommendation and one is being computed. Spam-clicking
        # Recommend used to surface a raw 409 body on the draft screen.
        pass
    return {"status": RUNNING}


@app.get("/api/recommendation")
async def get_recommendation() -> dict:
    return _service().recommendation.snapshot()


@app.get("/api/stream")
async def stream():
    service = _service()
    queue = service.subscribe()

    async def gen():
        try:
            while True:
                try:
                    frame = await asyncio.wait_for(queue.get(),
                                                   timeout=KEEPALIVE_SECONDS)
                except TimeoutError:
                    yield {"event": "ping", "data": "{}"}
                    continue
                import json as _json
                yield {"event": frame["event"],
                       "data": _json.dumps(frame["data"], default=str)}
        finally:
            service.unsubscribe(queue)

    return EventSourceResponse(gen())


# ------------------------------------------------------------------ workers
async def _launch(service: CockpitService) -> None:
    if service.recommendation.status == RUNNING:
        raise RunInFlight("a recommendation is already running")
    import time as _time

    generation = service.generation
    service.recommendation = type(service.recommendation)(
        status=RUNNING, started_at=_time.monotonic(), generation=generation,
        payload=service.recommendation.payload)
    args = service.build_run_args()
    service.publish("rec_started", {"generation": generation})
    asyncio.create_task(_run(service, args, generation))


async def _run(service: CockpitService, args: dict, generation: int) -> None:
    loop = asyncio.get_running_loop()
    try:
        rec, record = await loop.run_in_executor(
            None, service.run_recommendation, args)
    except Exception as exc:                          # noqa: BLE001
        # The generation check belongs here too, not only on the success path.
        # `start_session` and `archive_session` replace `service.recommendation`
        # wholesale, so an abandoned run that raises later would otherwise stamp
        # ERROR onto a BRAND NEW session's state and broadcast `rec_error` for a
        # recommendation that never failed.
        if generation != service.generation:
            logger.info("discarding failure from superseded run (gen %s != %s)",
                        generation, service.generation)
            return
        logger.exception("recommendation failed")
        service.recommendation.status = ERROR
        service.recommendation.detail = f"{type(exc).__name__}: {exc}"
        service.publish("rec_error", {"detail": service.recommendation.detail,
                                      "previous_is_stale": True})
        return

    if generation != service.generation:
        # Superseded by a pick or an undo. `allocate` has no cancellation
        # token, so the run was abandoned rather than stopped.
        logger.info("discarding superseded recommendation (gen %s != %s)",
                    generation, service.generation)
        service.recommendation.status = IDLE_IF_STALE
        if service.session is not None and service.session.is_my_turn():
            await _launch(service)
        return

    payload = service.payload_for(rec, generation)
    service.recommendation.status = READY
    service.recommendation.payload = payload
    service.publish("rec_ready", payload)

    if record is not None:
        asyncio.create_task(_narrate(service, record, generation))


IDLE_IF_STALE = "idle"


async def _narrate(service: CockpitService, record, generation: int) -> None:
    """Separate from the run so a slow model can never sit on the pick clock."""
    loop = asyncio.get_running_loop()
    try:
        narration = await asyncio.wait_for(
            loop.run_in_executor(None, service.narrate, record),
            timeout=service.web_cfg.narration_timeout_seconds + 2.0)
    except Exception as exc:                          # noqa: BLE001
        logger.info("narration unavailable: %s", exc)
        return
    if generation != service.generation:
        return
    body = schemas.narration_payload(narration)
    if service.recommendation.payload is not None:
        service.recommendation.payload["narration"] = body
    service.publish("narration_ready",
                    {"generation": generation, "narration": body})


async def _maybe_autorecommend(service: CockpitService) -> None:
    if not service.web_cfg.auto_recommend or service.session is None:
        return
    if service.source is not None and service.source.name == "replay" \
            and not service.web_cfg.replay_auto_recommend:
        return
    if service.session.is_my_turn() and service.recommendation.status != RUNNING:
        try:
            await _launch(service)
        except RunInFlight:
            pass


async def _poll_loop(application: FastAPI) -> None:
    """Supervised: a poller that dies silently stops the draft feed with no
    visible symptom, so failures are reported and retried with backoff."""
    backoff = 1.0
    while True:
        service = getattr(application.state, "service", None)
        interval = (service.web_cfg.poll_interval_seconds if service else 3.0)
        try:
            if service is not None and service.source is not None \
                    and service.session is not None:
                # OFF the loop thread. YahooSource.poll makes a synchronous
                # 10s-timeout HTTP call; running it inline would stall every
                # other request — including picks — for the duration. Ships
                # disabled today, but this is live code the moment credentials
                # land. Applying the events stays ON the loop (single writer).
                loop = asyncio.get_running_loop()
                events = await loop.run_in_executor(None, service.source.poll)
                for event in events:
                    try:
                        service.apply_event(event)
                    except DataError as exc:
                        # A duplicate the operator already typed. Drop it; a
                        # feed disagreement must never be fatal.
                        logger.info("dropping source event: %s", exc)
                        continue
                    service.publish("state", service.session_payload())
                    await _maybe_autorecommend(service)
                service.publish("source_status",
                                {"name": service.source.name,
                                 "state": service.source.status.state,
                                 "detail": service.source.status.detail})
            backoff = 1.0
        except asyncio.CancelledError:
            raise
        except Exception as exc:                      # noqa: BLE001
            logger.exception("poller error")
            backoff = min(30.0, backoff * 2)
            if service is not None:
                service.publish("source_status",
                                {"name": "poller", "state": "degraded",
                                 "detail": str(exc)})
            await asyncio.sleep(backoff)
            continue
        await asyncio.sleep(interval)


# --------------------------------------------------------------- rankings
# "My Board": the user's own tier rankings. Standalone by design — nothing here
# reaches a recommendation, so no route below may import from the decision path.
#
# EVERY ROUTE IN THIS SECTION MUST STAY ABOVE THE STATIC MOUNT BELOW.
# `Mount("/")` matches every path and Starlette takes the first match in
# registration order, so a route added after it silently 404s wherever
# `web/dist` exists — which is a built laptop, not CI.
# `test_rankings_routes_are_reachable.py` guards this.
def _rankings(request_app: FastAPI | None = None):
    store = getattr(request_app or app, "state", None)
    store = getattr(store, "rankings", None)
    if store is None:
        raise HTTPException(503, "rankings store unavailable")
    return store


def _rankings_data(request_app: FastAPI | None = None):
    data = getattr(request_app or app, "state", None)
    data = getattr(data, "rankings_data", None)
    if data is None:
        raise HTTPException(503, "rankings data unavailable")
    return data


def _rankings_error(exc: Exception) -> HTTPException:
    from src.app.rankings.store import RevConflict, SlugExists

    if isinstance(exc, SlugExists):
        return HTTPException(409, str(exc))
    if isinstance(exc, RevConflict):
        # `exc.payload` was built inside the worker thread that raised, because
        # board_payload scans the whole bundle to compute stale ids and this is
        # the CONFLICT path — it fires exactly when the loop is contended.
        return HTTPException(409, {
            "error": "rev_conflict", "board": exc.payload,
        })
    return HTTPException(400, str(exc))


async def _board_payload(board) -> dict:
    return await asyncio.to_thread(
        schemas.board_payload, board, _rankings_data())


@app.get("/api/rankings")
async def list_boards() -> dict:
    store = _rankings()
    return {"boards": await asyncio.to_thread(store.list)}


@app.post("/api/rankings", status_code=201)
async def create_board(body: BoardCreateRequest) -> dict:
    from src.app.rankings import seed as seed_mod
    from src.app.rankings.schema import RankingsError, empty_scopes, parse

    store, data = _rankings(), _rankings_data()
    # `is not None`, not truthiness: `scopes={}` is a legitimate "I supplied
    # scopes" that happens to be empty, and reading it as "no scopes" would
    # silently seed over the caller's explicit choice.
    if body.seed_method and body.scopes is not None:
        raise HTTPException(
            400, "pass seed_method or scopes, not both")

    def _build():
        scopes = empty_scopes()
        seeded: dict[str, Any] = {"bundle_snapshot_id":
                                  data.snapshots.get("bundle", "")}
        if body.scopes is not None:
            parsed = parse({"board_id": "tmp", "name": body.name,
                            "scopes": body.scopes})
            scopes = dict(parsed.scopes)
        elif body.seed_method:
            # Only `overall` is seeded at creation: it is the scope you start
            # from, and seeding all seven would do six people's work uninvited.
            scopes["overall"] = seed_mod.seed_scope(
                data, "overall", body.seed_method, tier_size=body.tier_size)
            seeded["method"] = body.seed_method
        return store.create(body.name, scopes, seeded)

    try:
        board = await asyncio.to_thread(_build)
    except RankingsError as exc:
        raise _rankings_error(exc) from exc
    return await _board_payload(board)


@app.get("/api/rankings/catalogue")
async def board_catalogue() -> dict:
    """Every bundle player, so `PlayerRow` can name the ids a board stores.

    Registered before `/api/rankings/{board_id}` — FastAPI matches in order and
    a path parameter would otherwise swallow the literal `catalogue`.
    """
    from src.app.rankings.detail import catalogue

    data = _rankings_data()
    players = await asyncio.to_thread(catalogue, data)
    return {"players": players, "target_season": data.target_season,
            "snapshots": dict(data.snapshots)}


@app.get("/api/rankings/{board_id}")
async def get_ranking_board(board_id: str) -> dict:
    from src.app.rankings.schema import RankingsError

    store = _rankings()
    try:
        board = await asyncio.to_thread(store.load, board_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"no board {board_id!r}") from exc
    except RankingsError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        # A corrupt file the client cannot influence is a SERVER fault; 422 is
        # FastAPI's code for request validation and would misdirect the reader.
        raise HTTPException(500, f"board {board_id!r} is unreadable: {exc}") \
            from exc
    return await _board_payload(board)


@app.put("/api/rankings/{board_id}")
async def put_board(board_id: str, body: BoardPutRequest) -> dict:
    return await _write_board(board_id, body)


@app.post("/api/rankings/{board_id}/flush")
async def flush_board(board_id: str, body: BoardPutRequest) -> dict:
    """A POST alias for `PUT`, because `beforeunload` cannot issue a PUT with
    `sendBeacon` and `fetch(keepalive)` is more reliable against a closing tab
    when the method matches what the browser expects to queue."""
    return await _write_board(board_id, body)


async def _write_board(board_id: str, body: BoardPutRequest) -> dict:
    from src.app.rankings.schema import RankingsError, parse, to_json

    store = _rankings()

    def _save():
        current = store.load(board_id)
        # The client owns `name` and `scopes`; everything else is server-owned,
        # so a stale or hostile body cannot rewrite provenance or the revision.
        merged = to_json(current)
        merged["name"] = body.name or current.name
        merged["scopes"] = body.scopes
        return store.save(parse(merged), expected_rev=body.expected_rev)

    try:
        board = await asyncio.to_thread(_save)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"no board {board_id!r}") from exc
    except RankingsError as exc:
        raise _rankings_error(exc) from exc
    return await _board_payload(board)


@app.post("/api/rankings/{board_id}/seed")
async def seed_board(board_id: str, body: SeedRequest) -> dict:
    from src.app.rankings import seed as seed_mod
    from src.app.rankings.schema import RankingsError, Scope

    store, data = _rankings(), _rankings_data()

    def _seed():
        current = store.load(board_id)
        scope = seed_mod.seed_scope(data, body.scope, body.method,
                                    tier_size=body.tier_size, board=current)
        scopes: dict[str, Scope] = {**current.scopes, body.scope: scope}
        from dataclasses import replace as _replace

        return store.save(_replace(current, scopes=scopes),
                          expected_rev=body.expected_rev)

    try:
        board = await asyncio.to_thread(_seed)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"no board {board_id!r}") from exc
    except RankingsError as exc:
        raise _rankings_error(exc) from exc
    return await _board_payload(board)


@app.delete("/api/rankings/{board_id}")
async def delete_board(board_id: str) -> dict:
    from src.app.rankings.schema import RankingsError

    store = _rankings()
    try:
        await asyncio.to_thread(store.delete, board_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"no board {board_id!r}") from exc
    except RankingsError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"deleted": True}


@app.get("/api/players/{player_id}/detail")
async def player_detail(player_id: str) -> dict:
    from src.app.rankings.detail import player_detail as build

    data = _rankings_data()
    payload = await asyncio.to_thread(build, player_id, data)
    if payload is None:
        raise HTTPException(404, f"{player_id!r} is not on the board")
    return payload


# The built frontend, when present. Mounted last so it cannot shadow /api.
if DIST.exists():                                     # pragma: no cover
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(DIST), html=True), name="web")
else:
    @app.get("/")
    async def no_frontend() -> JSONResponse:
        return JSONResponse({
            "detail": "frontend not built; run `cd web && npm install && "
                      "npm run build`, or use the API directly",
            "api": "/docs",
        })
