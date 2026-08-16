"""The rankings HTTP surface.

Driven through `httpx.ASGITransport` with `app.state` injected, matching
`test_web_api.py`. Starlette's sync `TestClient` would trigger `lifespan` ->
`build_service()` -> `Board.from_bundle()`, which raises on any clone without a
bundle — and it would not exercise the real async path these routes run on.

The point of the feature is that it works BEFORE draft night, so nothing here
starts a session.
"""
from __future__ import annotations

import json

import httpx
import pytest

from src.app.web import api as api_mod


@pytest.fixture()
async def board_client(rankings_store, rankings_data, service):
    """No session started — a research tool must not require a live draft."""
    api_mod.app.state.service = service
    api_mod.app.state.rankings = rankings_store
    api_mod.app.state.rankings_data = rankings_data
    transport = httpx.ASGITransport(app=api_mod.app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://cockpit") as client:
        yield client
    api_mod.app.state.service = None
    api_mod.app.state.rankings = None
    api_mod.app.state.rankings_data = None


async def _create(client, name="My Board", **body):
    response = await client.post("/api/rankings", json={"name": name, **body})
    assert response.status_code == 201, response.text
    return response.json()


# ----------------------------------------------------------- independence
async def test_rankings_routes_do_not_require_an_active_session(board_client):
    """The cockpit 404s without a session. This must not — the whole feature
    is pre-draft research."""
    assert (await board_client.get("/api/session")).status_code == 404
    assert (await board_client.get("/api/rankings")).status_code == 200


# --------------------------------------------------------------- lifecycle
async def test_the_full_lifecycle_preserves_the_users_order(board_client):
    board = await _create(board_client, seed_method="adp")
    board_id = board["board_id"]

    scopes = board["scopes"]
    first = scopes["overall"]["tiers"][0]["player_ids"]
    moved = [first[1], first[0], *first[2:]]
    scopes["overall"]["tiers"][0]["player_ids"] = moved

    response = await board_client.put(
        f"/api/rankings/{board_id}",
        json={"expected_rev": board["rev"], "scopes": scopes})
    assert response.status_code == 200, response.text
    assert response.json()["scopes"]["overall"]["tiers"][0]["player_ids"] \
        == moved

    fetched = (await board_client.get(f"/api/rankings/{board_id}")).json()
    assert fetched["scopes"]["overall"]["tiers"][0]["player_ids"] == moved

    assert (await board_client.delete(
        f"/api/rankings/{board_id}")).status_code == 200
    assert (await board_client.get(
        f"/api/rankings/{board_id}")).status_code == 404


async def test_a_board_can_be_created_with_no_seed_at_all(board_client):
    board = await _create(board_client, name="Blank")
    assert all(scope["tiers"] == [] for scope in board["scopes"].values())


async def test_seed_and_scopes_together_are_refused(board_client):
    response = await board_client.post(
        "/api/rankings",
        json={"name": "X", "seed_method": "adp", "scopes": {}})
    assert response.status_code == 400


async def test_a_colliding_name_is_409(board_client):
    """409, not 400: the request was well-formed, it lost to an existing name.
    The client shows a retype prompt for one and a bug report for the other."""
    await _create(board_client, name="Dup")
    response = await board_client.post("/api/rankings", json={"name": "dup!"})
    assert response.status_code == 409
    assert "already exists" in response.text


# --------------------------------------------------------------- conflicts
async def test_a_stale_rev_put_is_409_and_returns_the_current_board(
        board_client):
    board = await _create(board_client, seed_method="adp")
    board_id = board["board_id"]

    first = await board_client.put(
        f"/api/rankings/{board_id}",
        json={"expected_rev": 1, "scopes": board["scopes"]})
    assert first.status_code == 200
    assert first.json()["rev"] == 2

    stale = await board_client.put(
        f"/api/rankings/{board_id}",
        json={"expected_rev": 1, "scopes": board["scopes"]})
    assert stale.status_code == 409
    detail = stale.json()["detail"]
    assert detail["error"] == "rev_conflict"
    assert detail["board"]["rev"] == 2, "the client cannot resolve what it " \
                                        "cannot see"


async def test_the_flush_alias_writes_like_a_put(board_client):
    """`beforeunload` cannot issue a PUT with sendBeacon; the POST alias is
    what keeps the last edit."""
    board = await _create(board_client, seed_method="adp")
    response = await board_client.post(
        f"/api/rankings/{board['board_id']}/flush",
        json={"expected_rev": 1, "scopes": board["scopes"]})
    assert response.status_code == 200
    assert response.json()["rev"] == 2


# ------------------------------------------------------------------ seeding
async def test_seeding_a_scope_replaces_only_that_scope(board_client):
    board = await _create(board_client, seed_method="adp")
    board_id = board["board_id"]
    before = board["scopes"]["overall"]["tiers"]

    response = await board_client.post(
        f"/api/rankings/{board_id}/seed",
        json={"scope": "WR", "method": "engine_tiers", "expected_rev": 1})
    assert response.status_code == 200, response.text
    after = response.json()
    assert after["scopes"]["WR"]["tiers"], "WR was not seeded"
    assert after["scopes"]["overall"]["tiers"] == before, "overall moved"


async def test_an_illegal_scope_method_pair_is_400_with_a_reason(board_client):
    board = await _create(board_client)
    response = await board_client.post(
        f"/api/rankings/{board['board_id']}/seed",
        json={"scope": "overall", "method": "engine_value", "expected_rev": 1})
    assert response.status_code == 400
    assert "quarterbacks" in response.text


# -------------------------------------------------------------- containment
@pytest.mark.parametrize("hostile", ["..", "A" * 80, "has space", "UPPER"])
async def test_a_hostile_board_id_never_reaches_the_filesystem(board_client,
                                                               hostile):
    response = await board_client.get(f"/api/rankings/{hostile}")
    assert response.status_code in (400, 404)


# ------------------------------------------------------------------- stale
async def test_put_preserves_a_player_absent_from_the_bundle(board_client):
    """A rebuilt bundle must not silently delete rows from saved research."""
    board = await _create(board_client, seed_method="adp")
    board_id = board["board_id"]
    scopes = board["scopes"]
    scopes["overall"]["tiers"][0]["player_ids"].append("ghost-99")

    saved = await board_client.put(
        f"/api/rankings/{board_id}",
        json={"expected_rev": 1, "scopes": scopes})
    assert saved.status_code == 200
    assert "ghost-99" in saved.json()["scopes"]["overall"]["tiers"][0][
        "player_ids"]
    assert "ghost-99" in saved.json()["stale_player_ids"]


async def test_every_board_response_carries_stale_player_ids(board_client):
    """Attaching it only to GET would make the greyed rows vanish on save."""
    board = await _create(board_client, seed_method="adp")
    board_id = board["board_id"]
    assert "stale_player_ids" in board

    for response in (
        await board_client.get(f"/api/rankings/{board_id}"),
        await board_client.put(f"/api/rankings/{board_id}",
                               json={"expected_rev": 1,
                                     "scopes": board["scopes"]}),
        await board_client.post(f"/api/rankings/{board_id}/seed",
                                json={"scope": "RB", "method": "adp",
                                      "expected_rev": 2}),
    ):
        assert "stale_player_ids" in response.json(), response.text


# ----------------------------------------------------------------- detail
async def test_detail_for_a_dst_has_no_matrix_sections_and_is_not_an_error(
        board_client, rankings_data):
    dst = rankings_data.board[rankings_data.board["position"] == "DST"]
    response = await board_client.get(
        f"/api/players/{dst.iloc[0]['player_id']}/detail")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["has_matrix_row"] is False
    assert payload["production"] is None and payload["role"] is None
    assert payload["projection"]["p10"] is None, "K/DST have no band"
    assert payload["projection"]["value"] is not None


async def test_detail_is_json_serializable_for_every_shape(board_client,
                                                           rankings_data):
    """NaN and numpy scalars are the crash. `json.dumps(np.int64(14))` raises
    TypeError, and Starlette renders with allow_nan=False, so a leak is a 500."""
    board = rankings_data.board
    sample = [board[board["position"] == pos].iloc[0]["player_id"]
              for pos in ("K", "DST", "WR", "QB")]
    for player_id in sample:
        response = await board_client.get(f"/api/players/{player_id}/detail")
        assert response.status_code == 200, response.text
        json.loads(json.dumps(response.json()))          # no NaN, no numpy


async def test_detail_never_serves_the_training_label(board_client,
                                                      rankings_data):
    """`fppg` in the target-season row is the LABEL. Null today; real leakage
    the moment in-season data lands."""
    wr = rankings_data.board[rankings_data.board["position"] == "WR"]
    payload = (await board_client.get(
        f"/api/players/{wr.iloc[0]['player_id']}/detail")).json()
    assert "fppg" not in payload["production"]
    assert "prior_fppg" in payload["production"]
    assert json.dumps(payload).count('"fppg"') == 0


async def test_detail_carries_the_market_and_the_engine_tier(board_client,
                                                             rankings_data):
    wr = rankings_data.board[rankings_data.board["position"] == "WR"]
    payload = (await board_client.get(
        f"/api/players/{wr.iloc[0]['player_id']}/detail")).json()
    assert payload["market"]["matched"] is True
    assert payload["market"]["ranks"]["yahoo"] > 0
    # Fractional on purpose — it is an average of platform ranks.
    assert payload["market"]["consensus_rank"] % 1 != 0
    assert payload["engine_position_tier"] in (1, 2, 3)


async def test_a_float_column_that_means_a_bool_is_coerced(board_client,
                                                           rankings_data):
    """`is_projected_starter` is float64 with NaNs in the real matrix."""
    modeled = rankings_data.board[
        rankings_data.board["position"].isin(("QB", "RB", "WR", "TE"))]
    kinds = set()
    for player_id in modeled["player_id"].head(4):
        role = (await board_client.get(
            f"/api/players/{player_id}/detail")).json()["role"]
        kinds.add(type(role["is_projected_starter"]).__name__)
        assert isinstance(role["depth_chart_rank"], int)
    assert kinds <= {"bool", "NoneType"}


async def test_an_unknown_player_is_404(board_client):
    assert (await board_client.get(
        "/api/players/nobody-99/detail")).status_code == 404


# -------------------------------------------------------------- catalogue
async def test_the_catalogue_names_every_board_player(board_client,
                                                      rankings_data):
    """Without this, a 260-row board cannot render a single name."""
    payload = (await board_client.get("/api/rankings/catalogue")).json()
    assert len(payload["players"]) == len(rankings_data.board)
    row = payload["players"][0]
    assert row["name"] and row["position"]
    assert "value" in row and "vor" in row and "adp" in row


async def test_the_catalogue_is_not_shadowed_by_the_board_id_route(
        board_client):
    """`/api/rankings/catalogue` must not be read as board_id='catalogue'."""
    response = await board_client.get("/api/rankings/catalogue")
    assert response.status_code == 200
    assert "players" in response.json()


async def test_the_catalogue_includes_drafted_players(board_client, service,
                                                      rankings_data):
    """`/api/board` filters drafted players out. A research board must not
    lose rows because someone got picked."""
    service.start_session(seat=3, source="manual", resume=False)
    taken = str(rankings_data.board.iloc[0]["player_id"])
    service.board.take(taken)

    payload = (await board_client.get("/api/rankings/catalogue")).json()
    assert taken in {p["player_id"] for p in payload["players"]}
