"""Yahoo OAuth 2.0: authorize URL, token exchange/refresh, team-key lookup.

No `scope` param on the authorize URL: Yahoo's authorize endpoint returns
`error=invalid_scope` for the commonly-documented `fspt-r` scope on
self-serve apps (confirmed against a live app, 2026-08-03) — omitting
`scope` entirely is what let the OAuth flow complete. Fantasy Sports data
access itself is gated separately, at the app level, behind Yahoo's manual
API-access application (sports.yahoo.com/developer/access/) — until that's
approved, fetch_my_team_key (and any real draftresults call) returns a 401
`additional_authorization_required` regardless of scope or token validity.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import TYPE_CHECKING, Callable, Optional

import httpx
from authlib.integrations.httpx_client import OAuth2Client

from api.crypto import decrypt, encrypt

if TYPE_CHECKING:
    from api.db_models import YahooConnection

AUTHORIZE_URL = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"
FANTASY_BASE = "https://fantasysports.yahooapis.com/fantasy/v2"


class YahooUpstreamError(Exception):
    """A non-2xx response from Yahoo's own APIs (auth server or Fantasy
    Sports API) — maps to HTTP 502 (api/main.py); distinct from our own
    domain errors, since the failure is upstream, not a client mistake."""


def _env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"{name} environment variable is not set")
    return val


def _client(*, redirect_uri: Optional[str] = None) -> OAuth2Client:
    return OAuth2Client(
        _env("YAHOO_CLIENT_ID"),
        _env("YAHOO_CLIENT_SECRET"),
        redirect_uri=redirect_uri or _env("YAHOO_REDIRECT_URI"),
    )


def authorize_url(state: str) -> str:
    """Yahoo authorize URL for a login redirect. See module docstring for
    why no `scope` param is sent."""
    client = _client()
    uri, _ = client.create_authorization_url(AUTHORIZE_URL, state=state)
    return uri


def exchange_code(code: str) -> dict:
    """Exchange an OAuth authorization code for tokens. Raises whatever
    authlib/httpx raises on a non-2xx token-endpoint response (caller
    catches this — see api/routers/auth_yahoo.py's callback)."""
    return _client().fetch_token(TOKEN_URL, code=code)


def refresh(refresh_token: str) -> dict:
    return _client().refresh_token(TOKEN_URL, refresh_token=refresh_token)


def _extract_team_keys(data: dict) -> list[str]:
    """Yahoo's JSON is deeply nested and numerically-indexed; walk it for
    every `team_key` value rather than hardcoding index paths, which are
    brittle against Yahoo's per-response shape variability."""
    found: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            tk = node.get("team_key")
            if isinstance(tk, str):
                found.append(tk)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    return found


def fetch_my_team_key(access_token: str, league_key: str) -> Optional[str]:
    """The team_key (format '{game_key}.l.{league_id}.t.{team_id}') the
    connected user owns in `league_key`, or None if they own no team there
    (e.g. a league_key typo). Raises YahooUpstreamError on a non-2xx from
    Yahoo (including the current 'app not entitled' 401 — see module
    docstring)."""
    url = f"{FANTASY_BASE}/users;use_login=1/games;game_keys=nfl/teams?format=json"
    resp = httpx.get(url, headers={"Authorization": f"Bearer {access_token}"})
    if resp.status_code != 200:
        raise YahooUpstreamError(
            f"team lookup failed: HTTP {resp.status_code}: {resp.text[:300]}")
    for team_key in _extract_team_keys(resp.json()):
        if team_key.startswith(f"{league_key}."):
            return team_key
    return None


def make_token_provider(conn: "YahooConnection") -> Callable[[], str]:
    """Zero-arg callable returning a live access token for `conn`,
    refreshing (and re-encrypting onto `conn`) if `conn.expires_at` has
    passed. Mutates `conn` in place — the CALLER commits, matching
    api/draft_service.py's reassign-then-commit pattern (e.g.
    api/draft_service.py:127)."""

    def provider() -> str:
        if conn.expires_at <= dt.datetime.utcnow():
            rt = decrypt(conn.refresh_token_enc)
            try:
                tokens = refresh(rt)
            except Exception as exc:
                raise YahooUpstreamError(f"token refresh failed: {exc}") from exc
            conn.access_token_enc = encrypt(tokens["access_token"])
            conn.refresh_token_enc = encrypt(tokens.get("refresh_token", rt))
            conn.expires_at = dt.datetime.utcnow() + dt.timedelta(
                seconds=tokens["expires_in"])
        return decrypt(conn.access_token_enc)

    return provider
