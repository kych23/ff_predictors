"""Yahoo OAuth login/callback — connects a draft session to a Yahoo league.

See api/yahoo_oauth.py's module docstring for the invalid_scope/access-gate
findings from the 2026-08-03 feasibility spike (docs/superpowers/plans/
2026-07-07-yahoo-oauth-live-sync.md).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from api import yahoo_oauth
from api.crypto import encrypt
from api.db_models import DraftSession, YahooConnection
from api.deps import get_db
from api.settings import frontend_url

router = APIRouter(prefix="/auth/yahoo")


@router.get("/login")
def login(session_id: str, league_key: str):
    state = f"{session_id}:{league_key}"
    return RedirectResponse(yahoo_oauth.authorize_url(state), status_code=307)


def _redirect(session_id: str | None, connect_error: str | None = None,
             connected: bool = False) -> RedirectResponse:
    params = []
    if session_id:
        params.append(f"session={session_id}")
    if connect_error:
        params.append(f"connect_error={connect_error}")
    if connected:
        params.append("connected=1")
    qs = "&".join(params)
    return RedirectResponse(f"{frontend_url()}/draft?{qs}", status_code=307)


@router.get("/callback")
def callback(code: str, state: str, db: Session = Depends(get_db)):
    parts = state.split(":", 1)
    if len(parts) != 2:
        return _redirect(None, connect_error="invalid_request")
    session_id, league_key = parts
    sess = db.get(DraftSession, session_id)
    if sess is None:
        return _redirect(None, connect_error="invalid_request")

    try:
        tokens = yahoo_oauth.exchange_code(code)
    except Exception:
        return _redirect(session_id, connect_error="oauth_failed")
    # Yahoo's OAuth2 spec allows omitting refresh_token on a repeat
    # authorization without an explicit offline-access prompt; treat any
    # missing required field the same as an exchange failure rather than
    # letting a KeyError below turn into a raw 500 mid-redirect.
    if not all(k in tokens for k in ("access_token", "refresh_token", "expires_in")):
        return _redirect(session_id, connect_error="oauth_failed")

    try:
        team_key = yahoo_oauth.fetch_my_team_key(tokens["access_token"], league_key)
    except yahoo_oauth.YahooUpstreamError:
        return _redirect(session_id, connect_error="oauth_failed")
    if team_key is None:
        return _redirect(session_id, connect_error="team_not_found")

    conn = db.query(YahooConnection).filter_by(session_id=session_id).first()
    if conn is None:
        conn = YahooConnection(session_id=session_id)
        db.add(conn)
    conn.league_key = league_key
    conn.team_key = team_key
    # Yahoo's actual token response (verified live, 2026-08-03) carries no
    # `xoauth_yahoo_guid` field despite older docs/examples assuming one;
    # fall back to the id_token's `sub` claim's presence as a placeholder —
    # this field is provenance-only, never used for `mine` resolution
    # (team_key match handles that), so an inexact value here is non-fatal.
    conn.yahoo_guid = tokens.get("xoauth_yahoo_guid") or (tokens.get("id_token") or "unknown")[:64]
    conn.access_token_enc = encrypt(tokens["access_token"])
    conn.refresh_token_enc = encrypt(tokens["refresh_token"])
    import datetime as dt
    conn.expires_at = dt.datetime.utcnow() + dt.timedelta(seconds=tokens["expires_in"])

    sess.platform = "yahoo"
    db.commit()
    return _redirect(session_id, connected=True)
