"""Yahoo OAuth login/callback routes + make_token_provider (all Yahoo HTTP
calls mocked — no live network, no dependency on the pending API-access
approval). See docs/superpowers/plans/2026-07-07-yahoo-oauth-live-sync.md
for the live feasibility-spike findings that shaped this design."""
from __future__ import annotations

import datetime as dt

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from api import yahoo_oauth
from api.db_models import DraftSession, YahooConnection
from api.deps import get_board_for, get_db, get_snapshot_id
from api.main import create_app
from tests.test_api_replay import make_board


@pytest.fixture(autouse=True)
def _token_encryption_key(monkeypatch):
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())


@pytest.fixture()
def client(db_session):
    app = create_app()
    board = make_board()
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_board_for] = lambda: (lambda season: board)
    app.dependency_overrides[get_snapshot_id] = lambda: None
    return TestClient(app)


def _seed_session(db_session, session_id="sess123") -> DraftSession:
    sess = DraftSession(session_id=session_id, season=2026, draft_position=2,
                        platform="manual")
    db_session.add(sess)
    db_session.commit()
    return sess


def test_login_redirects_to_yahoo_with_state(client, monkeypatch):
    monkeypatch.setattr(yahoo_oauth, "authorize_url",
                        lambda state: f"https://yahoo.example/authorize?state={state}")
    r = client.get("/auth/yahoo/login", params={"session_id": "s1", "league_key": "461.l.1"},
                   follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "https://yahoo.example/authorize?state=s1:461.l.1"


def test_callback_stores_encrypted_connection_and_sets_platform(client, monkeypatch, db_session):
    _seed_session(db_session, "sess123")
    monkeypatch.setattr(yahoo_oauth, "exchange_code", lambda code: {
        "access_token": "at", "refresh_token": "rt", "expires_in": 3600,
    })
    monkeypatch.setattr(yahoo_oauth, "fetch_my_team_key",
                        lambda access_token, league_key: f"{league_key}.t.3")

    r = client.get("/auth/yahoo/callback", params={"code": "abc", "state": "sess123:461.l.1"},
                   follow_redirects=False)
    assert r.status_code == 307
    assert "connected=1" in r.headers["location"]
    assert "session=sess123" in r.headers["location"]

    conn = db_session.query(YahooConnection).filter_by(session_id="sess123").one()
    assert conn.league_key == "461.l.1"
    assert conn.team_key == "461.l.1.t.3"
    assert conn.access_token_enc != b"at"  # actually encrypted, not stored raw
    assert conn.access_token_enc  # non-empty ciphertext

    sess = db_session.get(DraftSession, "sess123")
    assert sess.platform == "yahoo"


def test_callback_handles_null_id_token(client, monkeypatch, db_session):
    """Yahoo's real token response omits xoauth_yahoo_guid; the id_token
    fallback must tolerate id_token being explicitly None, not just absent."""
    _seed_session(db_session, "sess123")
    monkeypatch.setattr(yahoo_oauth, "exchange_code", lambda code: {
        "access_token": "at", "refresh_token": "rt", "expires_in": 3600,
        "id_token": None,
    })
    monkeypatch.setattr(yahoo_oauth, "fetch_my_team_key",
                        lambda access_token, league_key: f"{league_key}.t.3")

    r = client.get("/auth/yahoo/callback", params={"code": "abc", "state": "sess123:461.l.1"},
                   follow_redirects=False)
    assert r.status_code == 307
    assert "connected=1" in r.headers["location"]
    conn = db_session.query(YahooConnection).filter_by(session_id="sess123").one()
    assert conn.yahoo_guid == "unknown"


def test_callback_repeat_updates_not_duplicates(client, monkeypatch, db_session):
    _seed_session(db_session, "sess123")
    monkeypatch.setattr(yahoo_oauth, "exchange_code", lambda code: {
        "access_token": "at1", "refresh_token": "rt1", "expires_in": 3600,
    })
    monkeypatch.setattr(yahoo_oauth, "fetch_my_team_key",
                        lambda access_token, league_key: f"{league_key}.t.3")
    client.get("/auth/yahoo/callback", params={"code": "abc", "state": "sess123:461.l.1"})

    monkeypatch.setattr(yahoo_oauth, "exchange_code", lambda code: {
        "access_token": "at2", "refresh_token": "rt2", "expires_in": 3600,
    })
    client.get("/auth/yahoo/callback", params={"code": "def", "state": "sess123:461.l.1"})

    rows = db_session.query(YahooConnection).filter_by(session_id="sess123").all()
    assert len(rows) == 1


def test_callback_team_not_found(client, monkeypatch, db_session):
    _seed_session(db_session, "sess123")
    monkeypatch.setattr(yahoo_oauth, "exchange_code", lambda code: {
        "access_token": "at", "refresh_token": "rt", "expires_in": 3600,
    })
    monkeypatch.setattr(yahoo_oauth, "fetch_my_team_key",
                        lambda access_token, league_key: None)

    r = client.get("/auth/yahoo/callback", params={"code": "abc", "state": "sess123:461.l.1"},
                   follow_redirects=False)
    assert "connect_error=team_not_found" in r.headers["location"]
    assert db_session.query(YahooConnection).filter_by(session_id="sess123").first() is None


def test_callback_exchange_failure(client, monkeypatch, db_session):
    _seed_session(db_session, "sess123")

    def _raise(code):
        raise RuntimeError("bad code")
    monkeypatch.setattr(yahoo_oauth, "exchange_code", _raise)

    r = client.get("/auth/yahoo/callback", params={"code": "abc", "state": "sess123:461.l.1"},
                   follow_redirects=False)
    assert "connect_error=oauth_failed" in r.headers["location"]


def test_callback_missing_refresh_token_treated_as_oauth_failed(client, monkeypatch, db_session):
    """Yahoo can omit refresh_token on a repeat authorization without an
    offline-access prompt — must not KeyError, must redirect oauth_failed."""
    _seed_session(db_session, "sess123")
    monkeypatch.setattr(yahoo_oauth, "exchange_code", lambda code: {
        "access_token": "at", "expires_in": 3600,  # no refresh_token
    })
    r = client.get("/auth/yahoo/callback", params={"code": "abc", "state": "sess123:461.l.1"},
                   follow_redirects=False)
    assert "connect_error=oauth_failed" in r.headers["location"]
    assert db_session.query(YahooConnection).filter_by(session_id="sess123").first() is None


def test_callback_invalid_state(client):
    r = client.get("/auth/yahoo/callback", params={"code": "abc", "state": "not-a-valid-state"},
                   follow_redirects=False)
    assert "connect_error=invalid_request" in r.headers["location"]


def test_callback_unknown_session(client):
    r = client.get("/auth/yahoo/callback", params={"code": "abc", "state": "ghost:461.l.1"},
                   follow_redirects=False)
    assert "connect_error=invalid_request" in r.headers["location"]


def test_make_token_provider_returns_token_without_refresh_when_fresh(monkeypatch):
    conn = YahooConnection(session_id="s", yahoo_guid="g", league_key="461.l.1",
                           access_token_enc=yahoo_oauth.encrypt("live-token"),
                           refresh_token_enc=yahoo_oauth.encrypt("rt"),
                           expires_at=dt.datetime.utcnow() + dt.timedelta(hours=1))
    called = {"refresh": False}
    def _fail_if_called(refresh_token):
        called["refresh"] = True
        raise AssertionError("refresh should not be called for a non-expired token")
    monkeypatch.setattr(yahoo_oauth, "refresh", _fail_if_called)

    token = yahoo_oauth.make_token_provider(conn)()
    assert token == "live-token"
    assert called["refresh"] is False


def test_make_token_provider_refreshes_expired_token(monkeypatch):
    conn = YahooConnection(session_id="s", yahoo_guid="g", league_key="461.l.1",
                           access_token_enc=yahoo_oauth.encrypt("stale-token"),
                           refresh_token_enc=yahoo_oauth.encrypt("old-rt"),
                           expires_at=dt.datetime.utcnow() - dt.timedelta(seconds=1))
    monkeypatch.setattr(yahoo_oauth, "refresh", lambda refresh_token: {
        "access_token": "fresh-token", "refresh_token": "new-rt", "expires_in": 3600,
    })

    token = yahoo_oauth.make_token_provider(conn)()
    assert token == "fresh-token"
    assert yahoo_oauth.decrypt(conn.access_token_enc) == "fresh-token"
    assert conn.expires_at > dt.datetime.utcnow()
