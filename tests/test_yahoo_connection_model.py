"""YahooConnection row roundtrip (api/db_models.py)."""
import datetime as dt

from api.db_models import YahooConnection


def test_yahoo_connection_roundtrip(db_session):
    row = YahooConnection(
        session_id="s1",
        yahoo_guid="guid1",
        league_key="461.l.123456",
        team_key="461.l.123456.t.3",
        access_token_enc=b"enc-access",
        refresh_token_enc=b"enc-refresh",
        expires_at=dt.datetime(2026, 1, 1),
    )
    db_session.add(row)
    db_session.commit()

    fetched = db_session.query(YahooConnection).filter_by(session_id="s1").one()
    assert fetched.connection_id
    assert fetched.league_key == "461.l.123456"
    assert fetched.team_key == "461.l.123456.t.3"
    assert fetched.access_token_enc == b"enc-access"
    assert fetched.refresh_token_enc == b"enc-refresh"


def test_session_id_is_unique(db_session):
    db_session.add(YahooConnection(
        session_id="dup", yahoo_guid="g", league_key="461.l.1",
        access_token_enc=b"a", refresh_token_enc=b"r",
        expires_at=dt.datetime(2026, 1, 1),
    ))
    db_session.commit()
    db_session.add(YahooConnection(
        session_id="dup", yahoo_guid="g2", league_key="461.l.2",
        access_token_enc=b"a2", refresh_token_enc=b"r2",
        expires_at=dt.datetime(2026, 1, 1),
    ))
    import pytest
    from sqlalchemy.exc import IntegrityError
    with pytest.raises(IntegrityError):
        db_session.commit()
