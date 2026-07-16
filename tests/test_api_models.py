"""DraftSession persists on SQLite (portable types only) and history round-trips."""
from api.db_models import DraftSession


def test_session_roundtrip_with_history(db_session):
    s = DraftSession(season=2026, draft_position=4)
    db_session.add(s)
    db_session.commit()
    assert s.session_id and len(s.session_id) == 32
    assert s.platform == "manual"
    assert s.status == "active"
    assert s.history == []

    s.history = s.history + [[["pick", "P0001", True]]]
    db_session.commit()
    db_session.expire_all()

    loaded = db_session.get(DraftSession, s.session_id)
    assert loaded.history == [[["pick", "P0001", True]]]
    assert loaded.created_at is not None
