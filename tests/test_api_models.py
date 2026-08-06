"""DraftSession persists on SQLite (portable types only) and history round-trips."""

import pytest

# §9.4: this module is pinned to the v1 config shape. api/, web/ and the
# weekly start/sit surface are FROZEN for the build window, so v1 stays
# alive underneath them and these tests are deselected from the default
# run rather than migrated. Thaw is a §22.2 follow-up.
pytestmark = pytest.mark.v1_frozen
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
