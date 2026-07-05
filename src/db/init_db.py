from sqlalchemy import text

from .models import Base
from .session import get_engine


def init_db() -> None:
    """Create tables and indexes if they don't exist (idempotent)."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)

    with engine.begin() as conn:
        conn.execute(text("SELECT 1"))


