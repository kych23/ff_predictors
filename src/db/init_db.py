from sqlalchemy import text

from .models import Base
from .session import engine


def init_db() -> None:
    """Create tables and indexes if they don't exist (idempotent)."""
    Base.metadata.create_all(bind=engine)

    with engine.begin() as conn:
        conn.execute(text("SELECT 1"))


