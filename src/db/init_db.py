from sqlalchemy import text

from .models import Base
from .session import get_engine


def init_db() -> None:
    """Create tables and indexes if they don't exist (idempotent).

    Product tables (api.db_models) share this Base but live in the api package;
    import them here so create_all provisions them too. Lazy/local import keeps
    the src -> api dependency out of module load (api imports src, not the
    reverse) while still making this the single schema-provisioning entry point.
    """
    engine = get_engine()
    try:
        import api.db_models  # noqa: F401 — registers DraftSession on Base.metadata
    except Exception:
        pass  # api package optional in bare research/pipeline environments
    Base.metadata.create_all(bind=engine)

    with engine.begin() as conn:
        conn.execute(text("SELECT 1"))


