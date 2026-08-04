from sqlalchemy import text

from .models import Base
from .session import RESEARCH, SERVING, _resolve_url, get_engine


def init_db() -> None:
    """Create tables and indexes if they don't exist (idempotent).

    Provisions BOTH engines: the research DB (full schema, pipeline target) and,
    when it is a distinct database, the serving DB (Supabase). They share one
    Base, so create_all covers the full schema on each — serving simply won't
    hold research *data* (that's the publish step's job), but having the tables
    present means a fresh serving DB is publishable without a manual bootstrap.

    Product tables (api.db_models) share this Base but live in the api package;
    import them so create_all provisions them too. The lazy/local import keeps
    the src -> api dependency out of module load (api imports src, not reverse).
    """
    try:
        import api.db_models  # noqa: F401 — registers DraftSession on Base.metadata
    except Exception:
        pass  # api package optional in bare research/pipeline environments

    roles = [RESEARCH]
    if _resolve_url(SERVING) != _resolve_url(RESEARCH):
        roles.append(SERVING)
    for role in roles:
        engine = get_engine(role)
        Base.metadata.create_all(bind=engine)
        with engine.begin() as conn:
            conn.execute(text("SELECT 1"))
