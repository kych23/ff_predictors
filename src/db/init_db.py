from sqlalchemy import text

from .models import Base
from .session import RESEARCH, SERVING, _resolve_url, get_engine

#: Tables that exist ONLY in the research database (§20.2).
#:
#: `api/db_models.py` shares this Base, so `create_all` provisions everything
#: on both engines by default. That is fine for the pipeline tables — an empty
#: table costs nothing and lets a fresh serving DB be published into without a
#: manual bootstrap. It is NOT fine for these: they accumulate a row per
#: simulation run and per source blob, and the serving DB is a 500 MB free
#: tier holding a draft-night subset the API actually reads. Without this set,
#: "research DB only" is a sentence in a design doc rather than a property of
#: the schema.
RESEARCH_ONLY_TABLES = frozenset({
    "blob_manifest",
    "sim_runs",
    "weekly_stats_raw",
    "season_features",
    "weekly_labels",
    "season_labels",
})


def init_db() -> None:
    """Create tables and indexes if they don't exist (idempotent).

    Provisions BOTH engines: the research DB (full schema, pipeline target) and,
    when it is a distinct database, the serving DB (Supabase). They share one
    Base, so create_all covers the full schema on each EXCEPT
    ``RESEARCH_ONLY_TABLES`` — serving won't hold research *data* (that's the
    publish step's job), but having the shared tables present means a fresh
    serving DB is publishable without a manual bootstrap.

    Product tables (api.db_models) share this Base but live in the api package;
    import them so create_all provisions them too. The lazy/local import keeps
    the src -> api dependency out of module load (api imports src, not reverse).

    ``LedgerBase`` is deliberately absent: the decision ledger is local SQLite,
    created by ``app/cockpit/ledger.py`` itself, because the recommendation
    path must not require a network (§19). It reaches the research DB only
    afterwards, through ``scripts/import_ledger.py``.
    """
    try:
        import api.db_models  # noqa: F401 — registers DraftSession on Base.metadata
    except Exception:
        pass  # api package optional in bare research/pipeline environments

    roles = [RESEARCH]
    serving_is_distinct = _resolve_url(SERVING) != _resolve_url(RESEARCH)
    if serving_is_distinct:
        roles.append(SERVING)

    for role in roles:
        engine = get_engine(role)
        if role == SERVING and serving_is_distinct:
            tables = [t for name, t in Base.metadata.tables.items()
                      if name not in RESEARCH_ONLY_TABLES]
            Base.metadata.create_all(bind=engine, tables=tables)
        else:
            Base.metadata.create_all(bind=engine)
        with engine.begin() as conn:
            conn.execute(text("SELECT 1"))
