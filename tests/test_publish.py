"""publish_serving_core copies the serving subset research -> serving and leaves
research untouched. Two SQLite engines stand in for the two Postgres DBs; only
portable-typed tables are exercised (JSONB tables are Postgres-only and skipped
by the has_table guard)."""
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from src.db.models import Adp, Player, Projection, WeeklyProjection
from src.db.publish import publish_serving_core

PORTABLE = [Player.__table__, Projection.__table__, Adp.__table__,
            WeeklyProjection.__table__]


def _engine():
    eng = create_engine("sqlite:///:memory:", future=True)
    for t in PORTABLE:
        t.create(eng)
    return eng


def _seed(session):
    session.add(Player(player_id="P1", name="Star", position="RB", latest_team="SF"))
    session.add(Projection(player_id="P1", season=2026, model_version="draft_v1.x",
                           position="RB", p10=1.0, p50=2.0, p90=3.0, snapshot_id="s"))
    session.add(Projection(player_id="P1", season=2024, model_version="draft_v1.x",
                           position="RB", p10=4.0, p50=5.0, p90=6.0, snapshot_id="s"))
    session.add(Adp(source="ffc", season=2024, format="ppr", teams=12, player_id="P1",
                    adp=5.0, snapshot_id="s"))
    session.add(WeeklyProjection(player_id="P1", season=2026, week=1,
                                 model_version="weekly_fwd_v1.x", position="RB",
                                 p10=1.0, p50=2.0, p90=3.0, snapshot_id="s"))
    session.commit()


def test_publish_copies_serving_subset():
    research_eng, serving_eng = _engine(), _engine()
    RS = sessionmaker(bind=research_eng, future=True)
    SS = sessionmaker(bind=serving_eng, future=True)
    research, serving = RS(), SS()
    _seed(research)

    counts = publish_serving_core(research, serving)
    serving.commit()

    # both projection seasons (demo 2024 + live 2026) land on serving
    assert counts["projections"] == 2
    assert counts["players"] == 1
    assert counts["adp"] == 1
    assert counts["weekly_projections"] == 1
    assert serving.execute(select(func.count()).select_from(Projection.__table__)).scalar() == 2
    assert serving.execute(select(func.count()).select_from(Player.__table__)).scalar() == 1
    # research untouched
    assert research.execute(select(func.count()).select_from(Projection.__table__)).scalar() == 2


def test_publish_is_idempotent():
    research_eng, serving_eng = _engine(), _engine()
    RS = sessionmaker(bind=research_eng, future=True)
    SS = sessionmaker(bind=serving_eng, future=True)
    research, serving = RS(), SS()
    _seed(research)
    publish_serving_core(research, serving); serving.commit()
    publish_serving_core(research, serving); serving.commit()
    assert serving.execute(select(func.count()).select_from(Projection.__table__)).scalar() == 2
