from __future__ import annotations

from typing import Dict

import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.db.init_db import init_db
from src.db.models import Feature, Label
from src.db.session import SessionLocal
from .clean_data import clean_player_stats


def _chunk_iter(df: pd.DataFrame, size: int = 5_000):
    for start in range(0, len(df), size):
        yield df.iloc[start : start + size]


def persist_priors_to_features(priors: Dict[str, pd.DataFrame]) -> int:
    """Upsert priors into the features table as feature_json per player-week.

    Returns number of rows attempted.
    """
    init_db()
    session: Session = SessionLocal()
    total = 0
    try:
        for df in priors.values():
            if df.empty:
                continue
            prior_cols = [c for c in df.columns if any(s in c for s in ("_avg_", "_ewm"))] + ["gp_prior"]
            id_cols = [c for c in ["player_id", "season", "week", "opponent_team"] if c in df.columns]
            keep = id_cols + prior_cols
            df2 = df[keep].copy()

            records = []
            for _, r in df2.iterrows():
                feature_json = {k: float(r[k]) for k in prior_cols if pd.notna(r[k])}
                records.append({
                    "player_id": r["player_id"],
                    "season": int(r["season"]),
                    "week": int(r["week"]),
                    "opponent_team": r.get("opponent_team", ""),
                    "feature_json": feature_json,
                })

            for chunk in _chunk_iter(pd.DataFrame(records)):
                if chunk.empty:
                    continue
                stmt = insert(Feature).values(chunk.to_dict(orient="records"))
                stmt = stmt.on_conflict_do_update(
                    constraint="features_pkey",
                    set_={
                        "opponent_team": stmt.excluded.opponent_team,
                        "feature_json": stmt.excluded.feature_json,
                    },
                )
                session.execute(stmt)
                total += len(chunk)
        session.commit()
        return total
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def persist_labels_from_clean(years: list[int]) -> int:
    """Upsert labels (PPR) from cleaned stats into labels table."""
    init_db()
    df = clean_player_stats(years)
    if "fantasy_points_ppr" not in df.columns:
        return 0
    labels = df.rename(columns={"fantasy_points_ppr": "fantasy_points"})[
        ["player_id", "season", "week", "fantasy_points"]
    ].copy()

    session: Session = SessionLocal()
    total = 0
    try:
        for chunk in _chunk_iter(labels):
            if chunk.empty:
                continue
            stmt = insert(Label).values(chunk.to_dict(orient="records"))
            stmt = stmt.on_conflict_do_update(
                constraint="labels_pkey",
                set_={"fantasy_points": stmt.excluded.fantasy_points},
            )
            session.execute(stmt)
            total += len(chunk)
        session.commit()
        return total
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    from .calc_features import calc_position_priors

    YEARS = list(range(2012, 2025))
    priors = calc_position_priors(YEARS)
    wrote = persist_priors_to_features(priors)
    print({"features_upserted": wrote})
    labels_written = persist_labels_from_clean(YEARS)
    print({"labels_upserted": labels_written})


