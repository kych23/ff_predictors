from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query
from sqlalchemy.orm import Session

from api.services.model_loader import get_cached_model
from db.models import Feature, Player
from db.session import SessionLocal


router = APIRouter()


@router.get("/predictions")
def top_predictions(season: int, week: int, position: Optional[str] = Query(default=None), limit: int = 50) -> List[dict]:
    session: Session = SessionLocal()
    try:
        model, feature_columns, version = get_cached_model(session)

        q = (
            session.query(Feature, Player.name, Player.position, Player.team_current)
            .join(Player, Player.player_id == Feature.player_id)
            .filter(Feature.season == season, Feature.week == week)
        )
        if position:
            q = q.filter(Player.position == position)
        rows = q.all()
        if not rows:
            return []

        # Build feature matrix
        features_df = pd.DataFrame([r.Feature.feature_json for r in rows])
        for col in feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0.0
        X = features_df[feature_columns].fillna(0.0)
        preds = np.asarray(model.predict(X))
        y_std = None
        if hasattr(model, "estimators_") and len(getattr(model, "estimators_", [])) > 1:
            tree_preds = np.column_stack([est.predict(X) for est in model.estimators_])
            y_std = np.std(tree_preds, axis=1, ddof=1)

        results = []
        for idx, (r, y) in enumerate(zip(rows, preds)):
            std_val = float(y_std[idx]) if y_std is not None else None
            ci_l = float(y - 1.96 * std_val) if std_val is not None else None
            ci_u = float(y + 1.96 * std_val) if std_val is not None else None
            results.append(
                {
                    "player_id": r.Feature.player_id,
                    "name": r.name,
                    "position": r.position,
                    "team": r.team_current,
                    "season": r.Feature.season,
                    "week": r.Feature.week,
                    "opponent_team": r.Feature.opponent_team,
                    "y_pred": float(y),
                    "y_std": std_val,
                    "ci_lower": ci_l,
                    "ci_upper": ci_u,
                    "model_version": version,
                }
            )
        results.sort(key=lambda x: x["y_pred"], reverse=True)
        return results[: max(1, min(limit, 500))]
    finally:
        session.close()


