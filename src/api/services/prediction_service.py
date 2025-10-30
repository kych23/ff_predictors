from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from api.schemas import PredictRequest, PredictResponse
from db.models import Feature
from db.session import SessionLocal
from .model_loader import get_cached_model


def predict_points(req: PredictRequest) -> PredictResponse:
    session: Session = SessionLocal()
    try:
        # Load model
        model, feature_columns, version = get_cached_model(session)

        # Fetch features for the requested key
        row = (
            session.query(Feature)
            .filter(
                Feature.player_id == req.player_id,
                Feature.season == req.season,
                Feature.week == req.week,
            )
            .first()
        )
        if row is None:
            raise RuntimeError("Features not found for requested player/week.")

        features = pd.DataFrame([row.feature_json])
        # Align to training columns
        for col in feature_columns:
            if col not in features.columns:
                features[col] = 0.0
        X = features[feature_columns].fillna(0.0)

        y_pred = float(np.asarray(model.predict(X))[0])

        # Approximate predictive std via tree ensemble variability if available
        y_std: float | None = None
        ci_lower: float | None = None
        ci_upper: float | None = None
        if hasattr(model, "estimators_") and len(getattr(model, "estimators_", [])) > 1:
            tree_preds = np.column_stack([est.predict(X) for est in model.estimators_])
            y_std = float(np.std(tree_preds, axis=1, ddof=1)[0]) if tree_preds.size else None
            if y_std is not None:
                ci_lower = float(y_pred - 1.96 * y_std)
                ci_upper = float(y_pred + 1.96 * y_std)

        return PredictResponse(y_pred=y_pred, y_std=y_std, model_version=version, ci_lower=ci_lower, ci_upper=ci_upper)
    finally:
        session.close()


