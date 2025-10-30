from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy.orm import Session

from db.init_db import init_db
from db.session import SessionLocal
import os


def _load_training_data(session: Session) -> pd.DataFrame:
    sql = """
        SELECT f.player_id, f.season, f.week, f.opponent_team,
               f.feature_json, l.fantasy_points
        FROM features f
        JOIN labels l
          ON l.player_id = f.player_id
         AND l.season = f.season
         AND l.week = f.week
    """
    df = pd.read_sql(sql, session.bind)
    return df


def _expand_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features_df = pd.json_normalize(df["feature_json"]).astype(float)
    target = df["fantasy_points"].astype(float)
    # Drop any constant or all-NaN columns
    features_df = features_df.dropna(axis=1, how="all")
    nunique = features_df.nunique()
    features_df = features_df.loc[:, nunique.gt(1)]
    features_df = features_df.fillna(0.0)
    return features_df, target


def train_and_register(model_dir: Path = Path("models")) -> Dict[str, Any]:
    init_db()
    model_dir.mkdir(parents=True, exist_ok=True)
    session: Session = SessionLocal()
    try:
        df = _load_training_data(session)
        if df.empty:
            raise RuntimeError("No training data (features×labels) found.")
        X, y = _expand_features(df)

        model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
        model.fit(X, y)

        preds = model.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        mae = float(mean_absolute_error(y, preds))
        metrics = {"rmse_in_sample": rmse, "mae_in_sample": mae, "n_rows": int(len(df))}

        version = str(int(time.time()))
        artifact_path = model_dir / f"model_{version}.pkl"
        bundle = {"model": model, "feature_columns": list(X.columns)}
        joblib.dump(bundle, artifact_path)

        # Also write/overwrite a latest pointer for the API
        latest_path = Path(os.getenv("MODEL_PATH", model_dir / "latest.pkl"))
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, latest_path)

        session.commit()
        return {"version": version, "artifact_path": str(artifact_path), "latest": str(latest_path), "metrics": metrics}
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    info = train_and_register()
    print(json.dumps(info, indent=2))


