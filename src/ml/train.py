from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple
import argparse

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


def _expand_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    features_df = pd.json_normalize(df["feature_json"]).astype(float)
    target = df["fantasy_points"].astype(float)
    # Drop any constant or all-NaN columns
    features_df = features_df.dropna(axis=1, how="all")
    nunique = features_df.nunique()
    features_df = features_df.loc[:, nunique.gt(1)]
    features_df = features_df.fillna(0.0)
    return features_df, target


def _train_val_split(df: pd.DataFrame, val_season: int | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if val_season is None:
        val_season = int(df["season"].max())
    train_df = df[df["season"] < val_season].copy()
    val_df = df[df["season"] == val_season].copy()
    if train_df.empty or val_df.empty:
        # Fallback: random 80/20 split within the full data
        msk = np.random.RandomState(42).rand(len(df)) < 0.8
        train_df, val_df = df[msk].copy(), df[~msk].copy()
    return train_df, val_df


def train_and_register(model_dir: Path = Path("models"), val_season: int | None = None, n_estimators: int = 500) -> Dict[str, Any]:
    init_db()
    model_dir.mkdir(parents=True, exist_ok=True)
    session: Session = SessionLocal()
    try:
        df = _load_training_data(session)
        if df.empty:
            raise RuntimeError("No training data (features×labels) found.")
        train_df, val_df = _train_val_split(df, val_season)
        X_train, y_train = _expand_features(train_df)
        X_val, y_val = _expand_features(val_df)

        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        preds_tr = model.predict(X_train)
        preds_val = model.predict(X_val)
        rmse_tr = float(np.sqrt(mean_squared_error(y_train, preds_tr)))
        mae_tr = float(mean_absolute_error(y_train, preds_tr))
        rmse_val = float(np.sqrt(mean_squared_error(y_val, preds_val)))
        mae_val = float(mean_absolute_error(y_val, preds_val))
        metrics = {
            "n_rows_total": int(len(df)),
            "n_rows_train": int(len(train_df)),
            "n_rows_val": int(len(val_df)),
            "rmse_train": rmse_tr,
            "mae_train": mae_tr,
            "rmse_val": rmse_val,
            "mae_val": mae_val,
            "val_season": int(val_df["season"].max()) if not val_df.empty else None,
        }

        version = str(int(time.time()))
        artifact_path = model_dir / f"model_{version}.pkl"
        bundle = {"model": model, "feature_columns": list(X_train.columns)}
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-season", type=int, default=None, help="Season to use for validation split (default: latest season)")
    parser.add_argument("--n-estimators", type=int, default=500, help="RandomForest n_estimators")
    args = parser.parse_args()
    info = train_and_register(val_season=args.val_season, n_estimators=args.n_estimators)
    print(json.dumps(info, indent=2))


