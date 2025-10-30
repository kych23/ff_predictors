from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple
import argparse

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session

from src.db.init_db import init_db
from src.db.session import SessionLocal
from .ml_models import get_model
from .eval import evaluate_regression


def _load_training_data(session: Session) -> pd.DataFrame:
    sql = """
        SELECT f.player_id, f.season, f.week, f.opponent_team,
               f.feature_json, l.fantasy_points, p.position
        FROM features f
        JOIN labels l
          ON l.player_id = f.player_id
         AND l.season = f.season
         AND l.week = f.week
        LEFT JOIN players p
          ON p.player_id = f.player_id
    """
    df = pd.read_sql(sql, session.bind)
    return df


def _expand_features(df: pd.DataFrame, priors_req: int = 1) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    feats = pd.json_normalize(df["feature_json"]).astype(float)
    prior_cols = [c for c in feats.columns if ("_avg_" in c or c.endswith("_ewm"))]
    if "gp_prior" in feats.columns:
        prior_cols.append("gp_prior")
    features_df = feats[prior_cols].copy()
    target = df["fantasy_points"].astype(float)
    mask = np.ones(len(features_df), dtype=bool)
    if "gp_prior" in features_df.columns:
        mask = features_df["gp_prior"].fillna(0).to_numpy() >= priors_req
    features_df = features_df.fillna(0.0)
    return features_df, target, mask


def build_training_frame(session: Session, priors_req: int = 1):
    """
    Returns X, y, meta after joining DB, expanding priors, filtering by priors_req.
    X: features, y: target, meta: player_id, season, week, opponent_team, position
    """
    df = _load_training_data(session)
    X, y, mask = _expand_features(df, priors_req=priors_req)
    df = df.loc[mask].reset_index(drop=True)
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    meta = df[["player_id", "season", "week", "opponent_team", "position"]].copy()
    return X, y, meta


def run_experiment(model_name: str, model_kwargs: dict, val_season: int | None = None, priors_req: int = 1):
    init_db()
    session = SessionLocal()
    X, y, meta = build_training_frame(session, priors_req=priors_req)
    if val_season is None:
        val_season = int(meta["season"].max())
    train_idx = meta["season"] < val_season
    val_idx = meta["season"] == val_season
    model = get_model(model_name, **(model_kwargs or {}))
    metrics, y_pred_train, y_pred_val = evaluate_regression(model, X[train_idx], y[train_idx], X[val_idx], y[val_idx])
    out = {
        "model": model_name,
        "params": model_kwargs,
        "val_season": val_season,
        "metrics": metrics,
    }
    session.close()
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (rf, linreg, ridge, etc.)")
    parser.add_argument("--model-kwargs", type=str, default="{}", help="JSON or dict string of model kwargs")
    parser.add_argument("--val-season", type=int, default=None, help="Season for validation split")
    parser.add_argument("--priors-req", type=int, default=1, help="Minimum required prior games")
    args = parser.parse_args()
    kwargs = json.loads(args.model_kwargs.replace("'", '"')) if args.model_kwargs else {}
    result = run_experiment(args.model, kwargs, val_season=args.val_season, priors_req=args.priors_req)
    print(json.dumps(result, indent=2))


