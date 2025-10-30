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
        SELECT f.player_id, f.season, f.week, f.team, f.opponent_team,
               f.player_features, f.matchup_features,
               l.fantasy_points, p.position
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


def _batch_load_training_data(session: Session, years=None) -> pd.DataFrame:
    """
    Pulls features x labels (plus position) one year at a time and concatenates.
    Returns single DataFrame as in _load_training_data, for use when
    direct join/load_training_data would timeout.
    years: list of years; if None, infers years from the db.
    """
    if years is None:
        # Find unique available years (seasons) in features table
        q = "SELECT DISTINCT season FROM features ORDER BY season"
        candidates = pd.read_sql(q, session.bind)
        years = list(candidates["season"].astype(int))
    dfs = []
    for year in years:
        sql = f"""
            SELECT f.player_id, f.season, f.week, f.team, f.opponent_team,
                   f.player_features, f.matchup_features,
                   l.fantasy_points, p.position
            FROM features f
            JOIN labels l
              ON l.player_id = f.player_id
             AND l.season = f.season
             AND l.week = f.week
            LEFT JOIN players p
              ON p.player_id = f.player_id
            WHERE f.season = {year}
        """
        df = pd.read_sql(sql, session.bind)
        print(f"Loaded season {year} rows: {len(df)}")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _expand_features(df: pd.DataFrame, priors_req: int = 1) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    # Player priors
    feats = pd.json_normalize(df["player_features"]).astype(float)
    prior_cols = [c for c in feats.columns if ("_avg_" in c or c.endswith("_ewm"))]
    if "gp_prior" in feats.columns:
        prior_cols.append("gp_prior")
    features_df = feats[prior_cols].copy()
    # Matchup features: bring through as is (may contain NaNs)
    if "matchup_features" in df.columns:
        m = pd.json_normalize(df["matchup_features"]).add_prefix("m_")
        # Select expected keys if present
        keep_m = [c for c in m.columns if c in {
            "m_team_implied_total", "m_game_spread_team_view", "m_temp", "m_wind", "m_is_indoor"
        }]
        if keep_m:
            features_df = pd.concat([features_df, m[keep_m]], axis=1)
    target = df["fantasy_points"].astype(float)
    mask = np.ones(len(features_df), dtype=bool)
    if "gp_prior" in features_df.columns:
        mask = features_df["gp_prior"].fillna(0).to_numpy() >= priors_req
    # Do not force-fill matchup NaNs here; leave to Pipeline/trees
    return features_df, target, mask


def build_training_frame(session: Session, priors_req: int = 1, batch: bool = False):
    """
    Returns X, y, meta after joining DB, expanding priors, filtering by priors_req.
    X: features, y: target, meta: player_id, season, week, opponent_team, position
    batch: if True, use batch loading to avoid SQL timeouts (slower, but robust for big joins)
    """
    if batch:
        df = _batch_load_training_data(session)
    else:
        df = _load_training_data(session)
    X, y, mask = _expand_features(df, priors_req=priors_req)
    df = df.loc[mask].reset_index(drop=True)
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    meta = df[["player_id", "season", "week", "team", "opponent_team", "position"]].copy()
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

