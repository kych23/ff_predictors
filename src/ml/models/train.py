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
from .eval import evaluate_regression

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


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




# ---------- New: flat loader utility ----------
def load_flat_player_weeks(session: Session, years: list[int] | None = None) -> pd.DataFrame:
    """Load weekly player rows and flatten JSONB into pf_/mf_ columns.

    Returns DataFrame with columns:
    player_id, season, week, team, opponent_team, position, fantasy_points,
    pf_* (player priors), mf_* (matchup features)
    """
    df = _batch_load_training_data(session, years) if years else _load_training_data(session)
    # Flatten player_features
    pf = pd.json_normalize(df["player_features"]).add_prefix("pf_")
    # Flatten matchup_features
    mf = pd.json_normalize(df["matchup_features"]).add_prefix("mf_") if "matchup_features" in df.columns else pd.DataFrame(index=df.index)
    # Expected matchup keys -> numeric/bool coercion
    for col in ["mf_team_implied_total", "mf_game_spread_team_view", "mf_temp", "mf_wind", "mf_opp_pos_fp_allowed_avg5"]:
        if col in mf.columns:
            mf[col] = pd.to_numeric(mf[col], errors="coerce")
    if "mf_is_indoor" in mf.columns:
        # Cast to boolean then to int for XGB/sklearn compatibility
        mf["mf_is_indoor"] = mf["mf_is_indoor"].astype("boolean").astype("Int8")

    flat = pd.concat([
        df[["player_id", "season", "week", "team", "opponent_team", "position", "fantasy_points"]].reset_index(drop=True),
        pf.reset_index(drop=True),
        mf.reset_index(drop=True),
    ], axis=1)
    return flat


# ---------- Split into position datasets and define feature lists ----------
def split_by_position(flat: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return dict with QB, K, and SKILL (RB/WR/TE) subsets sorted by season, week."""
    qb = flat[flat["position"].eq("QB")].sort_values(["season", "week"]).reset_index(drop=True)
    k = flat[flat["position"].eq("K")].sort_values(["season", "week"]).reset_index(drop=True)
    skill = flat[flat["position"].isin(["RB", "WR", "TE"])].sort_values(["season", "week"]).reset_index(drop=True)
    return {"QB": qb, "K": k, "SKILL": skill}


def per_position_feature_lists(flat: pd.DataFrame) -> dict[str, list[str]]:
    """Build per-position feature lists: all pf_* plus shared mf_* columns.

    Shared matchup features we expect: team implied, spread, defense allowed last-5, indoor, temp, wind.
    """
    pf_cols = [c for c in flat.columns if c.startswith("pf_")]
    mf_expected = [
        "mf_team_implied_total",
        "mf_game_spread_team_view",
        "mf_opp_pos_fp_allowed_avg5",
        "mf_is_indoor",
        "mf_temp",
        "mf_wind",
    ]
    mf_cols = [c for c in mf_expected if c in flat.columns]
    feat = pf_cols + mf_cols
    return {"QB": feat, "K": feat, "SKILL": feat}


# ---------- Expanding time-series CV ----------
def make_expanding_folds(meta: pd.DataFrame, min_years: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding (rolling-origin) folds by season.

    Each fold trains on all seasons < val_year and validates on val_year only.
    Requires meta to contain a 'season' column; returns list of (train_idx, val_idx) index arrays.
    """
    years = np.sort(meta["season"].unique())
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(min_years, len(years)):
        val_year = years[i]
        train_mask = meta["season"].astype(int) < int(val_year)
        val_mask = meta["season"].astype(int) == int(val_year)
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        folds.append((train_mask.values, val_mask.values))
    return folds


def position_expanding_folds(splits: dict[str, pd.DataFrame], min_years: int = 3) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Generate expanding folds per position using each subset's 'season'."""
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for pos, df in splits.items():
        meta = df[["season"]].copy()
        out[pos] = make_expanding_folds(meta, min_years=min_years)
    return out


# ---------- XGBoost CV wiring ----------
def _xgb_default_params() -> dict:
    return {
        "n_estimators": 600,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def xgb_cv_oof(df: pd.DataFrame, feature_cols: list[str], folds: list[tuple[np.ndarray, np.ndarray]], params: dict | None = None) -> tuple[np.ndarray, list]:
    """Train XGBRegressor across expanding folds and return OOF predictions and list of models."""
    from xgboost import XGBRegressor
    X = df[feature_cols].copy()
    # Ensure numeric dtypes for XGB
    for c in X.columns:
        if X[c].dtype == "boolean" or str(X[c].dtype).startswith("Int"):
            X[c] = X[c].astype("float")
    y = df["fantasy_points"].astype(float).to_numpy()
    oof = np.full(len(df), np.nan, dtype=float)
    models: list = []
    p = _xgb_default_params()
    if params:
        p.update(params)
    for train_mask, val_mask in folds:
        if val_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        model = XGBRegressor(**p)
        model.fit(X[train_mask], y[train_mask])
        oof[val_mask] = model.predict(X[val_mask])
        models.append(model)
    return oof, models


def train_xgb_per_position(splits: dict[str, pd.DataFrame], feat_lists: dict[str, list[str]], folds: dict[str, list[tuple[np.ndarray, np.ndarray]]], params: dict | None = None) -> dict[str, dict]:
    """Train XGB per position. Returns dict: pos -> {"oof": oof_array, "models": [models], "features": feature_cols}"""
    out: dict[str, dict] = {}
    for pos, df in splits.items():
        feats = feat_lists[pos]
        pos_folds = folds[pos]
        oof, models = xgb_cv_oof(df, feats, pos_folds, params=params)
        out[pos] = {"oof": oof, "models": models, "features": feats}
    return out


# ---------- Ridge meta model per position ----------
def train_ridge_meta_per_position(splits: dict[str, pd.DataFrame], mf_candidates: list[str] | None = None, alpha: float = 1.0) -> dict[str, dict]:
    """Train a Ridge meta-model per position using pred_xgb + selected mf_ features.

    mf_candidates defaults to [team implied, opp allowed avg5, spread].
    Returns pos -> {"model": pipeline, "features": meta_feature_cols}
    """


    if mf_candidates is None:
        mf_candidates = [
            "mf_team_implied_total",
            "mf_opp_pos_fp_allowed_avg5",
            "mf_game_spread_team_view",
        ]
    out: dict[str, dict] = {}
    for pos, df in splits.items():
        if "pred_xgb" not in df.columns:
            raise ValueError(f"pred_xgb missing in split {pos}. Train XGB first and attach OOF to splits[pos]['pred_xgb'].")
        meta_feats = ["pred_xgb"] + [c for c in mf_candidates if c in df.columns]
        Xm = df[meta_feats].copy()
        # Ensure numeric
        for c in Xm.columns:
            Xm[c] = pd.to_numeric(Xm[c], errors="coerce")
        ym = df["fantasy_points"].astype(float)
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=42))
        ])
        pipe.fit(Xm, ym)
        out[pos] = {"model": pipe, "features": meta_feats}
    return out

