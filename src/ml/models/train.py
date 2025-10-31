from __future__ import annotations

from typing import Tuple

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV

from src.db.init_db import init_db
from src.db.session import SessionLocal
from src.db.models import Prediction
from sqlalchemy.dialects.postgresql import insert
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
    """Return dict with QB and SKILL (RB/WR/TE) subsets sorted by season, week.

    Kickers are intentionally excluded from training.
    """
    qb = flat[flat["position"].eq("QB")].sort_values(["season", "week"]).reset_index(drop=True)
    skill = flat[flat["position"].isin(["RB", "WR", "TE"])].sort_values(["season", "week"]).reset_index(drop=True)
    return {"QB": qb, "SKILL": skill}


def per_position_feature_lists(flat: pd.DataFrame) -> dict[str, list[str]]:
    """Build per-position feature lists for QB and SKILL: all pf_* plus shared mf_* columns.

    Shared matchup features we expect: team implied, spread, defense allowed last-5, indoor, temp, wind.
    Kickers are intentionally excluded from training.
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
    return {"QB": feat, "SKILL": feat}


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
    """Return tuned XGBoost parameters from RandomizedSearchCV on aggregated data.
    
    Best params from notebook: colsample_bytree=0.684, learning_rate=0.022,
    max_depth=5, n_estimators=562, reg_alpha=0.04, reg_lambda=8.0, subsample=0.822
    These achieved MAE=4.15 (holdout) / 4.32 (rolling avg) vs 4.37 untuned.
    """
    return {
        "n_estimators": 562,
        "max_depth": 5,
        "learning_rate": 0.022,
        "subsample": 0.822,
        "colsample_bytree": 0.684,
        "reg_alpha": 0.04,
        "reg_lambda": 8.0,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }


def tune_xgb_hyperparameters(
    df: pd.DataFrame, 
    feature_cols: list[str],
    position: str | None = None,
    n_iter: int = 40,
    cv: int = 5,
    random_state: int = 42
) -> dict:
    """Tune XGBoost hyperparameters using RandomizedSearchCV.
    
    Returns dict of best parameters found.
    """
    from xgboost import XGBRegressor
    
    X = df[feature_cols].copy()
    # Ensure numeric dtypes for XGB
    for c in X.columns:
        if X[c].dtype == "boolean" or str(X[c].dtype).startswith("Int"):
            X[c] = X[c].astype("float")
    y = df["fantasy_points"].astype(float).to_numpy()
    
    # Define parameter distributions (position-specific if provided)
    if position == "QB":
        # QB needs different tuning: more regularization, lower learning rate
        param_dist = {
            "n_estimators": randint(400, 1200),
            "max_depth": randint(4, 10),
            "learning_rate": loguniform(5e-3, 2e-1),
            "subsample": uniform(0.7, 0.3),  # 0.7-1.0
            "colsample_bytree": uniform(0.7, 0.3),  # 0.7-1.0
            "reg_alpha": loguniform(1e-6, 1e1),  # L1
            "reg_lambda": loguniform(1e-3, 1e1),  # L2
        }
    elif position == "SKILL":
        # SKILL: keep broader ranges but slightly adjusted
        param_dist = {
            "n_estimators": randint(300, 1200),
            "max_depth": randint(3, 12),
            "learning_rate": loguniform(1e-3, 3e-1),
            "subsample": uniform(0.6, 0.4),  # 0.6-1.0
            "colsample_bytree": uniform(0.6, 0.4),  # 0.6-1.0
            "reg_alpha": loguniform(1e-8, 1e2),  # L1
            "reg_lambda": loguniform(1e-4, 1e2),  # L2
        }
    else:
        # Default shared parameters
        param_dist = {
            "n_estimators": randint(300, 1200),
            "max_depth": randint(3, 12),
            "learning_rate": loguniform(1e-3, 3e-1),
            "subsample": uniform(0.6, 0.4),  # 0.6-1.0
            "colsample_bytree": uniform(0.6, 0.4),  # 0.6-1.0
            "reg_alpha": loguniform(1e-8, 1e2),  # L1
            "reg_lambda": loguniform(1e-4, 1e2),  # L2
        }
    
    # Base params that aren't tuned
    # Set n_jobs=1 for XGBoost to avoid nested parallelism issues with RandomizedSearchCV
    base_params = {
        "random_state": random_state,
        "n_jobs": 1,  # Single-threaded XGBoost to work with RandomSearchCV parallelism
        "tree_method": "hist",
    }
    
    # Create XGBoost factory
    def xgb_factory(**kw):
        params = base_params.copy()
        params.update(kw)
        return XGBRegressor(**params)
    
    # Run RandomizedSearchCV with reduced parallelism to avoid worker timeout issues
    search = RandomizedSearchCV(
        xgb_factory(),
        param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=cv,
        random_state=random_state,
        n_jobs=2  # Reduced from -1 to avoid worker timeout warnings
    )
    search.fit(X, y)
    
    return search.best_params_


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


def train_xgb_per_position_with_tuning(
    splits: dict[str, pd.DataFrame],
    feat_lists: dict[str, list[str]],
    folds: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    tune: bool = True,
    n_iter: int = 40,
    cv: int = 5,
    random_state: int = 42
) -> dict[str, dict]:
    """Train XGB per position with optional per-position hyperparameter tuning.
    
    If tune=True, uses RandomizedSearchCV to find best params for each position.
    Returns dict: pos -> {"oof": oof_array, "models": [models], "features": feature_cols, "best_params": params}
    """
    out: dict[str, dict] = {}
    for pos, df in splits.items():
        feats = feat_lists[pos]
        pos_folds = folds[pos]
        
        # Tune hyperparameters for this position
        if tune:
            print(f"Tuning hyperparameters for {pos}...")
            best_params = tune_xgb_hyperparameters(df, feats, position=pos, n_iter=n_iter, cv=cv, random_state=random_state)
            print(f"  Best params for {pos}: {best_params}")
        else:
            best_params = None
        
        oof, models = xgb_cv_oof(df, feats, pos_folds, params=best_params)
        out[pos] = {"oof": oof, "models": models, "features": feats, "best_params": best_params}
    
    return out


def save_predictions_to_db(
    splits: dict[str, pd.DataFrame],
    xgb_out: dict[str, dict],
    model_version: str = "xgb_v1",
    session: Session | None = None
) -> int:
    """Save OOF predictions to the predictions table.
    
    Args:
        splits: Position dataframes containing player_id, season, week, opponent_team
        xgb_out: Output from train_xgb_per_position or train_xgb_per_position_with_tuning
        model_version: Version string for this model
        session: Optional database session (creates one if not provided)
    
    Returns:
        Number of predictions saved
    """
    close_session = False
    if session is None:
        session = SessionLocal()
        close_session = True
    
    total_saved = 0
    try:
        for pos, df in splits.items():
            if pos not in xgb_out:
                continue
            
            oof = xgb_out[pos]["oof"]
            
            # Create prediction records
            records = []
            for idx, row in df.iterrows():
                if pd.isna(oof[idx]):
                    continue
                
                records.append({
                    "player_id": str(row["player_id"]),
                    "season": int(row["season"]),
                    "week": int(row["week"]),
                    "opponent_team": str(row.get("opponent_team", "")),
                    "model_version": f"{model_version}_{pos}",
                    "y_pred": float(oof[idx]),
                    "y_std": None,  # Could compute from ensemble in future
                })
            
            if not records:
                continue
            
            # Batch insert with upsert
            for chunk in [records[i:i+5000] for i in range(0, len(records), 5000)]:
                stmt = insert(Prediction).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_prediction_by_model",
                    set_={
                        "y_pred": stmt.excluded.y_pred,
                        "y_std": stmt.excluded.y_std,
                    }
                )
                session.execute(stmt)
                total_saved += len(chunk)
        
        session.commit()
        return total_saved
    
    except Exception:
        session.rollback()
        raise
    finally:
        if close_session:
            session.close()

