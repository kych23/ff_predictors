#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path if running from scripts directory
if Path(__file__).parent.name == "scripts":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

import argparse
from typing import List

from src.db.init_db import init_db
from src.db.session import SessionLocal
from src.ml.models.train import (
    load_flat_player_weeks,
    split_by_position,
    per_position_feature_lists,
    position_expanding_folds,
    train_xgb_per_position,
    train_xgb_per_position_with_tuning,
    save_predictions_to_db,
)


def main(years: List[int] | None, priors_req: int, k: float, tune: bool = False, n_iter: int = 40, save_preds: bool = False) -> None:
    init_db()
    session = SessionLocal()
    try:
        flat = load_flat_player_weeks(session, years=years)
    finally:
        session.close()

    # Split and prepare folds
    splits = split_by_position(flat)
    feats = per_position_feature_lists(flat)
    folds = position_expanding_folds(splits, min_years=3)

    # Train XGB per position and evaluate OOF predictions
    if tune:
        print("Using RandomizedSearchCV for hyperparameter tuning...")
        xgb_out = train_xgb_per_position_with_tuning(splits, feats, folds, tune=True, n_iter=n_iter)
    else:
        xgb_out = train_xgb_per_position(splits, feats, folds)
    
    for pos, out in xgb_out.items():
        oof = out["oof"]
        print(f"{pos}: XGB OOF MAE (non-NaN) = ", end="")
        oof_mask = ~splits[pos]["fantasy_points"].isna() & ~pd.isna(oof)
        if oof_mask.any():
            y = splits[pos].loc[oof_mask, "fantasy_points"].astype(float).to_numpy()
            yhat = oof[oof_mask]
            import numpy as np  # local import to avoid global dependency when imported as module
            mae = np.abs(y - yhat).mean()
            acc = (np.abs(y - yhat) <= k).mean() * 100.0
            print(f"{round(float(mae), 4)} | Within-{k} Acc = {round(float(acc), 2)}%")
        else:
            print("n/a")
    
    # Save predictions to database if requested
    if save_preds:
        session = SessionLocal()
        try:
            saved = save_predictions_to_db(splits, xgb_out, model_version="xgb_oof", session=session)
            print(f"\nSaved {saved} predictions to database")
        finally:
            session.close()

if __name__ == "__main__":
    import pandas as pd
    import numpy as np  # noqa: F401
    parser = argparse.ArgumentParser(description="Test end-to-end training pipeline")
    parser.add_argument("--start", type=int, default=2012, help="Start season (inclusive)")
    parser.add_argument("--end", type=int, default=2025, help="End season (exclusive)")
    parser.add_argument("--priors-req", type=int, default=1, help="Min prior games required")
    parser.add_argument("--k", type=float, default=3.0, help="Within-k accuracy threshold in points")
    parser.add_argument("--tune", action="store_true", help="Use RandomizedSearchCV for hyperparameter tuning")
    parser.add_argument("--n-iter", type=int, default=40, help="Number of iterations for RandomizedSearchCV")
    parser.add_argument("--save", action="store_true", help="Save predictions to database")
    args = parser.parse_args()
    years = list(range(args.start, args.end))
    main(years, args.priors_req, args.k, tune=args.tune, n_iter=args.n_iter, save_preds=args.save)

