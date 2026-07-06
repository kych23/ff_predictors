"""Forward-serving path for the weekly model (project_weekly_forward).

Contract: train only on rows strictly before the target (season, week)
(as-of-kickoff), predict the target week's unlabeled feature rows, keep
quantiles monotonic after serving calibration.
"""
import numpy as np
import pandas as pd

from src.projection.weekly_dataset import build_weekly_matrix
from src.projection.weekly_train import forward_train_mask, project_weekly_forward

# small trees keep the two end-to-end tests fast; contract is unaffected
FAST_PARAMS = {"n_estimators": 30, "learning_rate": 0.1, "min_child_samples": 5}


def _weekly_rows(seasons, weeks, players_per_pos=6, poison_week=None,
                 poison_points=999.0, drop_feature=None):
    """Synthetic weekly features + labels (same shape as test_weekly_model)."""
    rng = np.random.RandomState(7)
    feat_rows, label_rows = [], []
    pid = 0
    for pos in ("QB", "RB", "WR", "TE"):
        for _ in range(players_per_pos):
            pid += 1
            player_id = f"00-{pid:04d}"
            base_fp = rng.uniform(5, 25)
            for season in seasons:
                for w in weeks:
                    feats = {
                        "season_p50": base_fp + rng.normal(0, 2),
                        "rolling_fppg": base_fp + rng.normal(0, 3),
                        "rolling_target_share": rng.uniform(0, 0.3),
                        "rolling_snap_share": rng.uniform(0.3, 1.0),
                        "rolling_carries_pg": rng.uniform(0, 15),
                        "wk_implied_pts": rng.uniform(18, 30),
                        "wk_game_total": rng.uniform(38, 55),
                        "wk_spread": rng.uniform(-10, 10),
                        "opp_dvp_fpa": rng.uniform(10, 35),
                    }
                    if drop_feature:
                        feats.pop(drop_feature, None)
                    feat_rows.append({
                        "player_id": player_id, "season": season, "week": w,
                        "position": pos, "features": feats,
                    })
                    fp = max(0.0, base_fp + rng.normal(0, 5))
                    if poison_week is not None and (season, w) == poison_week:
                        fp = poison_points
                    label_rows.append({
                        "player_id": player_id, "season": season, "week": w,
                        "fantasy_points": fp,
                    })
    return pd.DataFrame(feat_rows), pd.DataFrame(label_rows)


def _unlabeled_target(feats):
    empty = pd.DataFrame(columns=["player_id", "season", "week", "fantasy_points"])
    return build_weekly_matrix(feats, empty, require_label=False)


# --- forward_train_mask ---

def test_mask_keeps_only_strictly_prior_rows():
    meta = pd.DataFrame([
        {"season": 2024, "week": 17},   # prior season: keep
        {"season": 2025, "week": 2},    # earlier week: keep
        {"season": 2025, "week": 3},    # target week itself: drop
        {"season": 2025, "week": 4},    # later week: drop
        {"season": 2026, "week": 1},    # future season: drop
    ])
    mask = forward_train_mask(meta, 2025, 3)
    assert list(mask) == [True, True, False, False, False]


def test_mask_week_one_uses_only_prior_seasons():
    meta = pd.DataFrame([
        {"season": 2024, "week": 18},
        {"season": 2025, "week": 1},
    ])
    assert list(forward_train_mask(meta, 2025, 1)) == [True, False]


def test_mask_handles_string_seasons():
    meta = pd.DataFrame([
        {"season": "2024", "week": "3"},
        {"season": "2025", "week": "3"},
    ])
    assert list(forward_train_mask(meta, 2025, 3)) == [True, False]


# --- project_weekly_forward end-to-end ---

def test_forward_projections_schema_monotonic_and_no_label_leak():
    """One e2e pass: target week rows are excluded from training (poison labels
    of 999 in the target week must not inflate predictions), every target row
    gets a monotone, finite quantile triple."""
    seasons = list(range(2015, 2022))
    feats, labels = _weekly_rows(seasons, weeks=range(1, 5),
                                 poison_week=(2021, 4))
    ds = build_weekly_matrix(feats, labels)
    target_feats = feats[(feats["season"] == 2021) & (feats["week"] == 4)]
    target = _unlabeled_target(target_feats)

    out = project_weekly_forward(ds, target, 2021, 4, params=FAST_PARAMS)

    assert len(out) == len(target.meta)
    for col in ("player_id", "season", "week", "position", "p10", "p50", "p90"):
        assert col in out.columns
    assert out[["p10", "p50", "p90"]].notna().all().all()
    assert (out["p10"] <= out["p50"] + 1e-9).all()
    assert (out["p50"] <= out["p90"] + 1e-9).all()
    # honest labels top out ~30; training on the 999-point poison week would
    # drag P90 toward it
    assert out["p90"].max() < 100.0


def test_forward_predicts_with_missing_feature_column():
    """A forward week can lack a feature the model trained on (e.g. no DvP yet);
    prediction must still produce full rows."""
    seasons = [2019, 2020, 2021]
    feats, labels = _weekly_rows(seasons, weeks=range(1, 4))
    ds = build_weekly_matrix(feats, labels)
    target_feats, _ = _weekly_rows([2022], weeks=[1], drop_feature="opp_dvp_fpa")
    target = _unlabeled_target(target_feats)

    out = project_weekly_forward(ds, target, 2022, 1, params=FAST_PARAMS)
    assert len(out) == len(target.meta)
    assert out[["p10", "p50", "p90"]].notna().all().all()


def test_forward_empty_target_returns_empty():
    feats, labels = _weekly_rows([2020, 2021], weeks=range(1, 3))
    ds = build_weekly_matrix(feats, labels)
    target = _unlabeled_target(feats.iloc[0:0])
    out = project_weekly_forward(ds, target, 2022, 1, params=FAST_PARAMS)
    assert out.empty


def test_forward_no_prior_data_returns_empty():
    feats, labels = _weekly_rows([2024], weeks=[1])
    ds = build_weekly_matrix(feats, labels)
    target = _unlabeled_target(feats)
    # target is before all training data -> zero training rows
    out = project_weekly_forward(ds, target, 2020, 1, params=FAST_PARAMS)
    assert out.empty
