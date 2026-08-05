"""Fit the weekly-variance, correlation and K/DST artifacts (§12.3-§13.2).

    venv/bin/python scripts/fit_models.py [--start 2012] [--end 2025]

Writes data/artifacts/{weekly_sigma,correlation,kdst}_<snapshot_id>.*
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_league  # noqa: E402
from src.domain.scoring.dst import points_allowed_score, score_dst  # noqa: E402
from src.domain.scoring.engine import score_offense  # noqa: E402
from src.domain.scoring.kicker import score_kicker  # noqa: E402
from src.models import artifacts  # noqa: E402
from src.models.correlation.slot_matrix import fit_slot_correlation  # noqa: E402
from src.models.weekly.kdst import KDSTModel, fit_empirical  # noqa: E402
from src.models.weekly.panel import assign_slots, player_season_summary  # noqa: E402
from src.models.weekly.variance import fit_weekly_sigma  # noqa: E402
from src.platform.sources import nflverse  # noqa: E402
from src.platform.store.manifest import build_snapshot, write_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=2012)
    ap.add_argument("--end", type=int, default=2025)
    args = ap.parse_args()
    seasons = list(range(args.start, args.end + 1))

    cfg = load_league()
    entries = []

    print(f"[1/5] pulling player stats {args.start}-{args.end}...")
    stats_pull = nflverse.fetch("player_stats", seasons=seasons)
    entries.append(stats_pull.entry)
    stats = stats_pull.frame
    stats = stats[stats["season_type"] == "REG"].copy()

    skill = stats[stats["position"].isin(cfg.roster.modeled_positions)].copy()
    skill["points"] = score_offense(skill, cfg.scoring.offense)
    panel = skill[["season", "week", "team", "player_id", "position", "points"]]
    print(f"      {len(panel):,} skill player-weeks")

    print("[2/5] weekly sigma...")
    summary = player_season_summary(panel, cfg)
    snapshot_id = build_snapshot(entries)          # provisional; extended below
    sigma_model = fit_weekly_sigma(summary, cfg, snapshot_id=snapshot_id,
                                   model_version=cfg.model_version)
    print(f"      fitted on {sigma_model.n_observations:,} player-seasons")
    print(f"      log(sigma) = {sigma_model.intercept:.3f} "
          f"+ {sigma_model.log_mu_coef:.3f}*log(mu) + position")
    print(f"      Duan smearing factor {sigma_model.smearing:.4f}")
    for pos in cfg.roster.modeled_positions:
        mu = summary.loc[summary.position == pos, "mu"]
        if len(mu):
            typical = float(mu.quantile(0.75))
            pred = float(sigma_model.predict(np.array([pos]), np.array([typical]))[0])
            actual = float(summary.loc[summary.position == pos, "sigma"].median())
            print(f"        {pos}: predicted sigma at p75 mu={typical:5.1f} -> "
                  f"{pred:5.2f}   (median observed {actual:5.2f})")

    print("[3/5] slot correlation matrix...")
    slotted = assign_slots(panel)
    corr = fit_slot_correlation(slotted, snapshot_id=snapshot_id,
                                model_version=cfg.model_version)
    print(f"      slots {corr.slots}")
    print(f"      min eigenvalue {corr.min_eigenvalue:.4f}  "
          f"projected={corr.was_projected}")
    for a, b in [("QB1", "WR1"), ("QB1", "WR2"), ("QB1", "TE1"),
                 ("WR1", "WR2"), ("RB1", "RB2"), ("QB1", "RB1")]:
        n = corr.pair_counts[corr.index_of(a), corr.index_of(b)]
        print(f"        {a} x {b}: {corr.correlation(a, b):+.3f}  (n={n:,})")

    print("[4/5] K and DST empirical distributions...")
    kickers = stats[stats["position"] == "K"].copy()
    k_points = score_kicker(kickers, cfg.scoring.kicker)

    team_pull = nflverse.fetch("team_stats", seasons=seasons)
    entries.append(team_pull.entry)
    sched_pull = nflverse.fetch("schedules")
    entries.append(sched_pull.entry)
    dst_points = _dst_points(team_pull.frame, sched_pull.frame, cfg)

    kdst = KDSTModel(
        distributions={
            "K": fit_empirical(k_points, "K", grid=cfg.weekly.kdst_cdf_grid),
            "DST": fit_empirical(dst_points, "DST", grid=cfg.weekly.kdst_cdf_grid),
        },
        snapshot_id=snapshot_id, model_version=cfg.model_version,
    )
    for pos, dist in kdst.distributions.items():
        print(f"        {pos}: mean {dist.mean:5.2f}  sd {dist.sd:5.2f}  "
              f"n={dist.n_observations:,}")

    print("[5/5] writing artifacts...")
    snapshot_id = build_snapshot(entries)
    write_manifest(snapshot_id, entries)
    sigma_model = _restamp(sigma_model, snapshot_id)
    corr = _restamp(corr, snapshot_id)
    kdst = _restamp(kdst, snapshot_id)
    for path in (artifacts.save_weekly_sigma(sigma_model),
                 artifacts.save_correlation(corr),
                 artifacts.save_kdst(kdst)):
        print(f"      {path}")
    artifacts.load_all(snapshot_id).assert_consistent()
    print(f"      snapshot {snapshot_id} — provenance consistent")


def _restamp(obj, snapshot_id: str):
    import dataclasses
    return dataclasses.replace(obj, snapshot_id=snapshot_id)


def _dst_points(team_stats: pd.DataFrame, schedules: pd.DataFrame,
                cfg) -> pd.Series:
    ts = team_stats[team_stats["season_type"] == "REG"].copy()
    home = schedules[["season", "week", "home_team", "away_score"]].rename(
        columns={"home_team": "team", "away_score": "points_allowed"})
    away = schedules[["season", "week", "away_team", "home_score"]].rename(
        columns={"away_team": "team", "home_score": "points_allowed"})
    allowed = pd.concat([home, away], ignore_index=True)
    merged = ts.merge(allowed, on=["season", "week", "team"], how="inner")
    merged = merged.dropna(subset=["points_allowed"])
    return score_dst(merged, cfg.scoring.dst)


if __name__ == "__main__":
    main()
