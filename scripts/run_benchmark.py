#!/usr/bin/env python
"""CLI: produce the scoreboard (DrafterSpec.md §4.7).

Tier-1 ranking gate (engine vs ADP, NDCG@k) + Tier-3 risk calibration. Reports
overall AND data-complete-era (>= 2016) splits. Tier-2 draft-sim lives in
``benchmark/draft_sim.py`` and is exercised via ``--with-sim`` once a recommender
exists (M6).
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
from sqlalchemy import select

from src.benchmark.calibration import run_calibration_benchmark
from src.benchmark.draft_sim import run_draft_sim
from src.benchmark.ranking import run_ranking_benchmark
from src.benchmark.report import DATA_COMPLETE_ERA_START, format_gate
from src.config import load_config
from src.db.models import Adp, AdpCoverage, Projection, SeasonFeature, SeasonLabel
from src.db.session import SessionLocal, resolve_snapshot


def _read(session, model, cols) -> pd.DataFrame:
    rows = session.execute(select(model)).scalars().all()
    return pd.DataFrame([{c: getattr(r, c) for c in cols} for r in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the benchmark scoreboard (M4).")
    parser.add_argument("--snapshot-id", type=str, default=None,
                        help="Ingest snapshot to use (default: latest).")
    parser.add_argument("--with-sim", action="store_true", help="Run Tier-2 draft sim (slow)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    resolve_snapshot(args.snapshot_id)  # validate a snapshot exists + log it
    cfg = load_config()

    session = SessionLocal()
    try:
        proj = _read(session, Projection, ["player_id", "season", "p50", "p10", "p90", "position"])
        adp = _read(session, Adp, ["player_id", "season", "adp"])
        labels = _read(session, SeasonLabel, ["player_id", "season", "fppg"])
        cov = _read(session, AdpCoverage, ["season", "ranking_eligible", "sim_eligible"])
    finally:
        session.close()

    if proj.empty or adp.empty or labels.empty:
        print("Missing data (projections/adp/labels). Run M0-M3 first.")
        return

    ranking_eligible = set(cov[cov["ranking_eligible"]]["season"]) if not cov.empty else None

    def _report(tag: str, seasons):
        res = run_ranking_benchmark(proj, adp, labels, eligible_seasons=seasons)
        print(f"\n===== Tier-1 ranking: {tag} =====")
        print(res.per_season.to_string(index=False))
        for k, test in res.gate.items():
            print(format_gate(f"NDCG@{k}", test))

    _report("ALL eligible seasons", ranking_eligible)
    if ranking_eligible is not None:
        complete = {s for s in ranking_eligible if s >= DATA_COMPLETE_ERA_START}
        _report(f"data-complete era (>= {DATA_COMPLETE_ERA_START})", complete)

    # Tier-3 calibration uses projections joined to actual labels + the
    # bucket-driving fields from season_features (is_rookie, seasons_of_history,
    # team_changed) so coverage is reported BY player-type, not collapsed.
    session = SessionLocal()
    try:
        frows = session.execute(select(SeasonFeature)).scalars().all()
    finally:
        session.close()
    bucket_fields = pd.DataFrame([
        {"player_id": r.player_id, "season": r.season, "is_rookie": r.is_rookie,
         "seasons_of_history": (r.features or {}).get("seasons_of_history"),
         "team_changed": (r.features or {}).get("team_changed")}
        for r in frows
    ])
    oof = proj.merge(labels, on=["player_id", "season"], how="inner").rename(columns={"fppg": "y"})
    if not bucket_fields.empty:
        oof = oof.merge(bucket_fields, on=["player_id", "season"], how="left")
    if not oof.empty:
        print("\n===== Tier-3 risk calibration =====")
        print(run_calibration_benchmark(oof).to_string(index=False))

    # Tier-2 draft sim (slow — opt-in).
    if args.with_sim:
        sim_eligible = sorted(cov[cov["sim_eligible"]]["season"].tolist()) if not cov.empty else []
        print(f"\n===== Tier-2 draft sim: SIM_ELIGIBLE seasons {sim_eligible} =====")
        result = run_draft_sim(
            projections=proj,
            adp=adp,
            labels=labels,
            sim_eligible_seasons=sim_eligible,
            slots=[1, 3, 6, 9, 12],
            cfg=cfg,
        )
        if not result.per_season_slot.empty:
            print(result.per_season_slot.to_string(index=False))
        print(format_gate("Tier-2 roster value vs bots", result.gate))


if __name__ == "__main__":
    main()
