#!/usr/bin/env python
from __future__ import annotations

import argparse

from src.db.init_db import init_db
from src.db.session import SessionLocal
from src.db.upsert_data import upsert_players, upsert_teams, upsert_games
import nflreadpy as nfl
from src.ml.features.calc_features import calc_position_priors
from src.ml.features.persist import persist_priors_to_features, persist_labels_from_clean


def main(start: int, end: int) -> None:
    years = list(range(start, end))
    init_db()
    # 1) Players
    session = SessionLocal()
    try:
        players = nfl.load_players()
        players_df = players.to_pandas() if hasattr(players, "to_pandas") else players
        n_players = upsert_players(players_df, session)
        # 2) Teams & Games from schedules
        schedules = nfl.load_schedules(seasons=years)
        schedules_df = schedules.to_pandas() if hasattr(schedules, "to_pandas") else schedules
        n_teams = upsert_teams(schedules_df, session)
        n_games = upsert_games(schedules_df, session)
        session.commit()
        print(f"Inserted/updated players: {n_players}, teams: {n_teams}, games: {n_games}")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    # 3) Labels (uses cleaned stats logic)
    n_labels = persist_labels_from_clean(years)
    print(f"Inserted/updated labels rows: {n_labels}")

    # 4) Features: Build priors per position and persist with matchup features
    priors = calc_position_priors(years)
    for k, v in priors.items():
        print(f"priors[{k}]: rows={len(v)} cols={list(v.columns)[:6]} ...")
    n_features = persist_priors_to_features(priors, years=years)
    print(f"Inserted/updated features rows: {n_features}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recreate schema and seed features/labels from nflreadpy")
    parser.add_argument("--start", type=int, default=2012, help="Start season (inclusive)")
    parser.add_argument("--end", type=int, default=2025, help="End season (exclusive)")
    args = parser.parse_args()
    main(args.start, args.end)


