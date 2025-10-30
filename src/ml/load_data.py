import nflreadpy as nfl
import pandas as pd
from sqlalchemy.orm import Session

from db.init_db import init_db
from db.initial_load import upsert_games, upsert_labels, upsert_players, upsert_teams
from db.session import SessionLocal

YEARS: list[int] = list(range(2012, 2025))

def load_data_from_api() -> None:
    """Load data from nflreadpy API and upsert directly into Postgres."""
    init_db()
    session: Session = SessionLocal()
    try:
        # Players
        players_polars = nfl.load_players()
        players = players_polars.to_pandas()
        upsert_players(players, session)

        # Schedules -> teams and games
        schedules_polars = nfl.load_schedules(seasons=YEARS)
        schedules = schedules_polars.to_pandas()
        upsert_teams(schedules, session)
        upsert_games(schedules, session)

        # Labels from player_stats
        stats_polars = nfl.load_player_stats(seasons=YEARS)
        player_stats = stats_polars.to_pandas()
        labels_cols = [c for c in ["player_id", "season", "week", "fantasy_points", "fantasy_points_ppr"] if c in player_stats.columns]
        upsert_labels(player_stats[labels_cols], session)

        session.commit()
        print("Successfully loaded players/teams/games/labels data from nflreadpy")
        
    except Exception as e:
        print(f"Error loading data from nflreadpy: {e}")
        session.rollback()
    finally:
        session.close()


if __name__ == "__main__":
    load_data_from_api()
