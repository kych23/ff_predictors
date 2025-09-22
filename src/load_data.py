import nflreadpy as nfl
import pandas as pd

YEARS = list(range(2012, 2025))

def fetch_and_save():
    player_stats = nfl.load_player_stats(seasons=YEARS)
    player_stats.write_parquet("data/raw/player_stats.parquet", partition_by=["season"])

    pbp = nfl.load_pbp(seasons=YEARS)
    pbp.write_parquet("data/raw/pbp.parquet", partition_by=["season"])

    schedules = nfl.load_schedules(seasons=YEARS)
    schedules.write_parquet("data/raw/schedules.parquet", partition_by=["season"])

    snaps = nfl.load_snap_counts(seasons=YEARS)
    snaps.write_parquet("data/raw/snap_counts.parquet", partition_by=["season"])
    
    players = nfl.load_players()
    players.write_parquet("data/raw/players.parquet")
    
    injuries = nfl.load_injuries(seasons=YEARS)
    injuries.write_parquet("data/raw/injuries.parquet", partition_by=["season"])

if __name__ == "__main__":
    fetch_and_save()
    print("Raw data fetched successfully!")
