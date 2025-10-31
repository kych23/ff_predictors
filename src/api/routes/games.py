from __future__ import annotations

from typing import List
from datetime import datetime, timedelta

from fastapi import APIRouter
from sqlalchemy import distinct, func
from sqlalchemy.orm import Session

from db.models import Game
from db.session import SessionLocal


router = APIRouter()


def get_nfl_season_start(season: int) -> datetime:
    """Get the NFL season start date for a given year.
    
    NFL regular season typically starts in early September.
    For 2024 season: Sept 5, 2024
    For 2025 season: Sept 4, 2025
    """
    # NFL season typically starts on the Thursday after Labor Day
    # Common pattern: first Thursday of September
    if season == 2024:
        return datetime(2024, 9, 5)
    elif season == 2025:
        return datetime(2025, 9, 4)
    else:
        # Fallback: estimate first Thursday of September
        # Start with Sept 1, find first Thursday
        sept_1 = datetime(season, 9, 1)
        days_until_thursday = (3 - sept_1.weekday()) % 7
        # If Sept 1 is Thursday-Sunday, add 4 to get to next Thursday
        if days_until_thursday == 0:
            days_until_thursday = 0
        return sept_1 + timedelta(days=days_until_thursday)


def calculate_current_week(season: int) -> int:
    """Calculate the current NFL week based on the season start date.
    
    NFL weeks typically start on Thursdays, but the first week calculation
    should account for Thursday-Sunday being part of Week 1.
    """
    now = datetime.now()
    season_start = get_nfl_season_start(season)
    
    # If we're before the season starts, return 1
    if now < season_start:
        return 1
    
    # Calculate days since season start
    days_since_start = (now - season_start).days
    
    # NFL weeks start on Thursday
    # Days 0-3 (Thu-Sun of first week) = Week 1
    # Days 4-10 = Week 2, etc.
    if days_since_start <= 3:
        return 1
    
    # After first week, calculate normally
    # Subtract 4 to account for first Thursday-Sunday, then divide by 7
    current_week = ((days_since_start - 4) // 7) + 2
    
    # NFL season has 18 regular season weeks
    return min(current_week, 18)


@router.get("/current")
def get_current_week() -> dict:
    """Get the current NFL season and week based on calendar dates."""
    from datetime import datetime
    now = datetime.now()
    
    # Determine current season
    # NFL season runs from September to January/February
    # If we're before August, we're in the previous season
    if now.month >= 8:
        current_season = now.year
    else:
        current_season = now.year - 1
    
    # Calculate current week based on actual NFL season start dates
    current_week = calculate_current_week(current_season)
    
    # Fallback: check database for most recent data if calculated week seems off
    # This handles cases where season hasn't started yet or we're in offseason
    session: Session = SessionLocal()
    try:
        max_season_result = session.query(func.max(Game.season)).first()
        if max_season_result and max_season_result[0]:
            db_season = max_season_result[0]
            # Use database season if it's more recent than our calculated one
            if db_season >= current_season:
                max_week_result = session.query(func.max(Game.week)).filter(Game.season == db_season).first()
                if max_week_result and max_week_result[0]:
                    db_week = max_week_result[0]
                    # Use calculated week if it's reasonable, otherwise use DB week
                    if 1 <= current_week <= 18:
                        return {"season": current_season, "week": current_week}
                    else:
                        return {"season": db_season, "week": db_week}
        
        # Return calculated values
        return {"season": current_season, "week": current_week}
    finally:
        session.close()


@router.get("/weeks")
def list_weeks(season: int) -> List[int]:
    session: Session = SessionLocal()
    try:
        rows = session.query(distinct(Game.week)).filter(Game.season == season).order_by(Game.week).all()
        return [w[0] for w in rows]
    finally:
        session.close()


@router.get("/opponents")
def list_opponents(season: int, week: int) -> List[str]:
    session: Session = SessionLocal()
    try:
        rows = session.query(distinct(Game.home_team)).filter(Game.season == season, Game.week == week).all()
        rows_away = session.query(distinct(Game.away_team)).filter(Game.season == season, Game.week == week).all()
        teams = {r[0] for r in rows} | {r[0] for r in rows_away}
        return sorted(teams)
    finally:
        session.close()


