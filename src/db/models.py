from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func


Base = declarative_base()


class Player(Base):
    __tablename__ = "players"

    player_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    position = Column(String, nullable=False)
    team_current = Column(String, nullable=True)


class Team(Base):
    __tablename__ = "teams"

    team_name = Column(String, primary_key=True)


class Game(Base):
    __tablename__ = "games"

    game_id = Column(String, primary_key=True)
    season = Column(Integer, nullable=False, index=True)
    week = Column(Integer, nullable=False, index=True)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)

    __table_args__ = (
        Index("ix_games_season_week", "season", "week"),
    )


class Feature(Base):
    __tablename__ = "features"

    player_id = Column(String, primary_key=True)
    season = Column(Integer, primary_key=True)
    week = Column(Integer, primary_key=True)
    opponent_team = Column(String, nullable=False, index=True)
    player_features = Column(JSONB, nullable=False)
    matchup_features = Column(JSONB, nullable=True)
    team = Column(String, nullable=True, index=True)
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("ix_features_season_week", "season", "week"),
    )


class Label(Base):
    __tablename__ = "labels"

    player_id = Column(String, primary_key=True)
    season = Column(Integer, primary_key=True)
    week = Column(Integer, primary_key=True)
    fantasy_points = Column(Float, nullable=False)

    __table_args__ = (
        Index("ix_labels_season_week", "season", "week"),
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    player_id = Column(String, nullable=False)
    season = Column(Integer, nullable=False)
    week = Column(Integer, nullable=False)
    opponent_team = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    y_pred = Column(Float, nullable=False)
    y_std = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

    __table_args__ = (
        Index("ix_predictions_season_week", "season", "week"),
        Index("ix_predictions_player", "player_id"),
        UniqueConstraint(
            "player_id", "season", "week", "model_version", name="uq_prediction_by_model"
        ),
    )


    


