"""Season-grain database schema (DrafterSpec.md §4.3).

The old player-week grain (features/labels/predictions/games/teams) is retired.
The unit of modeling is one (player, season) row (§4.0). Raw staging tables carry
``snapshot_id`` + ``extracted_at`` so training/benchmarking pin a frozen snapshot
and stay reproducible against nflverse retro-corrections (§4.0).
"""
from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
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
    """Canonical player dimension keyed by gsis_id (§4.0 canonical ID)."""

    __tablename__ = "players"

    player_id = Column(String, primary_key=True)  # gsis_id
    name = Column(String, nullable=False)
    position = Column(String, nullable=True)
    college = Column(String, nullable=True)
    rookie_year = Column(Integer, nullable=True)
    draft_year = Column(Integer, nullable=True)
    draft_round = Column(Integer, nullable=True)
    draft_pick = Column(Integer, nullable=True)
    draft_team = Column(String, nullable=True)
    birth_date = Column(String, nullable=True)
    latest_team = Column(String, nullable=True)


class PlayerIdMap(Base):
    """gsis_id <-> external source IDs, built from the ff_playerids crosswalk."""

    __tablename__ = "player_id_map"

    gsis_id = Column(String, primary_key=True)
    fantasypros_id = Column(String, nullable=True, index=True)
    sleeper_id = Column(String, nullable=True)
    espn_id = Column(String, nullable=True)
    pfr_id = Column(String, nullable=True, index=True)
    cfb_id = Column(String, nullable=True, index=True)  # cfbref/cfb player id for college bridge
    merge_name = Column(String, nullable=True, index=True)
    years_exp = Column(Integer, nullable=True)


class WeeklyStatsRaw(Base):
    """Raw weekly box-score components; staging input for label computation (§4.4.1).

    Stores only raw component stats — NEVER the precomputed fantasy_points_ppr.
    The full component set lives in ``stats`` JSONB so scoring.py can be the one
    place rules become numbers.
    """

    __tablename__ = "weekly_stats_raw"

    player_id = Column(String, primary_key=True)
    season = Column(Integer, primary_key=True)
    week = Column(Integer, primary_key=True)
    season_type = Column(String, primary_key=True, default="REG")
    team = Column(String, nullable=True)
    position = Column(String, nullable=True)
    stats = Column(JSONB, nullable=False)
    snapshot_id = Column(String, nullable=False, index=True)
    extracted_at = Column(DateTime, nullable=False, server_default=func.now())

    __table_args__ = (Index("ix_weekly_stats_raw_season", "season"),)


class SeasonLabel(Base):
    """Per (player, season) target: games_played, season total, per-game (§4.4)."""

    __tablename__ = "season_labels"

    player_id = Column(String, primary_key=True)
    season = Column(Integer, primary_key=True)
    games_played = Column(Integer, nullable=False)
    fantasy_points_total = Column(Float, nullable=False)
    fppg = Column(Float, nullable=False)
    snapshot_id = Column(String, nullable=False, index=True)

    __table_args__ = (Index("ix_season_labels_season", "season"),)


class SeasonFeature(Base):
    """Per (player, season) preseason-safe feature row (§4.5).

    Features stored as opaque JSONB (fluid set); leakage is enforced at
    construction time, not by SQL (§4.0).
    """

    __tablename__ = "season_features"

    player_id = Column(String, primary_key=True)
    season = Column(Integer, primary_key=True)
    position = Column(String, nullable=True)  # as-of season Y (§4.0)
    is_rookie = Column(Boolean, nullable=False, default=False)
    as_of_date = Column(String, nullable=True)
    features = Column(JSONB, nullable=False)
    snapshot_id = Column(String, nullable=False, index=True)

    __table_args__ = (Index("ix_season_features_season", "season"),)


class Projection(Base):
    """Per (player, season, model_version) quantile projections (§4.6)."""

    __tablename__ = "projections"

    player_id = Column(String, primary_key=True)
    season = Column(Integer, primary_key=True)
    model_version = Column(String, primary_key=True)
    position = Column(String, nullable=True)
    p10 = Column(Float, nullable=False)
    p50 = Column(Float, nullable=False)
    p90 = Column(Float, nullable=False)
    pos_rank = Column(Integer, nullable=True)
    snapshot_id = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

    __table_args__ = (Index("ix_projections_season", "season"),)


class Adp(Base):
    """Market data — quarantined in its own table so the wall is structural (§4.0/§4.3).

    Only ingest/recommender/benchmark may read this; projection/features never do.
    """

    __tablename__ = "adp"

    source = Column(String, primary_key=True, default="ffc")
    season = Column(Integer, primary_key=True)
    format = Column(String, primary_key=True)
    teams = Column(Integer, primary_key=True)
    player_id = Column(String, primary_key=True)  # gsis_id (resolved via crosswalk)
    name = Column(String, nullable=True)
    position = Column(String, nullable=True)
    adp = Column(Float, nullable=False)
    adp_stdev = Column(Float, nullable=True)
    n_drafts = Column(Integer, nullable=True)
    snapshot_id = Column(String, nullable=False, index=True)
    extracted_at = Column(DateTime, nullable=False, server_default=func.now())

    __table_args__ = (Index("ix_adp_season_format", "season", "format", "teams"),)


class WeeklyLabel(Base):
    """Per (player, week) fantasy points scored — weekly grain of SeasonLabel."""

    __tablename__ = "weekly_labels"

    player_id = Column(String, primary_key=True)
    season = Column(Integer, primary_key=True)
    week = Column(Integer, primary_key=True)
    position = Column(String, nullable=True)
    team = Column(String, nullable=True)
    fantasy_points = Column(Float, nullable=False)
    snapshot_id = Column(String, nullable=False, index=True)

    __table_args__ = (Index("ix_weekly_labels_season", "season"),)


class WeeklyFeature(Base):
    """Per (player, week) feature vector for the weekly projection model."""

    __tablename__ = "weekly_features"

    player_id = Column(String, primary_key=True)
    season = Column(Integer, primary_key=True)
    week = Column(Integer, primary_key=True)
    position = Column(String, nullable=True)
    features = Column(JSONB, nullable=False)
    snapshot_id = Column(String, nullable=False, index=True)

    __table_args__ = (Index("ix_weekly_features_season", "season"),)


class WeeklyProjection(Base):
    """Per (player, week) quantile projections from the weekly model."""

    __tablename__ = "weekly_projections"

    player_id = Column(String, primary_key=True)
    season = Column(Integer, primary_key=True)
    week = Column(Integer, primary_key=True)
    model_version = Column(String, primary_key=True)
    position = Column(String, nullable=True)
    p10 = Column(Float, nullable=False)
    p50 = Column(Float, nullable=False)
    p90 = Column(Float, nullable=False)
    snapshot_id = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

    __table_args__ = (Index("ix_weekly_projections_season", "season"),)


class IngestSnapshot(Base):
    """One row per ingest run for reproducibility (§4.0)."""

    __tablename__ = "ingest_snapshots"

    snapshot_id = Column(String, primary_key=True)
    extracted_at = Column(DateTime, nullable=False, server_default=func.now())
    nflreadpy_version = Column(String, nullable=True)
    notes = Column(String, nullable=True)
    coverage = Column(JSONB, nullable=True)  # per-season ADP coverage counts + eligibility


class AdpCoverage(Base):
    """Per-season ADP coverage counts + benchmark eligibility flags (§4.0)."""

    __tablename__ = "adp_coverage"

    season = Column(Integer, primary_key=True)
    format = Column(String, primary_key=True)
    teams = Column(Integer, primary_key=True)
    n_players = Column(Integer, nullable=False)
    ranking_eligible = Column(Boolean, nullable=False, default=False)
    sim_eligible = Column(Boolean, nullable=False, default=False)
    snapshot_id = Column(String, nullable=False)
