"""Column availability registry (§11.3).

Every column a feature may read declares *when it became knowable*. The default
is refusal, not permission: an unregistered column cannot be read at all, which
is what stops a new source quietly entering the feature set without anyone
deciding its cutoff.

Availability literals and the timestamp each implies:

===============  ==========================================================
``season_end``   Week-1 kickoff of ``season + 1`` — season aggregates
``week_end``     kickoff of ``(season, week + 1)`` — weekly rows
``preseason``    Week-1 kickoff of ``season`` minus ``offset_days``
``static``       always available — combine, draft picks, birthdates
===============  ==========================================================

**Only Week-1 betting lines are ``preseason``.** Later-week lines exist in the
same frame and are emphatically not preseason-available; §12.5 conditions K/DST
on the Week-1 line for exactly this reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from src.core.errors import UnregisteredColumnError

Availability = Literal["season_end", "week_end", "preseason", "static"]


@dataclass(frozen=True)
class ColumnSpec:
    source: str
    column: str
    availability: Availability
    offset_days: int = 0


REGISTRY: dict[tuple[str, str], ColumnSpec] = {}


def register(
    source: str, columns: tuple[str, ...], availability: Availability,
    *, offset_days: int = 0,
) -> None:
    for column in columns:
        REGISTRY[(source, column)] = ColumnSpec(source, column, availability, offset_days)


def availability_of(source: str, column: str) -> ColumnSpec:
    spec = REGISTRY.get((source, column))
    if spec is None:
        raise UnregisteredColumnError(
            f"column {column!r} of source {source!r} has no availability "
            f"declaration. Register it in platform.asof.registry before a "
            f"feature reads it — the default is refusal, not permission."
        )
    return spec


def is_registered(source: str, column: str) -> bool:
    return (source, column) in REGISTRY


# --------------------------------------------------------------------------- #
# nflverse registrations, verified against nflreadpy 0.1.3 on 2026-08-04
# --------------------------------------------------------------------------- #
_WEEKLY_STAT_COLUMNS: Final = (
    "passing_yards", "passing_tds", "passing_interceptions", "attempts",
    "completions", "rushing_yards", "rushing_tds", "carries", "receptions",
    "receiving_yards", "receiving_tds", "targets", "special_teams_tds",
    "passing_2pt_conversions", "rushing_2pt_conversions",
    "receiving_2pt_conversions", "rushing_fumbles_lost",
    "receiving_fumbles_lost", "sack_fumbles_lost", "fumble_recovery_tds",
    "target_share", "air_yards_share", "wopr", "racr", "pacr",
    "fg_made_0_19", "fg_made_20_29", "fg_made_30_39", "fg_made_40_49",
    "fg_made_50_59", "fg_made_60_", "fg_missed", "pat_made",
)
register("player_stats", _WEEKLY_STAT_COLUMNS, "week_end")
register("player_stats", ("player_id", "player_display_name", "position",
                          "team", "season", "week", "season_type",
                          "opponent_team"), "week_end")

_TEAM_STAT_COLUMNS: Final = (
    "def_sacks", "def_interceptions", "def_safeties", "def_tds",
    "fumble_recovery_opp", "fg_blocked", "pat_blocked", "pt_blocked",
    "pt_return_tds",
)
register("team_stats", _TEAM_STAT_COLUMNS, "week_end")
register("team_stats", ("team", "season", "week", "season_type"), "week_end")

# Week-1 lines only; the accessor filters to week 1 before this applies.
register("schedules", ("total_line", "spread_line", "home_team", "away_team",
                       "gameday", "gametime", "week", "season", "game_type"),
         "preseason", offset_days=1)
# Final scores are outcomes, never preseason.
register("schedules", ("home_score", "away_score", "result", "total"), "week_end")

register("rosters", ("player_id", "position", "team", "status", "season",
                     "week", "depth_chart_position"), "preseason", offset_days=1)
register("depth_charts", ("player_id", "position", "team", "depth_team",
                          "season", "week"), "preseason", offset_days=1)
register("injuries", ("player_id", "team", "season", "week", "report_status",
                      "practice_status"), "week_end")

register("combine", ("player_id", "pfr_id", "ht", "wt", "forty", "bench",
                     "vertical", "broad_jump", "cone", "shuttle", "season"),
         "static")
register("draft_picks", ("player_id", "pfr_player_id", "season", "round",
                         "pick", "team"), "static")
register("players", ("gsis_id", "display_name", "position", "birth_date",
                     "draft_year", "draft_round", "draft_pick"), "static")
