"""One player's card: projection, market, prior production, role, capital.

Two things here are load-bearing and neither is obvious.

**The allowlist is the leakage guarantee.** The matrix row served is the TARGET
season's, and that row's `fppg` is the training LABEL — null today, real the
moment in-season data lands. A "serve only `prior_`-prefixed columns" rule would
be wrong twice over: it would forbid two thirds of this payload (`role`,
`capital`, `college` and `team_ctx` carry no prefix) while still reading from
the same row. So the guarantee is an explicit allowlist that names every column
and contains no label. `test_detail_never_serves_the_training_label` pins it.

**Types must be coerced, not just NaN-guarded.** `_json_float` handles floats;
it does not handle the rest of what is actually in that frame. Measured:
`is_projected_starter` is float64 WITH NaNs but reads as a boolean;
`depth_chart_rank`, `draft_round`, `draft_pick_overall`,
`same_position_competition`, `vacated_targets` and `college_conference_tier`
are float64 but mean integers; `is_undrafted`, `has_college_stats` and
`has_depth_data` are numpy int64 but mean booleans. `json.dumps(np.int64(14))`
raises `TypeError`, `int(float("nan"))` raises `ValueError`, and Starlette's
`JSONResponse` renders with `allow_nan=False`, so a leaked NaN is a hard 500
rather than a malformed body. `_scalar` is the one door every value goes
through.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd

from .data import RankingsData

#: Prior production, by position. A quarterback has no catch rate and a
#: receiver has no sack rate; showing either as a null row is noise.
PRODUCTION: Mapping[str, tuple[str, ...]] = {
    "QB": ("prior_fppg", "prior_pass_attempts_per_game", "prior_completion_pct",
           "prior_yards_per_attempt", "prior_pass_td_rate", "prior_int_rate",
           "prior_sack_rate", "prior_rush_yards_per_game"),
    "RB": ("prior_fppg", "prior_carries_per_game", "prior_touches_per_game",
           "prior_yards_per_carry", "prior_rush_td_rate",
           "prior_targets_per_game", "prior_catch_rate", "prior_snap_share"),
    "WR": ("prior_fppg", "prior_targets_per_game", "prior_target_share",
           "prior_catch_rate", "prior_yards_per_target",
           "prior_yards_per_reception", "prior_rec_td_rate",
           "prior_snap_share"),
}
PRODUCTION["TE"] = PRODUCTION["WR"]

ROLE = ("depth_chart_rank", "is_projected_starter", "same_position_competition",
        "has_depth_data", "vacated_targets", "vacated_targets_share",
        "vacated_carries", "vacated_carries_share")
CAPITAL = ("draft_round", "draft_pick_overall", "is_undrafted",
           "draft_capital_score", "seasons_of_history", "team_changed",
           "age_at_season_start")
COLLEGE = ("college_rec_yds_z", "college_rec_td_z", "college_rush_yds_z",
           "college_rush_td_z", "college_scrimmage_yds_z",
           "college_conference_tier", "has_college_stats")
TEAM_CTX = ("team_ctx_implied_pts", "team_ctx_game_total", "team_ctx_spread")

#: Anything not named here is a float. Split out because the dtype on disk does
#: not tell you the meaning — six of these are float64 columns that mean ints.
_KINDS: Mapping[str, str] = {
    "depth_chart_rank": "int", "same_position_competition": "int",
    "vacated_targets": "int", "vacated_carries": "int",
    "draft_round": "int", "draft_pick_overall": "int",
    "seasons_of_history": "int", "college_conference_tier": "int",
    "is_projected_starter": "bool", "has_depth_data": "bool",
    "is_undrafted": "bool", "has_college_stats": "bool", "team_changed": "bool",
}

MARKET_PLATFORMS = ("yahoo", "espn", "sleeper", "underdog", "cbs", "ffpc")


def _scalar(value: Any, kind: str = "float") -> Any:
    """The single coercion door. Missing is `None` for every kind."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value) if kind == "str" else None
    if not math.isfinite(number):
        return None
    if kind == "int":
        return int(number)
    if kind == "bool":
        return bool(number)
    return number


def _block(row: Mapping[str, Any] | None,
           columns: tuple[str, ...]) -> dict[str, Any] | None:
    if row is None:
        return None
    return {name: _scalar(row.get(name), _KINDS.get(name, "float"))
            for name in columns}


def _matrix_row(data: RankingsData, player_id: str) -> dict[str, Any] | None:
    if data.matrix.empty or player_id not in data.matrix.index:
        return None
    row = data.matrix.loc[player_id]
    if isinstance(row, pd.DataFrame):          # duplicate ids; take the first
        row = row.iloc[0]
    return row.to_dict()


def _market(data: RankingsData, player_id: str,
            board_row: Mapping[str, Any]) -> dict[str, Any]:
    market: dict[str, Any] = {
        "adp": _scalar(board_row.get("adp")),
        "adp_stdev": _scalar(board_row.get("adp_stdev")),
        "matched": False, "consensus_rank": None, "rank_spread": None,
        "ranks": {},
    }
    key = data.match_keys.get(player_id)
    if not key or data.ranks.empty or key not in data.ranks.index:
        return market

    row = data.ranks.loc[key]
    if isinstance(row, pd.DataFrame):
        return market                          # ambiguous; claim nothing
    market["matched"] = True
    # Fractional on purpose: consensus_rank is an AVERAGE of platform ranks
    # (Smith-Njigba is 5.8), so typing it int would truncate real signal.
    market["consensus_rank"] = _scalar(row.get("consensus_rank"))
    market["rank_spread"] = _scalar(row.get("rank_spread"), "int")
    market["ranks"] = {
        platform: rank
        for platform in MARKET_PLATFORMS
        if (rank := _scalar(row.get(f"rank_{platform}"), "int")) is not None
    }
    return market


def _engine_tier(data: RankingsData, player_id: str) -> int | None:
    if data.tiers.empty:
        return None
    hit = data.tiers[data.tiers["player_id"].astype(str) == player_id]
    if hit.empty:
        return None
    return _scalar(hit.iloc[0].get("tier"), "int")


def player_detail(player_id: str, data: RankingsData) -> dict[str, Any] | None:
    """The full card, or None when the id is not on the board.

    Not-in-the-bundle is a 404 rather than a partial payload: a board can only
    ever hold bundle ids, so the caller is asking about someone who is not in
    this draft.
    """
    player_id = str(player_id)
    board = data.board
    if board.empty:
        return None
    hit = board[board["player_id"].astype(str) == player_id]
    if hit.empty:
        return None
    row = hit.iloc[0].to_dict()

    matrix_row = _matrix_row(data, player_id)
    position = str(row.get("position", ""))
    production_cols = PRODUCTION.get(position)

    college = None
    if matrix_row is not None and _scalar(
            matrix_row.get("has_college_stats"), "bool"):
        college = _block(matrix_row, COLLEGE)

    floor = data.replacement.get(position)
    value = _scalar(row.get("value"))

    return {
        "player_id": player_id,
        "name": str(row.get("player_name", "")),
        "position": position,
        "team": row.get("team") if isinstance(row.get("team"), str) else None,
        "bye_week": _scalar(row.get("bye_week"), "int"),
        "projection": {
            "value": value,
            "vor": None if value is None or floor is None else value - floor,
            "p10": _scalar(row.get("value_p10")),
            "p90": _scalar(row.get("value_p90")),
            "source": str(row.get("value_source", "")),
            "coverage": str(row.get("coverage", "")),
        },
        "market": _market(data, player_id, row),
        "production": (_block(matrix_row, production_cols)
                       if production_cols else None),
        "role": _block(matrix_row, ROLE),
        "capital": _block(matrix_row, CAPITAL),
        "college": college,
        "team_context": _block(matrix_row, TEAM_CTX),
        "engine_position_tier": _engine_tier(data, player_id),
        "has_matrix_row": matrix_row is not None,
    }


def catalogue(data: RankingsData) -> list[dict[str, Any]]:
    """Every bundle player, unfiltered by draft state.

    `PlayerRow` needs a name for each of 260 stored ids, and neither a board
    response (ids only) nor the one-player detail route can supply that.
    `/api/board` cannot either: it omits `value` and it filters out drafted
    players, so rows would silently vanish from a research board mid-draft.
    """
    if data.board.empty:
        return []
    frame = data.board
    floors = data.replacement
    out = []
    for row in frame.itertuples(index=False):
        position = str(getattr(row, "position", ""))
        value = _scalar(getattr(row, "value", None))
        floor = floors.get(position)
        out.append({
            "player_id": str(row.player_id),
            "name": str(getattr(row, "player_name", "")),
            "position": position,
            "team": getattr(row, "team", None)
            if isinstance(getattr(row, "team", None), str) else None,
            "bye_week": _scalar(getattr(row, "bye_week", None), "int"),
            "value": value,
            "vor": None if value is None or floor is None else value - floor,
            "adp": _scalar(getattr(row, "adp", None)),
        })
    return out
