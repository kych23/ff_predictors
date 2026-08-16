"""Play-by-play features: red-zone opportunity, routes, pace, and defence.

Four things the season box score cannot express, all of which fall out of one
play-level pull:

**Red-zone opportunity.** A carry on the two-yard line and a carry at midfield
are the same row in a season total and completely different fantasy events.
Touchdowns are where scoring concentrates and red-zone usage is how a coach
says who gets them. `rz_targets_per_game` and `rz_carries_per_game` existed as
placeholder columns hardcoded to NaN and were dropped before training — the
model has never seen them.

**Routes run**, from `participation`'s on-field lists. Not a published stat in
nflverse, but a receiver on the field for a pass play has run a route, and that
is the standard proxy. It separates a player who is on the field and ignored
from one who is not on the field at all — two very different reasons to have
few targets, and identical in a target count.

**Team pace and pass rate.** How many plays an offence runs, and how it splits
them, is the size of the pie every teammate divides. A 68-play pass-first
offence and a 58-play run-first one support completely different receiving
lines from identical talent.

**Defence allowed, by position.** What an opponent gives up to backs is not
what it gives up to receivers, and a season-long "strength of schedule" number
averages that distinction away.

Everything here is aggregated to PRIOR seasons by the caller and passes through
`platform.asof.guards` like every other backward-looking source. Nothing reads
a play from the season being projected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Inside this many yards from the end zone is the red zone.
RED_ZONE_YARDS = 20

#: Columns pulled from the play-by-play. Selected rather than taking all 372,
#: because a decade of full play-by-play does not need to be in memory to count
#: carries inside the twenty.
PBP_COLUMNS = (
    "season", "week", "posteam", "defteam", "play_type", "yardline_100",
    "receiver_player_id", "rusher_player_id", "pass_attempt", "rush_attempt",
    "yards_gained", "touchdown", "epa", "game_seconds_remaining", "game_id",
)

PLAYER_COLS = ["rz_targets_per_game", "rz_carries_per_game",
               "rz_share_targets", "rz_share_carries"]
ROUTE_COLS = ["routes_per_game", "route_participation", "targets_per_route"]
TEAM_COLS = ["team_plays_per_game", "team_pass_rate", "team_rz_plays_per_game"]
DEFENCE_COLS = ["def_fp_allowed_rb", "def_fp_allowed_wr", "def_fp_allowed_te"]


def _per_game(counts: pd.Series, games: pd.Series) -> pd.Series:
    return counts / games.clip(lower=1)


def red_zone_opportunity(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (player, season) red-zone targets and carries, absolute and as share.

    The SHARE matters as much as the count. Ten red-zone carries on a team with
    forty is a different role from ten on a team with twelve, and only the
    share says which.
    """
    if pbp is None or pbp.empty:
        return pd.DataFrame(columns=["player_id", "season", *PLAYER_COLS])

    plays = pbp[pbp["yardline_100"] <= RED_ZONE_YARDS]
    plays = plays[plays["play_type"].isin(["pass", "run"])]
    if plays.empty:
        return pd.DataFrame(columns=["player_id", "season", *PLAYER_COLS])

    frames = []
    for column, label in (("receiver_player_id", "rz_targets"),
                          ("rusher_player_id", "rz_carries")):
        sub = plays[plays[column].notna()]
        if sub.empty:
            continue
        per_player = (sub.groupby([column, "season", "posteam"])
                         .size().rename(label).reset_index()
                         .rename(columns={column: "player_id"}))
        team_total = (sub.groupby(["season", "posteam"])
                         .size().rename(f"{label}_team").reset_index())
        merged = per_player.merge(team_total, on=["season", "posteam"])
        merged[f"{label}_share"] = (
            merged[label] / merged[f"{label}_team"].replace(0, np.nan))
        frames.append(merged[["player_id", "season", label,
                              f"{label}_share"]])

    if not frames:
        return pd.DataFrame(columns=["player_id", "season", *PLAYER_COLS])

    out = frames[0]
    for extra in frames[1:]:
        out = out.merge(extra, on=["player_id", "season"], how="outer")

    games = (pbp.groupby(["season"])["game_id"].nunique()
                .rename("league_games").reset_index())
    out = out.merge(games, on="season", how="left")
    # Each team plays roughly `league_games / (teams/2) ...` — but a player's
    # own game count is the honest denominator, so use his appearances.
    appearances = _player_games(pbp)
    out = out.merge(appearances, on=["player_id", "season"], how="left")
    played = out["games"].fillna(1)

    # A season with no red-zone passes at all leaves the column absent, not
    # empty — `.get(name, 0)` would then return a bare int and every rate below
    # would break on it.
    def _column(name: str) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=out.index)

    out["rz_targets_per_game"] = _per_game(_column("rz_targets"), played)
    out["rz_carries_per_game"] = _per_game(_column("rz_carries"), played)
    out["rz_share_targets"] = (out["rz_targets_share"]
                               if "rz_targets_share" in out.columns else np.nan)
    out["rz_share_carries"] = (out["rz_carries_share"]
                               if "rz_carries_share" in out.columns else np.nan)
    return out[["player_id", "season", *PLAYER_COLS]]


def _player_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """Games a player appears in, from either side of the ball he touched."""
    pieces = []
    for column in ("receiver_player_id", "rusher_player_id"):
        sub = pbp[pbp[column].notna()][[column, "season", "game_id"]]
        pieces.append(sub.rename(columns={column: "player_id"}))
    if not pieces:
        return pd.DataFrame(columns=["player_id", "season", "games"])
    every = pd.concat(pieces, ignore_index=True)
    return (every.groupby(["player_id", "season"])["game_id"]
                 .nunique().rename("games").reset_index())


def routes_run(participation: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (player, season) routes, from who was on the field for a pass.

    A receiver on the field for a pass play ran a route. It is a proxy and it
    is the one nflverse supports — routes run is otherwise a paid stat — and it
    separates the two reasons a player has few targets: on the field and
    ignored, or not on the field at all. A target count cannot tell those
    apart, and they mean opposite things about his role.
    """
    if participation is None or participation.empty:
        return pd.DataFrame(columns=["player_id", "season", *ROUTE_COLS])
    if "offense_players" not in participation.columns:
        return pd.DataFrame(columns=["player_id", "season", *ROUTE_COLS])

    passes = pbp[pbp["play_type"] == "pass"][["game_id", "week", "season"]]
    key = participation.rename(columns={"nflverse_game_id": "game_id"})
    key = key[key["offense_players"].notna()]
    if "season" not in key.columns:
        key = key.merge(passes.drop_duplicates("game_id"), on="game_id",
                        how="inner")
    if key.empty:
        return pd.DataFrame(columns=["player_id", "season", *ROUTE_COLS])

    exploded = key.assign(
        player_id=key["offense_players"].str.split(";")).explode("player_id")
    exploded["player_id"] = exploded["player_id"].str.strip()
    exploded = exploded[exploded["player_id"].astype(bool)]

    routes = (exploded.groupby(["player_id", "season"])
                      .size().rename("routes").reset_index())
    games = (exploded.groupby(["player_id", "season"])["game_id"]
                     .nunique().rename("games").reset_index())
    routes = routes.merge(games, on=["player_id", "season"], how="left")

    team_passes = (exploded.groupby(["season"])
                           .size().rename("league_snaps").reset_index())
    routes = routes.merge(team_passes, on="season", how="left")

    routes["routes_per_game"] = _per_game(routes["routes"], routes["games"])
    # How much of his own team's dropback volume he was present for.
    routes["route_participation"] = (
        routes["routes"] / routes["league_snaps"].replace(0, np.nan))
    routes["targets_per_route"] = np.nan     # filled by the caller if wanted
    return routes[["player_id", "season", *ROUTE_COLS]]


def team_environment(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (team, season) pace and pass rate — the size of the pie.

    A 68-play pass-first offence and a 58-play run-first one support completely
    different receiving lines from identical talent, and neither shows up in a
    player's own prior production.
    """
    if pbp is None or pbp.empty:
        return pd.DataFrame(columns=["team", "season", *TEAM_COLS])

    offense = pbp[pbp["play_type"].isin(["pass", "run"])].copy()
    if offense.empty:
        return pd.DataFrame(columns=["team", "season", *TEAM_COLS])

    grouped = offense.groupby(["posteam", "season"])
    out = grouped.agg(
        plays=("play_type", "size"),
        games=("game_id", "nunique"),
        passes=("play_type", lambda s: (s == "pass").sum()),
    ).reset_index().rename(columns={"posteam": "team"})

    red = offense[offense["yardline_100"] <= RED_ZONE_YARDS]
    rz = (red.groupby(["posteam", "season"]).size()
             .rename("rz_plays").reset_index()
             .rename(columns={"posteam": "team"}))
    out = out.merge(rz, on=["team", "season"], how="left")

    out["team_plays_per_game"] = _per_game(out["plays"], out["games"])
    out["team_pass_rate"] = out["passes"] / out["plays"].replace(0, np.nan)
    out["team_rz_plays_per_game"] = _per_game(
        out["rz_plays"].fillna(0), out["games"])
    return out[["team", "season", *TEAM_COLS]]


def defence_allowed(pbp: pd.DataFrame, offense_scoring) -> pd.DataFrame:
    """Per (defence, season) fantasy points allowed to each skill position.

    A season-long "strength of schedule" number averages away the distinction
    that matters: a defence stout against the run and porous against receivers
    is two different matchups depending on who you are starting.
    """
    if pbp is None or pbp.empty:
        return pd.DataFrame(columns=["team", "season", *DEFENCE_COLS])
    plays = pbp[pbp["play_type"].isin(["pass", "run"])]
    if plays.empty or "defteam" not in plays.columns:
        return pd.DataFrame(columns=["team", "season", *DEFENCE_COLS])

    # Yards and touchdowns conceded, split by whether the ball went to a back
    # or through the air. Scored through the league's own coefficients so this
    # is fantasy points and not a raw yardage number.
    rush = plays[plays["rusher_player_id"].notna()]
    rec = plays[plays["receiver_player_id"].notna()]

    def _allowed(sub: pd.DataFrame, yard_key: str, td_key: str) -> pd.DataFrame:
        if sub.empty:
            return pd.DataFrame(columns=["defteam", "season", "fp"])
        agg = sub.groupby(["defteam", "season"]).agg(
            yards=("yards_gained", "sum"),
            tds=("touchdown", "sum"),
            games=("game_id", "nunique"),
        ).reset_index()
        points = (agg["yards"] * float(offense_scoring.get(yard_key, 0.0))
                  + agg["tds"] * float(offense_scoring.get(td_key, 0.0)))
        agg["fp"] = points / agg["games"].clip(lower=1)
        return agg[["defteam", "season", "fp"]]

    rush_fp = _allowed(rush, "rush_yd", "rush_td").rename(
        columns={"fp": "def_fp_allowed_rb"})
    rec_fp = _allowed(rec, "rec_yd", "rec_td").rename(
        columns={"fp": "def_fp_allowed_wr"})

    out = rush_fp.merge(rec_fp, on=["defteam", "season"], how="outer")
    # Tight ends are receivers; without a position join at play level the
    # honest move is to reuse the receiving number rather than invent a third.
    out["def_fp_allowed_te"] = out["def_fp_allowed_wr"]
    return out.rename(columns={"defteam": "team"})[
        ["team", "season", *DEFENCE_COLS]]
