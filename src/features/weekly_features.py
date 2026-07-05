"""Weekly in-season feature builders for the weekly projection pipeline.

Three feature blocks:
  * Opponent DvP (defense vs position) — EWMA-smoothed FPA by defense × position
  * Per-week Vegas context — implied team totals, spread, game total for a specific week
  * Rolling player stats — EWMA over recent games (snap share, targets, carries, FPPG)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.labels.scoring import score_dataframe

# An offseason decays prior-season signal like this many missed games. With
# halflife=3, the last game of season Y-1 carries an extra 0.5^(6/3) = 0.25x
# weight at Week 1 of season Y — recent form matters, but a hot December
# shouldn't dominate a September projection after roster/scheme turnover.
OFFSEASON_GAP_GAMES = 6.0


def _gap_aware_ewma_last(grp: pd.DataFrame, cols, *, halflife: float,
                         season_gap: float = OFFSEASON_GAP_GAMES) -> dict:
    """EWMA final value over ordered games, with extra decay across offseasons.

    Builds a synthetic game clock (consecutive games 1 apart, +``season_gap``
    when the season changes) and takes the weight-0.5^(dist/halflife) mean of
    each column, ignoring NaNs. ``grp`` must be sorted chronologically.
    """
    seasons = pd.to_numeric(grp["season"], errors="coerce").to_numpy()
    n = len(grp)
    t = np.arange(n, dtype=float)
    if n > 1:
        t += np.cumsum(np.r_[0.0, (seasons[1:] != seasons[:-1]) * season_gap])
    dist = t[-1] - t
    w = 0.5 ** (dist / halflife)
    out = {}
    for c in cols:
        v = pd.to_numeric(grp[c], errors="coerce").to_numpy(dtype=float)
        m = ~np.isnan(v)
        out[c] = float((w[m] * v[m]).sum() / w[m].sum()) if m.any() else np.nan
    return out


def build_weekly_vegas(schedules: pd.DataFrame, target_season: int,
                       target_week: int) -> pd.DataFrame:
    """Per-team Vegas context for a specific week.

    Returns (team, opponent, wk_implied_pts, wk_game_total, wk_spread).
    One row per team (two rows per game). Empty if no lines available.
    """
    from ._utils import schedule_to_team_lines

    empty = pd.DataFrame(columns=["team", "opponent", "wk_implied_pts",
                                  "wk_game_total", "wk_spread"])
    if schedules is None or schedules.empty:
        return empty
    s = schedules.copy()
    s["season"] = pd.to_numeric(s.get("season"), errors="coerce")
    s["week"] = pd.to_numeric(s.get("week"), errors="coerce")
    s = s[(s["season"] == target_season) & (s["week"] == target_week)]
    if "game_type" in s.columns:
        s = s[s["game_type"] == "REG"]
    for c in ("total_line", "spread_line"):
        if c not in s.columns:
            return empty
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s.dropna(subset=["total_line", "spread_line", "home_team", "away_team"])
    if s.empty:
        return empty

    rename = {"implied_pts": "wk_implied_pts", "game_total": "wk_game_total",
              "spread": "wk_spread"}
    return schedule_to_team_lines(s).rename(columns=rename)


def build_opponent_dvp(weekly_labels: pd.DataFrame, schedules: pd.DataFrame,
                       target_season: int, target_week: int,
                       halflife_games: float = 3.0) -> pd.DataFrame:
    """Fantasy points allowed by each defense to each position, EWMA-smoothed.

    `weekly_labels` must be pre-filtered via prior_weeks() by the caller.
    Returns (team, position, dvp_fpa) where team is the DEFENSE.
    """
    empty = pd.DataFrame(columns=["team", "position", "dvp_fpa"])
    if weekly_labels is None or weekly_labels.empty:
        return empty
    if schedules is None or schedules.empty:
        return empty

    wl = weekly_labels[["player_id", "season", "week", "position", "team",
                        "fantasy_points"]].copy()
    sched = schedules[["season", "week", "home_team", "away_team"]].copy()
    sched["season"] = pd.to_numeric(sched["season"], errors="coerce")
    sched["week"] = pd.to_numeric(sched["week"], errors="coerce")

    merged = wl.merge(sched, on=["season", "week"], how="inner")
    home_mask = merged["team"] == merged["home_team"]
    merged["opponent"] = np.where(home_mask, merged["away_team"], merged["home_team"])
    merged = merged[home_mask | (merged["team"] == merged["away_team"])]

    if merged.empty:
        return empty

    game_totals = (
        merged.groupby(["opponent", "position", "season", "week"])["fantasy_points"]
        .sum()
        .reset_index()
    )
    game_totals = game_totals.sort_values(["opponent", "position", "season", "week"])

    results = []
    for (opp, pos), grp in game_totals.groupby(["opponent", "position"]):
        vals = _gap_aware_ewma_last(grp, ["fantasy_points"], halflife=halflife_games)
        results.append({"team": opp, "position": pos, "dvp_fpa": vals["fantasy_points"]})
    return pd.DataFrame(results) if results else empty


def build_rolling_player_stats(player_stats: pd.DataFrame,
                               snap_counts: pd.DataFrame,
                               pfr_to_gsis: dict,
                               halflife_games: float = 3.0) -> pd.DataFrame:
    """Rolling EWMA over recent games for each player.

    Caller must pre-filter both inputs via prior_weeks() AND season_type == 'REG'.

    Returns one row per player_id with rolling_fppg, rolling_target_share,
    rolling_snap_share, rolling_carries_pg.
    """
    empty = pd.DataFrame(columns=["player_id", "rolling_fppg", "rolling_target_share",
                                  "rolling_snap_share", "rolling_carries_pg"])
    if player_stats is None or player_stats.empty:
        return empty

    ps = player_stats.copy()
    ps["player_id"] = ps["player_id"].astype(str)
    ps["season"] = pd.to_numeric(ps["season"], errors="coerce")
    ps["week"] = pd.to_numeric(ps["week"], errors="coerce")
    ps["fppg_raw"] = score_dataframe(ps)

    for c in ["target_share", "carries"]:
        if c not in ps.columns:
            ps[c] = np.nan
        else:
            ps[c] = pd.to_numeric(ps[c], errors="coerce")

    if snap_counts is not None and not snap_counts.empty and pfr_to_gsis:
        sc = snap_counts.copy()
        sc["player_id"] = sc["pfr_player_id"].astype("string").map(pfr_to_gsis)
        sc = sc[sc["player_id"].notna()]
        if "offense_pct" in sc.columns:
            sc["offense_pct"] = pd.to_numeric(sc["offense_pct"], errors="coerce")
            sc["season"] = pd.to_numeric(sc["season"], errors="coerce")
            sc["week"] = pd.to_numeric(sc["week"], errors="coerce")
            sc["player_id"] = sc["player_id"].astype(str)
            sc_lookup = sc[["player_id", "season", "week", "offense_pct"]].rename(
                columns={"offense_pct": "snap_share"})
            ps = ps.merge(sc_lookup, on=["player_id", "season", "week"], how="left")
        else:
            ps["snap_share"] = np.nan
    else:
        ps["snap_share"] = np.nan

    ps = ps.sort_values(["player_id", "season", "week"])
    results = []
    for pid, grp in ps.groupby("player_id"):
        last = _gap_aware_ewma_last(
            grp, ["fppg_raw", "target_share", "snap_share", "carries"],
            halflife=halflife_games)
        results.append({
            "player_id": pid,
            "rolling_fppg": last["fppg_raw"],
            "rolling_target_share": last["target_share"],
            "rolling_snap_share": last["snap_share"],
            "rolling_carries_pg": last["carries"],
        })
    return pd.DataFrame(results) if results else empty
