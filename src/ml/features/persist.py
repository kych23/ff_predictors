from __future__ import annotations

from typing import Dict

import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.db.init_db import init_db
from src.db.models import Feature, Label
from src.db.session import SessionLocal
from .clean_data import clean_player_stats
import nflreadpy as nfl


def _chunk_iter(df: pd.DataFrame, size: int = 5_000):
    for start in range(0, len(df), size):
        yield df.iloc[start : start + size]


def _build_team_week_features(years: list[int]) -> pd.DataFrame:
    """Create per-team/week matchup features from schedules.

    Output columns: season, week, team, opponent_team, is_home,
    team_implied_total, game_spread_team_view, temp, wind, is_indoor
    """
    sch = nfl.load_schedules(seasons=years)
    sch = sch.to_pandas() if hasattr(sch, "to_pandas") else sch
    use = sch[[
        "season", "week", "home_team", "away_team",
        "total_line", "total", "spread_line", "roof", "temp", "wind",
    ]].copy()
    use["total_for_calc"] = use["total_line"].fillna(use["total"])  # prefer total_line
    # Implied totals from home spread convention
    use["home_itt"] = use["total_for_calc"] / 2 - use["spread_line"] / 2
    use["away_itt"] = use["total_for_calc"] / 2 + use["spread_line"] / 2
    # is_indoor from roof
    use["is_indoor"] = use["roof"].astype(str).str.lower().isin(["dome", "closed"]) 

    # Build home rows
    home = use.rename(columns={
        "home_team": "team",
        "away_team": "opponent_team",
    })[
        ["season", "week", "team", "opponent_team", "spread_line", "home_itt", "temp", "wind", "is_indoor"]
    ].copy()
    home["is_home"] = True
    home["team_implied_total"] = home["home_itt"]
    home["game_spread_team_view"] = home["spread_line"]

    # Build away rows
    away = use.rename(columns={
        "away_team": "team",
        "home_team": "opponent_team",
    })[
        ["season", "week", "team", "opponent_team", "spread_line", "away_itt", "temp", "wind", "is_indoor"]
    ].copy()
    away["is_home"] = False
    away["team_implied_total"] = away["away_itt"]
    away["game_spread_team_view"] = -away["spread_line"]

    tw = pd.concat([
        home[["season","week","team","opponent_team","is_home","team_implied_total","game_spread_team_view","temp","wind","is_indoor"]],
        away[["season","week","team","opponent_team","is_home","team_implied_total","game_spread_team_view","temp","wind","is_indoor"]],
    ], ignore_index=True)
    return tw


def _build_def_pos_allowed_avg5(years: list[int]) -> pd.DataFrame:
    """Compute rolling 5-week average fantasy points allowed by a defense to a position.

    Returns columns: season, week, opponent_team (defense), position, opp_pos_fp_allowed_avg5
    """
    stats = clean_player_stats(years)
    # Expect columns: player_id, season, week, team, opponent_team, position, fantasy_points_ppr
    if "fantasy_points_ppr" not in stats.columns:
        return pd.DataFrame()
    wk = (
        stats
        .groupby(["season", "week", "opponent_team", "position"], as_index=False)["fantasy_points_ppr"]
        .mean()
        .rename(columns={"fantasy_points_ppr": "wk_fp_allowed"})
    )
    # Rolling mean per defense-position within a season
    def _per_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(["season", "week"]).copy()
        # leakage-safe: shift by 1 before rolling
        prior = g["wk_fp_allowed"].shift(1)
        g["opp_pos_fp_allowed_avg5"] = prior.rolling(5, min_periods=1).mean()
        return g
    allow = (
        wk.groupby(["season", "opponent_team", "position"], group_keys=False)
        .apply(_per_group)
    )
    return allow[["season", "week", "opponent_team", "position", "opp_pos_fp_allowed_avg5"]]


def persist_priors_to_features(priors: Dict[str, pd.DataFrame], years: list[int] | None = None) -> int:
    """Upsert priors into the features table as player_features per player-week,
    and attach matchup_features computed from schedules.

    Returns number of rows attempted.
    """
    init_db()
    session: Session = SessionLocal()
    total = 0
    try:
        years = years or []
        tw = _build_team_week_features(years) if years else pd.DataFrame()
        allow = _build_def_pos_allowed_avg5(years) if years else pd.DataFrame()
        for df in priors.values():
            if df.empty:
                continue
            prior_cols = [c for c in df.columns if any(s in c for s in ("_avg_", "_ewm"))] + ["gp_prior"]
            id_cols = [c for c in ["player_id", "season", "week", "opponent_team", "team", "position"] if c in df.columns]
            keep = id_cols + prior_cols
            df2 = df[keep].copy()

            # Join matchup features if available
            if not tw.empty and "team" in df2.columns:
                df2 = df2.merge(
                    tw,
                    how="left",
                    on=["season", "week", "team"],
                    suffixes=("", "_m")
                )
            # Join opponent allowed by position if available
            if not allow.empty and {"opponent_team", "position"}.issubset(df2.columns):
                df2 = df2.merge(
                    allow,
                    how="left",
                    on=["season", "week", "opponent_team", "position"],
                )

            records = []
            for _, r in df2.iterrows():
                player_features = {k: float(r[k]) for k in prior_cols if pd.notna(r[k])}
                matchup_features = None
                if "team_implied_total" in df2.columns:
                    matchup_features = {
                        "team_implied_total": (float(r["team_implied_total"]) if pd.notna(r.get("team_implied_total")) else None),
                        "game_spread_team_view": (float(r["game_spread_team_view"]) if pd.notna(r.get("game_spread_team_view")) else None),
                        "temp": (float(r["temp"]) if pd.notna(r.get("temp")) else None),
                        "wind": (float(r["wind"]) if pd.notna(r.get("wind")) else None),
                        "is_indoor": (bool(r["is_indoor"]) if pd.notna(r.get("is_indoor")) else None),
                        "opp_pos_fp_allowed_avg5": (float(r["opp_pos_fp_allowed_avg5"]) if pd.notna(r.get("opp_pos_fp_allowed_avg5")) else None),
                    }
                records.append({
                    "player_id": r["player_id"],
                    "season": int(r["season"]),
                    "week": int(r["week"]),
                    "opponent_team": r.get("opponent_team", ""),
                    "team": r.get("team"),
                    "player_features": player_features,
                    "matchup_features": matchup_features,
                })

            for chunk in _chunk_iter(pd.DataFrame(records)):
                if chunk.empty:
                    continue
                stmt = insert(Feature).values(chunk.to_dict(orient="records"))
                stmt = stmt.on_conflict_do_update(
                    constraint="features_pkey",
                    set_={
                        "opponent_team": stmt.excluded.opponent_team,
                        "team": stmt.excluded.team,
                        "player_features": stmt.excluded.player_features,
                        "matchup_features": stmt.excluded.matchup_features,
                    },
                )
                session.execute(stmt)
                total += len(chunk)
        session.commit()
        return total
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def persist_labels_from_clean(years: list[int]) -> int:
    """Upsert labels (PPR) from cleaned stats into labels table."""
    init_db()
    df = clean_player_stats(years)
    if "fantasy_points_ppr" not in df.columns:
        return 0
    labels = df.rename(columns={"fantasy_points_ppr": "fantasy_points"})[
        ["player_id", "season", "week", "fantasy_points"]
    ].copy()

    session: Session = SessionLocal()
    total = 0
    try:
        for chunk in _chunk_iter(labels):
            if chunk.empty:
                continue
            stmt = insert(Label).values(chunk.to_dict(orient="records"))
            stmt = stmt.on_conflict_do_update(
                constraint="labels_pkey",
                set_={"fantasy_points": stmt.excluded.fantasy_points},
            )
            session.execute(stmt)
            total += len(chunk)
        session.commit()
        return total
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()