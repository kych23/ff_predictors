"""Independent play-by-play aggregation, for the §21.1 reconciliation gate.

The gate's purpose is to validate the data pull, the ID mapping and the scoring
rules *simultaneously*, before months of model work rest on them. Its primary
form compares computed points against Yahoo's reported totals; Yahoo is blocked
(§11.5), so the doc's degradation applies: rebuild the same components from a
completely different data path — play-by-play — and require the two to agree.

This is genuinely weaker than the Yahoo form, and the weakness is specific: it
no longer validates the ID crosswalk against the platform the league actually
scores on. It *does* validate every column mapping in
``domain.scoring.engine``, which is where a silent scoring error would live —
a mis-summed fumble component or a missed two-point column produces a
disagreement here even though both paths come from nflverse.

Any run that uses this form must carry ``RECONCILIATION_DEGRADED``.
"""
from __future__ import annotations

from typing import Final

import pandas as pd

#: Components rebuilt from PBP, keyed to the same names `score_offense` uses.
#: Everything here is derived from play rows and player-id columns, sharing no
#: code with the weekly-aggregate path.
PBP_COMPONENTS: Final = (
    "passing_yards", "passing_tds", "passing_interceptions",
    "rushing_yards", "rushing_tds",
    "receptions", "receiving_yards", "receiving_tds",
)


def _sum_by(df: pd.DataFrame, id_col: str, value: pd.Series, name: str
            ) -> pd.DataFrame:
    frame = pd.DataFrame({
        "player_id": df[id_col], "season": df["season"], name: value,
    }).dropna(subset=["player_id"])
    return frame.groupby(["player_id", "season"], as_index=False)[name].sum()


def aggregate_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    """Rebuild season component totals from raw play rows.

    Deliberately verbose rather than clever: the point of a reconciliation gate
    is that the two paths share no logic, so this does not reuse any helper
    from the weekly-aggregate path.
    """
    df = pbp[pbp["season_type"] == "REG"].copy()
    for col in ("passing_yards", "rushing_yards", "receiving_yards",
                "pass_touchdown", "rush_touchdown", "interception",
                "complete_pass", "two_point_attempt"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Two-point conversions are scored on their own play rows and carry no
    # yardage, so they must be excluded from the yardage/reception totals that
    # player_stats reports separately.
    scrimmage = df[df["two_point_attempt"] == 0].copy()

    # LATERALS. Verified against 2024 play rows: `receiving_yards` is ALREADY
    # the original receiver's credited amount and `lateral_receiving_yards` is
    # the lateral receiver's — the two are ADDITIVE, not overlapping. (R.Bell
    # 12 + J.Brendel 2 on the same play, and player_stats credits exactly 12
    # and 2.) Ignoring the lateral leg entirely produced exactly 20
    # receiving_yards and 3 rushing_yards mismatches in 2024 — precisely the
    # lateral counts — which is what the gate caught on its first run.
    for col in ("lateral_receiving_yards", "lateral_rushing_yards"):
        if col in scrimmage.columns:
            scrimmage[col] = pd.to_numeric(scrimmage[col],
                                           errors="coerce").fillna(0.0)
        else:
            scrimmage[col] = 0.0

    parts = [
        _sum_by(scrimmage, "passer_player_id", scrimmage["passing_yards"],
                "passing_yards"),
        _sum_by(scrimmage, "passer_player_id",
                scrimmage["pass_touchdown"], "passing_tds"),
        _sum_by(scrimmage, "passer_player_id", scrimmage["interception"],
                "passing_interceptions"),
        _sum_by(scrimmage, "rusher_player_id", scrimmage["rushing_yards"],
                "rushing_yards"),
        _sum_by(scrimmage, "lateral_rusher_player_id",
                scrimmage["lateral_rushing_yards"], "rushing_yards_lat"),
        _sum_by(scrimmage, "rusher_player_id", scrimmage["rush_touchdown"],
                "rushing_tds"),
        _sum_by(scrimmage, "receiver_player_id", scrimmage["complete_pass"],
                "receptions"),
        _sum_by(scrimmage, "receiver_player_id", scrimmage["receiving_yards"],
                "receiving_yards"),
        _sum_by(scrimmage, "lateral_receiver_player_id",
                scrimmage["lateral_receiving_yards"], "receiving_yards_lat"),
        # Receiving TDs are credited to whoever CROSSED THE LINE, not to the
        # original receiver. On a pass caught by A and lateraled to B who
        # scores, `receiver_player_id` is A but the TD is B's — which showed up
        # as St. Brown over-credited by 2 and Gibbs under by 1, the exact
        # lateral pair. `td_player_id` is the scorer.
        _sum_by(scrimmage[scrimmage["pass_touchdown"] == 1],
                "td_player_id",
                scrimmage.loc[scrimmage["pass_touchdown"] == 1, "pass_touchdown"],
                "receiving_tds"),
    ]

    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=["player_id", "season"], how="outer")
    out = out.fillna(0.0)

    # fold the lateral legs back into the player's own totals
    for base in ("receiving_yards", "rushing_yards"):
        lat = f"{base}_lat"
        if lat in out.columns:
            out[base] = out[base] + out[lat]
            out = out.drop(columns=[lat])
    return out


def compare(weekly_totals: pd.DataFrame, pbp_totals: pd.DataFrame, *,
            tolerance: float = 0.01) -> pd.DataFrame:
    """Per player-season, per component: the two paths and their difference."""
    merged = weekly_totals.merge(pbp_totals, on=["player_id", "season"],
                                 how="inner", suffixes=("_weekly", "_pbp"))
    rows = []
    for comp in PBP_COMPONENTS:
        a, b = f"{comp}_weekly", f"{comp}_pbp"
        if a not in merged or b not in merged:
            continue
        diff = merged[a] - merged[b]
        rows.append({
            "component": comp,
            "n": len(merged),
            "n_mismatched": int((diff.abs() > tolerance).sum()),
            "max_abs_diff": float(diff.abs().max()),
            "total_weekly": float(merged[a].sum()),
            "total_pbp": float(merged[b].sum()),
        })
    return pd.DataFrame(rows)
