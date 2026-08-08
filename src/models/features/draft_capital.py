"""Where the NFL drafted him — the strongest public rookie signal.

Rookies have no prior-season production, so every `prior_*` feature is null for
them and the model has almost nothing to separate one from another. Measured on
the shipped artifact: rookie RBs projected with a standard deviation of 0.74
against 4.00 for veterans, and a maximum (7.84) below the veteran median plus
one SD. A first-round back and an undrafted back came out nearly identical.

Draft capital fixes the discrimination, not the level. A team that spends the
12th overall pick on a back has told the market what role he is getting, and
that intent is knowable in April — months before a fantasy draft, so this is
**as-of safe by construction**. Nothing here reads a snap, a target or a game.

The college features this replaces (`has_college_stats`, `is_udfa`,
`has_combine`) are hardcoded to 0 since CFBD was descoped, so they contribute
nothing today; `is_undrafted` below is the honest version of `is_udfa`.

Round is kept ALONGSIDE overall pick rather than folded into it. They are not
redundant: pick number is close to linear in expected production, while round
carries the discrete cliff at the end of day two that teams themselves behave
around.
"""
from __future__ import annotations

import pandas as pd

#: Emitted for every player, rookie or not. Veterans carry their own original
#: draft position, which stays mildly predictive years later.
CAPITAL_COLS = ["draft_round", "draft_pick_overall", "is_undrafted",
                "draft_capital_score"]

#: Stand-in slot for a player with no draft row. Undrafted free agents are
#: worse in expectation than the last pick of the seventh round, so a value
#: past the end of the draft is the right encoding — not a null the tree would
#: have to learn to split around, and not zero, which would read as "first
#: overall".
UNDRAFTED_PICK = 300
UNDRAFTED_ROUND = 8


def build_draft_capital(player_ids_table: pd.DataFrame,
                        player_ids: pd.Series) -> pd.DataFrame:
    """One row per player id: round, overall pick, and a decayed score.

    Source is **`ff_playerids`, not `draft_picks`**, and the difference matters
    more than it looks. `draft_picks` keys on `gsis_id`, which nflverse has not
    yet backfilled for the newest class: 2022-2025 are 100% gsis format, 2026
    is 0% (provisional ids like ``MEN516487``). Training would therefore have
    had full coverage and serving almost none — the model learns to trust the
    feature, then meets a season where every rookie looks undrafted.

    Measured when this was wired to `draft_picks`: rookie p50 spread COLLAPSED
    from std 0.74 to 0.47 and the maximum fell from 7.84 to 4.75. The feature
    made projections worse, which is the signature of train/serve skew rather
    than of a bad feature.

    `ff_playerids` carries `draft_round`/`draft_ovr` keyed on `gsis_id`
    directly, at ~200-215 players per year INCLUDING 2026 — the same coverage
    on both sides of the split, which is the property that actually matters.
    """
    ids = pd.Series(list(player_ids), dtype=str).rename("player_id")
    out = pd.DataFrame({"player_id": ids})

    table = player_ids_table
    if table is None or table.empty or "gsis_id" not in table:
        out["draft_round"] = UNDRAFTED_ROUND
        out["draft_pick_overall"] = UNDRAFTED_PICK
        out["is_undrafted"] = 1
        out["draft_capital_score"] = 0.0
        return out

    picks = table.dropna(subset=["gsis_id"]).copy()
    picks["player_id"] = picks["gsis_id"].astype(str)
    picks["draft_round"] = pd.to_numeric(picks.get("draft_round"), errors="coerce")
    picks["draft_pick_overall"] = pd.to_numeric(picks.get("draft_ovr"),
                                                errors="coerce")
    picks = picks.dropna(subset=["draft_pick_overall"])
    picks = (picks.sort_values("draft_pick_overall")
                  .drop_duplicates("player_id", keep="first"))

    out = out.merge(picks[["player_id", "draft_round", "draft_pick_overall"]],
                    on="player_id", how="left")
    out["is_undrafted"] = out["draft_pick_overall"].isna().astype(int)
    out["draft_round"] = out["draft_round"].fillna(UNDRAFTED_ROUND)
    out["draft_pick_overall"] = out["draft_pick_overall"].fillna(UNDRAFTED_PICK)

    # Monotone decreasing in pick number, steep early and flat late — which is
    # how draft-pick value behaves. A raw pick number makes the tree spend
    # splits recovering that curvature.
    out["draft_capital_score"] = (
        1.0 / (1.0 + out["draft_pick_overall"] / 32.0)).astype(float)
    return out
