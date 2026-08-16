"""Building a starting tier list, so nobody drags 260 players from nothing.

**The methods are not uniform across scopes, and that is the point.** The
obvious design — offer every method everywhere — produces boards that are
quietly wrong, in two different ways measured on the live 260-player bundle:

*Raw `value` cannot rank `overall`.* Per-game points are not comparable across
positions when only one quarterback starts. Sorted by `value`, the top 40 is 24
QBs, 9 RBs and 7 WRs; Daniel Jones (ADP 150.8) lands 18th overall. Spearman
against ADP over the top-150 is -0.575. Subtracting the per-position
replacement level fixes it: the top 40 becomes 20 WR / 19 RB / 1 TE, headed by
Nacua (ADP 3.9), Chase (3.1) and Gibbs (1.7), and Spearman rises to -0.826. So
`overall` gets `engine_vor` and `adp`, never raw `value`.

*Engine tiers cannot rank `overall` either.* The artifact is tiered per
POSITION — `rank` restarts at 1 for each — so a "tier 2 RB" and a "tier 2 WR"
are unrelated numbers. There is no overall tiering anywhere in the repo.

*Neither engine method can rank K or DST.* All 21 kickers share
`value == 8.051787` and all 26 defences share `6.440885`, so the sort would be
`player_id` order wearing a projection's name. Those two scopes get `adp` and
`engine_tiers` only.

A note on the VOR caveat, since it is visible in the output: with replacement
at the 10th percentile within position, QB replacement lands at 14.7 because
all 29 QBs bunch tightly, so no QB reaches the top 40 at all. That is later
than the market has them. It is a seed the user re-drags, and `adp` is the
default precisely because the market's view of positional scarcity is the more
familiar starting point.
"""
from __future__ import annotations

import pandas as pd

from .data import RankingsData, vor
from .schema import (
    TIER_COLORS,
    Board,
    RankingsError,
    Scope,
    Tier,
)

METHODS = ("adp", "engine_vor", "engine_value", "engine_tiers", "from_overall")

#: Twelve for `overall` because 260 players at six per tier is 44 tiers, which
#: nobody can scroll. Six within a position, where the lists are short.
DEFAULT_TIER_SIZE = {"overall": 12}
POSITION_TIER_SIZE = 6

#: Which methods are legal where. Everything else is a 400 with the reason.
_LEGAL: dict[str, frozenset[str]] = {
    "overall": frozenset({"adp", "engine_vor"}),
    **{pos: frozenset({"adp", "engine_vor", "engine_value", "engine_tiers",
                       "from_overall"})
       for pos in ("QB", "RB", "WR", "TE")},
    **{pos: frozenset({"adp", "engine_tiers", "from_overall"})
       for pos in ("K", "DST")},
}

_WHY = {
    ("overall", "engine_value"):
        "raw per-game value is not comparable across positions — it puts 24 "
        "quarterbacks in the top 40. Use engine_vor.",
    ("overall", "engine_tiers"):
        "the engine's tiers are per position; there is no overall tiering.",
    ("overall", "from_overall"):
        "from_overall seeds a position FROM the overall list.",
    ("K", "engine_value"): "every kicker shares one projected value.",
    ("K", "engine_vor"): "every kicker shares one projected value.",
    ("DST", "engine_value"): "every defence shares one projected value.",
    ("DST", "engine_vor"): "every defence shares one projected value.",
}


def tier_size_for(scope: str, requested: int | None) -> int:
    if requested is not None:
        return int(requested)
    return DEFAULT_TIER_SIZE.get(scope, POSITION_TIER_SIZE)


def check_method(scope: str, method: str) -> None:
    """Raise with a reason a user can act on, not just a rejection."""
    if method not in METHODS:
        raise RankingsError(
            f"unknown seed method {method!r}; expected one of {METHODS}")
    legal = _LEGAL.get(scope)
    if legal is None:
        raise RankingsError(f"unknown scope {scope!r}")
    if method not in legal:
        why = _WHY.get((scope, method), "not available for this scope")
        raise RankingsError(f"{method!r} cannot seed {scope!r}: {why}")


def _eligible(data: RankingsData, scope: str) -> pd.DataFrame:
    players = data.board
    if players.empty:
        return players
    if scope == "overall":
        return players.copy()
    return players[players["position"] == scope].copy()


def _chunk(ordered: list[str], size: int) -> tuple[Tier, ...]:
    size = max(int(size), 1)
    tiers = []
    for index in range(0, len(ordered), size):
        number = len(tiers) + 1
        tiers.append(Tier(id=f"t-{number}", label=f"Tier {number}",
                          color=TIER_COLORS[(number - 1) % len(TIER_COLORS)],
                          player_ids=tuple(ordered[index:index + size])))
    return tuple(tiers)


def _ordered_by(frame: pd.DataFrame, column: str, *,
                ascending: bool) -> list[str]:
    """Sort, with nulls last either way and `player_id` as the final tiebreak.

    The tiebreak is not decoration: all 26 defences share one projected value,
    so without it a reseed could return a different board than the last one.
    """
    working = frame.copy()
    working["_missing"] = working[column].isna()
    working = working.sort_values(
        ["_missing", column, "player_id"],
        ascending=[True, ascending, True], kind="mergesort")
    return [str(pid) for pid in working["player_id"]]


def seed_scope(data: RankingsData, scope: str, method: str, *,
               tier_size: int | None = None,
               board: Board | None = None) -> Scope:
    check_method(scope, method)
    frame = _eligible(data, scope)
    if frame.empty:
        return Scope()
    size = tier_size_for(scope, tier_size)

    if method == "adp":
        return Scope(tiers=_chunk(_ordered_by(frame, "adp", ascending=True),
                                  size))

    if method == "engine_value":
        return Scope(tiers=_chunk(_ordered_by(frame, "value", ascending=False),
                                  size))

    if method == "engine_vor":
        frame = frame.assign(_vor=vor(data, frame))
        return Scope(tiers=_chunk(_ordered_by(frame, "_vor", ascending=False),
                                  size))

    if method == "engine_tiers":
        return Scope(tiers=_seed_from_artifact(data, frame))

    return Scope(tiers=_chunk(_seed_from_overall(frame, board, scope), size))


def _seed_from_artifact(data: RankingsData, frame: pd.DataFrame) -> tuple[Tier, ...]:
    """One tier per distinct artifact tier, ordered by its own `rank`.

    Bundle players absent from the artifact land in a trailing "Unranked" tier
    rather than vanishing — a seed that silently drops players would leave the
    user's board quietly incomplete.
    """
    if data.tiers.empty:
        raise RankingsError(
            "no engine tier artifact is loaded; run scripts/build_bundle.py "
            "or seed from ADP")

    joined = frame.merge(
        data.tiers[["player_id", "tier", "rank"]].assign(
            player_id=lambda d: d["player_id"].astype(str)),
        on="player_id", how="left")

    tiers: list[Tier] = []
    ranked = joined[joined["tier"].notna()]
    for number, (_, rows) in enumerate(
            ranked.sort_values(["tier", "rank", "player_id"],
                               kind="mergesort").groupby("tier", sort=True), 1):
        tiers.append(Tier(
            id=f"t-{number}", label=f"Tier {number}",
            color=TIER_COLORS[(number - 1) % len(TIER_COLORS)],
            player_ids=tuple(str(p) for p in rows["player_id"])))

    unranked = joined[joined["tier"].isna()]
    if not unranked.empty:
        tiers.append(Tier(
            id=f"t-{len(tiers) + 1}", label="Unranked",
            color=TIER_COLORS[-1],
            player_ids=tuple(sorted(str(p) for p in unranked["player_id"]))))
    return tuple(tiers)


def _seed_from_overall(frame: pd.DataFrame, board: Board | None,
                       scope: str) -> list[str]:
    """This position, in the order the user already put them in `overall`.

    The one bridge between scopes, and deliberately one-way. Players at the
    position who are not in `overall` are appended rather than dropped.
    """
    if board is None:
        raise RankingsError("from_overall needs the board it is seeding")
    order = [pid for tier in board.scopes["overall"].tiers
             for pid in tier.player_ids]
    if not order:
        raise RankingsError(
            f"cannot seed {scope!r} from an empty overall list; seed overall "
            f"first")
    eligible = {str(pid) for pid in frame["player_id"]}
    ranked = [pid for pid in order if pid in eligible]
    rest = sorted(eligible - set(ranked))
    return ranked + rest
