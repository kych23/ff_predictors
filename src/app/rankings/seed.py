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

**No seed produces fixed-size tiers.** Metric seeds cut where the ordering
cliffs (`natural_breaks`); `engine_tiers` takes the engine artifact's own
boundaries; `from_overall` carries your overall tiers across unchanged. The
configured size is a MAXIMUM that only splits a genuinely flat run, and
`from_overall` ignores it entirely — re-chunking there would discard the
grouping the action exists to copy.

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

#: MAXIMUM tier size, not a fixed one — tiers are cut where the ordering
#: cliffs (see `natural_breaks`) and the cap only splits genuinely flat runs.
#: Set high enough that a real tier is rarely truncated: measured on the live
#: board, 24 leaves the replacement-level tail as the only capped region.
DEFAULT_TIER_SIZE = {"overall": 24}
POSITION_TIER_SIZE = 12

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


#: A gap this many times the LOCAL typical gap ends a tier. A ratio, not an
#: absolute, because the unit differs per metric and per depth.
GAP_RATIO = 2.5
#: Gaps either side used to estimate what "typical" means around here.
WINDOW = 15
#: A tier of one is a claim you rarely mean.
MIN_TIER = 2


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def natural_breaks(values: list[float], *, max_size: int,
                   gap_ratio: float = GAP_RATIO,
                   window: int = WINDOW) -> list[int]:
    """Indices at which a new tier starts, cut where the ordering CLIFFS.

    Fixed-size chunks are the obvious approach and they are wrong: they assert
    that the 12th and 13th players differ because of where the boundary landed,
    when the real structure is that a run of near-equal players ends and the
    next one is a step down. A tier means "anyone in here is interchangeable to
    me" — a statement about DISTANCE — so the breaks must come from distances.

    **The comparison is LOCAL, and that is the whole design.** A global
    threshold fails badly on ADP, whose gaps grow with depth: consecutive picks
    are 0.1 apart at the top and 5+ apart by pick 150, so any global cutoff
    finds no cliff among the elite and slices only the tail. Measured on the
    live board, a global rule at 2.5 sd produced a 29-player "Tier 1" running
    from Gibbs to pick 30 — worse than useless, because it looks considered.

    So each gap is compared to the median gap in a window around it. A cliff is
    a gap `gap_ratio` times its neighbourhood. That is scale-free: it works the
    same on ADP picks, on points over replacement, and on any metric added
    later, without a per-metric constant to tune.

    `max_size` remains a scrolling cap. Where a metric is genuinely flat — the
    replacement-level tail, where everyone really is equivalent — there is no
    cliff to find and the cap is what splits it. That cut is arbitrary, and it
    is the only arbitrary one.
    """
    if len(values) <= 1:
        return [0]

    gaps = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]

    # The window must be a FRACTION of the data, never the whole of it: on a
    # short list a fixed 15 spans everything and the test silently degenerates
    # to the global rule it exists to replace.
    span = max(3, min(window, len(gaps) // 4))

    starts = [0]
    size = 1
    for i, gap in enumerate(gaps):
        lo, hi = max(0, i - span), min(len(gaps), i + span + 1)
        # Excluding itself: a gap is judged against its NEIGHBOURS. Left in,
        # a large gap raises the very baseline it is measured against.
        neighbours = gaps[lo:i] + gaps[i + 1:hi]
        local = _median(neighbours)
        # A flat neighbourhood has no scale to be large relative to; only the
        # cap can break it, which is correct — those players ARE equivalent.
        cliff = local > 0 and gap >= gap_ratio * local and size >= MIN_TIER
        if cliff or size >= max_size:
            starts.append(i + 1)
            size = 1
        else:
            size += 1
    return starts


def _tiers_from(ordered: list[str], starts: list[int]) -> tuple[Tier, ...]:
    bounds = [*starts, len(ordered)]
    tiers = []
    for number, (lo, hi) in enumerate(zip(bounds, bounds[1:], strict=False), 1):
        if lo >= hi:
            continue
        tiers.append(Tier(id=f"t-{number}", label=f"Tier {number}",
                          color=TIER_COLORS[(number - 1) % len(TIER_COLORS)],
                          player_ids=tuple(ordered[lo:hi])))
    return tuple(tiers)


def _by_metric(frame: pd.DataFrame, column: str, *, ascending: bool,
               max_size: int) -> tuple[Tier, ...]:
    """Sort on a metric, then cut where that metric cliffs."""
    working = frame.copy()
    working["_missing"] = working[column].isna()
    working = working.sort_values(
        ["_missing", column, "player_id"],
        ascending=[True, ascending, True], kind="mergesort")

    ordered = [str(pid) for pid in working["player_id"]]
    # Unpriced players carry no distance to anyone, so they cannot participate
    # in gap detection. They trail in their own tier — a real state ("the
    # market has no opinion"), not a rank.
    ranked = working[~working["_missing"]]
    unranked = [str(pid) for pid in working.loc[working["_missing"], "player_id"]]

    values = [float(v) for v in ranked[column]]
    tiers = _tiers_from(
        ordered[: len(values)],
        natural_breaks(values, max_size=max_size) if values else [0])

    if unranked:
        number = len(tiers) + 1
        tiers = (*tiers, Tier(
            id=f"t-{number}", label="Unpriced",
            color=TIER_COLORS[-1], player_ids=tuple(unranked)))
    return tiers


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
        return Scope(tiers=_by_metric(frame, "adp", ascending=True,
                                      max_size=size))

    if method == "engine_value":
        return Scope(tiers=_by_metric(frame, "value", ascending=False,
                                      max_size=size))

    if method == "engine_vor":
        frame = frame.assign(_vor=vor(data, frame))
        return Scope(tiers=_by_metric(frame, "_vor", ascending=False,
                                      max_size=size))

    if method == "engine_tiers":
        return Scope(tiers=_seed_from_artifact(data, frame))

    # NOT chunked: from_overall carries the overall tier structure across.
    return Scope(tiers=_seed_from_overall(frame, board, scope))


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
                       scope: str) -> tuple[Tier, ...]:
    """This position, carrying its OVERALL tiers across intact.

    The tier structure is the point of the action, so it is preserved rather
    than rebuilt: each overall tier contributes its members at this position,
    keeping that tier's id, label and colour. If Henry and Gibbs share overall
    Tier 2 then they share RB Tier 2, in the same colour, and a glance at
    either scope tells you the same thing.

    Re-chunking the flattened order — which this used to do — would have
    discarded exactly the grouping the user built, then imposed boundaries
    every six players that mean nothing.

    Overall tiers with nobody at this position contribute nothing, so the
    labels can skip: an RB list reading "Tier 1, Tier 3, Tier 4" is telling
    you there were no backs in your second tier. That gap is information, not
    a rendering bug, which is why the numbering is not resequenced.
    """
    if board is None:
        raise RankingsError("from_overall needs the board it is seeding")
    overall = board.scopes["overall"].tiers
    if not any(t.player_ids for t in overall):
        raise RankingsError(
            f"cannot seed {scope!r} from an empty overall list; seed overall "
            f"first")

    eligible = {str(pid) for pid in frame["player_id"]}
    tiers: list[Tier] = []
    placed: set[str] = set()
    for source in overall:
        members = tuple(pid for pid in source.player_ids if pid in eligible)
        if not members:
            continue
        placed.update(members)
        tiers.append(Tier(id=source.id, label=source.label,
                          color=source.color, player_ids=members))

    # Anyone at this position you never ranked overall. Appended in a tier of
    # their own rather than dropped, and labelled for what they are.
    rest = tuple(sorted(eligible - placed))
    if rest:
        tiers.append(Tier(id=f"t-{len(overall) + 1}", label="Not in overall",
                          color=TIER_COLORS[-1], player_ids=rest))
    return tuple(tiers)
