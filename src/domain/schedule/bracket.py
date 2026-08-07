"""Regular-season schedule and playoff bracket (§16.3).

Produces the two rank arrays every rank-type reducer consumes:
``regular_rank`` (record-based) and ``final_rank`` (bracket outcome), both
complete over 1..T so that ``rank: 5`` and ``rank: -1`` are equally
well-defined.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.core.config.league import ScheduleConfig


def round_robin(teams: int, weeks: int) -> list[list[tuple[int, int]]]:
    """Circle-method pairings, repeating from week 1 once exhausted.

    A 12-team round robin is 11 weeks; a 14-week season therefore replays
    weeks 1-3, which is Yahoo's default behavior.
    """
    if teams % 2:
        raise ValueError(f"odd team count {teams} not supported")
    rotation = list(range(teams))
    base: list[list[tuple[int, int]]] = []
    for _ in range(teams - 1):
        half = teams // 2
        base.append([(rotation[i], rotation[teams - 1 - i]) for i in range(half)])
        rotation = [rotation[0]] + [rotation[-1]] + rotation[1:-1]
    return [base[w % len(base)] for w in range(weeks)]


def regular_season_ranks(
    team_week: np.ndarray, cfg: ScheduleConfig, *, tiebreakers: Sequence[str] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(regular_rank, wins)``, both shaped ``(T, R)``.

    ``team_week`` is ``(T, W, R)``. Only the regular season counts.
    Tiebreakers are applied in the configured order; ``head_to_head`` resolves
    only exact two-way ties (it is undefined for three-way ties, which fall
    through to the next key), and team index is the final deterministic
    fallback.
    """
    order = list(tiebreakers or cfg.seeding_tiebreakers)
    n_teams, _, n_reps = team_week.shape
    rs = team_week[:, : cfg.regular_season_weeks, :]

    wins = np.zeros((n_teams, n_reps), dtype="float64")
    schedule = round_robin(n_teams, cfg.regular_season_weeks)
    for week, pairings in enumerate(schedule):
        for a, b in pairings:
            a_score, b_score = rs[a, week, :], rs[b, week, :]
            wins[a] += (a_score > b_score) + 0.5 * (a_score == b_score)
            wins[b] += (b_score > a_score) + 0.5 * (a_score == b_score)

    points_for = rs.sum(axis=1)

    # lexsort applies the LAST key first, so build the key list reversed.
    keys: list[np.ndarray] = [np.tile(np.arange(n_teams)[:, None], (1, n_reps))]
    for name in reversed(order):
        if name == "wins":
            keys.append(-wins)
        elif name == "points_for":
            keys.append(-points_for)
        elif name == "head_to_head":
            continue  # only well-defined for two-way ties; see docstring
        else:
            raise ValueError(f"unknown seeding tiebreaker {name!r}")

    regular_rank = np.empty((n_teams, n_reps), dtype="int16")
    for r in range(n_reps):
        order_idx = np.lexsort(tuple(k[:, r] for k in keys))
        regular_rank[order_idx, r] = np.arange(1, n_teams + 1)
    return regular_rank, wins


def final_ranks(
    team_week: np.ndarray, regular_rank: np.ndarray, cfg: ScheduleConfig
) -> np.ndarray:
    """Single-elimination bracket -> complete ``final_rank`` (T, R), 1..T.

    Seeds 1..playoff_teams by ``regular_rank``; the top ``playoff_byes`` seeds
    skip round 1. Highest remaining seed plays lowest. Matchup winner is the
    higher total over ``matchup_length_weeks``; ties break to the better seed.

    Ranking, complete so every rank a prize can name is defined:
      1                      champion
      2                      runner-up
      3..4                   semifinal losers, by seed
      5..playoff_teams       earlier losers, by round eliminated (later is
                             better), then by seed
      playoff_teams+1..T     non-playoff teams, by regular_rank
    """
    n_teams, _, n_reps = team_week.shape
    out = np.empty((n_teams, n_reps), dtype="int16")
    weeks = list(cfg.playoff_weeks)
    m = cfg.matchup_length_weeks

    for r in range(n_reps):
        seed_of = {int(regular_rank[t, r]): t for t in range(n_teams)}
        alive = [seed_of[s] for s in range(1, cfg.playoff_teams + 1)]
        seed_by_team = {seed_of[s]: s for s in range(1, cfg.playoff_teams + 1)}
        eliminated: list[tuple[int, int]] = []  # (round_index, team)

        byes = [t for t in alive[: cfg.playoff_byes]]
        active = [t for t in alive[cfg.playoff_byes:]]
        round_idx = 0
        week_cursor = 0

        while len(active) + len(byes) > 1:
            active.sort(key=lambda t: seed_by_team[t])
            pairs = [(active[i], active[len(active) - 1 - i])
                     for i in range(len(active) // 2)]
            winners: list[int] = []
            wk = weeks[week_cursor: week_cursor + m]
            week_cursor += m
            for hi, lo in pairs:
                hi_pts = team_week[hi, [w - 1 for w in wk], r].sum()
                lo_pts = team_week[lo, [w - 1 for w in wk], r].sum()
                win, lose = (hi, lo) if hi_pts >= lo_pts else (lo, hi)
                winners.append(win)
                eliminated.append((round_idx, lose))
            active = winners + byes
            byes = []
            round_idx += 1

        champion = active[0]
        out[champion, r] = 1
        # later elimination round is a better finish; ties broken by seed
        eliminated.sort(key=lambda rt: (-rt[0], seed_by_team[rt[1]]))
        for placing, (_, team) in enumerate(eliminated, start=2):
            out[team, r] = placing

        missed = sorted(
            (t for t in range(n_teams) if t not in seed_by_team),
            key=lambda t: int(regular_rank[t, r]),
        )
        for offset, team in enumerate(missed):
            out[team, r] = cfg.playoff_teams + 1 + offset

    return out


def resolve_rank(rank: int, teams: int) -> int:
    """Normalize a configured rank to 1..teams (negatives count from last)."""
    return rank if rank > 0 else teams + 1 + rank
