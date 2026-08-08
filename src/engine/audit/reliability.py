"""Reliability study (§21.4) — how much of the ranking is signal?

Refit the projection model several times on resampled data, run every refit
against the same recorded draft states, and measure how much the *decision*
moves. A model whose top-84 ordering reshuffles between refits is not ranking
players; it is ranking noise, and any claim built on it inherits that.

**The resampling unit is players within season, never whole seasons.** A
season-block bootstrap breaks `make_expanding_folds`, which needs strictly
ordered distinct seasons and silently dedupes duplicates — and a four-season
half emits ZERO folds against `min_train_seasons: 5`, leaving the interval
calibrator nothing to fit. Resampling player rows inside each season preserves
the fold structure and split-conformal exchangeability.

**The number this produces is an UPPER bound on stability.** It varies the
player sample only. It does not vary the seasons observed, the feature set, the
hyperparameters, or the scoring config, so the true reliability of the whole
pipeline is lower than what is reported here. The README repeats this, and so
does the printed output — a stability figure quoted without its scope is the
kind of number that gets believed for years.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

#: Depths reported. 12 is roughly one round; 84 is the draftable pool.
DEFAULT_K: tuple[int, ...] = (12, 84)


@dataclass(frozen=True)
class ReliabilityReport:
    by_round: pd.DataFrame
    overall: pd.DataFrame
    n_refits: int
    n_states: int
    #: Always True for this construction — carried in the artifact so a
    #: downstream reader cannot lose the caveat.
    is_upper_bound: bool = True

    def describe(self) -> str:
        lines = [f"{self.n_refits} refits x {self.n_states} states "
                 f"(UPPER BOUND: player resampling only)"]
        for _, r in self.overall.iterrows():
            lines.append(f"  k={int(r['k']):3d}  spearman {r['spearman']:.3f}"
                         f"  exact@1 {r['exact_match_at_1']:.3f}")
        return "\n".join(lines)


def resample_players_within_season(frame: pd.DataFrame, rng: np.random.Generator,
                                   *, season_col: str = "season"
                                   ) -> pd.DataFrame:
    """Bootstrap rows inside each season, preserving every season's presence.

    Each season keeps its own row count, so the expanding folds see exactly the
    seasons they saw before — which is the whole reason this is the resampling
    unit.
    """
    if season_col not in frame.columns:
        raise KeyError(f"{season_col!r} is required to resample within season")
    parts = []
    for _, group in frame.groupby(season_col, sort=True):
        idx = rng.integers(0, len(group), len(group))
        parts.append(group.iloc[idx])
    return pd.concat(parts, ignore_index=True)


def _top_k_agreement(a: Sequence[str], b: Sequence[str], k: int) -> float:
    """Spearman over the union of the two top-k sets.

    Ranking only the intersection would score two orderings that share three
    players as highly as two that share eighty. The union, with absent players
    ranked last, penalizes disagreement about *membership* as well as order —
    which is the disagreement that actually changes a pick.
    """
    top_a, top_b = list(a)[:k], list(b)[:k]
    union = list(dict.fromkeys(top_a + top_b))
    rank_a = {p: i for i, p in enumerate(a)}
    rank_b = {p: i for i, p in enumerate(b)}
    fallback = max(len(a), len(b)) + 1
    ra = [rank_a.get(p, fallback) for p in union]
    rb = [rank_b.get(p, fallback) for p in union]
    if len(union) < 3 or len(set(ra)) < 2 or len(set(rb)) < 2:
        return float(ra == rb)
    rho, _ = spearmanr(ra, rb)
    return float(rho) if np.isfinite(rho) else 0.0


def reliability(states: Sequence[dict],
                rank_fn: Callable[[dict, int], Sequence[str]],
                *, n_refits: int = 2, ks: tuple[int, ...] = DEFAULT_K
                ) -> ReliabilityReport:
    """Compare orderings across refits on identical states.

    `rank_fn(state, refit_index) -> ordered player_ids` is the seam: the
    statistics here do not care whether a refit costs three seconds or three
    hours, which is what makes them testable without one.
    """
    if n_refits < 2:
        raise ValueError("reliability needs at least two refits to compare")
    if not states:
        raise ValueError("no recorded draft states to evaluate")

    rows = []
    for state in states:
        orders = [list(rank_fn(state, i)) for i in range(n_refits)]
        for i in range(n_refits):
            for j in range(i + 1, n_refits):
                for k in ks:
                    rows.append({
                        "state_id": state.get("state_id", ""),
                        "round": int(state.get("round", 0)),
                        "pair": f"{i}v{j}",
                        "k": k,
                        "spearman": _top_k_agreement(orders[i], orders[j], k),
                        "exact_match_at_1": float(
                            bool(orders[i]) and bool(orders[j])
                            and orders[i][0] == orders[j][0]),
                    })

    detail = pd.DataFrame(rows)
    by_round = (detail.groupby(["round", "k"], as_index=False)
                      [["spearman", "exact_match_at_1"]].mean())
    overall = (detail.groupby("k", as_index=False)
                     [["spearman", "exact_match_at_1"]].mean())
    return ReliabilityReport(by_round=by_round, overall=overall,
                             n_refits=n_refits, n_states=len(states))
