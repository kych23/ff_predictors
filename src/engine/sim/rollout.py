"""Complete the draft from a board state (§14.3, AD-12).

Evaluating "what is candidate X worth" requires a full roster, so the rest of
the draft has to be played out: my remaining picks and all eleven opponents'.

**Opponents.** The §14.2 gate failed (permutation p = 0.135), so per-manager
policies were not built. Opponents draft from the league-mean softmax over ADP
— which is what the gate's null result licenses, and no less than the evidence
supports.

**My own future picks (AD-12).** Filled with the tier-2 VONA heuristic, not a
recursive call into the decision engine. Recursion is unaffordable on a 60-second
clock.

    Bias, stated: a non-recursive self-policy UNDERSTATES the current pick's
    value, because it cannot exploit the option value of adapting later picks
    to what actually falls. Under CRN the bias is common to all candidates and
    largely cancels in the DIFFERENCE, which is what the decision uses. It does
    not cancel in the absolute E[$] level — one more reason §5.1 forbids reading
    that level as a measurement.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.constants import RngKind
from src.engine.decision import replacement as replacement_mod
from src.engine.decision import roster_state as rs_mod
from src.engine.decision import vona
from src.engine.sim import rng as rng_mod


@dataclass(frozen=True)
class RolloutResult:
    rosters: list[np.ndarray]      # board-row indices per seat
    my_seat: int


def unfilled_mandatory(counts: dict[str, int], cfg) -> dict[str, int]:
    """Starting slots this roster still cannot fill.

    Flex is deliberately excluded: it is satisfied by any eligible position, so
    it never *forces* a specific pick.
    """
    need: dict[str, int] = {}
    for slot, n in cfg.roster.slots.items():
        if slot == "BENCH" or slot in cfg.roster.flex_eligibility:
            continue
        short = int(n) - counts.get(slot, 0)
        if short > 0:
            need[slot] = short
    return need


def _legal(board: pd.DataFrame, taken: set[int], counts: dict[str, int],
           cfg, caps: dict[str, int], picks_left: int) -> np.ndarray:
    """Row indices still draftable for a manager with these position counts.

    Two constraints, and the second one matters more than it looks:

    * **Caps** stop a manager taking a sixth kicker.
    * **Needs** force the endgame. Once remaining picks equal unfilled mandatory
      slots, only those positions are legal. Without this, ADP-ordered drafting
      routinely produces rosters with NO quarterback — measured on the real
      board, 2 of 12 seats had none and every seat fielded 6-7 of 9 starters.
      Those rosters score ~0 in the empty slots, so E[$] ends up measuring
      roster-construction failure rather than player value. Real drafters, and
      Yahoo's autodrafter, both fill their slots (§14.1).
    """
    mask = ~board.index.isin(taken)
    over = [p for p, c in counts.items()
            if c >= int(caps.get(p, cfg.roster.rounds))]
    if over:
        mask &= ~board["position"].isin(over).to_numpy()

    need = unfilled_mandatory(counts, cfg)
    if need and sum(need.values()) >= picks_left:
        forced = mask & board["position"].isin(list(need)).to_numpy()
        if forced.any():
            return board.index.to_numpy()[forced]
    return board.index.to_numpy()[mask]


def softmax_adp_pick(board: pd.DataFrame, candidates: np.ndarray, pick_no: int,
                     tau: float, u: float) -> int:
    """League-mean policy: softmax over (adp - pick_no) / tau.

    Larger tau is flatter, i.e. more reaching. Sampled by inverse-CDF on a
    single addressed uniform so the draw is CRN-stable.
    """
    adp = board.loc[candidates, "adp"].to_numpy(dtype="float64")
    adp = np.where(np.isnan(adp), pick_no + 60.0, adp)
    score = -(adp - pick_no) / max(tau, 1e-6)
    score -= score.max()
    p = np.exp(score)
    p /= p.sum()
    return int(candidates[np.searchsorted(np.cumsum(p), u)])


def rollout(board: pd.DataFrame, cfg, strategy, *, my_seat: int,
            already_taken: dict[int, int] | None = None,
            forced: tuple[int, int] | None = None,
            root: str, rep: int) -> RolloutResult:
    """Play the draft to completion.

    ``already_taken`` maps board row -> seat for picks that have happened.
    ``forced`` is (seat, row): the candidate under evaluation, taken at my
    next turn.
    """
    tau = float(strategy.opponents.defaults["params"]["tau"])
    caps = dict(strategy.max_per_position)
    rosters: list[list[int]] = [[] for _ in range(cfg.teams)]
    taken: set[int] = set()
    counts = [dict() for _ in range(cfg.teams)]

    for row, seat in (already_taken or {}).items():
        rosters[seat].append(row)
        taken.add(row)
        pos = board.at[row, "position"]
        counts[seat][pos] = counts[seat].get(pos, 0) + 1

    forced_row = forced[1] if forced else None
    pick_no = len(taken) + 1
    total_picks = cfg.teams * cfg.roster.rounds

    while pick_no <= total_picks:
        rnd = (pick_no - 1) // cfg.teams
        idx = (pick_no - 1) % cfg.teams
        seat = idx if rnd % 2 == 0 else cfg.teams - 1 - idx

        picks_left = cfg.roster.rounds - len(rosters[seat])
        candidates = _legal(board, taken, counts[seat], cfg, caps, picks_left)
        if len(candidates) == 0:
            pick_no += 1
            continue

        if forced_row is not None and seat == my_seat and forced_row not in taken:
            row = forced_row
            forced_row = None
        elif seat == my_seat:
            row = _my_pick(board, candidates, cfg, counts[seat], pick_no)
        else:
            u = float(rng_mod.uniforms(root, RngKind.DRAFT, n=rep + 1,
                                       a=seat, b=pick_no)[rep])
            row = softmax_adp_pick(board, candidates, pick_no, tau, u)

        rosters[seat].append(row)
        taken.add(row)
        pos = board.at[row, "position"]
        counts[seat][pos] = counts[seat].get(pos, 0) + 1
        pick_no += 1

    return RolloutResult(rosters=[np.array(r) for r in rosters], my_seat=my_seat)


def _my_pick(board: pd.DataFrame, candidates: np.ndarray, cfg,
             counts: dict[str, int], pick_no: int) -> int:
    """AD-12: my future picks use the tier-2 heuristic, not recursion.

    Best value over replacement among the legal candidates, which is VONA's
    marginal term without the wait term — the wait term needs a next-pick
    horizon that a rollout does not carry cheaply.
    """
    sub = board.loc[candidates]
    modeled = sub[sub["position"].isin(cfg.roster.modeled_positions)]
    if modeled.empty:
        return int(sub["value"].idxmax())
    return int(modeled["value"].idxmax())
