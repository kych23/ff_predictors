"""Does the recommender beat ADP bots? (§21, the gate `expected_dollars_vs_adp_bots`)

    venv/bin/python scripts/audit_vs_adp.py --replicates 12

The project's central claim has never been measured. `power_assumptions.yaml`
declares this gate with `effect_provenance: assumed` — the $4 figure there is
the effect size used to SIZE the test, not a result of running it. This script
runs it.

**A/B on the same random stream, not seat-vs-seat.**

The obvious design — compare my seat's dollars to the other eleven in the same
draft — is bad twice over. The payout pot is conserved, so my seat's dollars
and the bots' are mechanically linked and the "comparison" is partly an
identity. And twelve seats from one draft are not twelve observations; the
pre-registered gate accounts for that with a Kish design effect of 4.85,
collapsing 200 drafts to ~41 effective.

Instead each replicate drafts the SAME room twice:

    arm A   the engine occupies seat s
    arm B   an ADP bot occupies seat s

Same seed, so the eleven opponents behave identically in both arms; same drawn
seasons, so both rosters are scored against the same football. The difference
is then attributable to the seat's policy and nothing else, and each replicate
yields ONE independent paired observation. No clustering, no design effect —
n is simply the number of replicates.

**READ THIS BEFORE QUOTING THE NUMBER: the engine is graded on its own model.**

The engine chooses each pick by maximising E[$] under this simulator, and this
script then scores the resulting roster with THE SAME simulator and the same
fitted parameters. A large edge is therefore close to guaranteed by
construction — it measures whether the optimiser optimises, not whether it
drafts well in reality. Every assumption it is wrong about (the projections,
the correlation matrix, the hazard, the opponent model) is an assumption the
scoring shares, so a mistake cancels instead of showing up as a loss.

What this DOES establish: the decision path is internally consistent, and the
roster-construction value the model believes it adds. That is a real sanity
check — an engine that could NOT beat ADP bots under its own objective would be
broken — but it is not evidence of a real-world edge and must never be quoted
as one.

A claim about reality needs a HOLD-OUT: build a bundle as-of a past season,
draft against bots, and score on that season's ACTUAL results. That is the
honest version of this test and it is not what runs here.

**Seat variance is enormous.** Measured on an all-bot draft: seats ranged from
$9.10 to $70.40 against a $32.00 baseline, from draft position and softmax luck
alone. The A/B pairing removes the seed but not this — once my seat deviates,
the pool diverges and the opponents' identical uniforms select different
players. Expect a wide interval.

**What the defaults measure is a LOWER BOUND.** Running the engine at draft-night
settings costs ~10 s per pick, or ~50 minutes per replicate-pair. The defaults
here are cheaper (fewer replications, shorter budget), which can only make the
engine worse than it is in the cockpit. The settings used are stamped into the
output and the artifact so a number can never be quoted without them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.cockpit.build import build_tiers  # noqa: E402
from src.app.cockpit.ladder import recommend  # noqa: E402
from src.core.config import load_league, load_strategy  # noqa: E402
from src.core.config.slots import build_slot_plan  # noqa: E402
from src.core.constants import RngKind  # noqa: E402
from src.domain.payout.compile import compile_payout  # noqa: E402
from src.engine.decision.board import Board  # noqa: E402
from src.engine.sim import kernel  # noqa: E402
from src.engine.sim import rng as rng_mod  # noqa: E402
from src.engine.sim.bundle_build import (  # noqa: E402
    build_projection_bundle,
    load_correlation_matrix,
)
from src.engine.sim.draws import draw_points  # noqa: E402
from src.engine.sim.rollout import _legal, softmax_adp_pick  # noqa: E402

ARTIFACTS = Path("data/artifacts")


@dataclass(frozen=True)
class Replicate:
    seed: int
    seat: int
    engine_dollars: float
    bot_dollars: float

    @property
    def delta(self) -> float:
        return self.engine_dollars - self.bot_dollars


def snake_seat(pick_no: int, teams: int) -> int:
    rnd, idx = divmod(pick_no - 1, teams)
    return idx if rnd % 2 == 0 else teams - 1 - idx


def play_draft(frame: pd.DataFrame, cfg, strategy, *, my_seat: int,
               engine_seat: bool, root: str, rep: int,
               reps: int, shortlist: int, budget_s: float,
               ) -> list[np.ndarray]:
    """One full draft. `engine_seat` swaps my policy; everything else is fixed.

    Opponents use the league-mean softmax over ADP — the same policy the
    rollout uses, which is what the §14.2 gate's null result licenses.
    """
    tau = float(strategy.opponents.defaults["params"]["tau"])
    caps = dict(strategy.max_per_position)
    teams, rounds = cfg.teams, cfg.roster.rounds
    rosters: list[list[int]] = [[] for _ in range(teams)]
    counts: list[dict[str, int]] = [{} for _ in range(teams)]
    taken: set[int] = set()
    ids = frame["player_id"].astype(str).tolist()

    for pick_no in range(1, teams * rounds + 1):
        seat = snake_seat(pick_no, teams)
        picks_left = rounds - len(rosters[seat])
        legal = _legal(frame, taken, counts[seat], cfg, caps, picks_left)
        if len(legal) == 0:
            continue

        if seat == my_seat and engine_seat:
            row = engine_pick(frame, ids, cfg, strategy, taken, rosters,
                              my_seat=my_seat, reps=reps, shortlist=shortlist,
                              budget_s=budget_s, legal=legal)
        else:
            u = float(rng_mod.uniforms(root, RngKind.DRAFT, n=rep + 1,
                                       a=seat, b=pick_no)[rep])
            row = softmax_adp_pick(frame, legal, pick_no, tau, u)

        rosters[seat].append(row)
        taken.add(row)
        pos = frame.at[row, "position"]
        counts[seat][pos] = counts[seat].get(pos, 0) + 1

    return [np.array(r) for r in rosters]


def engine_pick(frame, ids, cfg, strategy, taken, rosters, *, my_seat,
                reps, shortlist, budget_s, legal) -> int:
    """The real cockpit path: build the ladder, take its leader."""
    board = Board.from_bundle()
    for row in taken:
        try:
            board.take(ids[row])
        except Exception:                  # noqa: BLE001 — never stop a draft
            pass
    by_seat = {s: [ids[r] for r in rows] for s, rows in enumerate(rosters)}
    tiers, _ = build_tiers(
        board, cfg, strategy, my_seat + 1, reps, shortlist,
        my_roster=[ids[r] for r in rosters[my_seat]], by_seat=by_seat)
    rec = recommend(tiers, budget_s=budget_s, demote_after_s=budget_s * 0.8)
    row_of = {pid: i for i, pid in enumerate(ids)}
    row = row_of.get(str(rec.leader))
    # The ladder can only ever name a board id; if it somehow names one that is
    # not legal here, fall back rather than draft an illegal player.
    return row if row is not None and row in set(legal) else int(legal[0])


def score(rosters: list[np.ndarray], points: np.ndarray, proj, plan, cfg,
          strategy, seat: int) -> float:
    """E[$] for one seat, against a FIXED set of drawn seasons."""
    weeks = points.shape[1]
    masks = kernel.build_masks(proj, rosters, plan, weeks)
    team_week = kernel.evaluate_rosters(points, masks)
    outcome = kernel.season_outcome(team_week, cfg, my_team_index=seat)
    objective = compile_payout(strategy.payout, cfg)
    return float(objective(outcome)[seat].mean())


def summarize(reps_out: list[Replicate], expected_effect: float) -> dict:
    deltas = np.array([r.delta for r in reps_out], dtype=float)
    n = len(deltas)
    mean = float(deltas.mean())
    sd = float(deltas.std(ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    # Paired design: one independent observation per replicate. No cluster
    # correction, unlike the seat-vs-seat design the gate was sized for.
    lo, hi = (mean - 1.96 * se, mean + 1.96 * se) if n > 1 else (np.nan, np.nan)
    mde = 1.96 * se + 0.84 * se if n > 1 else float("nan")   # 80% power, two-sided
    return {"n": n, "mean_delta": mean, "sd": sd, "se": se,
            "ci95": [lo, hi], "mde_at_80pct_power": mde,
            "expected_effect": expected_effect,
            "wins": int((deltas > 0).sum()),
            # Two DIFFERENT questions, conflated in the first version of this
            # script. "Is the observed effect distinguishable from zero" is the
            # confidence interval. "Could this design have found the effect it
            # was sized for" is the MDE. A run can answer the first decisively
            # while failing the second — which is exactly what happens here,
            # because the observed effect is an order of magnitude larger than
            # the one the gate was written for.
            "separated_from_zero": bool(n > 1 and lo > 0),
            "powered_for_expected": bool(n > 1 and mde <= abs(expected_effect))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates", type=int, default=12)
    ap.add_argument("--reps", type=int, default=128,
                    help="simulator replications per engine pick (draft night uses 512)")
    ap.add_argument("--shortlist", type=int, default=4)
    ap.add_argument("--budget", type=float, default=4.0,
                    help="seconds per engine pick (draft night uses 25)")
    ap.add_argument("--score-reps", type=int, default=512,
                    help="replications used to SCORE final rosters; cheap")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_league()
    strategy = load_strategy(cfg)
    b = Board.from_bundle()
    frame = b.players.reset_index(drop=True).copy()
    frame["player_id"] = frame["player_id"].astype(str)
    weeks = cfg.schedule.regular_season_weeks + len(cfg.schedule.playoff_weeks)
    proj = build_projection_bundle(frame, cfg, weeks)
    corr = load_correlation_matrix()
    plan = build_slot_plan(cfg.roster.slots, cfg.roster.flex_eligibility)
    root = rng_mod.seed_root(b.snapshot_id, cfg.model_version,
                             strategy.strategy_hash)

    print("§21 ENGINE vs ADP BOTS")
    print(f"  bundle {b.snapshot_id}, {len(frame)} players")
    print(f"  {args.replicates} replicates, A/B on the same stream")
    print(f"  ENGINE SETTINGS reps={args.reps} shortlist={args.shortlist} "
          f"budget={args.budget}s  (draft night: 512 / 6 / 25s)")
    print("  a cheaper engine can only understate the real one\n")

    out: list[Replicate] = []
    started = time.perf_counter()
    for r in range(args.replicates):
        seat = r % cfg.teams          # rotate: seat position matters
        # ONE set of drawn seasons per replicate, shared by both arms, so the
        # rosters are judged against identical football.
        points = draw_points(proj, corr.cholesky, root, reps=args.score_reps)

        t0 = time.perf_counter()
        a = play_draft(frame, cfg, strategy, my_seat=seat, engine_seat=True,
                       root=root, rep=r, reps=args.reps,
                       shortlist=args.shortlist, budget_s=args.budget)
        bmt = play_draft(frame, cfg, strategy, my_seat=seat, engine_seat=False,
                         root=root, rep=r, reps=args.reps,
                         shortlist=args.shortlist, budget_s=args.budget)

        eng = score(a, points, proj, plan, cfg, strategy, seat)
        bot = score(bmt, points, proj, plan, cfg, strategy, seat)
        out.append(Replicate(seed=r, seat=seat, engine_dollars=eng,
                             bot_dollars=bot))
        print(f"  rep {r:2d}  seat {seat + 1:2d}   engine ${eng:7.2f}   "
              f"bot ${bot:7.2f}   delta {eng - bot:+7.2f}   "
              f"[{time.perf_counter() - t0:.0f}s]")

    stats = summarize(out, expected_effect=4.0)
    elapsed = time.perf_counter() - started

    print(f"\n  n = {stats['n']} paired drafts, {elapsed / 60:.1f} min")
    print(f"  engine beat the bot seat in {stats['wins']} of {stats['n']}")
    print(f"  mean delta  ${stats['mean_delta']:+.2f}")
    if stats["n"] > 1:
        print(f"  95% CI      [${stats['ci95'][0]:+.2f}, ${stats['ci95'][1]:+.2f}]")
        print(f"  MDE @80%    ${stats['mde_at_80pct_power']:.2f}  "
              f"(expected effect ${stats['expected_effect']:.2f})")

    print("\n  SELF-GRADED. The engine maximises E[$] under this simulator and")
    print("  is scored by the same one, so an edge here shows the optimiser")
    print("  works — not that it drafts well against reality. A real claim")
    print("  needs a hold-out season scored on actual results.")

    print("\n  VERDICT")
    ratio = (abs(stats["mean_delta"]) / stats["expected_effect"]
             if stats["expected_effect"] else float("inf"))
    if stats["n"] < 2:
        print("    too few replicates to say anything.")
    elif stats["separated_from_zero"]:
        print("    Beats ADP bots UNDER ITS OWN MODEL, decisively at this n:")
        print(f"    the interval excludes zero and it won "
              f"{stats['wins']}/{stats['n']}.")
        if ratio > 3:
            print(f"\n    BUT the effect is {ratio:.0f}x the ${stats['expected_effect']:.2f} "
                  f"this gate was pre-registered for.")
            print("    An estimate that far above its own prior is evidence the")
            print("    test is measuring self-consistency, not skill. Treat the")
            print("    SIGN as meaningful and the MAGNITUDE as an artefact.")
    elif not stats["powered_for_expected"]:
        print("    INCONCLUSIVE. The interval includes zero and the minimum")
        print(f"    detectable effect (${stats['mde_at_80pct_power']:.2f}) exceeds "
              f"the ${stats['expected_effect']:.2f} this was")
        print("    sized for, so this means 'not enough drafts', NOT 'no edge'.")
    else:
        print("    No detectable edge, and the test WAS powered to find one.")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"vs_adp_{b.snapshot_id}.json"
    path.write_text(json.dumps({
        "snapshot_id": b.snapshot_id,
        "engine": {"reps": args.reps, "shortlist": args.shortlist,
                   "budget_seconds": args.budget,
                   "score_reps": args.score_reps},
        "replicates": [r.__dict__ | {"delta": r.delta} for r in out],
        **stats,
    }, indent=2))
    print(f"\n  wrote {path}")
    print("  settings are stamped in: a cheaper engine is a lower bound, and")
    print("  the number must never be quoted without them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
