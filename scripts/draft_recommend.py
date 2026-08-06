"""Draft-night recommendation across the full degradation ladder (§7.3, §19).

    venv/bin/python scripts/draft_recommend.py --slot 4
    venv/bin/python scripts/draft_recommend.py --slot 4 --taken "Jahmyr Gibbs,Bijan Robinson"

Offline: reads the bundle and the fitted artifacts, nothing else.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.cockpit.ladder import Recommendation, recommend  # noqa: E402
from src.app.narration import NarrationConfig, narrate  # noqa: E402
from src.app.narration.attribution import build_record  # noqa: E402
from src.core.config import load_league, load_strategy  # noqa: E402
from src.core.config.slots import build_slot_plan  # noqa: E402
from src.domain.payout.compile import compile_payout  # noqa: E402
from src.engine.decision import recommend as tier2  # noqa: E402
from src.engine.decision import roster_state as rs_mod  # noqa: E402
from src.engine.decision.allocate import allocate  # noqa: E402
from src.engine.decision.board import Board  # noqa: E402
from src.engine.sim import kernel, rng as rng_mod  # noqa: E402
from src.engine.sim.draws import draw_points  # noqa: E402
from src.engine.sim.rollout import rollout  # noqa: E402
from src.models.correlation.slot_matrix import from_prior  # noqa: E402
from src.models.correlation import slot_matrix as sm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from simulate import build_projection_bundle, load_correlation_matrix  # noqa: E402


def _load_posterior():
    """Bootstrap correlation posterior, if one has been fitted.

    Absent it the kernel still runs on the point estimate — but epistemic_se
    then reflects only rollout variation, and the recommendation says so via a
    stale flag rather than quietly reporting a number that means something
    else.
    """
    path = sorted(Path("data/artifacts").glob("correlation_posterior_*.npz"))
    if not path:
        return None
    import numpy as _np
    with _np.load(path[-1], allow_pickle=False) as data:
        slots = tuple(json.loads(str(data["slots"])))
        mats = data["matrices"]
    draws = tuple(
        sm.SlotCorrelation(
            slots=slots, matrix=m, cholesky=_np.linalg.cholesky(m),
            pair_counts=_np.zeros((len(slots), len(slots)), dtype=int),
            min_eigenvalue=float(_np.linalg.eigvalsh(m).min()),
            was_projected=False, source="bootstrap",
        )
        for m in mats
    )
    return sm.CorrelationPosterior(point=draws[0], draws=draws)


def build_tiers(board: Board, cfg, strategy, slot: int, reps: int,
                shortlist: int, my_roster: list[str] | None = None,
                by_seat: dict[int, list[str]] | None = None):
    """Construct the four tiers as closures over shared state.

    `by_seat` maps seat -> the player_ids that seat has already taken, and it
    is load-bearing. Two things depend on it:

    * **`rollout` resumes the draft instead of restarting it.** `already_taken`
      seeds each seat's roster and position counts, which is what makes
      `_legal` force a mandatory slot late and what puts `pick_no` at the right
      place in the snake.
    * **The projection bundle spans the FULL board**, not just what is
      available, so players I already hold still contribute to my team's
      weekly totals.

    Before this existed, `taken_rows` was an empty dict on every call: every
    rollout began at pick 1 with twelve empty rosters, so E[$] answered "which
    player is best on a team built from scratch" at every pick. Measured over a
    full 180-pick rehearsal, that produced 8 WR, 5 RB, 2 DST and **no QB, TE or
    K** — a roster that cannot field a legal lineup, recommended with complete
    confidence at every step.
    """
    weeks = cfg.schedule.regular_season_weeks + len(cfg.schedule.playoff_weeks)

    # Full board with a stable integer index. Rollout, the projection bundle
    # and the candidate rows all address players by this row number.
    frame = board.players.reset_index(drop=True).copy()
    frame["player_id"] = frame["player_id"].astype(str)
    row_of = {pid: i for i, pid in enumerate(frame["player_id"])}
    drafted_ids = {str(p) for p in board.drafted}
    avail = frame[~frame["player_id"].isin(drafted_ids)]

    taken_rows: dict[int, int] = {}
    for seat, pids in (by_seat or {}).items():
        for pid in pids:
            row = row_of.get(str(pid))
            if row is not None:
                taken_rows[row] = int(seat)

    mine = [{"player_id": str(p), "position": str(frame.at[row_of[str(p)],
                                                          "position"])}
            for p in (my_roster or []) if str(p) in row_of]
    rs = rs_mod.RosterState(cfg=cfg, draft_position=slot, my_roster=mine,
                            drafted=set(drafted_ids))
    position = len(drafted_ids) + 1
    my_pick = next((p for p in rs.my_picks if p >= position), rs.my_picks[-1])
    flags = board.stale_flags(cfg)

    def tier2_fn(_budget: float) -> Recommendation:
        result = tier2.score(avail, rs, cfg, strategy, current_pick=my_pick,
                             stale_flags=tuple(flags))
        live = result.ranked[result.ranked["sink_reason"] == ""]
        leader = live.iloc[0]
        return Recommendation(
            tier=2, leader=str(leader["player_id"]),
            leader_name=str(leader["player_name"]),
            ranked=live.head(10), indifference_set=[str(leader["player_id"])],
            stale_flags=flags + ["tier2_vona_only"],
        )

    def tier3_fn(_budget: float) -> Recommendation:
        ranked = avail.sort_values("adp").head(10)
        leader = ranked.iloc[0]
        return Recommendation(
            tier=3, leader=str(leader["player_id"]),
            leader_name=str(leader["player_name"]), ranked=ranked,
            indifference_set=[str(leader["player_id"])],
            stale_flags=flags + ["static_adp_list"],
        )

    def tier0_fn(budget: float) -> Recommendation:
        deadline = time.perf_counter() + budget
        # The FITTED matrix when one exists. `from_prior` is the pre-fit
        # fallback; preferring it once an artifact is on disk means running the
        # draft on assumed correlations. (Same bug was live in simulate.py.)
        corr = load_correlation_matrix()
        proj = build_projection_bundle(frame, cfg, weeks)
        plan = build_slot_plan(cfg.roster.slots, cfg.roster.flex_eligibility)
        objective = compile_payout(strategy.payout, cfg)
        root = rng_mod.seed_root(board.snapshot_id, cfg.model_version,
                                 strategy.strategy_hash)

        shortlist_rows = tier2.score(
            avail, rs, cfg, strategy, current_pick=my_pick,
        ).ranked
        live = shortlist_rows[shortlist_rows["sink_reason"] == ""]
        # Address candidates by PLAYER_ID, then map to full-board rows.
        # `tier2.score` reindexes its input, so its row labels enumerate the
        # AVAILABLE subset — identical to full-board rows only while the two
        # frames are the same object, which stopped being true when the
        # projection bundle moved to the full board. Reading them as full-board
        # rows silently nominated already-drafted players (observed: the
        # engine recommending a player taken at pick 1, at pick 165).
        candidates = [str(row_of[str(pid)])
                      for pid in live.head(shortlist)["player_id"].astype(str)
                      if str(pid) in row_of and str(pid) not in drafted_ids]

        # THETA DRAWS. Each parameter draw uses a different correlation matrix
        # from the bootstrap posterior, so `epistemic_se` measures parameter
        # uncertainty rather than rollout variation. §15.1 keeps the aleatory
        # streams fixed across draws (c = 0), so the ONLY thing varying between
        # draws is theta — which is what makes the two-level decomposition
        # mean anything.
        posterior = _load_posterior()
        cache: dict[int, np.ndarray] = {}

        def points_for(draw: int) -> np.ndarray:
            if draw not in cache:
                theta = (posterior.draw(draw) if posterior is not None else corr)
                cache[draw] = draw_points(proj, theta.cholesky, root, reps=reps)
            return cache[draw]

        def _run(candidate: str, draw: int):
            row = int(candidate)
            res = rollout(frame, cfg, strategy, my_seat=slot - 1,
                          already_taken=taken_rows,
                          forced=(slot - 1, row), root=root, rep=draw)
            masks = kernel.build_masks(proj, res.rosters, plan, weeks)
            team_week = kernel.evaluate_rosters(points_for(draw), masks)
            outcome = kernel.season_outcome(team_week, cfg,
                                            my_team_index=slot - 1)
            return outcome, team_week

        def evaluate(candidate: str, draw: int) -> np.ndarray:
            outcome, _ = _run(candidate, draw)
            return objective(outcome)[slot - 1]

        def attribution(leader: str, runner: str, n_draws: int):
            """§18's record, and §17.3's `separating_axis`.

            Both need the per-prize decomposition, which `evaluate` throws
            away, so the top two are re-run here.

            **Averaged over the COMMON DRAW PREFIX, not over draw 0.** A single
            draw is a sample, and comparing it against an allocator mean taken
            over fifty draws produces a narration that contradicts the
            recommendation it is explaining. Observed live: the table read
            "Gibbs over Robinson, total -3.70" while the ranking had Gibbs
            ahead by +5.59. The common prefix is also the only range over which
            the CRN pairing is valid, since halving leaves the survivor with
            more draws than the candidate it beat.
            """
            parts_a: dict[str, np.ndarray] = {}
            parts_b: dict[str, np.ndarray] = {}
            tw_a = tw_b = None
            for draw in range(max(n_draws, 1)):
                out_a, week_a = _run(leader, draw)
                out_b, week_b = _run(runner, draw)
                for store, decomposed in ((parts_a, objective.decompose(out_a)),
                                          (parts_b, objective.decompose(out_b))):
                    for prize, arr in decomposed.items():
                        store[prize] = (arr if prize not in store
                                        else store[prize] + arr)
                tw_a = week_a if tw_a is None else tw_a + week_a
                tw_b = week_b if tw_b is None else tw_b + week_b

            n = max(n_draws, 1)
            return ({k: v / n for k, v in parts_a.items()},
                    {k: v / n for k, v in parts_b.items()},
                    tw_a / n, tw_b / n)

        result = allocate(
            candidates, evaluate,
            initial_draws=strategy.simulation.initial_draws_per_candidate,
            max_draws=strategy.simulation.outer_parameter_draws,
            indifference_zone=float(
                strategy.simulation.decision["indifference_zone_dollars"]),
            deadline=deadline,
        )
        # ROW -> PLAYER_ID at the boundary. Inside tier 0 a candidate is a
        # board row (that is what `rollout` and `evaluate` address); outside,
        # every tier must speak player_ids, because the ledger, the cockpit
        # and tiers 2/3 all do. Leaking rows out of here made `rec.leader`
        # mean something different depending on which tier answered.
        pid_of = {str(i): str(frame.at[i, "player_id"]) for i in frame.index}
        names = {str(i): frame.at[i, "player_name"] for i in frame.index}

        # §17.3: the prize with the largest absolute mean difference between
        # leader and runner-up. Empty when there is no runner-up to separate
        # from, rather than a misleading default.
        separating_axis, record = "", None
        ordered = sorted(result.estimates.items(), key=lambda kv: -kv[1].mean)
        if len(ordered) > 1:
            runner = next(c for c, _ in ordered if c != result.leader)
            common = min(result.estimates[result.leader].draws_used,
                         result.estimates[runner].draws_used)
            try:
                parts_a, parts_b, tw_a, tw_b = attribution(
                    result.leader, runner, common)
                record = build_record(
                    pid_of[result.leader], pid_of[runner], leader_parts=parts_a,
                    runner_parts=parts_b, seat=slot - 1,
                    leader_team_week=tw_a, runner_team_week=tw_b,
                    aleatory_se=result.estimates[result.leader].aleatory_se,
                    epistemic_se=result.estimates[result.leader].epistemic_se,
                    roster_slot_affected=str(frame.at[int(result.leader),
                                                      "position"]),
                    names={pid_of[c]: n for c, n in names.items()},
                )
                separating_axis = max(record.delta_by_prize,
                                      key=lambda p: abs(record.delta_by_prize[p]))
                # COHERENCE GUARD. The attribution is a re-run, so it can
                # disagree with the allocator when the common prefix is short.
                # A narration that explains why the leader wins while its own
                # numbers say it loses is worse than no narration: it reads as
                # authoritative and contradicts the pick it accompanies.
                if record.total_delta < 0:
                    print(f"  (attribution disagrees with the ranking over the "
                          f"{common}-draw common prefix: "
                          f"{record.total_delta:+.2f}; suppressing narration)")
                    record = None
            except Exception as exc:      # noqa: BLE001 — never cost a pick
                print(f"  (attribution unavailable: {exc})")

        tier0_fn.record = record
        ranked = pd.DataFrame([
            {"player_name": names[c], "position": frame.at[int(c), "position"],
             "adp": frame.at[int(c), "adp"], "E_dollars": e.mean,
             "aleatory_se": e.aleatory_se, "epistemic_se": e.epistemic_se,
             "draws": e.draws_used}
            for c, e in sorted(result.estimates.items(), key=lambda kv: -kv[1].mean)
        ])
        return Recommendation(
            tier=0, leader=pid_of[result.leader],
            leader_name=names[result.leader],
            ranked=ranked,
            indifference_set=[pid_of[c] for c in result.indifference_set],
            dollars={pid_of[c]: e.mean for c, e in result.estimates.items()},
            aleatory_se={pid_of[c]: e.aleatory_se
                         for c, e in result.estimates.items()},
            epistemic_se={pid_of[c]: e.epistemic_se
                          for c, e in result.estimates.items()},
            p_best=result.p_best,
            draws_used=max(e.draws_used for e in result.estimates.values()),
            stopped_because=result.stopped_because,
            separating_axis=separating_axis,
            stale_flags=flags + ([] if posterior is not None
                                 else ["epistemic_se_is_rollout_only"]),
        )

    return {0: tier0_fn, 2: tier2_fn, 3: tier3_fn}, my_pick


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slot", type=int, required=True)
    ap.add_argument("--taken", default="")
    ap.add_argument("--reps", type=int, default=512)
    ap.add_argument("--shortlist", type=int, default=4)
    ap.add_argument("--budget", type=float, default=25.0)
    args = ap.parse_args()

    cfg = load_league()
    strategy = load_strategy(cfg)
    board = Board.from_bundle()
    for name in [x.strip() for x in args.taken.split(",") if x.strip()]:
        hits = board.find(name, limit=1)
        if not hits.empty:
            board.take(hits.iloc[0]["player_id"])

    tiers, my_pick = build_tiers(board, cfg, strategy, args.slot,
                                 args.reps, args.shortlist)
    print(f"seat {args.slot}, on the clock at pick {my_pick}, "
          f"{len(board.available())} available")
    print(f"budget {args.budget:.0f}s, demote after "
          f"{strategy.simulation.decision['demote_to_tier1_after_seconds']}s\n")

    rec = recommend(
        tiers, budget_s=args.budget,
        demote_after_s=float(
            strategy.simulation.decision["demote_to_tier1_after_seconds"]),
    )
    print(rec.describe())
    print()
    cols = [c for c in ("player_name", "position", "adp", "E_dollars",
                        "aleatory_se", "epistemic_se", "draws", "vona_score")
            if c in rec.ranked.columns]
    print(rec.ranked[cols].round(3).to_string(index=False))
    if rec.tier == 0:
        print(f"\n  p(best) {rec.p_best:.2f}   stopped: {rec.stopped_because}   "
              f"draws {rec.draws_used}")
        if rec.separating_axis:
            print(f"  separating axis: {rec.separating_axis}")

    # §18 narration. Every quantitative statement is parsed back out and
    # checked against the attribution record before it is printed; anything
    # unentailed falls back to the table, which states the same content and
    # needs no model at all.
    record = getattr(tiers.get(0), "record", None)
    if record is not None:
        narration = narrate(record, NarrationConfig.from_strategy(strategy))
        print(f"\nWHY ({narration.source}"
              f"{'' if narration.verified else ', UNVERIFIED'}):")
        print("  " + narration.text.replace("\n", "\n  "))
        if narration.reason:
            print(f"  [{narration.reason}]")


if __name__ == "__main__":
    main()
