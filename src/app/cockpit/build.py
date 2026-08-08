"""Tier construction for the cockpit (§7.3, §17).

MOVED from ``scripts/draft_recommend.py``. It lives in `app` rather than
`engine` for a concrete reason: it calls ``build_record`` from
``src.app.narration.attribution``, and `engine` (rank 4) importing `app`
(rank 5) is exactly what ``tests/test_layer_deps.py`` rejects. This is cockpit
orchestration, not engine math, so `app` is the honest home — and from here
both `engine` and the narration package are legal imports.

The move exists so a web backend can import it; ``scripts/`` is not a package
and the scripts reach each other through a ``sys.path`` insert.

Three behavioural additions over the script version, all recorded in
notes/draft-cockpit-web.md:

* ``unresolved_count`` — a pick whose player could not be identified never
  reaches ``board.drafted``, so ``position``, ``RosterState`` and ``rollout``
  are each short by that count and the snake seat drifts for the rest of the
  draft. One integer, threaded to all three.
* tier-0 ``ranked`` now carries ``player_id``. ``Recommendation.draws_used`` is
  a SCALAR, so per-candidate draw counts exist only in this frame; without an
  id the API would have to join on ``player_name``, which the identity spine
  documents as genuinely non-unique.
* ``artifacts_dir`` is threaded to the bundle builder so a test can run without
  ``data/artifacts`` (which is gitignored).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.app.narration.attribution import build_record
from src.core.config.slots import build_slot_plan
from src.domain.payout.compile import compile_payout
from src.engine.decision import recommend as tier2
from src.engine.decision import roster_state as rs_mod
from src.engine.decision.allocate import allocate
from src.engine.decision.board import Board
from src.engine.decision.recommendation import Recommendation
from src.engine.decision.survival import survival_probability
from src.engine.sim import kernel
from src.engine.sim import rng as rng_mod
from src.engine.sim.bundle_build import (
    build_projection_bundle,
    load_correlation_matrix,
    load_posterior,
)
from src.engine.sim.draws import draw_points
from src.engine.sim.rollout import rollout


def _player_facts(rows: list[str], proj, frame, pid_of: dict) -> dict:
    """Verifiable football facts for the narration (§18).

    Every value here comes off the board or a fitted model, so the gate can
    check a clause about it. That is the whole point: "he misses fewer games"
    is only worth saying if someone can tell whether it is true.

    * `games`        expected games from the fitted availability hazard
    * `consistency`  weekly sigma — LOWER is steadier, stated that way in the prompt
    * `adp`          market draft position

    Not included, and not an oversight: team changes, target share, depth-chart
    moves. The bundle carries a current team and no history, so none of it is
    checkable, and an unverifiable clause is what the gate exists to reject.
    """
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        i = int(row)
        facts: dict[str, float] = {}
        try:
            facts["games"] = round(float(proj.games_hazard[i].sum()), 1)
            facts["consistency"] = round(float(proj.weekly_sigma[i]), 1)
        except Exception:                  # noqa: BLE001 — never cost a pick
            pass
        adp = frame.at[i, "adp"] if "adp" in frame.columns else None
        if adp is not None and not pd.isna(adp):
            facts["adp"] = round(float(adp), 1)
        if facts:
            out[pid_of[row]] = facts
    return out


def _cell(frame: pd.DataFrame, row: int, column: str):
    """Optional board column, or None — boards vary by source and tier."""
    if column not in frame.columns:
        return None
    value = frame.at[row, column]
    return None if pd.isna(value) else value


def _bye_of(frame: pd.DataFrame, row: int) -> int | None:
    value = _cell(frame, row, "bye_week")
    if value is None:
        return None
    week = int(value)
    return week if week > 0 else None      # 0 encodes "no bye on record"


def my_roster_rows(frame: pd.DataFrame, row_of: dict[str, int],
                   my_roster: list[str] | None) -> list[dict]:
    """My drafted players as `RosterState` rows, board columns attached.

    `team` and `bye_week` ride along for the NARRATION only. They do not feed
    the objective — `sim.kernel.starter_values` masks byes week by week off
    the bundle's own `bye_weeks`, so a clustered-bye roster already loses
    dollars in the simulation. Carrying them here is what lets the engine say
    WHY, instead of silently docking a player and explaining something else.
    """
    rows = []
    for pid in (my_roster or []):
        row = row_of.get(str(pid))
        if row is None:
            continue
        rows.append({
            "player_id": str(pid),
            "position": str(frame.at[row, "position"]),
            "team": _cell(frame, row, "team"),
            "bye_week": _bye_of(frame, row),
        })
    return rows


def _survival_probabilities(frame: pd.DataFrame, rows: list[str],
                            pid_of: dict, next_pick: int | None) -> dict:
    """P(each candidate is still on the board at MY next turn).

    The record has carried this field since §18 and the production caller
    never passed it, so `allowed_subjects` returned an empty `probability`
    list, the schema enum could not contain a probability subject, and
    `verify`'s probability rule was unreachable. The narration's only market
    input was switched off — the same failure as `bye_conflicts`, on the fact
    that most often decides a pick: whether waiting is even an option.

    None when there is no next pick. At the last turn nothing survives to a
    turn that does not exist, and 1.0 would read as "he is certain to last".
    """
    if next_pick is None:
        return {}
    out: dict[str, float] = {}
    for row in rows:
        i = int(row)
        adp = _cell(frame, i, "adp")
        if adp is None:
            continue
        out[pid_of[row]] = float(survival_probability(
            float(adp), _cell(frame, i, "adp_stdev"), next_pick))
    return out


def _bye_conflicts(rs, frame: pd.DataFrame, row: int) -> list[int]:
    """Weeks where taking this player ADDS to an existing bye pile-up.

    Only weeks my roster already occupies count. A player's bye in isolation is
    not a conflict — every player has one, and flagging all of them would make
    the signal meaningless.

    This is explanation, not scoring. `kernel.starter_values` already zeroes a
    player in his bye week, so a clustered-bye roster loses real dollars in the
    simulation whether or not this list is ever populated. Before this was
    wired, the engine acted on bye clustering and then declined to mention it.
    """
    bye = _bye_of(frame, row)
    if bye is None:
        return []
    return [bye] if rs.my_bye_week_counts().get(bye, 0) > 0 else []


def build_tiers(board: Board, cfg, strategy, slot: int, reps: int,
                shortlist: int, my_roster: list[str] | None = None,
                unresolved_count: int = 0,
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

    mine = my_roster_rows(frame, row_of, my_roster)
    rs = rs_mod.RosterState(cfg=cfg, draft_position=slot, my_roster=mine,
                            drafted=set(drafted_ids),
                            unresolved_count=unresolved_count)
    # An unresolved pick is a real pick whose player we could not identify.
    # It never reaches `board.drafted` (there is no id to take), so every
    # counter derived from the board is short by exactly that many unless it
    # is added back explicitly. See notes/draft-cockpit-web.md.
    position = len(drafted_ids) + unresolved_count + 1
    my_pick = next((p for p in rs.my_picks if p >= position), rs.my_picks[-1])
    flags = board.stale_flags(cfg)

    def tier2_fn(_budget: float) -> Recommendation:
        result = tier2.score(avail, rs, cfg, strategy, current_pick=my_pick,
                             stale_flags=tuple(flags), preseason_board=frame)
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
            preseason_board=frame,
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

        # OUTER DRAWS. Each one uses a different correlation matrix from the
        # bootstrap posterior AND a different opponent-draft realization —
        # `_run` passes `rep=draw` into `rollout`, whose softmax uniforms are
        # addressed on `rep`.
        #
        # This comment used to claim "the ONLY thing varying between draws is
        # theta". That was not true, and the gap is not small. Pinning the
        # draft to a single realization and re-running collapses the reported
        # `epistemic_se` on the two deepest arms from 0.885 to 0.248 and from
        # 0.799 to 0.190 — roughly **4x**, so most of what the field is named
        # after is opponent-draft variation, not parameter uncertainty.
        #
        # The ESTIMATE is unaffected and remains correct: E[$] must average
        # over opponent behaviour as well as over theta, and the outer draws
        # sample that joint distribution, so `total_se` is a valid standard
        # error of the mean under the design actually run. What is wrong is the
        # NAME. Anything reasoning about theta specifically — §5.3's elasticity
        # argument, "is the nuisance parameter larger than the decision" — is
        # reading a number dominated by something else.
        #
        # Separating the two properly needs a crossed theta x draft design,
        # which multiplies the work per candidate and does not fit the pick
        # clock. Recorded rather than papered over.
        posterior = load_posterior()
        cache: dict[int, np.ndarray] = {}

        def points_for(draw: int) -> np.ndarray:
            if draw not in cache:
                theta = (posterior.draw(draw) if posterior is not None else corr)
                cache[draw] = draw_points(proj, theta.cholesky, root, reps=reps)
            return cache[draw]

        def _run(candidate: str, draw: int):
            row = int(candidate)
            res = rollout(frame, cfg, strategy, my_seat=slot - 1,
                          pick_offset=unresolved_count,
                          already_taken=taken_rows,
                          forced=(slot - 1, row), root=root, rep=draw)
            masks = kernel.build_masks(proj, res.rosters, plan, weeks)
            points = points_for(draw)
            team_week = kernel.evaluate_rosters(points, masks)
            team_week = kernel.apply_waiver_floor(points, masks, team_week,
                                                  res.rosters, proj, cfg,
                                                  strategy,
                                                  in_decision_path=True)
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
                    bye_conflicts=_bye_conflicts(rs, frame,
                                                 int(result.leader)),
                    survival_probabilities=_survival_probabilities(
                        frame, [result.leader, runner], pid_of,
                        rs.next_my_pick(after_overall=my_pick)),
                    player_facts=_player_facts(
                        [result.leader, runner], proj, frame, pid_of),
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
            {"player_id": pid_of[c], "player_name": names[c], "position": frame.at[int(c), "position"],
             "adp": frame.at[int(c), "adp"], "E_dollars": e.mean,
             "aleatory_se": e.aleatory_se, "epistemic_se": e.epistemic_se,
             "draws": e.draws_used}
            # Leader first, then by DEPTH, then mean. Sorting purely by mean
            # floats arms eliminated after two draws above the one the
            # allocator spent fifty on, which reads as the engine contradicting
            # its own recommendation.
            for c, e in sorted(
                result.estimates.items(),
                key=lambda kv: (kv[0] != result.leader, -kv[1].draws_used,
                                -kv[1].mean))
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


