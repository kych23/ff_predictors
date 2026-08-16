"""Recommendation -> JSON (§17.3).

Three rules do the real work here, all of them learned the hard way.

**Identity comes from `rec.dollars` keys, never from `ranked`.** Tier-0 `ranked`
carried no `player_id` until the `build_tiers` move added one, and joining on
`player_name` is unsafe — the identity spine documents names as genuinely
non-unique. The dict keys are the only trustworthy identity on that path.

**Every float passes through `_json_float`.** `p_best` is NaN on tiers 2/3,
`adp` is nullable on the board, and the SE dicts are empty on demoted tiers.
This is not only a rendering concern: the ledger's `canonical_json` runs with
``allow_nan=False`` and *raises* on a NaN, so the same helper guards that write.

**The leader is a field, not a row position.** `ranked` is sorted by raw
`E_dollars` while the leader is chosen accounting for uncertainty, so the
recommended player can legitimately sort second — observed live: St. Brown at
$44.81 on 2 draws above the recommended Gibbs at $44.64 on 50. Any client that
renders `candidates[0]` as "the pick" would be wrong. `leader` is authoritative;
order is presentation.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.engine.decision import confidence
from src.engine.decision.recommendation import Recommendation


def _json_float(value: Any) -> float | None:
    """Non-finite -> None. JSON has no NaN, and the ledger refuses one."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _board_lookup(board: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in board.itertuples(index=False):
        out[str(row.player_id)] = {
            "name": str(getattr(row, "player_name", "")),
            "position": str(getattr(row, "position", "")),
            "team": getattr(row, "team", None),
            "adp": _json_float(getattr(row, "adp", None)),
        }
    return out


def _tier0_candidates(rec: Recommendation, lookup: dict) -> list[dict]:
    """Driven by `rec.dollars`; `ranked` supplies per-candidate `draws`."""
    draws_by_id: dict[str, int] = {}
    if "player_id" in getattr(rec.ranked, "columns", []):
        for row in rec.ranked.itertuples(index=False):
            draws_by_id[str(row.player_id)] = int(getattr(row, "draws", 0) or 0)

    indifferent = {str(p) for p in rec.indifference_set}
    out = []
    for pid, dollars in rec.dollars.items():
        pid = str(pid)
        meta = lookup.get(pid, {})
        out.append({
            "player_id": pid,
            "name": meta.get("name", pid),
            "position": meta.get("position", ""),
            "team": meta.get("team"),
            "adp": meta.get("adp"),
            "e_dollars": _json_float(dollars),
            "aleatory_se": _json_float(rec.aleatory_se.get(pid)),
            "epistemic_se": _json_float(rec.epistemic_se.get(pid)),
            "total_se": _json_float(rec.total_se(pid)),
            "draws": draws_by_id.get(pid),
            "vona_score": None,
            "in_indifference_set": pid in indifferent,
        })
    # Leader first, then by DEPTH, then dollars. A two-draw estimate carries a
    # standard error two to three times the leader's, so ranking it on its
    # point estimate alone puts an arm the allocator dismissed at the top of
    # the screen — the engine appearing to argue with itself.
    leader = str(rec.leader) if rec.leader is not None else None
    out.sort(key=lambda c: (c["player_id"] != leader,
                            -(c["draws"] or 0),
                            c["e_dollars"] is None,
                            -(c["e_dollars"] or 0.0)))
    return out


def _fallback_candidates(rec: Recommendation, lookup: dict) -> list[dict]:
    """Tiers 2 and 3 return board-shaped frames that already carry `player_id`
    but have no dollar estimate at all."""
    frame = rec.ranked
    if frame is None or not len(frame):
        return []
    indifferent = {str(p) for p in rec.indifference_set}
    out = []
    for row in frame.itertuples(index=False):
        pid = str(getattr(row, "player_id", ""))
        meta = lookup.get(pid, {})
        out.append({
            "player_id": pid,
            "name": str(getattr(row, "player_name", meta.get("name", pid))),
            "position": str(getattr(row, "position", meta.get("position", ""))),
            "team": meta.get("team"),
            "adp": _json_float(getattr(row, "adp", meta.get("adp"))),
            "e_dollars": None,
            "aleatory_se": None,
            "epistemic_se": None,
            "total_se": None,
            "draws": None,
            "vona_score": _json_float(getattr(row, "vona_score", None)),
            "in_indifference_set": pid in indifferent,
        })
    return out


def recommendation_payload(rec: Recommendation, board: pd.DataFrame, *,
                           snapshot_id: str, generation: int,
                           reps: int, shortlist: int,
                           budget_seconds: float,
                           current_pick: int | None = None) -> dict:
    """The `rec_ready` body. Narration is attached later, never inline.

    `engine` is stamped in because `reps` and `budget_seconds` are
    decision-affecting but cannot live inside `strategy_hash` (they are
    machine- and time-dependent). Without it, two runs could differ in leader
    while carrying an identical `decision_version`.
    """
    lookup = _board_lookup(board)
    candidates = (_tier0_candidates(rec, lookup) if rec.dollars
                  else _fallback_candidates(rec, lookup))

    leader = str(rec.leader) if rec.leader is not None else None
    if leader is not None and not any(c["player_id"] == leader for c in candidates):
        meta = lookup.get(leader, {})
        candidates.insert(0, {
            "player_id": leader, "name": rec.leader_name or meta.get("name", leader),
            "position": meta.get("position", ""), "team": meta.get("team"),
            "adp": meta.get("adp"), "e_dollars": None, "aleatory_se": None,
            "epistemic_se": None, "total_se": None, "draws": None,
            "vona_score": None, "in_indifference_set": True,
        })

    leader_row = next((c for c in candidates
                       if c["player_id"] == leader), None)
    conf = confidence.score(
        tier=int(rec.tier),
        p_best=rec.p_best,
        stopped_because=rec.stopped_because,
        adp=(leader_row or {}).get("adp"),
        current_pick=current_pick,
        indifference_size=len(rec.indifference_set),
        has_projection=bool((leader_row or {}).get("e_dollars") is not None),
        stale_flags=tuple(rec.stale_flags),
    )

    return {
        "snapshot_id": snapshot_id,
        "generation": generation,
        "tier": int(rec.tier),
        "confidence": {"score": conf.score, "label": conf.label,
                       "drivers": conf.drivers},
        "leader": leader,
        "leader_name": rec.leader_name,
        "elapsed_s": _json_float(rec.elapsed_s),
        "p_best": _json_float(rec.p_best),
        "draws_used": int(rec.draws_used),
        "stopped_because": rec.stopped_because,
        "separating_axis": rec.separating_axis,
        "stale_flags": list(rec.stale_flags),
        "indifference_set": [str(p) for p in rec.indifference_set],
        "engine": {"reps": reps, "shortlist": shortlist,
                   "budget_seconds": budget_seconds},
        "candidates": candidates,
        "narration": None,
    }


def narration_payload(narration) -> dict:
    """`verified` is a property, not a field — a client cannot compute it."""
    return {
        "text": narration.text,
        "source": narration.source,
        "verified": bool(narration.verified),
        "reason": narration.reason,
        "model": narration.model,
        "latency_ms": _json_float(narration.latency_ms),
    }


def ledger_recommendation(payload: dict) -> dict:
    """What goes in the hash-chained ledger: everything except `candidates`.

    The candidate list is unbounded and would bloat every entry. It is also the
    part most likely to smuggle a DataFrame into `canonical_json`, which uses
    ``default=str`` and would silently stringify one into the hash rather than
    refusing it.
    """
    return {k: v for k, v in payload.items() if k != "candidates"}


def board_payload(board, data) -> dict:
    """A rankings board, plus which of its players the current bundle lost.

    `stale_player_ids` rides on EVERY board-returning response, not just `GET`.
    The client swaps the response straight into state, so attaching it only to
    the initial read would make the greyed rows vanish on the first save — a
    board would appear to heal itself and then not.

    Staleness is set membership against the live bundle and nothing else. An
    earlier design compared the board's recorded snapshot to the training
    matrix's; snapshot ids are per artifact family, so that comparison would
    have been false on every board ever saved.
    """
    from src.app.rankings.schema import to_json

    payload = to_json(board)
    known = set(data.board["player_id"].astype(str)) if not data.board.empty \
        else set()
    payload["stale_player_ids"] = sorted(board.player_ids() - known)
    return payload
