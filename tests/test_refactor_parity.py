"""The move of `build_tiers` and `build_projection_bundle` changed no answers.

This is the one test that makes that refactor safe. `build_tiers` moved from
`scripts/draft_recommend.py` to `src/app/cockpit/build.py` and the projection
bundle builder moved from `scripts/simulate.py` to
`src/engine/sim/bundle_build.py`, so that a web backend could import them —
`scripts/` is not a package. A move that silently changes a recommendation is
worse than no move at all.

**Why the assertion is the leader and not E[$].** `allocate` is bounded by wall
clock, so the number of draws — and therefore the dollar estimate to several
decimals — legitimately varies with machine speed. Asserting E[$] to 3 dp would
produce a test that fails on a slow CI box for a reason that has nothing to do
with correctness. Instead: the same leader, plus `stopped_because != "deadline"`
to prove the run separated on evidence rather than being truncated by the
budget. If the run ever *is* truncated, the leader comparison is meaningless and
the test says so rather than passing by luck.

The golden was generated from the PRE-MOVE path and is committed at
`tests/fixtures/parity_golden.json`. It carries the `snapshot_id` it was
produced against, so a bundle rebuild invalidates it loudly instead of silently.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "fixtures" / "parity_golden.json"
BUNDLE = ROOT / "data" / "bundles" / "draft_night_bundle.parquet"

pytestmark = pytest.mark.slow


def _seat_of(i: int, teams: int) -> int:
    rnd, idx = divmod(i, teams)
    return idx if rnd % 2 == 0 else teams - 1 - idx


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip(f"no parity golden at {GOLDEN}")
    return json.loads(GOLDEN.read_text())


@pytest.fixture(scope="module")
def replayed(golden):
    """Re-run the golden scenario through the POST-move import path."""
    if not BUNDLE.exists():
        pytest.skip("no draft_night_bundle.parquet; run scripts/build_bundle.py")

    from src.app.cockpit.build import build_tiers
    from src.app.cockpit.ladder import recommend
    from src.core.config import load_league, load_strategy
    from src.engine.decision.board import Board
    from src.models.artifacts import code_digest

    cfg = load_league()
    strategy = load_strategy(cfg)
    board = Board.from_bundle()
    if board.snapshot_id != golden["snapshot_id"]:
        pytest.skip(
            f"golden was built against {golden['snapshot_id']} but the bundle "
            f"is {board.snapshot_id}; regenerate the golden")
    # Data identity is not enough. `snapshot_id` is a Merkle root over the
    # SOURCE DATA, so changing feature or model code reprojects every player
    # under an unchanged id. Without this second check the golden would go on
    # comparing against numbers a different engine produced — and pass, which
    # is the one outcome that makes a regression guard worse than none.
    current = code_digest()
    if golden.get("code_digest") not in (None, current):
        pytest.skip(
            f"golden was fitted by code {golden['code_digest']} but this tree "
            f"is {current}; regenerate the golden")

    by_seat: dict[int, list[str]] = {}
    for i, pid in enumerate(golden["drafted"]):
        by_seat.setdefault(_seat_of(i, cfg.teams), []).append(pid)
        board.take(pid)

    tiers, _ = build_tiers(
        board, cfg, strategy, golden["seat"], golden["reps"],
        golden["shortlist"],
        my_roster=list(by_seat.get(golden["seat"] - 1, [])), by_seat=by_seat)
    return recommend(tiers, budget_s=golden["budget_s"], demote_after_s=55.0)


def test_the_run_separated_rather_than_timing_out(replayed):
    """Guards the guard: a deadline-truncated run makes the leader comparison
    below meaningless, so it must not pass silently."""
    assert replayed.stopped_because != "deadline", (
        f"run hit the budget ({replayed.stopped_because}); the leader "
        f"comparison proves nothing on this machine")


def test_the_moved_path_reaches_the_same_tier(golden, replayed):
    assert replayed.tier == golden["tier"]


def test_the_moved_path_recommends_the_same_player(golden, replayed):
    """The whole point of the refactor test."""
    assert replayed.leader == golden["leader"], (
        f"moved path recommends {replayed.leader_name!r}, "
        f"pre-move path recommended {golden['leader_name']!r}")


def test_dollar_estimates_stay_within_simulation_noise(golden, replayed):
    """Not equality — the allocator is wall-clock bounded. But the leader's
    value should not move by more than a few times its own reported SE."""
    pid = golden["leader"]
    before = golden["dollars"][pid]
    after = replayed.dollars[pid]
    tolerance = max(4.0 * replayed.total_se(pid), 1.0)
    assert abs(after - before) <= tolerance, (
        f"E[$] moved {before:.2f} -> {after:.2f}, beyond {tolerance:.2f}")


def test_tier0_ranked_carries_player_id(replayed):
    """Added by the move so the API can id-join per-candidate `draws`.
    `Recommendation.draws_used` is a scalar, and joining on `player_name` is
    unsafe — the identity spine documents names as genuinely non-unique."""
    if replayed.tier != 0:
        pytest.skip("not a tier-0 answer")
    assert "player_id" in replayed.ranked.columns
    assert "draws" in replayed.ranked.columns
    assert set(replayed.ranked["player_id"]) <= set(replayed.dollars)
