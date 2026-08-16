"""Shared fixtures for the web-cockpit tests.

`data/` is gitignored, so a fresh clone has no `draft_night_bundle.parquet` and
no fitted artifacts. Backend tests therefore run against a synthetic `Board`
built here — small, deterministic, and complete enough to exercise every
position and the ADP ordering. The real bundle is used only by the manual
walkthrough and by `test_refactor_parity.py`, which skips without it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.engine.decision.board import Board

SNAPSHOT = "sn2_synthetic_test"

#: Enough at each position to fill twelve rosters' starters plus a bench.
_COUNTS = {"QB": 8, "RB": 14, "WR": 16, "TE": 8, "K": 4, "DST": 4}


@pytest.fixture
def synthetic_players() -> pd.DataFrame:
    rows = []
    adp = 1.0
    for pos, n in _COUNTS.items():
        for i in range(n):
            rows.append({
                "player_id": f"{pos.lower()}-{i:02d}",
                "player_name": f"{pos} Player {i}",
                "position": pos,
                "team": f"T{i % 8:02d}",
                "bye_week": 5 + (i % 6),
                "value": float(24 - i) if pos in ("RB", "WR") else float(16 - i),
                "value_source": "quantile_model",
                "coverage": "full",
                "adp": adp + i * 1.5,
                "adp_stdev": 4.0,
            })
        adp += 12.0
    frame = pd.DataFrame(rows).sort_values("adp").reset_index(drop=True)
    frame["value"] = frame["value"].clip(lower=1.0)
    return frame


@pytest.fixture
def synthetic_board(synthetic_players) -> Board:
    return Board(players=synthetic_players, snapshot_id=SNAPSHOT,
                 value_source="quantile_model", built_at="2026-08-06T00:00:00Z")


@pytest.fixture
def web_cfg(tmp_path):
    from src.core.config.web import EngineConfig, WebConfig

    return WebConfig(
        source="manual", auto_recommend=False,
        session_path=tmp_path / "session.json",
        ledger_path=tmp_path / "ledger.sqlite",
        # Without this, any store built from `web_cfg` writes boards into the
        # real repo tree at data/web/rankings.
        rankings_dir=tmp_path / "rankings",
        engine=EngineConfig(reps=8, budget_seconds=2.0),
    )


@pytest.fixture
def service(synthetic_board, web_cfg):
    """A cockpit wired to the synthetic board, with no session started yet."""
    from src.app.web.service import CockpitService
    from src.core.config import load_league, load_strategy

    cfg = load_league()
    svc = CockpitService(cfg=cfg, strategy=load_strategy(cfg),
                         web_cfg=web_cfg, board=synthetic_board)
    yield svc
    svc.close()


@pytest.fixture
def started(service):
    service.start_session(seat=3, source="manual", resume=False)
    return service


@pytest.fixture
async def client(started):
    """ASGI transport, not Starlette's sync TestClient: it exercises the real
    async path rather than a sync bridge, which is the path production uses."""
    import httpx

    from src.app.web import api as api_mod

    api_mod.app.state.service = started
    transport = httpx.ASGITransport(app=api_mod.app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://cockpit") as c:
        yield c
    api_mod.app.state.service = None


def _fake_recommendation(leader="rb-00", tier=0):
    """Mirrors the observed live case where the recommended player does NOT
    sort first: a rival with a higher point estimate but far fewer draws."""
    from src.engine.decision.recommendation import Recommendation

    ranked = pd.DataFrame([
        {"player_id": "wr-00", "player_name": "WR Player 0", "position": "WR",
         "adp": 13.0, "E_dollars": 44.81, "aleatory_se": 2.96,
         "epistemic_se": 2.43, "draws": 2},
        {"player_id": "rb-00", "player_name": "RB Player 0", "position": "RB",
         "adp": 1.0, "E_dollars": 44.64, "aleatory_se": 1.42,
         "epistemic_se": 1.34, "draws": 50},
    ])
    return Recommendation(
        tier=tier, leader=leader, leader_name="RB Player 0", ranked=ranked,
        indifference_set=["rb-00", "wr-00"],
        dollars={"rb-00": 44.64, "wr-00": 44.81},
        aleatory_se={"rb-00": 1.42, "wr-00": 2.96},
        epistemic_se={"rb-00": 1.34, "wr-00": 2.43},
        p_best=0.61, draws_used=50, stopped_because="separated",
        separating_axis="champion", elapsed_s=9.6,
    )


@pytest.fixture
def fake_recommendation():
    return _fake_recommendation


# ------------------------------------------------------ rankings fixtures
# `synthetic_players` is deliberately NOT extended with value_p10/value_p90.
# `recommend._risk_adjusted_value` reads both, so adding them would shift every
# non-median-alpha round and could change the outcome of every existing test
# built on `synthetic_board`. The rankings surface gets its own frame instead.
@pytest.fixture
def rankings_players(synthetic_players) -> pd.DataFrame:
    """The synthetic board plus the columns the rankings surface reads.

    K and DST carry NaN bands, mirroring the live bundle where all 47 of them
    (plus 2 WR) have no calibrated interval. One row carries a null ADP so
    "nulls sort last" has something to assert against — the live bundle has
    none, and every synthetic row would otherwise have one.
    """
    frame = synthetic_players.copy()
    frame["value_p10"] = frame["value"] - 3.0
    frame["value_p90"] = frame["value"] + 3.0
    unbanded = frame["position"].isin(("K", "DST"))
    frame.loc[unbanded, ["value_p10", "value_p90"]] = float("nan")
    frame.loc[unbanded, "coverage"] = "no_prior_season"
    frame.loc[unbanded, "value_source"] = "kdst_empirical"
    frame.loc[frame.index[-1], "adp"] = float("nan")
    return frame.reset_index(drop=True)


@pytest.fixture
def rankings_matrix(rankings_players) -> pd.DataFrame:
    """One row per QB/RB/WR/TE at the target season — K and DST have none,
    exactly as in the real matrix. `fppg` is present and null, because it is
    the training label and the detail endpoint must never serve it."""
    modeled = rankings_players[
        rankings_players["position"].isin(("QB", "RB", "WR", "TE"))]
    rows = []
    for i, (_, player) in enumerate(modeled.iterrows()):
        rows.append({
            "player_id": player["player_id"], "season": 2026,
            "position": player["position"],
            "fppg": float("nan"),                    # the label
            "prior_fppg": 10.0 + i * 0.1,
            "prior_targets_per_game": 5.0, "prior_target_share": 0.18,
            "prior_carries_per_game": 3.0, "prior_touches_per_game": 8.0,
            "prior_snap_share": 0.7, "prior_catch_rate": 0.65,
            "prior_yards_per_target": 8.1, "prior_yards_per_reception": 12.0,
            "prior_rec_td_rate": 0.05, "prior_yards_per_carry": 4.3,
            "prior_rush_td_rate": 0.03, "prior_rush_yards_per_game": 15.0,
            "prior_pass_attempts_per_game": 30.0, "prior_completion_pct": 0.64,
            "prior_yards_per_attempt": 7.2, "prior_pass_td_rate": 0.045,
            "prior_int_rate": 0.02, "prior_sack_rate": 0.06,
            # float64 with a NaN, like the real column — the bool coercion path
            "is_projected_starter": float("nan") if i == 1 else 1.0,
            "depth_chart_rank": 1.0, "same_position_competition": 2.0,
            "has_depth_data": 1, "vacated_targets": 41.0,
            "vacated_targets_share": 0.09, "vacated_carries": 12.0,
            "vacated_carries_share": 0.04,
            "draft_round": 2.0, "draft_pick_overall": 55.0,
            "is_undrafted": 0, "draft_capital_score": 0.8,
            "seasons_of_history": 0 if i == 0 else 3,
            "team_changed": 0, "age_at_season_start": 24.5,
            "college_rec_yds_z": 1.4, "college_rec_td_z": 0.9,
            "college_rush_yds_z": -0.2, "college_rush_td_z": 0.1,
            "college_scrimmage_yds_z": 1.1, "college_conference_tier": 3.0,
            "has_college_stats": 1 if i == 0 else 0,
            "team_ctx_implied_pts": 23.4, "team_ctx_game_total": 46.5,
            "team_ctx_spread": -1.5,
        })
    frame = pd.DataFrame(rows)
    return frame.set_index("player_id", drop=False)


@pytest.fixture
def rankings_tiers(rankings_players) -> pd.DataFrame:
    """Per-position tiers, `rank` restarting at 1 per position — the shape the
    real artifact has, and the reason `engine_tiers` cannot seed `overall`."""
    rows = []
    for position, group in rankings_players.groupby("position"):
        ordered = group.sort_values("value", ascending=False)
        for rank, (_, player) in enumerate(ordered.iterrows(), 1):
            rows.append({
                "player_id": player["player_id"],
                "player_name": player["player_name"], "position": position,
                "tier": 1 if position in ("K", "DST") else min(
                    (rank - 1) // 3 + 1, 3),
                "rank": rank, "adp": player["adp"],
            })
    return pd.DataFrame(rows)


@pytest.fixture
def rankings_ranks(rankings_players) -> pd.DataFrame:
    """A market export covering most of the board, so the miss path is live."""
    from src.core.names import normalize_name

    covered = rankings_players.iloc[: int(len(rankings_players) * 0.8)]
    rows = [{
        "player_name": player["player_name"],
        "match_key": normalize_name(str(player["player_name"])),
        "position": player["position"], "team": player["team"],
        "consensus_rank": float(i + 1) + 0.5,     # genuinely fractional
        "rank_spread": 5, "rank_sd": 2.0, "n_platforms": 6,
        "rank_yahoo": i + 1, "rank_espn": i + 2, "rank_sleeper": i + 1,
        "rank_underdog": i + 3, "rank_cbs": i + 1, "rank_ffpc": i + 2,
    } for i, (_, player) in enumerate(covered.iterrows())]
    return pd.DataFrame(rows).set_index("match_key", drop=False)


@pytest.fixture
def rankings_data(rankings_players, rankings_matrix, rankings_tiers,
                  rankings_ranks):
    from src.app.rankings.data import RankingsData, _match_keys, _replacement

    return RankingsData(
        board=rankings_players, matrix=rankings_matrix, ranks=rankings_ranks,
        tiers=rankings_tiers, replacement=_replacement(rankings_players),
        match_keys=_match_keys(rankings_players), target_season=2026,
        snapshots={"bundle": SNAPSHOT, "matrix": "sn2_m", "tiers": "sn2_t"},
    )


@pytest.fixture
def rankings_store(web_cfg):
    from src.app.rankings.store import RankingsStore

    return RankingsStore(web_cfg.rankings_dir)
