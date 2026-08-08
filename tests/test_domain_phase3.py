"""Domain-layer tests: scoring, lineup, bracket, payout (§21.3, §21.7)."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.config import load_league, load_strategy
from src.core.config.slots import build_slot_plan
from src.core.errors import ConfigError
from src.domain.payout.compile import compile_payout
from src.domain.payout.reducers import REDUCERS, SeasonOutcome
from src.domain.roster.lineup import greedy_lineup, lineup_points
from src.domain.roster.matroid import CounterOracle, MatchingOracle
from src.domain.schedule.bracket import final_ranks, regular_season_ranks, round_robin
from src.domain.scoring.dst import implied_points, points_allowed_score, score_dst
from src.domain.scoring.engine import score_offense, score_stat_line
from src.domain.scoring.kicker import map_buckets_to_bands, score_kicker
from tests.oracles.ilp_lineup import ilp_best_lineup

LIVE_SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1, "BENCH": 6}
LIVE_FLEX = {"FLEX": ["RB", "WR", "TE"]}


@pytest.fixture(scope="module")
def league():
    return load_league()


@pytest.fixture(scope="module")
def plan():
    return build_slot_plan(LIVE_SLOTS, LIVE_FLEX)


# ---------------------------------------------------------------- scoring
def test_offense_scores_a_known_line(league):
    """A hand-computed line, so the coefficients are checked not just the wiring."""
    line = {
        "passing_yards": 300, "passing_tds": 2, "passing_interceptions": 1,
        "rushing_yards": 20, "rushing_tds": 1,
        "receptions": 0, "receiving_yards": 0, "receiving_tds": 0,
    }
    # 300*.04 + 2*4 - 1 + 20*.1 + 6 = 12 + 8 - 1 + 2 + 6 = 27
    assert score_stat_line(line, league.scoring.offense) == pytest.approx(27.0)


def test_multi_column_components_are_summed(league):
    """Fumbles and 2pt each split across three columns; missing one is a
    silent scoring error no total-level test would catch."""
    each = {"rushing_fumbles_lost": 1, "receiving_fumbles_lost": 1,
            "sack_fumbles_lost": 1}
    assert score_stat_line(each, league.scoring.offense) == pytest.approx(-6.0)
    two_pt = {"passing_2pt_conversions": 1, "rushing_2pt_conversions": 1,
              "receiving_2pt_conversions": 1}
    assert score_stat_line(two_pt, league.scoring.offense) == pytest.approx(6.0)


def test_off_fumble_return_td_is_scored(league):
    assert score_stat_line({"fumble_recovery_tds": 1},
                           league.scoring.offense) == pytest.approx(6.0)


def test_unknown_scoring_key_raises(league):
    with pytest.raises(KeyError, match="no column mapping"):
        score_offense(pd.DataFrame([{"passing_yards": 1}]), {"bogus_key": 1.0})


def test_rows_without_stat_columns_score_zero(league):
    df = pd.DataFrame(index=range(3))
    out = score_offense(df, league.scoring.offense)
    assert len(out) == 3 and (out == 0).all()


def test_kicker_bands_map_onto_nflverse_buckets(league):
    mapping = map_buckets_to_bands(league.scoring.kicker["fg_bands"])
    assert mapping["fg_made_0_19"] == 3
    assert mapping["fg_made_20_29"] == 3
    assert mapping["fg_made_30_39"] == 3
    assert mapping["fg_made_40_49"] == 4
    assert mapping["fg_made_50_59"] == 5
    assert mapping["fg_made_60_"] == 5


def test_kicker_refuses_bands_that_straddle_a_bucket():
    """player_stats carries counts, not distances, so a band boundary inside a
    bucket is unrecoverable — fail loudly rather than mis-score."""
    with pytest.raises(ConfigError, match="straddles"):
        map_buckets_to_bands([[0, 24, 3], [25, 49, 4], [50, 99, 5]])


def test_kicker_scores_a_known_row(league):
    df = pd.DataFrame([{"fg_made_30_39": 2, "fg_made_40_49": 1,
                        "fg_made_50_59": 1, "pat_made": 3}])
    # 2*3 + 1*4 + 1*5 + 3*1 = 18
    assert score_kicker(df, league.scoring.kicker).iloc[0] == pytest.approx(18.0)


def test_points_allowed_tiers(league):
    tiers = league.scoring.dst["points_allowed_tiers"]
    got = points_allowed_score(pd.Series([0, 3, 10, 17, 24, 30, 45]), tiers)
    assert list(got) == [10, 7, 4, 1, 0, -1, -4]


def test_dst_scores_a_known_row(league):
    df = pd.DataFrame([{
        "def_sacks": 3, "def_interceptions": 2, "fumble_recovery_opp": 1,
        "def_tds": 1, "def_safeties": 0, "fg_blocked": 1,
        "pat_blocked": 0, "pt_blocked": 0, "pt_return_tds": 0,
        "points_allowed": 10,
    }])
    # 3 + 4 + 2 + 6 + 0 + 2 + 4(tier) = 21
    assert score_dst(df, league.scoring.dst).iloc[0] == pytest.approx(21.0)


def test_dst_requires_points_allowed(league):
    with pytest.raises(KeyError, match="points_allowed"):
        score_dst(pd.DataFrame([{"def_sacks": 1}]), league.scoring.dst)


def test_implied_points_sign_convention():
    """nflverse signs spread_line from the HOME perspective, POSITIVE meaning
    home favored.

    Measured over 1,408 completed games (2020-2024):
    corr(spread_line, home margin) = +0.439, and on the 333 games with
    |spread| > 7 the home team won 84.6% of the time when the line was
    positive against 21.5% when it was negative.

    The previous version of this test asserted the opposite and passed,
    because the function it guarded had the same inversion. A test that
    encodes a convention has to be checked against the data, not against the
    implementation it is protecting.
    """
    total = pd.Series([45.0, 45.0])
    spread = pd.Series([7.0, 7.0])          # POSITIVE = home favored by 7
    home = implied_points(total, spread, is_home=pd.Series([True, True]))
    away = implied_points(total, spread, is_home=pd.Series([False, False]))
    assert home.iloc[0] == pytest.approx(26.0), "the favourite scores more"
    assert away.iloc[0] == pytest.approx(19.0)
    assert (home + away).iloc[0] == pytest.approx(45.0)


def test_both_implied_points_implementations_agree():
    """The actual defect: one formula, two implementations, no shared source.
    Whichever is edited next, they must not drift apart again."""
    from src.models.features._utils import schedule_to_team_lines

    sched = pd.DataFrame([{"home_team": "AAA", "away_team": "BBB",
                           "total_line": 45.0, "spread_line": 7.0}])
    lines = schedule_to_team_lines(sched).set_index("team")
    home = implied_points(pd.Series([45.0]), pd.Series([7.0]),
                          is_home=pd.Series([True])).iloc[0]
    away = implied_points(pd.Series([45.0]), pd.Series([7.0]),
                          is_home=pd.Series([False])).iloc[0]
    assert lines.at["AAA", "implied_pts"] == pytest.approx(home)
    assert lines.at["BBB", "implied_pts"] == pytest.approx(away)


# ---------------------------------------------------------------- lineup
def test_greedy_matches_ilp_on_random_rosters(plan):
    rng = random.Random(20260804)
    positions_pool = ["QB", "RB", "WR", "TE", "K", "DST"]
    for _ in range(300):
        n = rng.randint(1, 14)
        pos = np.array([rng.choice(positions_pool) for _ in range(n)])
        val = np.array([round(rng.uniform(-5, 30), 2) for _ in range(n)])
        mask = greedy_lineup(val, pos, plan)
        assert val[mask].sum() == pytest.approx(ilp_best_lineup(val, pos, plan)), (
            f"positions={list(pos)} values={list(val)}"
        )


def test_greedy_never_starts_a_negative_player(plan):
    pos = np.array(["QB", "RB", "RB", "WR", "WR", "TE", "K", "DST"])
    val = np.array([10.0, 8.0, 6.0, 7.0, 5.0, 4.0, 3.0, -4.0])
    mask = greedy_lineup(val, pos, plan)
    assert not mask[-1], "a negative-value DST was started with an open slot"


def test_greedy_respects_the_3rb_3wr_counterexample(plan):
    pos = np.array(["RB"] * 3 + ["WR"] * 3)
    val = np.array([30.0, 29.0, 28.0, 27.0, 26.0, 25.0])
    mask = greedy_lineup(val, pos, plan)
    assert mask.sum() == 5, "only RB,RB,WR,WR,FLEX can be filled"


def test_lineup_points_selects_and_scores_separately(plan):
    """Non-clairvoyance: selection uses projections, scoring uses draws.

    Ten players for nine slots, so exactly one is benched. The benched player
    is the one that blows up — a clairvoyant lineup would have started him.
    """
    pos = np.array(["QB", "RB", "RB", "RB", "WR", "WR", "TE", "K", "DST", "RB"])
    projection = np.array([20.0, 15.0, 14.0, 11.0, 13.0, 12.0, 10.0, 8.0, 7.0, 1.0])
    realized = np.zeros(10)
    realized[-1] = 99.0
    assert lineup_points(projection, realized, pos, plan) == pytest.approx(0.0)

    # and the clairvoyant version, which the simulator must never use, does
    # capture it — that gap is the bias §15.3 exists to avoid
    assert lineup_points(realized, realized, pos, plan) == pytest.approx(99.0)


def test_both_oracles_agree(plan):
    counter, matching = CounterOracle(plan), MatchingOracle(plan)
    rng = random.Random(7)
    for _ in range(400):
        counter.reset()
        matching.reset()
        for _ in range(12):
            pos = rng.choice(list(plan.positions))
            c_ok, m_ok = counter.can_add(pos), matching.can_add(pos)
            assert c_ok == m_ok, f"disagree on {pos}"
            if c_ok:
                counter.add(pos)
                matching.add(pos)


# ---------------------------------------------------------------- schedule
def test_round_robin_is_a_valid_schedule():
    weeks = round_robin(12, 14)
    assert len(weeks) == 14
    for pairings in weeks:
        teams = [t for pair in pairings for t in pair]
        assert sorted(teams) == list(range(12)), "every team plays exactly once"


def test_round_robin_repeats_after_exhaustion():
    weeks = round_robin(12, 14)
    assert weeks[11] == weeks[0], "week 12 replays week 1 (Yahoo default)"


def _fake_team_week(rng, teams=12, weeks=17, reps=3):
    return rng.normal(100, 15, size=(teams, weeks, reps))


def test_final_ranks_are_a_complete_permutation(league):
    rng = np.random.default_rng(0)
    tw = _fake_team_week(rng)
    regular, _ = regular_season_ranks(tw, league.schedule)
    final = final_ranks(tw, regular, league.schedule)
    for r in range(tw.shape[2]):
        assert sorted(final[:, r]) == list(range(1, 13)), (
            "final_rank must be a complete 1..T permutation so rank: 5 and "
            "rank: -1 are both defined"
        )
        assert sorted(regular[:, r]) == list(range(1, 13))


def test_non_playoff_teams_rank_below_playoff_teams(league):
    rng = np.random.default_rng(1)
    tw = _fake_team_week(rng)
    regular, _ = regular_season_ranks(tw, league.schedule)
    final = final_ranks(tw, regular, league.schedule)
    for r in range(tw.shape[2]):
        made = [t for t in range(12) if regular[t, r] <= league.schedule.playoff_teams]
        missed = [t for t in range(12) if regular[t, r] > league.schedule.playoff_teams]
        assert max(final[t, r] for t in made) < min(final[t, r] for t in missed)


# ---------------------------------------------------------------- payout
def _outcome(rng, teams=12, weeks=17, reps=40, league=None):
    tw = rng.normal(100, 15, size=(teams, weeks, reps))
    regular, _ = regular_season_ranks(tw, league.schedule)
    final = final_ranks(tw, regular, league.schedule)
    return SeasonOutcome(
        team_week=tw, final_rank=final, regular_rank=regular,
        season_points=tw[:, : league.schedule.regular_season_weeks, :].sum(axis=1),
        my_team_index=0,
    )


def test_total_paid_equals_the_pot_in_every_replication(league):
    """The property that actually matters: conservation must hold per draw,
    not merely in expectation."""
    st = load_strategy(league)
    obj = compile_payout(st.payout, league)
    rng = np.random.default_rng(42)
    sim = _outcome(rng, league=league)
    paid = obj(sim).sum(axis=0)
    assert np.allclose(paid, st.payout.pot(league.teams)), (
        f"pot leak: min={paid.min():.2f} max={paid.max():.2f} "
        f"expected={st.payout.pot(league.teams):.2f}"
    )


def test_decompose_sums_to_the_total(league):
    st = load_strategy(league)
    obj = compile_payout(st.payout, league)
    rng = np.random.default_rng(7)
    sim = _outcome(rng, league=league)
    parts = obj.decompose(sim)
    assert set(parts) == set(obj.prize_ids)
    assert np.allclose(sum(parts.values()), obj(sim))


def test_weekly_high_score_pays_once_per_week(league):
    st = load_strategy(league)
    obj = compile_payout(st.payout, league)
    rng = np.random.default_rng(3)
    sim = _outcome(rng, league=league)
    weekly = obj.decompose(sim)["weekly_high"]
    amount, weeks = _weekly_prize(st)
    # one payout per qualifying week, distributed across teams each week
    assert np.allclose(weekly.sum(axis=0), weeks * amount)


def _weekly_prize(st):
    """The live weekly amount and its week count, read off the config.

    Hardcoding "14 x $6" made a payout change (weekly 6 -> 9) read as two
    broken reducers. What these tests assert is that a per-week prize pays
    once per week and that split ties conserve it — neither claim is about
    the dollar figure.
    """
    prize = next(p for p in st.payout.prizes if p.type == "weekly_high_score")
    weeks = prize.weeks.end - prize.weeks.start + 1
    return prize.amount.resolve(st.payout.pot(12)), weeks


def test_split_ties_conserve(league):
    """A deliberate exact tie must still pay exactly the prize amount."""
    st = load_strategy(league)
    obj = compile_payout(st.payout, league)
    tw = np.zeros((12, 17, 1))
    tw[:, 0, 0] = 50.0  # all twelve tie in week 1
    regular, _ = regular_season_ranks(tw, league.schedule)
    sim = SeasonOutcome(
        team_week=tw, final_rank=final_ranks(tw, regular, league.schedule),
        regular_rank=regular, season_points=tw[:, :14, :].sum(axis=1),
        my_team_index=0,
    )
    week1 = obj.decompose(sim)["weekly_high"]
    amount, weeks = _weekly_prize(st)
    assert week1.sum() == pytest.approx(weeks * amount)


def test_my_dollars_selects_my_row(league):
    st = load_strategy(league)
    obj = compile_payout(st.payout, league)
    rng = np.random.default_rng(11)
    sim = _outcome(rng, league=league)
    assert np.allclose(obj.my_dollars(sim), obj(sim)[sim.my_team_index])


def test_needs_bracket_is_false_without_rank_prizes(league):
    """No rank prize -> the kernel skips bracket and schedule entirely."""
    from src.core.config.payout_schema import Amount, PrizeRule, WeekRange
    from src.core.config.strategy import PayoutConfig

    payout = PayoutConfig(
        buy_in=32.0, currency="USD",
        prizes=(PrizeRule(id="w", type="weekly_high_score",
                          amount=Amount(dollars=384 / 14),
                          weeks=WeekRange(1, 14)),),
    )
    assert not compile_payout(payout, league).needs_bracket


def test_registry_matches_core_vocabulary():
    from src.core.config.payout_schema import PRIZE_TYPES
    assert set(REDUCERS) == set(PRIZE_TYPES)


# --------------------------------------------- real-data reconciliation
@pytest.mark.slow
def test_scoring_reconciles_against_nflverse_with_explained_deltas(league):
    """Score 2024 from raw columns and account for EVERY delta vs nflverse.

    This is the §21.1 gate in miniature. nflverse's `fantasy_points_ppr`
    encodes a different league's rules, which is exactly why the design
    forbids using it as a source — but it makes an excellent independent
    cross-check, because every disagreement must be attributable to a named
    scoring difference rather than waved away.

    Measured 2026-08-04: 5,849 rows, 0 unexplained.
    """
    nfl = pytest.importorskip("nflreadpy")
    df = nfl.load_player_stats(seasons=[2024]).to_pandas()
    df = df[(df.season_type == "REG") & df.position.isin(["QB", "RB", "WR", "TE"])]

    ours = score_offense(df, league.scoring.offense)
    theirs = pd.to_numeric(df["fantasy_points_ppr"], errors="coerce").fillna(0.0)

    ints = pd.to_numeric(df["passing_interceptions"], errors="coerce").fillna(0.0)
    frtd = pd.to_numeric(df["fumble_recovery_tds"], errors="coerce").fillna(0.0)

    # our int coefficient is -1; nflverse's PPR column uses the standard -2,
    # so we score +1 per interception relative to it. And we award
    # off_fumble_return_td, which it does not.
    explained = ints * (league.scoring.offense["int"] - (-2.0)) \
        + frtd * league.scoring.offense["off_fumble_return_td"]

    residual = (ours - theirs - explained).abs()
    assert residual.max() < 1e-9, (
        f"{int((residual > 1e-9).sum())} rows disagree with nflverse beyond the "
        f"two known scoring differences; max residual {residual.max():.4f}"
    )


def test_head_to_head_is_configured_but_unreachable():
    """It is accepted and skipped. That is fine — it can only decide a tie
    surviving BOTH wins and points_for, and points_for is a continuous sum of
    drawn scores; measured 0 such ties in 20,000 simulated seasons. What was
    wrong is that the skip was silent, so an auditor reading league.yaml would
    believe a rule was being applied that never runs."""
    from src.core.config import load_league

    league = load_league()
    assert "head_to_head" in league.schedule.seeding_tiebreakers
    src = Path("src/domain/schedule/bracket.py").read_text()
    assert "NOT IMPLEMENTED" in src, "the skip must say so out loud"
    cfg_src = Path("config/league.yaml").read_text()
    assert "NOT implemented" in cfg_src, "and so must the config that lists it"
