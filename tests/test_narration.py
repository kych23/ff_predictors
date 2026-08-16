"""Narration and the entailment gate (§18).

The gate is adversarial by design, so these tests are written as attacks. Each
one is a way a plausible model output could state something the record does not
support; every one of them must be caught, because a gate that passes a
hallucinated number is worse than no gate — it launders the claim as verified.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.app.narration.attribution import AttributionRecord, build_record
from src.app.narration.backends import Clause, response_schema
from src.app.narration.render import (
    NarrationConfig,
    narrate,
    record_payload,
    render_table,
)
from src.app.narration.verify import (
    extract_claims,
    gate,
)

TOL = {"dollars": 0.05, "probability": 0.02, "week": 0.0, "slot": 0.0}


@pytest.fixture
def record():
    return AttributionRecord(
        pair=("p1", "p2"),
        delta_by_prize={"champion": 2.40, "weekly_high": 0.60},
        delta_weeks=[(1, 0.8), (2, -0.3), (3, 1.5)],
        roster_slot_affected="RB2",
        aleatory_se=0.30, epistemic_se=0.40,
        survival_probabilities={"p1": 0.25, "p2": 0.80},
        bye_conflicts=[9],
        names={"p1": "Jahmyr Gibbs", "p2": "Bijan Robinson"},
    )


def _claim(subject, quantity, comparator, value):
    return (f'<claim subject="{subject}" quantity="{quantity}" '
            f'comparator="{comparator}" value="{value}"/>')


# ------------------------------------------------------------- extraction
def test_claims_are_parsed_not_inferred(record):
    text = f"Gibbs is worth {_claim('champion', 'dollars', 'approx', '2.40')}."
    claims = extract_claims(text)
    assert len(claims) == 1
    assert claims[0].subject == "champion"
    assert claims[0].numeric == pytest.approx(2.40)


def test_self_closing_and_open_forms_both_parse():
    assert len(extract_claims(
        '<claim subject="a" quantity="dollars" comparator="eq" value="1"/>'
        '<claim subject="b" quantity="dollars" comparator="eq" value="2">'
    )) == 2


# ------------------------------------------------- untagged quantities
def test_a_bare_numeral_in_prose_is_a_failure(record):
    """Checking tagged claims alone would let 'he scores 18 a game' straight
    through — the exact statement the gate exists to stop."""
    text = f"Gibbs is worth {_claim('champion', 'dollars', 'approx', '2.40')} " \
           f"and scores 18 a game."
    result = gate(text, record, tolerances=TOL)
    assert not result.ok
    assert "18" in " ".join(result.untagged_numerals)


def test_spelled_numbers_are_caught_too(record):
    """A model told not to write digits writes 'twice' instead, and the
    sentence is exactly as unverifiable."""
    text = f"Gibbs is worth {_claim('champion', 'dollars', 'approx', '2.40')}, " \
           f"roughly twice the alternative."
    result = gate(text, record, tolerances=TOL)
    assert not result.ok
    assert any("twice" in u for u in result.untagged_numerals)


def test_prose_with_no_quantities_is_fine(record):
    text = (f"Gibbs separates on the championship prize, worth "
            f"{_claim('champion', 'dollars', 'approx', '2.40')} more.")
    assert gate(text, record, tolerances=TOL).ok


def test_percent_signs_and_money_are_caught(record):
    for bad in ("a 25% chance", "worth $3 more"):
        result = gate(f"Something. {bad}.", record, tolerances=TOL)
        assert not result.ok, bad


# ------------------------------------------------------------ entailment
def test_a_correct_dollar_claim_passes(record):
    assert gate(_claim("champion", "dollars", "approx", "2.40"), record,
                tolerances=TOL).ok


def test_a_wrong_dollar_claim_fails(record):
    result = gate(_claim("champion", "dollars", "approx", "9.99"), record,
                  tolerances=TOL)
    assert not result.ok
    assert "not entailed" in result.failures[0]


def test_tolerance_is_respected_on_both_sides(record):
    assert gate(_claim("champion", "dollars", "approx", "2.44"), record,
                tolerances=TOL).ok
    assert not gate(_claim("champion", "dollars", "approx", "2.50"), record,
                    tolerances=TOL).ok


def test_an_invented_subject_is_caught(record):
    """The most dangerous hallucination: a confident sentence about a prize
    that does not exist."""
    result = gate(_claim("consolation_bracket", "dollars", "approx", "2.40"),
                  record, tolerances=TOL)
    assert not result.ok
    assert "invented a subject" in result.failures[0]


def test_directional_claims_must_clear_the_tolerance(record):
    """'greater than 2.40' when the actual is 2.40 is technically arguable and
    rhetorically false."""
    assert gate(_claim("champion", "dollars", "gt", "1.00"), record,
                tolerances=TOL).ok
    assert not gate(_claim("champion", "dollars", "gt", "2.40"), record,
                    tolerances=TOL).ok
    assert gate(_claim("champion", "dollars", "lt", "5.00"), record,
                tolerances=TOL).ok


def test_probabilities_resolve_by_display_name(record):
    assert gate(_claim("Jahmyr Gibbs", "probability", "approx", "0.25"),
                record, tolerances=TOL).ok
    assert gate(_claim("p1", "probability", "approx", "0.25"), record,
                tolerances=TOL).ok


def test_week_claims_must_be_weeks_the_record_mentions(record):
    assert gate(_claim("delta", "week", "eq", "3"), record, tolerances=TOL).ok
    assert gate(_claim("bye", "week", "eq", "9"), record, tolerances=TOL).ok
    assert not gate(_claim("delta", "week", "eq", "14"), record,
                    tolerances=TOL).ok


def test_slot_claims_are_exact_strings(record):
    assert gate(_claim("slot", "slot", "eq", "RB2"), record, tolerances=TOL).ok
    assert not gate(_claim("slot", "slot", "eq", "RB1"), record,
                    tolerances=TOL).ok


def test_an_unknown_quantity_kind_cannot_be_verified(record):
    """There is no entailment rule for it, so it must fail rather than pass by
    default — silently accepting the unknown is how a gate becomes theatre."""
    result = gate(_claim("champion", "vibes", "approx", "2.40"), record,
                  tolerances=TOL)
    assert not result.ok
    assert "unknown quantity" in result.failures[0]


def test_a_non_numeric_value_fails(record):
    assert not gate(_claim("champion", "dollars", "approx", "a lot"), record,
                    tolerances=TOL).ok


def test_standard_errors_are_addressable(record):
    assert gate(_claim("total_se", "dollars", "approx", "0.50"), record,
                tolerances=TOL).ok
    assert gate(_claim("total", "dollars", "approx", "3.00"), record,
                tolerances=TOL).ok


# -------------------------------------------------------------- fallback
def test_the_table_needs_no_model_and_states_the_record(record):
    text = render_table(record)
    assert "Jahmyr Gibbs" in text and "Bijan Robinson" in text
    assert "champion" in text and "RB2" in text
    assert "+3.00" in text


class _FakeBackend:
    """A backend returning fixed clauses.

    Tests pass this EXPLICITLY rather than letting `resolve_backend` run:
    otherwise the suite's result depends on whether an Ollama daemon happens to
    be running on the machine, which is how a green CI and a red laptop get to
    disagree.
    """

    name = "fake"

    def __init__(self, clauses, error=None):
        self._clauses, self._error = clauses, error
        self.calls = 0

    def available(self) -> bool:
        return True

    def clauses(self, record, *, model, max_tokens):
        self.calls += 1
        if self._error:
            raise self._error
        return self._clauses


def test_a_passing_narration_is_returned_as_prose(record):
    backend = _FakeBackend([
        Clause("Gibbs leads on the title prize by", "champion", "dollars",
               "approx")])
    out = narrate(record, NarrationConfig(), backend=backend)
    assert out.source == "fake"
    assert out.verified
    assert "<claim" not in out.text
    assert "2.40" in out.text
    assert "Gibbs leads on the title prize" in out.text


def test_the_model_never_supplies_the_number(record):
    """THE property that makes a small local model safe here. The clause
    carries a subject, not a value, so the figure printed is read off the
    record and cannot be wrong however badly the model behaves."""
    backend = _FakeBackend([
        Clause("the gap on the title", "champion", "dollars", "approx")])
    out = narrate(record, NarrationConfig(), backend=backend)
    assert "2.40" in out.text
    assert out.verified


def test_a_clause_naming_a_nonexistent_subject_is_dropped(record):
    """Schema-constrained decoding should prevent it. If one slips through, a
    sentence about a fact that does not exist is worse than a missing one."""
    backend = _FakeBackend([
        Clause("the consolation bracket favours him", "consolation", "dollars",
               "approx"),
        Clause("and the title prize does too", "champion", "dollars",
               "approx"),
    ])
    out = narrate(record, NarrationConfig(), backend=backend)
    assert "consolation" not in out.text
    assert "title prize" in out.text
    assert out.verified


def test_directional_clauses_clear_the_tolerance(record):
    """`_entails` rejects `gt X` when the actual only equals X. The assembler
    has to anchor the threshold strictly inside the true value or every
    directional clause the model writes would fail its own gate."""
    for comparator in ("gt", "lt"):
        out = narrate(record, NarrationConfig(),
                      backend=_FakeBackend([
                          Clause("the title gap is", "champion", "dollars",
                                 comparator)]))
        assert out.verified, f"{comparator}: {out.reason}"


def test_a_clause_containing_a_bare_numeral_still_fails_the_gate(record):
    """Defence in depth. The schema tells the model not to write numbers; the
    gate is what makes it true."""
    backend = _FakeBackend([
        Clause("he is worth 18 points a game and leads by", "champion",
               "dollars", "approx")])
    out = narrate(record, NarrationConfig(), backend=backend)
    assert out.source == "table"
    assert not out.gate_result.ok


def test_a_backend_failure_never_raises(record):
    out = narrate(record, NarrationConfig(),
                  backend=_FakeBackend([], error=RuntimeError("connection refused")))
    assert out.source == "table"
    assert "connection refused" in out.reason


def test_empty_clauses_fall_back_rather_than_printing_nothing(record):
    out = narrate(record, NarrationConfig(), backend=_FakeBackend([]))
    assert out.source == "table"
    assert "no usable clauses" in out.reason


def test_disabled_narration_skips_the_call(record):
    backend = _FakeBackend([Clause("x", "champion", "dollars", "approx")])
    out = narrate(record, NarrationConfig(enabled=False), backend=backend)
    assert out.source == "table" and backend.calls == 0


def test_strict_mode_raises_instead_of_falling_back(record):
    backend = _FakeBackend([
        Clause("he scores 18 a game", "champion", "dollars", "approx")])
    cfg = NarrationConfig(on_gate_fail="raise")
    with pytest.raises(ValueError, match="entailment gate"):
        narrate(record, cfg, backend=backend)


def test_no_backend_available_says_how_to_get_one(record):
    out = narrate(record, NarrationConfig(backend="table"))
    assert out.source == "table"
    assert "ollama" in out.reason.lower()


# ------------------------------------------------------------- §19 offline
def test_local_and_table_are_offline_hosted_is_not(record):
    """§19: the recommendation path should not need the network. Which backend
    produced a line is the difference between honouring that and not."""
    local = narrate(record, NarrationConfig(),
                    backend=_FakeBackend([Clause("x", "champion", "dollars",
                                                 "approx")]))
    object.__setattr__(local, "source", "ollama")
    assert local.offline
    object.__setattr__(local, "source", "anthropic")
    assert not local.offline
    object.__setattr__(local, "source", "table")
    assert local.offline


def test_the_schema_enumerates_only_real_subjects(record):
    """Constrained decoding is what turns 'do not invent a prize' from a prompt
    request into an impossibility."""
    schema = response_schema(record)
    enum = schema["properties"]["clauses"]["items"]["properties"]["subject"]["enum"]
    assert "champion" in enum and "weekly_high" in enum
    assert "consolation_bracket" not in enum
    for subject in enum:
        assert (record.dollars_for(subject) is not None
                or record.probability_for(subject) is not None)


def test_the_local_backend_only_talks_to_loopback():
    """A configurable host would quietly reintroduce the network dependency
    §19 exists to remove, and nobody would notice until draft night."""
    from src.app.narration.backends import OLLAMA_URL
    assert OLLAMA_URL.startswith("http://127.0.0.1")


def test_backend_preference_puts_local_first():
    import inspect

    from src.app.narration.backends import resolve_backend
    source = inspect.getsource(resolve_backend)
    assert source.index("ollama") < source.index("anthropic_backend")


def test_the_payload_contains_only_what_the_gate_can_check(record):
    """If the model is shown a number the record cannot resolve, it will use
    it and the gate will reject the whole response."""
    payload = record_payload(record)
    assert "champion" in payload and "RB2" in payload
    assert "delta_weeks" not in payload      # only the top ones are exposed
    import json
    parsed = json.loads(payload)
    for prize in parsed["delta_by_prize"]:
        assert record.dollars_for(prize) is not None
    for player in parsed["survival_probabilities"]:
        assert record.probability_for(player) is not None


def test_config_reads_the_shipped_strategy_file():
    from src.core.config import load_league, load_strategy
    cfg = NarrationConfig.from_strategy(load_strategy(load_league()))
    assert cfg.model
    assert set(cfg.tolerances) >= {"dollars", "probability"}


# ----------------------------------------------------------- attribution
def test_build_record_subtracts_the_two_decompositions():
    seat, reps, weeks, teams = 3, 50, 4, 12
    leader_parts = {"champion": np.full((teams, reps), 10.0)}
    runner_parts = {"champion": np.full((teams, reps), 6.0)}
    lead_tw = np.full((teams, weeks, reps), 110.0)
    run_tw = np.full((teams, weeks, reps), 100.0)

    rec = build_record("a", "b", leader_parts=leader_parts,
                       runner_parts=runner_parts, seat=seat,
                       leader_team_week=lead_tw, runner_team_week=run_tw,
                       aleatory_se=0.3, epistemic_se=0.4)
    assert rec.delta_by_prize["champion"] == pytest.approx(4.0)
    assert rec.total_se == pytest.approx(0.5)
    assert all(v == pytest.approx(10.0) for _, v in rec.delta_weeks)


def test_a_prize_present_on_only_one_side_still_appears():
    rec = build_record("a", "b",
                       leader_parts={"weekly_high": np.full((2, 5), 3.0)},
                       runner_parts={}, seat=0,
                       leader_team_week=np.zeros((2, 3, 5)),
                       runner_team_week=np.zeros((2, 3, 5)),
                       aleatory_se=0.0, epistemic_se=0.0)
    assert rec.delta_by_prize["weekly_high"] == pytest.approx(3.0)


def test_top_weeks_ranks_by_magnitude_not_sign(record):
    assert record.top_weeks(2) == [(3, 1.5), (1, 0.8)]


# ------------------------------------------- the local-backend contract
def test_the_prompt_shows_the_model_no_numbers(record):
    """THE property that made a local model viable, and the one most likely to
    be undone by a well-meaning edit.

    Shown `champion: 2.40`, a 7B model writes "leads by $2.40" into its prose,
    the gate sees an untagged numeral, and the narration is discarded.
    Measured pass rate with a numeric payload: 0/12. With a qualitative one:
    29/30. If someone adds the values back "so the model has more context",
    narration silently stops appearing and the table shows instead — a
    regression with no error message anywhere.
    """
    from src.app.narration.backends import prompt_payload

    payload = prompt_payload(record)
    # The slot label ("RB2") is a category the gate exempts by exact equality,
    # so it is the one string here allowed to carry a digit.
    scrubbed = payload.replace(record.roster_slot_affected, "")
    digits = [ch for ch in scrubbed if ch.isdigit()]
    assert not digits, (
        f"the prompt leaked digits {set(digits)} — every value must reach the "
        f"model as direction and magnitude, never as a number"
    )
    # it still has to say something useful
    assert "champion" in payload
    assert "leader ahead" in payload


def test_the_qualitative_brief_preserves_direction_and_ordering(record):
    """Removing the numbers must not remove the meaning: the model still needs
    to know who is ahead and which prize matters most."""
    import json

    from src.app.narration.backends import prompt_payload

    parsed = json.loads(prompt_payload(record))
    prizes = [p["subject"] for p in parsed["prizes_by_importance"]]
    assert prizes[0] == "champion", "largest |delta| must lead"
    directions = {p["subject"]: p["direction"]
                  for p in parsed["prizes_by_importance"]}
    assert directions["champion"] == "leader ahead"
    assert parsed["overall"]["direction"] == "leader ahead"


def test_a_negative_prize_reads_as_leader_behind():
    """A split — ahead on one prize, behind on another — is the case worth
    narrating. Collapsing the sign would make the sentence wrong."""
    import json

    from src.app.narration.attribution import AttributionRecord
    from src.app.narration.backends import prompt_payload

    split = AttributionRecord(
        pair=("a", "b"), delta_by_prize={"champion": 3.0, "weekly_high": -1.0},
        delta_weeks=[(1, 0.5)], roster_slot_affected="RB",
        aleatory_se=0.2, epistemic_se=0.2)
    parsed = json.loads(prompt_payload(split))
    directions = {p["subject"]: p["direction"]
                  for p in parsed["prizes_by_importance"]}
    assert directions["champion"] == "leader ahead"
    assert directions["weekly_high"] == "leader behind"


def test_quantity_is_derived_from_the_subject_not_taken_from_the_model(record):
    """Exposing subject and quantity as independent enums makes invalid pairs
    expressible, and the model emits them: `champion` with `probability`
    appeared on 5 of 30 records, every one dropped as unresolvable. The schema
    therefore has no `quantity` field at all."""
    from src.app.narration.backends import quantity_of, response_schema

    properties = response_schema(record)["properties"]["clauses"]["items"]["properties"]
    assert "quantity" not in properties
    assert quantity_of(record, "champion") == "dollars"
    assert quantity_of(record, "Jahmyr Gibbs") == "probability"
    assert quantity_of(record, "not_a_subject") is None


def test_directional_margins_exceed_the_gate_tolerance(record):
    """`gt` requires `actual > claimed + tolerance`, so a purely proportional
    margin fails on small gaps — 10% of $0.34 is under the $0.05 dollars
    tolerance, and every directional clause on a near-tie failed its own gate
    that way."""
    from src.app.narration.render import assemble

    tiny = AttributionRecord(
        pair=("a", "b"), delta_by_prize={"champion": 0.34},
        delta_weeks=[(1, 0.1)], roster_slot_affected="RB",
        aleatory_se=0.1, epistemic_se=0.1)
    text = assemble([Clause("he leads", "champion", "dollars", "gt")], tiny,
                    TOL)
    assert gate(text, tiny, tolerances=TOL).ok


# ------------------------------------------------- football facts (§18)
def _facts_record():
    return AttributionRecord(
        pair=("p1", "p2"),
        delta_by_prize={"champion": 2.0, "weekly_high": 0.4},
        delta_weeks=[(1, 0.5)], roster_slot_affected="RB",
        aleatory_se=1.5, epistemic_se=1.2,
        survival_probabilities={"p1": 0.2},
        names={"p1": "Breece Hall", "p2": "Sam LaPorta"},
        player_facts={"p1": {"games": 15.234, "consistency": 5.1, "adp": 12.0},
                      "p2": {"games": 13.4}})


def test_a_football_fact_resolves_to_its_number():
    rec = _facts_record()
    assert rec.fact_for("Breece Hall games") == pytest.approx(15.234)
    assert rec.fact_for("Sam LaPorta games") == pytest.approx(13.4)
    assert rec.fact_for("Breece Hall targets") is None


def test_facts_become_selectable_subjects():
    """The model can only cite what the enum offers — a prompt line alone
    cannot make it talk football."""
    from src.app.narration.backends import allowed_subjects

    subjects = allowed_subjects(_facts_record())
    assert "Breece Hall games" in subjects["player_fact"]
    assert "Breece Hall consistency" in subjects["player_fact"]


@pytest.mark.parametrize("comparator", ["approx", "gt", "lt"])
def test_a_fact_claim_passes_its_own_gate(comparator):
    from src.app.narration.backends import Clause
    from src.app.narration.render import NarrationConfig, assemble
    from src.app.narration.verify import gate

    rec, cfg = _facts_record(), NarrationConfig()
    tagged = assemble([Clause(text="Breece Hall stays on the field",
                              subject="Breece Hall games",
                              quantity="player_fact", comparator=comparator)],
                      rec, cfg.tolerances)
    assert gate(tagged, rec, tolerances=cfg.tolerances).ok


def test_facts_render_in_their_own_units_not_dollars():
    """A durability claim printed as dollars is the right subject wearing the
    wrong unit — observed: 'steadier week to week less than $0.82'."""
    from src.app.narration.backends import Clause
    from src.app.narration.render import NarrationConfig, _readable, assemble

    rec, cfg = _facts_record(), NarrationConfig()
    # The prose must MENTION its own fact — coherence now requires it, so a
    # placeholder like "x" is correctly rejected before it can be rendered.
    for subject, unit, prose in (
            ("Breece Hall games", "games", "Hall is on the field more"),
            ("Breece Hall consistency", "pts/wk swing",
             "Hall is steadier week to week"),
            ("Breece Hall adp", "ADP", "Hall goes earlier by ADP")):
        text = _readable(assemble(
            [Clause(text=prose, subject=subject, quantity="player_fact",
                    comparator="approx")], rec, cfg.tolerances))
        assert unit in text and "$" not in text


def test_player_fact_is_a_recognised_quantity():
    """The gate refuses any quantity without an entailment rule, so adding the
    subject without adding the kind silently rejected every fact claim."""
    from src.app.narration.attribution import QUANTITIES

    assert "player_fact" in QUANTITIES


def test_partial_tolerances_do_not_wipe_the_defaults():
    """A config naming only `dollars` used to zero every other tolerance, and a
    zero tolerance turns `approx` into exact float equality — so the narration
    failed the gate on rounding alone."""
    from types import SimpleNamespace

    from src.app.narration.render import NarrationConfig

    strategy = SimpleNamespace(simulation=SimpleNamespace(
        narration={"tolerances": {"dollars": 0.5}}))
    tol = NarrationConfig.from_strategy(strategy).tolerances
    assert tol["dollars"] == 0.5
    assert tol["player_fact"] == pytest.approx(0.05)
    assert tol["probability"] == pytest.approx(0.02)


def test_football_guidance_is_absent_when_there_are_no_facts():
    """Without facts the enum offers only money, and pushing a football framing
    produces football words bolted to a dollar figure — measured live as
    'both are durable $2.24'."""
    from src.app.narration.backends import system_prompt

    bare = AttributionRecord(
        pair=("p1", "p2"), delta_by_prize={"champion": 2.0},
        delta_weeks=[(1, 0.5)], roster_slot_affected="RB",
        aleatory_se=1.0, epistemic_se=1.0, names={"p1": "A", "p2": "B"})
    assert "durability" not in system_prompt(bare)
    assert "durability" in system_prompt(_facts_record())


def test_unverifiable_context_is_not_offered():
    """Team changes, target share and depth-chart moves are absent BY DESIGN:
    the bundle has a current team and no history, so the gate could never
    check them.

    Asserted as a PROPERTY, not a frozen list. This used to pin the set to
    exactly {games, consistency, adp}, which made adding `ceiling` and `floor`
    — both read straight off the calibrated band, both checkable — look like a
    violation of a rule that is actually about verifiability.
    """
    from src.app.narration.attribution import PLAYER_FACTS
    from src.app.narration.verify import FACT_VOCABULARY

    forbidden = {"team_change", "new_team", "target_share", "depth_chart",
                 "holdout", "coaching", "snap_share", "trade"}
    assert not (set(PLAYER_FACTS) & forbidden)

    # Every offered fact must have a coherence rule, or the gate cannot tell
    # whether a sentence about it matches its own number.
    assert set(PLAYER_FACTS) <= set(FACT_VOCABULARY), (
        f"no coherence vocabulary for "
        f"{set(PLAYER_FACTS) - set(FACT_VOCABULARY)}")


def test_every_offered_fact_is_produced_by_the_cockpit():
    """A fact in the vocabulary that nothing populates is a subject the model
    may name and the gate will then reject — the `survival_probabilities`
    failure in a different costume."""
    import inspect

    from src.app.cockpit.build import _player_facts
    from src.app.narration.attribution import PLAYER_FACTS

    source = inspect.getsource(_player_facts)
    for fact in PLAYER_FACTS:
        assert f'facts["{fact}"]' in source, f"{fact} is never populated"


# ------------------------------------------ text/subject coherence (§18)
def _coh(text, subject, quantity):
    from src.app.narration.verify import Claim, check_coherence

    return check_coherence(text, Claim(subject=subject, quantity=quantity,
                                       comparator="approx", value="0"))


def test_a_football_sentence_may_not_carry_a_dollar_figure():
    """Observed live: 'both are durable with expected games played less than
    $2.20'. Every numeric check passed — $2.20 really was a prize delta — because
    nothing compared the PROSE to the subject."""
    assert _coh("both are durable with expected games played",
                "champion", "dollars") is not None


def test_a_durability_sentence_must_carry_a_games_number():
    assert _coh("Breece Hall stays healthy", "Breece Hall games",
                "player_fact") is None
    assert _coh("Breece Hall stays healthy", "Breece Hall consistency",
                "player_fact") is not None


def test_a_consistency_sentence_must_carry_a_consistency_number():
    assert _coh("steadier week to week", "Breece Hall consistency",
                "player_fact") is None
    assert _coh("steadier week to week", "Breece Hall games",
                "player_fact") is not None


@pytest.mark.parametrize("word", ["both", "neither", "each", "either"])
def test_a_collective_word_cannot_carry_one_players_fact(word):
    """The inserted number is one player's, so 'both are durable' asserts
    something about a player it never measured."""
    assert _coh(f"{word} are durable", "Breece Hall games",
                "player_fact") is not None


def test_a_collective_word_is_fine_on_a_prize_comparison():
    """A prize delta IS a comparison of the two, so 'both' is correct there."""
    assert _coh("both are close on the title", "champion", "dollars") is None


def test_an_incoherent_clause_is_dropped_not_rendered():
    """Drop the clause, keep the rest — failing the whole narration to the
    table costs the good sentences too."""
    from src.app.narration.backends import Clause
    from src.app.narration.render import NarrationConfig, assemble

    rec, cfg = _facts_record(), NarrationConfig()
    text = assemble([
        Clause(text="both are durable", subject="champion",
               quantity="dollars", comparator="approx"),
        Clause(text="Breece Hall edges the title", subject="champion",
               quantity="dollars", comparator="approx"),
    ], rec, cfg.tolerances)
    assert "durable" not in text
    assert "edges the title" in text


def test_a_player_fact_renders_the_actual_value_not_the_claimed_bound():
    """A one-player number beside a comparative sentence reads as a
    contradiction when shown as a bound.

    Observed live: "Nabers is on the field MORE weeks than Bowers (UNDER 13.3
    games)". Both halves were true — the gate had verified the real 13.2
    against the claimed bound — but "more" next to "under" is nonsense to
    read. The record's own figure cannot disagree with the sentence, and it
    says more than a bound does.
    """
    from src.app.narration.backends import Clause
    from src.app.narration.render import NarrationConfig, _readable, assemble

    rec, cfg = _facts_record(), NarrationConfig()
    actual = rec.fact_for("Breece Hall games")
    assert actual is not None

    text = _readable(assemble(
        [Clause(text="Hall is on the field more weeks", quantity="player_fact",
                subject="Breece Hall games", comparator="lt")],
        rec, cfg.tolerances), rec)

    assert f"{actual:.1f} games" in text
    assert "under" not in text.lower(), "a bound word contradicts 'more'"


def test_dollars_and_probabilities_still_show_their_comparator():
    """Only per-player facts switch to the actual value. A dollar delta is a
    genuine estimate with a direction worth stating."""
    from src.app.narration.backends import Clause
    from src.app.narration.render import NarrationConfig, _readable, assemble

    rec, cfg = _facts_record(), NarrationConfig()
    text = _readable(assemble(
        [Clause(text="Hall leads overall", quantity="dollars",
                subject="total", comparator="gt")], rec, cfg.tolerances), rec)
    # `assemble` states a BOUND for dollars (2.40 -> "over $2.16"), which is
    # the honest form for an estimate carrying simulation error. Only
    # per-player facts switch to the exact figure.
    assert "$" in text
    assert "over" in text.lower()


def test_rendering_without_a_record_still_works():
    """`_readable` is called with the record everywhere in production, but it
    must not become the only way to format text."""
    from src.app.narration.backends import Clause
    from src.app.narration.render import NarrationConfig, _readable, assemble

    rec, cfg = _facts_record(), NarrationConfig()
    text = _readable(assemble(
        [Clause(text="Hall is on the field more", quantity="player_fact",
                subject="Breece Hall games", comparator="gt")],
        rec, cfg.tolerances))
    assert "games" in text


# ------------------------------------------ survival direction (§18)
def _prob_claim(value):
    from src.app.narration.verify import Claim
    return Claim(subject="Nacua", quantity="probability",
                 comparator="approx", value=str(value))


@pytest.mark.parametrize("phrase", [
    "Nacua will not be there when you pick again",
    "Nacua won't be there when you pick again, so take him now",
    "Nacua will be gone by your next turn",
])
def test_claiming_a_player_is_gone_at_high_survival_is_rejected(phrase):
    """The failure both models produced: "Nacua will not be there when you
    pick again (83%)". The number was right and the sentence said the
    opposite, and nothing checked that they agreed."""
    from src.app.narration.verify import check_coherence

    assert check_coherence(phrase, _prob_claim(0.83)) is not None


@pytest.mark.parametrize("phrase", [
    "Nacua will still be there when you pick again",
    "Nacua is likely to last, so you can wait",
])
def test_claiming_a_player_lasts_at_low_survival_is_rejected(phrase):
    from src.app.narration.verify import check_coherence

    assert check_coherence(phrase, _prob_claim(0.06)) is not None


def test_a_correct_survival_claim_passes():
    """The rule must not simply reject every survival sentence."""
    from src.app.narration.verify import check_coherence

    gone = "Nacua will not be there when you pick again, so take him now"
    assert check_coherence(gone, _prob_claim(0.11)) is None

    lasts = "Nacua will still be there when you pick again"
    assert check_coherence(lasts, _prob_claim(0.88)) is None


def test_a_survival_clause_with_no_directional_language_is_left_alone():
    """Only the direction is checked. A clause that states the probability
    without asserting a direction has nothing to contradict."""
    from src.app.narration.verify import check_coherence

    # No fact markers at all — "availability" would trip the `games`
    # vocabulary, which is the mismatch rule doing its job, not this one.
    neutral = "Nacua is the name to weigh here"
    assert check_coherence(neutral, _prob_claim(0.83)) is None
