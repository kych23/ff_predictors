"""Rendering (§18) — one backend call, gated, with a deterministic fallback.

Backend order is LOCAL first (§19: the recommendation path should not need
the network), then hosted, then the table. `backends.py` owns the prompt and
the schema; this module owns assembly, the gate and the fallback.

The fallback is not a degraded mode. `render_table` needs no network, no key
and no model, and it says everything the record contains; the LLM's only job is
to make that same content read like a sentence. If narration is unavailable or
its output fails the gate, nothing about the recommendation changes — which is
why `on_gate_fail: render_table` is safe to make automatic.

`narrate` never raises on a model or network failure. Draft night has a clock,
and a missing paragraph must not cost a pick.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

from src.app.narration.attribution import AttributionRecord
from src.app.narration.backends import (
    resolve_backend,
)
from src.app.narration.verify import (
    Claim,
    GateResult,
    check_coherence,
    gate,
    strip_claims,
)

logger = logging.getLogger(__name__)

#: Units for the football facts, so a durability claim does not print dollars.
#: How each fact is spelled next to its number. `consistency` is a weekly
#: standard deviation, and labelling it " pts/wk" made praise read as an
#: insult: "Flowers' steadiness means you can start him without checking
#: (under 9.5 pts/wk)" parses as "he scores under 9.5 a week". Naming the
#: quantity as a SWING fixes the sentence without touching the model.
_FACT_UNITS = {
    "games": " games",
    "consistency": " pts/wk swing",
    "adp": " ADP",
    "ceiling": " pts/g upside",
    "floor": " pts/g floor",
}



@dataclass(frozen=True)
class NarrationConfig:
    enabled: bool = True
    #: Model id for whichever backend resolves. Local ids look like
    #: "qwen2.5:14b"; hosted ones like "claude-opus-5".
    #:
    #: 14b, not 7b. Under the coherence gate the 7b benchmarks at 75% and is
    #: reported NOT usable — the table would show most of the time. The 14b
    #: passes 85% and writes three distinct clauses to the 7b's 1.6. It costs
    #: ~11 s against ~4 s, which is free: narration is its own SSE event and
    #: never sits on the pick clock.
    model: str = "qwen2.5:14b"
    max_tokens: int = 400
    gate_on_entailment: bool = True
    on_gate_fail: str = "render_table"
    #: "auto" prefers the LOCAL backend, then hosted, then the table — because
    #: §19 wants the recommendation path offline, not because local is better.
    backend: str = "auto"
    tolerances: Mapping[str, float] = field(
        # `player_fact` needs a tolerance for the same reason `dollars` does:
        # the claim is written at two decimals while the record holds full
        # precision, so a zero tolerance makes every `approx` fact claim fail
        # exact-equality and the narration falls back to the table. Measured:
        # 7 of 20 records rejected on nothing but rounding.
        default_factory=lambda: {"dollars": 0.05, "probability": 0.02,
                                 "player_fact": 0.05,
                                 "week": 0.0, "slot": 0.0})

    @classmethod
    def from_strategy(cls, strategy) -> NarrationConfig:
        raw = dict(getattr(strategy.simulation, "narration", {}) or {})
        return cls(
            enabled=bool(raw.get("enabled", True)),
            model=str(raw.get("model", cls.model)),
            max_tokens=int(raw.get("max_tokens", 400)),
            gate_on_entailment=bool(raw.get("gate_on_entailment", True)),
            on_gate_fail=str(raw.get("on_gate_fail", "render_table")),
            backend=str(raw.get("backend", "auto")),
            # MERGED over the defaults, not substituted for them. A config that
            # names only `dollars` used to silently zero every other tolerance,
            # and a zero tolerance turns `approx` into exact float equality —
            # so the whole narration fails the gate on rounding alone. It also
            # means a new quantity works without editing strategy.yaml, which
            # matters because that file is hashed into every RNG seed.
            tolerances={**cls().tolerances, **dict(raw.get("tolerances", {}))},
        )


@dataclass(frozen=True)
class Narration:
    text: str
    source: str              # "ollama" | "anthropic" | "table"
    gate_result: GateResult | None = None
    reason: str = ""
    model: str = ""
    latency_ms: float = 0.0

    @property
    def verified(self) -> bool:
        return self.source == "table" or (self.gate_result is not None
                                          and self.gate_result.ok)

    @property
    def offline(self) -> bool:
        """Whether producing this text touched the network (§19)."""
        return self.source in ("table", "ollama")


def assemble(clauses, record: AttributionRecord,
             tolerances: Mapping[str, float] | None = None) -> str:
    """Turn model clauses into gated text, filling every value ourselves.

    **The model supplies wording; the record supplies arithmetic.** A clause
    carries a subject and a comparator, never a number, so the value written
    into the claim span is read straight off the record and cannot be wrong.
    This is what makes a 7B model safe on user-facing prose here: the usual fix
    for hallucinated figures is a bigger model, which makes them rarer; taking
    the number out of the model's output space makes them impossible.

    Clauses whose subject does not resolve are DROPPED rather than rendered
    with a placeholder — schema-constrained decoding should prevent it, and a
    sentence about a fact that does not exist is worse than a missing sentence.
    """
    parts = []
    for clause in clauses:
        if clause.quantity == "dollars":
            actual = record.dollars_for(clause.subject)
        elif clause.quantity == "player_fact":
            actual = record.fact_for(clause.subject)
        else:
            actual = record.probability_for(clause.subject)
        if actual is None:
            logger.warning("dropping clause with unresolvable subject %r",
                           clause.subject)
            continue
        text = clause.text.strip().rstrip(".,;: ")
        if not text:
            continue
        # Text and subject are chosen independently, so the model can write a
        # football sentence against a dollar figure and every numeric check
        # still passes. Drop the clause rather than failing the whole
        # narration — the other clauses are usually fine.
        incoherent = check_coherence(
            text, Claim(subject=clause.subject, quantity=clause.quantity,
                        comparator=clause.comparator, value="0"))
        if incoherent:
            logger.warning("dropping incoherent clause: %s", incoherent)
            continue
        # A directional claim needs a threshold the record actually clears, or
        # `verify._entails` rejects it: `gt` requires `actual > claimed + tol`.
        # The margin must therefore EXCEED the gate's own tolerance — a
        # proportional margin alone fails on small gaps, where 10% of $0.34 is
        # under the $0.05 dollars tolerance. Measured: every directional clause
        # on a near-tie record failed its own gate that way.
        if clause.comparator in ("gt", "lt"):
            tol = float((tolerances or {}).get(clause.quantity, 0.0))
            margin = max(abs(actual) * 0.10, 2 * tol, 0.01)
            value = actual - margin if clause.comparator == "gt" else actual + margin
        else:
            value = actual
        digits = 2 if clause.quantity in ("dollars", "player_fact") else 3
        parts.append(
            f'{text} <claim subject="{clause.subject}" '
            f'quantity="{clause.quantity}" comparator="{clause.comparator}" '
            f'value="{value:.{digits}f}"/>.'
        )
    return " ".join(parts)


def record_payload(record: AttributionRecord) -> str:
    """Exactly what the model is allowed to know. Nothing else is in scope."""
    return json.dumps({
        "leader": record.display(record.pair[0]),
        "runner_up": record.display(record.pair[1]),
        "delta_by_prize": {k: round(v, 4)
                           for k, v in record.delta_by_prize.items()},
        "total_dollars": round(record.total_delta, 4),
        "aleatory_se": round(record.aleatory_se, 4),
        "epistemic_se": round(record.epistemic_se, 4),
        "total_se": round(record.total_se, 4),
        "biggest_weeks": [[w, round(v, 4)] for w, v in record.top_weeks()],
        "roster_slot_affected": record.roster_slot_affected,
        "survival_probabilities": {record.display(k): round(v, 4)
                                   for k, v in
                                   record.survival_probabilities.items()},
        "bye_conflicts": record.bye_conflicts,
    }, indent=2)


def render_table(record: AttributionRecord) -> str:
    """Deterministic. No network, no key, no model — and no unverified claim."""
    leader, runner = (record.display(c) for c in record.pair)
    lines = [f"{leader} over {runner}",
             f"  total          {record.total_delta:+7.2f}  "
             f"+/- {record.total_se:.2f}"]
    for prize, value in sorted(record.delta_by_prize.items(),
                               key=lambda kv: -abs(kv[1])):
        lines.append(f"  {prize:14s} {value:+7.2f}")
    if record.roster_slot_affected:
        lines.append(f"  slot affected  {record.roster_slot_affected}")
    top = record.top_weeks()
    if top:
        lines.append("  biggest weeks  "
                     + ", ".join(f"w{w} {v:+.2f}" for w, v in top))
    if record.bye_conflicts:
        lines.append("  bye conflicts  "
                     + ", ".join(f"w{w}" for w in record.bye_conflicts))
    if record.survival_probabilities:
        lines.append("  survival       " + ", ".join(
            f"{record.display(p)} {v:.0%}"
            for p, v in sorted(record.survival_probabilities.items(),
                               key=lambda kv: -kv[1])[:4]))
    return "\n".join(lines)


def narrate(record: AttributionRecord, cfg: NarrationConfig, *,
            client=None, backend=None) -> Narration:
    """One model call, gated. Falls back to the table on ANY problem.

    Backend order is local, then hosted, then table (§19 — the recommendation
    path should not need the network). The gate runs identically whichever
    produced the text; a local model gets no benefit of the doubt.
    """
    if not cfg.enabled:
        return Narration(render_table(record), "table", reason="disabled")

    if backend is None:
        backend = resolve_backend(cfg.backend, client=client)
    if backend is None:
        return Narration(
            render_table(record), "table",
            reason="no narration backend available (start Ollama with "
                   "`ollama serve` and `ollama pull qwen2.5:14b`, or set "
                   "ANTHROPIC_API_KEY)")

    start = time.perf_counter()
    try:
        clauses = backend.clauses(record, model=cfg.model,
                                  max_tokens=cfg.max_tokens)
        text = assemble(clauses, record, cfg.tolerances)
    except Exception as exc:  # noqa: BLE001 — a missing paragraph is not a bug
        logger.warning("narration call failed (%s); rendering the table", exc)
        return Narration(render_table(record), "table",
                         reason=f"{backend.name} call failed: {exc}",
                         model=cfg.model,
                         latency_ms=(time.perf_counter() - start) * 1000)
    elapsed = (time.perf_counter() - start) * 1000

    if not text.strip():
        return Narration(render_table(record), "table",
                         reason=f"{backend.name} returned no usable clauses",
                         model=cfg.model, latency_ms=elapsed)

    if not cfg.gate_on_entailment:
        return Narration(strip_claims(text).strip(), backend.name,
                         reason="gate disabled in config", model=cfg.model,
                         latency_ms=elapsed)

    result = gate(text, record, tolerances=cfg.tolerances)
    if not result.ok:
        logger.warning("narration failed the entailment gate: %s",
                       "; ".join(result.failures))
        if cfg.on_gate_fail != "render_table":
            raise ValueError(
                f"narration failed the entailment gate and on_gate_fail is "
                f"{cfg.on_gate_fail!r}: {result.describe()}"
            )
        return Narration(render_table(record), "table", gate_result=result,
                         reason="failed the entailment gate", model=cfg.model,
                         latency_ms=elapsed)

    return Narration(_readable(text, record), backend.name, gate_result=result,
                     model=cfg.model, latency_ms=elapsed)


def _readable(text: str, record=None) -> str:
    """Replace each verified claim span with its value, formatted for a human.

    Runs AFTER the gate, so formatting here cannot affect verification — the
    gate compared the raw value against the record already. A dollar figure
    reads as money and a probability as a percentage; leaving both as bare
    decimals is how "leads on the title 2.40" ends up on screen.
    """
    from src.app.narration.verify import ATTR_RE, CLAIM_RE

    def swap(match) -> str:
        attrs = dict(ATTR_RE.findall(match.group(1)))
        raw = attrs.get("value", "")
        try:
            value = float(raw)
        except ValueError:
            return raw
        quantity = attrs.get("quantity", "")
        subject = attrs.get("subject", "")
        comparator = attrs.get("comparator", "approx")
        if quantity == "dollars":
            shown = f"-${abs(value):.2f}" if value < 0 else f"${value:.2f}"
        elif quantity == "probability":
            shown = f"{value:.0%}"
        elif quantity == "player_fact":
            # A football fact carries its own unit, so the subject supplies it.
            # Rendering these as dollars is what produced "provides steadier
            # production week to week less than $0.82" — the right subject
            # wearing the wrong unit, which reads as nonsense.
            unit = _FACT_UNITS.get(subject.rsplit(" ", 1)[-1].lower(), "")

            # Show the player's ACTUAL value, not the bound the model claimed.
            # A one-player fact next to a comparative sentence reads as a
            # contradiction otherwise: "Nabers is on the field MORE weeks than
            # Bowers (UNDER 13.3 games)". Both halves were true — the gate had
            # verified 13.2 < 13.3 — but "more" beside "under" is nonsense to
            # read. The true figure cannot disagree with the sentence, and it
            # is strictly more informative than a bound.
            #
            # Safe only because this runs AFTER the gate. Substituting before
            # it would have the gate check the record against itself.
            actual = record.fact_for(subject) if record is not None else None
            if actual is not None:
                return f"({actual:.1f}{unit})"
            shown = f"{value:.1f}{unit}"
        else:
            return raw
        # PARENTHETICAL, not a trailing fragment. Inline reads as a run-on
        # with the figure bolted on — "steadier week to week less than 7.6
        # pts/wk" — because the model wrote a complete clause and the number
        # arrives after it. In brackets it reads as the evidence it is.
        #
        # `lt` says "under", not "less than": half these quantities are
        # lower-is-better (weekly swing), and "less than" reads as a shortfall
        # on a metric where a small number is the good news.
        word = {"gt": "over ", "lt": "under "}.get(comparator, "")
        return f"({word}{shown})"

    return " ".join(CLAIM_RE.sub(swap, text).split())
