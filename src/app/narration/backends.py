"""Narration backends (§18, §19) — local first, offline by default.

§19 says the cockpit is fully offline: "no Postgres, no Supabase, no network in
the recommendation path." A hosted API call breaks that, and breaks it at the
worst moment — draft night, on somebody's hotspot. A local model puts narration
back inside the boundary.

**The model never chooses a number.** It picks which facts to mention, in what
order, and how to word them; every VALUE is filled in from the
`AttributionRecord` afterwards. That is the whole reason a 7B model is safe
here. The usual objection to a small model on user-facing prose is unbounded
hallucination, and the usual mitigation is a bigger model, which only makes it
rarer. Removing the number from the model's output space makes a wrong number
impossible instead of unlikely — and what remains (word choice, emphasis) is
what small models are actually good at.

Two further constraints do the rest:

* **Schema-constrained decoding.** Subjects are a JSON-schema `enum` built from
  the record itself, so the model cannot name a prize, player or slot that does
  not exist. Ollama enforces the schema during sampling; this is not a prompt
  request.
* **The §18 entailment gate still runs.** Defence in depth: the assembled text
  goes through `verify.gate` exactly as a hosted model's would, so a bug in
  this file cannot put an unverified claim on screen.

Backends are ordered by preference, not by quality: `ollama` (local, offline),
then `anthropic` (hosted, needs a key), then the table. The table is not a
degraded mode — it states the same record and needs no model at all.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from src.app.narration.attribution import AttributionRecord

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_LOCAL_MODEL = "qwen2.5:7b"

#: How many claim-bearing clauses a narration may contain. Three sentences is
#: the §18 budget; beyond that the model starts restating the table.
MAX_CLAUSES = 3

SYSTEM_PROMPT = """\
You explain a fantasy-football draft recommendation that has ALREADY been made. \
You are not deciding anything. The pick is fixed.

You will receive an attribution record: why the leading player beats the \
runner-up, decomposed by prize and by week.

Return JSON matching the schema. For each clause:
  - `text` is a short phrase with NO NUMBERS IN IT AT ALL. Not digits, not \
words like "two", "twice", "half" or "double". The number is inserted after \
you, automatically.
  - Refer to the players BY NAME. Never write "the two players", "the two \
backs" or "between the two" — say "both" or name them. This is the single \
most common way a response gets discarded.
  - `subject` names which fact the number refers to. Choose only from the \
allowed values.
  - `comparator` is how to read it: `approx` for a plain amount, `gt` or `lt` \
for a directional statement.

Write like an analyst talking to someone on the clock: what separates these two \
players, and where the separation comes from. Be specific and brief. Do not \
hedge, do not restate the whole table, do not give advice.\
"""


@dataclass(frozen=True)
class Clause:
    text: str
    subject: str
    quantity: str
    comparator: str


class Backend(Protocol):
    name: str

    def available(self) -> bool:
        ...

    def clauses(self, record: AttributionRecord, *, model: str,
                max_tokens: int) -> list[Clause]:
        ...


def allowed_subjects(record: AttributionRecord) -> dict[str, list[str]]:
    """Everything the model may refer to, grouped by quantity kind.

    Built from the record, so the enum cannot contain a subject the gate would
    later reject as invented. This is the same list twice — once as a
    generation constraint, once as a verification rule — which is the point.
    """
    dollars = sorted(record.delta_by_prize) + ["total", "total_se"]
    probability = sorted(record.display(p)
                         for p in record.survival_probabilities)
    return {"dollars": dollars, "probability": probability}


def quantity_of(record: AttributionRecord, subject: str) -> str | None:
    """Which quantity kind a subject denotes. A subject determines it."""
    for quantity, values in allowed_subjects(record).items():
        if subject in values:
            return quantity
    return None


def response_schema(record: AttributionRecord) -> dict[str, Any]:
    """The generation constraint.

    **`quantity` is NOT a field the model fills.** A subject determines its own
    kind — `champion` is always dollars, a player name is always a probability
    — so exposing them as two independent enums makes invalid combinations
    expressible, and the model duly emits them. Measured: `champion` paired
    with `probability` on 5 of 30 records, every one of which resolved to None
    and was dropped, costing three narrations outright. Deriving the quantity
    removes the entire failure class rather than validating against it.
    """
    every_subject = [s for values in allowed_subjects(record).values()
                     for s in values]
    return {
        "type": "object",
        "properties": {
            "clauses": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_CLAUSES,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "subject": {"type": "string", "enum": every_subject},
                        "comparator": {"type": "string",
                                       "enum": ["approx", "gt", "lt"]},
                    },
                    "required": ["text", "subject", "comparator"],
                },
            }
        },
        "required": ["clauses"],
    }


def _magnitude(value: float, scale: float) -> str:
    """Size of a gap, in words, relative to the decision's own scale."""
    ratio = abs(value) / scale if scale > 0 else 0.0
    if ratio < 0.25:
        return "negligible"
    if ratio < 1.0:
        return "slight"
    if ratio < 2.5:
        return "clear"
    return "large"


def prompt_payload(record: AttributionRecord,
                   indifference_zone: float = 2.0) -> str:
    """A QUALITATIVE brief. The model is shown no numbers at all.

    This is the fix for the failure mode that made the first version unusable:
    shown `champion: 0.18`, a 7B model writes "leads by $0.18" into its prose,
    the gate sees an untagged numeral, and the narration is discarded. Measured
    pass rate that way was **0%** — not because the model reasoned badly, but
    because copying a number you were just handed is the most natural thing in
    the world.

    Describing the same facts as direction and magnitude removes the digits
    from the model's input, so it cannot leak one into `text` however it
    behaves. The exact values are inserted afterwards by `render.assemble`,
    read straight off the record.
    """
    leader, runner = (record.display(c) for c in record.pair)
    prizes = sorted(record.delta_by_prize.items(), key=lambda kv: -abs(kv[1]))
    return json.dumps({
        "leader": leader,
        "runner_up": runner,
        "overall": {
            "direction": "leader ahead" if record.total_delta > 0
                         else "essentially tied",
            "magnitude": _magnitude(record.total_delta, indifference_zone),
            "certainty": ("confident"
                          if abs(record.total_delta) > 2 * record.total_se
                          else "within the noise"),
        },
        "prizes_by_importance": [
            {"subject": prize,
             "direction": ("leader ahead" if value > 0 else "leader behind"),
             "magnitude": _magnitude(value, indifference_zone)}
            for prize, value in prizes
        ],
        "roster_slot_affected": record.roster_slot_affected,
        "has_bye_conflict": bool(record.bye_conflicts),
        "gap_concentrated_in_a_few_weeks": bool(
            record.top_weeks() and abs(record.top_weeks()[0][1])
            > 2 * abs(record.total_delta)),
        "allowed_subjects": allowed_subjects(record),
    }, indent=2)


def _parse_clauses(raw: str, record: AttributionRecord) -> list[Clause]:
    """Quantity is DERIVED from the subject, never read from the model."""
    data = json.loads(raw)
    out = []
    for item in data.get("clauses", [])[:MAX_CLAUSES]:
        subject = str(item.get("subject", ""))
        quantity = quantity_of(record, subject)
        if quantity is None:
            logger.warning("dropping clause with unknown subject %r", subject)
            continue
        out.append(Clause(
            text=str(item.get("text", "")).strip(),
            subject=subject,
            quantity=quantity,
            comparator=str(item.get("comparator", "approx")),
        ))
    return out


class OllamaBackend:
    """Local model over Ollama's HTTP API. No key, no egress.

    127.0.0.1 only — this is deliberate. A backend that could be pointed at a
    remote host would quietly reintroduce the network dependency §19 exists to
    remove, and nobody would notice until draft night.
    """

    name = "ollama"

    def __init__(self, url: str = OLLAMA_URL, timeout: float = 30.0) -> None:
        self.url = url
        self.timeout = timeout

    def available(self) -> bool:
        try:
            import httpx
            r = httpx.get(f"{self.url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:  # noqa: BLE001 — absence is a normal state
            return False

    def installed_models(self) -> list[str]:
        try:
            import httpx
            r = httpx.get(f"{self.url}/api/tags", timeout=2.0)
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:  # noqa: BLE001
            return []

    def clauses(self, record: AttributionRecord, *, model: str,
                max_tokens: int) -> list[Clause]:
        import httpx

        response = httpx.post(
            f"{self.url}/api/chat",
            timeout=self.timeout,
            json={
                "model": model,
                "stream": False,
                # Ollama enforces this during sampling, so a subject outside
                # the enum is not merely discouraged — it cannot be emitted.
                "format": response_schema(record),
                # Low temperature: this is a formatting-disciplined task, not a
                # creative one, and the residual gate failures are the model
                # drifting into number-words like "three".
                "options": {"temperature": 0.2, "num_predict": max_tokens},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_payload(record)},
                ],
            },
        )
        response.raise_for_status()
        return _parse_clauses(response.json()["message"]["content"], record)


class AnthropicBackend:
    """Hosted fallback. Same contract, same constraint: no numbers from the
    model. Kept so the choice is a config line rather than a rewrite."""

    name = "anthropic"

    def __init__(self, client=None) -> None:
        self._client = client

    def available(self) -> bool:
        import os
        if self._client is not None:
            return True
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False
        try:
            import anthropic  # noqa: F401
            return True
        except ImportError:
            return False

    def clauses(self, record: AttributionRecord, *, model: str,
                max_tokens: int) -> list[Clause]:
        client = self._client
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        schema = json.dumps(response_schema(record), indent=2)
        response = client.messages.create(
            model=model, max_tokens=max_tokens,
            system=f"{SYSTEM_PROMPT}\n\nJSON schema:\n{schema}",
            messages=[{"role": "user", "content": prompt_payload(record)}],
        )
        text = "".join(b.text for b in response.content
                       if getattr(b, "type", "text") == "text")
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            raise ValueError("no JSON object in the response")
        return _parse_clauses(text[start:end + 1], record)


def resolve_backend(preference: str = "auto", *, client=None) -> Backend | None:
    """First available backend in preference order. None means table-only."""
    if preference == "table":
        return None
    ollama, anthropic_backend = OllamaBackend(), AnthropicBackend(client)
    if preference == "ollama":
        return ollama if ollama.available() else None
    if preference == "anthropic":
        return anthropic_backend if anthropic_backend.available() else None
    for backend in (ollama, anthropic_backend):
        if backend.available():
            return backend
    return None
