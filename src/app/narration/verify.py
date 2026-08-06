"""The entailment gate (§18) — claims are checked, never trusted.

Every quantitative statement in a narration must appear as an inline span:

    <claim subject="champion" quantity="dollars" comparator="gt" value="2.4"/>

and **no bare numeral may appear outside one**. Both halves matter. Checking
tagged claims alone would let a model write "roughly twice as valuable" or
"Gibbs scores 18 a game" in plain prose and sail through, which is precisely
the failure the gate exists to prevent.

Claims are extracted by **XML parse, not by an LLM**. Asking a model to check
another model's arithmetic reintroduces the thing being guarded against, and it
does so invisibly.

Entailment, one rule per quantity kind (§18):

    dollars      |claimed - actual| <= tolerances['dollars']
    probability  |claimed - actual| <= tolerances['probability']
    week         exact membership in delta_weeks
    slot         exact string equality with roster_slot_affected

Any unentailed claim blocks the render; the caller falls back to
`on_gate_fail: render_table`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping

from src.app.narration.attribution import QUANTITIES, AttributionRecord

CLAIM_RE = re.compile(
    r"<claim\s+([^>]*?)/?>",
    re.IGNORECASE | re.DOTALL,
)
ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')

#: A numeral in prose. Deliberately broad — it fires on "18", "2.4", "50%".
NUMERAL_RE = re.compile(r"\d")

#: Words that spell a number. A model told not to write digits will write
#: "twice" and "half" instead, and the resulting sentence is exactly as
#: unverifiable as "2x" would have been.
SPELLED_NUMBERS = frozenset({
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "dozen", "half", "third", "quarter",
    "twice", "double", "triple", "tenfold",
})

COMPARATORS = ("gt", "lt", "approx", "eq")


@dataclass(frozen=True)
class Claim:
    subject: str
    quantity: str
    comparator: str
    value: str
    raw: str = ""

    @property
    def numeric(self) -> float | None:
        try:
            return float(self.value)
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class GateResult:
    ok: bool
    claims: list[Claim] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    untagged_numerals: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if self.ok:
            return f"entailment gate passed: {len(self.claims)} claims verified"
        lines = [f"entailment gate FAILED ({len(self.failures)} problems)"]
        lines.extend(f"  {f}" for f in self.failures)
        return "\n".join(lines)


def extract_claims(text: str) -> list[Claim]:
    claims = []
    for match in CLAIM_RE.finditer(text):
        attrs = dict(ATTR_RE.findall(match.group(1)))
        claims.append(Claim(
            subject=attrs.get("subject", ""),
            quantity=attrs.get("quantity", "").lower(),
            comparator=attrs.get("comparator", "").lower(),
            value=attrs.get("value", ""),
            raw=match.group(0),
        ))
    return claims


def strip_claims(text: str) -> str:
    return CLAIM_RE.sub(" ", text)


def find_untagged_numerals(text: str, *, allowed: frozenset[str] = frozenset()
                           ) -> list[str]:
    """Numerals and number-words in prose outside any claim span.

    `allowed` exempts literal labels drawn from the record — a slot name like
    "RB2" or "FLEX2" carries a digit but is a CATEGORY, not a magnitude. It is
    verifiable by exact string equality against `roster_slot_affected`, which
    is precisely the rule the `slot` quantity uses, so permitting it costs
    nothing. Without the exemption every record whose affected slot is numbered
    fails the gate the moment the narrator mentions the position.
    """
    prose = strip_claims(text)
    hits = []
    for token in re.findall(r"[A-Za-z0-9$%.\-]+", prose):
        bare = token.strip(".,;:")
        if bare in allowed:
            continue
        if NUMERAL_RE.search(token):
            hits.append(token)
        elif bare.lower() in SPELLED_NUMBERS:
            hits.append(token)
    return hits


def _entails(claim: Claim, actual: float, tolerance: float) -> bool:
    claimed = claim.numeric
    if claimed is None:
        return False
    if claim.comparator == "approx":
        return abs(claimed - actual) <= tolerance
    if claim.comparator == "eq":
        return abs(claimed - actual) <= tolerance
    # A directional claim must be true AND not be a near-tie dressed as a
    # difference: "greater than 2.40" when the actual is 2.4001 is technically
    # right and rhetorically false.
    if claim.comparator == "gt":
        return actual > claimed + tolerance
    if claim.comparator == "lt":
        return actual < claimed - tolerance
    return False


def gate(text: str, record: AttributionRecord, *,
         tolerances: Mapping[str, float]) -> GateResult:
    """Verify every claim, and that nothing quantitative escaped a claim."""
    claims = extract_claims(text)
    failures: list[str] = []

    # Literal labels from the record are exempt: a slot like "RB2" is a
    # category verifiable by exact equality, not a magnitude.
    allowed_labels = frozenset(
        {record.roster_slot_affected} - {""})
    untagged = find_untagged_numerals(text, allowed=allowed_labels)
    if untagged:
        failures.append(
            f"untagged quantitative language outside a <claim> span: "
            f"{', '.join(untagged[:8])}"
        )

    for claim in claims:
        if claim.quantity not in QUANTITIES:
            failures.append(
                f"claim {claim.raw!r}: unknown quantity {claim.quantity!r} "
                f"(expected one of {', '.join(QUANTITIES)}) — there is no "
                f"entailment rule for it, so it cannot be verified"
            )
            continue
        if claim.comparator not in COMPARATORS:
            failures.append(
                f"claim {claim.raw!r}: unknown comparator "
                f"{claim.comparator!r}"
            )
            continue

        if claim.quantity == "slot":
            if claim.value.strip() != record.roster_slot_affected:
                failures.append(
                    f"slot claim {claim.value!r} != "
                    f"{record.roster_slot_affected!r}"
                )
            continue

        if claim.quantity == "week":
            numeric = claim.numeric
            if numeric is None or int(numeric) not in record.weeks():
                failures.append(
                    f"week claim {claim.value!r} is not a week the record "
                    f"mentions"
                )
            continue

        if claim.quantity == "dollars":
            actual = record.dollars_for(claim.subject)
        else:
            actual = record.probability_for(claim.subject)

        if actual is None:
            failures.append(
                f"claim about {claim.subject!r} has no counterpart in the "
                f"record — the narrator invented a subject"
            )
            continue

        tol = float(tolerances.get(claim.quantity, 0.0))
        if not _entails(claim, actual, tol):
            failures.append(
                f"{claim.quantity} claim {claim.comparator} {claim.value} for "
                f"{claim.subject!r} is not entailed by {actual:.4f} "
                f"(tolerance {tol})"
            )

    return GateResult(ok=not failures, claims=claims, failures=failures,
                      untagged_numerals=untagged)
