"""Narration (§18) — prose that cannot say anything the numbers do not.

The LLM never decides the pick. It is handed a finished `AttributionRecord`
and asked to explain it, and every quantitative statement it makes is parsed
back out and checked against that record before a human sees it.

The gate is the whole point. A prompt instruction to "only state numbers from
the record" is not a constraint, it is a hope — and an unenforced hope is worse
than no narration at all, because it launders unverified claims as verified.
"""
from src.app.narration.attribution import AttributionRecord
from src.app.narration.render import NarrationConfig, narrate, render_table
from src.app.narration.verify import Claim, GateResult, gate

__all__ = ["AttributionRecord", "Claim", "GateResult", "NarrationConfig",
           "gate", "narrate", "render_table"]
