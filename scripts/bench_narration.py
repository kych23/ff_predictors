"""Benchmark narration backends against the §18 entailment gate.

    venv/bin/python scripts/bench_narration.py --n 20
    venv/bin/python scripts/bench_narration.py --model qwen2.5:3b --n 20

The gate turns "is this model good enough?" from a judgement call into a
measurement. Every generated narration is verified against its own attribution
record, so a backend's quality is just: how often does it produce text that
survives, and how long does it take?

Reported per backend:
    pass rate     fraction whose text survives the gate
    latency       median and p90 milliseconds
    clause yield  how many claim-bearing clauses survive assembly
    failures      why the rejected ones were rejected

A backend below roughly 80% pass rate is not usable: the fallback table would
be showing most of the time, and an explanation that appears one pick in three
is worse than one that never appears, because you start waiting for it.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.narration.attribution import AttributionRecord  # noqa: E402
from src.app.narration.backends import (  # noqa: E402
    AnthropicBackend, OllamaBackend,
)
from src.app.narration.render import (  # noqa: E402
    NarrationConfig, narrate, render_table,
)

PRIZES = ("champion", "weekly_high", "season_points")


def synthetic_records(n: int, seed: int = 0) -> list[AttributionRecord]:
    """Records spanning the shapes a real draft produces.

    Deliberately includes the awkward ones: a near-tie, a candidate that wins
    on one prize and loses on another, and a bye conflict. A backend that only
    handles the clean case will look fine on averages and fail on the picks
    that actually need explaining.
    """
    rng = np.random.default_rng(seed)
    names = ["Jahmyr Gibbs", "Bijan Robinson", "Puka Nacua", "CeeDee Lamb",
             "Breece Hall", "Garrett Wilson", "Sam LaPorta", "Josh Allen"]
    out = []
    for i in range(n):
        a, b = rng.choice(len(names), 2, replace=False)
        champion = float(rng.normal(2.0, 2.5))
        weekly = float(rng.normal(0.4, 0.8))
        # one in four is a near-tie, where the honest narration is "these are
        # the same" and models tend to invent a difference anyway
        if i % 4 == 0:
            champion, weekly = champion * 0.05, weekly * 0.05
        # one in five splits: ahead on one prize, behind on the other
        if i % 5 == 0:
            weekly = -abs(weekly) if champion > 0 else abs(weekly)
        weeks = [(w + 1, float(rng.normal(0, 1.5))) for w in range(14)]
        out.append(AttributionRecord(
            pair=(f"p{a}", f"p{b}"),
            delta_by_prize={"champion": champion, "weekly_high": weekly,
                            "season_points": float(rng.normal(0.1, 0.3))},
            delta_weeks=weeks,
            roster_slot_affected=str(rng.choice(["RB", "WR", "TE", "QB"])),
            aleatory_se=float(abs(rng.normal(1.5, 0.4))),
            epistemic_se=float(abs(rng.normal(1.2, 0.4))),
            survival_probabilities={f"p{a}": float(rng.uniform(0.05, 0.5)),
                                    f"p{b}": float(rng.uniform(0.4, 0.95))},
            bye_conflicts=[int(rng.integers(5, 14))] if i % 3 == 0 else [],
            names={f"p{a}": names[a], f"p{b}": names[b]},
        ))
    return out


def run(backend, records, cfg) -> dict:
    passed, latencies, clauses, failures, samples = 0, [], [], Counter(), []
    for record in records:
        result = narrate(record, cfg, backend=backend)
        used_model = result.source == backend.name
        if used_model and result.verified:
            passed += 1
            clauses.append(len(result.gate_result.claims))
            if len(samples) < 3:
                samples.append(result.text)
        else:
            reason = result.reason or "gate"
            if result.gate_result is not None and result.gate_result.failures:
                reason = result.gate_result.failures[0][:90]
            failures[reason] += 1
        if result.latency_ms:
            latencies.append(result.latency_ms)
    n = max(len(records), 1)
    return {
        "pass_rate": passed / n,
        "median_ms": float(np.median(latencies)) if latencies else 0.0,
        "p90_ms": float(np.quantile(latencies, 0.9)) if latencies else 0.0,
        "mean_clauses": float(np.mean(clauses)) if clauses else 0.0,
        "failures": failures,
        "samples": samples,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--anthropic-model", default="claude-opus-5")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    records = synthetic_records(args.n, args.seed)
    print("§18 NARRATION BACKEND BENCHMARK")
    print(f"  {len(records)} synthetic attribution records "
          f"(includes near-ties, split prizes, bye conflicts)")
    print("  every narration is verified against its OWN record by the gate\n")

    candidates = []
    ollama = OllamaBackend()
    if ollama.available():
        installed = ollama.installed_models()
        if any(m.startswith(args.model.split(":")[0]) for m in installed):
            candidates.append((ollama, args.model))
        else:
            print(f"  ollama is up but {args.model!r} is not installed "
                  f"(have: {installed or 'nothing'})")
            print(f"  run: ollama pull {args.model}\n")
    else:
        print("  ollama not reachable — run `ollama serve`\n")

    hosted = AnthropicBackend()
    if hosted.available():
        candidates.append((hosted, args.anthropic_model))

    if not candidates:
        print("  no backend available; the table fallback is what ships.")
        print("\n  Table output for the first record:\n")
        print("    " + render_table(records[0]).replace("\n", "\n    "))
        return 0

    for backend, model in candidates:
        cfg = NarrationConfig(model=model, backend=backend.name)
        print(f"  {backend.name} / {model}")
        stats = run(backend, records, cfg)
        print(f"    pass rate     {stats['pass_rate']:.0%}")
        print(f"    latency       {stats['median_ms']:.0f} ms median, "
              f"{stats['p90_ms']:.0f} ms p90")
        print(f"    clauses       {stats['mean_clauses']:.1f} per narration")
        if stats["failures"]:
            print("    failures:")
            for reason, count in stats["failures"].most_common(4):
                print(f"      {count:3d}x {reason}")
        for sample in stats["samples"]:
            print(f"    e.g. {sample}")
        verdict = ("usable" if stats["pass_rate"] >= 0.80
                   else "NOT usable — the table would show most of the time")
        print(f"    verdict: {verdict}\n")

    print("  The table fallback needs no model and states the same record.")
    print("  A backend only earns its place by beating it on readability")
    print("  often enough that a reader stops expecting the table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
