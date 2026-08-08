"""The §21.5 power audit. Runs BEFORE any gate.

    venv/bin/python scripts/audit_power.py

Prints, for every gate the project intends to report: the nominal sample size,
the design effect that shrinks it, the minimum detectable effect, the power at
the effect actually expected, and the verdict — `claim` or `descriptive_only`.

This is not a pass/fail script. It exits 0 whether or not gates are
underpowered; being underpowered is a fact about the evidence, not a build
failure. It exits non-zero only when the assumptions file itself is
unusable, because an audit that silently audits nothing is worse than none.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.errors import ConfigError  # noqa: E402
from src.engine.audit.power import (  # noqa: E402
    MDE_POWER,
    evaluate,
    load_assumptions,
    required_n,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assumptions", type=Path, default=None)
    args = ap.parse_args()

    try:
        gates, settings = load_assumptions(args.assumptions)
    except ConfigError as exc:
        print(f"POWER AUDIT UNUSABLE: {exc}")
        return 1

    print("§21.5 POWER AUDIT")
    print(f"  alpha {settings['alpha']}  power floor {settings['power_floor']}"
          f"  MDE quoted at {MDE_POWER:.0%} power")
    print(f"  {len(gates)} gates declared\n")

    print(f"  {'gate':34s} {'n':>5s} {'deff':>6s} {'n_eff':>7s} "
          f"{'MDE':>9s} {'power':>6s}  verdict")
    print(f"  {'-' * 34} {'-' * 5} {'-' * 6} {'-' * 7} {'-' * 9} {'-' * 6}"
          f"  {'-' * 16}")

    verdicts = []
    for gate in gates:
        v = evaluate(gate, **settings)
        verdicts.append(v)
        mark = "*" if v.fully_assumed else " "
        print(f"  {gate.id:34s} {gate.n_nominal:5d} {gate.design_effect:6.2f} "
              f"{gate.n_effective:7.1f} {v.mde:9.4f} {v.power:6.2f}  "
              f"{v.verdict}{mark}")

    print("\n  * both the effect size and the SD are assumed, so the verdict")
    print("    is a statement about the assumption, not about the data.")

    claims = [v for v in verdicts if v.verdict == "claim"]
    weak = [v for v in verdicts if v.verdict == "descriptive_only"]

    print(f"\n  MAY CLAIM ({len(claims)}):")
    for v in claims:
        print(f"    {v.gate.id}")
        print(f"      detects effects down to {v.mde:.4f}; expected "
              f"{v.gate.expected_effect:.4f}")
        print(f"      equivalence bound: rules out differences beyond "
              f"{v.equivalence_bound:.4f}")

    print(f"\n  DESCRIPTIVE ONLY ({len(weak)}):")
    for v in weak:
        need = required_n(v.gate, alpha=settings["alpha"])
        print(f"    {v.gate.id}")
        print(f"      power {v.power:.2f} at the expected effect "
              f"{v.gate.expected_effect:.4f}; MDE is {v.mde:.4f}")
        print(f"      would need n = {need:,} {v.gate.unit}s for 80% power "
              f"(has {v.gate.n_nominal:,})")

    print("\n  Design effects are where the sample sizes actually go. The")
    print("  largest reduction here:")
    worst = max(gates, key=lambda g: g.design_effect)
    print(f"    {worst.id}: {worst.n_nominal} {worst.unit}s in clusters of "
          f"{worst.cluster_size} at rho={worst.intra_cluster_rho} "
          f"-> {worst.n_effective:.0f} effective")
    print("\n  Audit complete. No gate above may be reported without the")
    print("  verdict beside it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
