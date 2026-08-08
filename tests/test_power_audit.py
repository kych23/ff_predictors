"""Power audit (§21.5).

The audit's job is to stop the project overstating itself, so its own failure
mode is being wrong in the permissive direction. These tests pin the
arithmetic and, more importantly, pin the two places a power calculation
usually cheats: ignoring the design effect, and quoting an equivalence bound
for a gate that could not detect anything.
"""
from __future__ import annotations

import pytest
import yaml

from src.core.errors import ConfigError
from src.engine.audit.power import (
    Gate,
    evaluate,
    load_assumptions,
    power_table,
    required_n,
)


def _gate(**kw) -> Gate:
    base = dict(id="g", unit="season", n_nominal=100, cluster_size=1,
                intra_cluster_rho=0.0, expected_effect=0.5, sd=1.0)
    base.update(kw)
    return Gate(**base)


# --------------------------------------------------------- design effect
def test_clustering_shrinks_the_sample():
    """Twelve seats from one simulated draft share a board. Counting them as
    twelve independent observations is the single most common way a simulation
    study inflates its own sample."""
    flat = _gate(cluster_size=1, intra_cluster_rho=0.0)
    clustered = _gate(cluster_size=12, intra_cluster_rho=0.35)
    assert flat.n_effective == 100
    assert clustered.design_effect == pytest.approx(1 + 11 * 0.35)
    assert clustered.n_effective < 25
    assert clustered.standard_error > flat.standard_error


def test_zero_rho_is_a_no_op():
    assert _gate(cluster_size=12, intra_cluster_rho=0.0).design_effect == 1.0


def test_rho_of_one_is_rejected_not_silently_clipped():
    with pytest.raises(ConfigError, match="rho"):
        _gate(intra_cluster_rho=1.0)


# ------------------------------------------------------------ arithmetic
def test_power_rises_with_sample_size():
    small = evaluate(_gate(n_nominal=25))
    large = evaluate(_gate(n_nominal=400))
    assert large.power > small.power
    assert large.mde < small.mde


def test_mde_is_the_effect_detectable_at_eighty_percent():
    """Self-consistency: a gate whose expected effect EQUALS its own MDE must
    come back with power 0.80."""
    g = _gate()
    mde = evaluate(g).mde
    again = evaluate(_gate(expected_effect=mde))
    assert again.power == pytest.approx(0.80, abs=0.005)


def test_verdict_switches_at_the_floor():
    weak = evaluate(_gate(expected_effect=0.01), power_floor=0.5)
    strong = evaluate(_gate(expected_effect=1.0), power_floor=0.5)
    assert weak.verdict == "descriptive_only"
    assert strong.verdict == "claim"


def test_underpowered_gates_get_no_equivalence_bound():
    """'We can rule out a difference larger than X' is a real claim. A gate
    with 9% power has not earned it."""
    weak = evaluate(_gate(expected_effect=0.01))
    assert weak.equivalence_bound is None
    assert evaluate(_gate(expected_effect=1.0)).equivalence_bound is not None


def test_required_n_actually_reaches_the_target():
    g = _gate(n_nominal=10, cluster_size=6, intra_cluster_rho=0.2,
              expected_effect=0.2, sd=1.0)
    n = required_n(g, target_power=0.80)
    assert evaluate(_gate(**{**g.__dict__, "n_nominal": n})).power >= 0.795


def test_required_n_respects_the_design_effect():
    flat = required_n(_gate(cluster_size=1, intra_cluster_rho=0.0,
                            expected_effect=0.2))
    clustered = required_n(_gate(cluster_size=12, intra_cluster_rho=0.35,
                                 expected_effect=0.2))
    assert clustered > flat * 4


# ------------------------------------------------------------- the file
def test_shipped_assumptions_load_and_are_complete():
    gates, settings = load_assumptions()
    assert len(gates) >= 5
    assert 0 < settings["alpha"] < 1
    ids = {g.id for g in gates}
    assert {"ndcg_at_84_vs_adp", "expected_dollars_vs_adp_bots"} <= ids


def test_shipped_table_matches_the_designs_prediction():
    """§21.5 predicts: NDCG@84 and NDCG@36 are `claim`; E[$] vs ADP bots and
    per-bucket coverage are `descriptive_only`. If the shipped assumptions stop
    producing that, either the assumptions or the design changed and someone
    needs to say which."""
    table = power_table().set_index("gate")
    assert table.loc["ndcg_at_84_vs_adp", "verdict"] == "claim"
    assert table.loc["ndcg_at_36_vs_adp", "verdict"] == "claim"
    assert table.loc["expected_dollars_vs_adp_bots", "verdict"] == "descriptive_only"
    assert table.loc["quantile_coverage_by_bucket", "verdict"] == "descriptive_only"


def test_every_gate_declares_its_provenance():
    gates, _ = load_assumptions()
    for g in gates:
        assert g.sd_provenance in {"measured", "assumed"}
        assert g.effect_provenance in {"measured", "assumed"}


def test_missing_keys_are_rejected_by_name(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"gates": [{"id": "x", "unit": "season"}]}))
    with pytest.raises(ConfigError) as exc:
        load_assumptions(path)
    assert "n_nominal" in str(exc.value)


def test_duplicate_gate_ids_are_rejected(tmp_path):
    entry = {"id": "x", "unit": "season", "n_nominal": 10, "cluster_size": 1,
             "intra_cluster_rho": 0.0, "expected_effect": 0.1, "sd": 0.1}
    path = tmp_path / "dup.yaml"
    path.write_text(yaml.safe_dump({"gates": [entry, dict(entry)]}))
    with pytest.raises(ConfigError, match="duplicate"):
        load_assumptions(path)


def test_missing_file_is_reported_not_defaulted(tmp_path):
    with pytest.raises(ConfigError, match="no power assumptions"):
        load_assumptions(tmp_path / "nope.yaml")
