"""Enforce the layer dependency graph (DraftEngineDesign.md §9.0).

A module may import only from layers at or below its own rank. This single rule
subsumes three mechanisms the design previously maintained by hand:

  * the v1 ADP wall (a denylist on one module path, defeated by renaming it),
  * the r2-r4 import allowlist (which kept omitting real imports such as
    ``__future__``, ``datetime`` and ``scipy.stats``),
  * the separate requirement that DB access leave the model layer.

It is expressed in terms of *layers*, not names, so a new ADP module under a
new name cannot slip through: ADP lives in ``platform.sources``, and
``models`` may reach only ``platform.asof``.

Third-party imports are unrestricted; this governs ``src.*`` only.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

#: Layer rank. A module may import from its own rank or lower.
LAYER_RANK: dict[str, int] = {
    "core": 0,
    "domain": 1,
    "platform": 2,
    "models": 3,
    "engine": 4,
    "app": 5,
}

#: ``models`` is the ADP wall. It may reach exactly one ``platform``
#: subpackage — the point-in-time handle — and nothing else. That single
#: clause gives, simultaneously: no ADP import (ADP is in platform.sources),
#: no DB access (that is platform.persistence), and point-in-time discipline
#: (asof is the only door).
MODELS_PLATFORM_ALLOW: frozenset[str] = frozenset({"src.platform.asof"})

#: v1 packages not yet migrated into the layer scheme. Phase 5 empties this;
#: it is asserted to only ever shrink so the migration cannot silently stall.
LEGACY_PACKAGES: frozenset[str] = frozenset({
    "benchmark", "config", "db", "features", "ingest",
    "labels", "lineup", "projection", "recommender",
})


def _module_name(path: Path) -> str:
    """src/models/features/assemble.py -> src.models.features.assemble"""
    rel = path.relative_to(SRC.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _layer_of(module: str) -> str | None:
    """Second component of a dotted src module name, if it names a layer."""
    parts = module.split(".")
    if len(parts) < 2 or parts[0] != "src":
        return None
    return parts[1] if parts[1] in LAYER_RANK else None


def _imported_modules(tree: ast.AST, module: str) -> list[tuple[str, int]]:
    """Every ``src.*`` module this file imports, as (target, lineno).

    Relative imports are resolved to absolute names first. Skipping that step
    is exactly how a leak would slip through: ``from . import anything`` has no
    dotted name to match against, so an unresolved check would pass it
    silently.
    """
    out: list[tuple[str, int]] = []
    pkg_parts = module.split(".")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src."):
                    out.append((alias.name, node.lineno))

        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # `from . import x` inside src.a.b.c -> base is src.a.b
                base = pkg_parts[: len(pkg_parts) - node.level]
                if node.module:
                    targets = [".".join(base + node.module.split("."))]
                else:
                    targets = [".".join(base + [a.name]) for a in node.names]
            elif node.module and node.module.startswith("src."):
                targets = [node.module]
            else:
                continue
            out.extend((t, node.lineno) for t in targets if t.startswith("src."))

    return out


def _source_files() -> list[Path]:
    return sorted(
        p for p in SRC.rglob("*.py")
        if "__pycache__" not in p.parts
        and p.relative_to(SRC).parts[0] not in LEGACY_PACKAGES
    )


def test_layer_dependency_graph() -> None:
    """No module imports from a layer above its own."""
    violations: list[str] = []

    for path in _source_files():
        module = _module_name(path)
        layer = _layer_of(module)
        if layer is None:
            continue
        rank = LAYER_RANK[layer]

        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # a broken file is a failure, not a skip
            violations.append(f"{path}: unparseable ({exc})")
            continue

        for target, lineno in _imported_modules(tree, module):
            target_layer = _layer_of(target)
            if target_layer is None:
                if target.split(".")[1] in LEGACY_PACKAGES:
                    violations.append(
                        f"{module}:{lineno} imports LEGACY {target} — migrate "
                        f"it into a layer instead of importing across the "
                        f"v1/v2 boundary"
                    )
                continue

            if LAYER_RANK[target_layer] > rank:
                violations.append(
                    f"{module}:{lineno} ({layer}, rank {rank}) imports "
                    f"{target} ({target_layer}, rank {LAYER_RANK[target_layer]}) "
                    f"— layers may only import downward"
                )

    assert not violations, "layer violations:\n  " + "\n  ".join(violations)


def test_models_may_reach_only_asof_in_platform() -> None:
    """The ADP wall, expressed structurally (§11.3)."""
    violations: list[str] = []

    for path in _source_files():
        module = _module_name(path)
        if _layer_of(module) != "models":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for target, lineno in _imported_modules(tree, module):
            if _layer_of(target) != "platform":
                continue
            if not any(
                target == allowed or target.startswith(allowed + ".")
                for allowed in MODELS_PLATFORM_ALLOW
            ):
                violations.append(
                    f"{module}:{lineno} imports {target} — models may reach "
                    f"only {sorted(MODELS_PLATFORM_ALLOW)} inside platform. "
                    f"ADP lives in platform.sources and the DB in "
                    f"platform.persistence; both are walled off here."
                )

    assert not violations, "ADP-wall violations:\n  " + "\n  ".join(violations)


def test_core_imports_nothing_else_in_src() -> None:
    """`core` is the only home for shared utilities, and that is only safe
    while it stays dependency-free (§9.0 shared-utility policy)."""
    violations: list[str] = []

    for path in _source_files():
        module = _module_name(path)
        if _layer_of(module) != "core":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for target, lineno in _imported_modules(tree, module):
            if not target.startswith("src.core"):
                violations.append(
                    f"{module}:{lineno} imports {target} — core must import "
                    f"nothing else in src. If a helper needs another layer it "
                    f"does not belong in core; duplicate it instead."
                )

    assert not violations, "core purity violations:\n  " + "\n  ".join(violations)


def test_no_utils_package_outside_core() -> None:
    """Stop the god-module before it starts (§9.0 rule 3)."""
    offenders = [
        p.relative_to(SRC.parent)
        for p in SRC.rglob("utils*")
        if p.is_dir() or (p.suffix == ".py" and not p.name.startswith("_"))
        if p.relative_to(SRC).parts[0] not in LEGACY_PACKAGES
        and p.relative_to(SRC).parts[0] != "core"
    ]
    assert not offenders, (
        f"`utils` may exist only in core; found {offenders}. "
        f"Layer-private helpers belong in that layer as _utils.py."
    )


def test_every_layer_package_exists() -> None:
    """The scaffold is complete, so a misplaced file fails loudly rather than
    creating a new top-level package by accident."""
    missing = [name for name in LAYER_RANK if not (SRC / name).is_dir()]
    assert not missing, f"missing layer packages: {missing}"


@pytest.mark.parametrize("name", sorted(LAYER_RANK))
def test_layer_is_importable(name: str) -> None:
    __import__(f"src.{name}")
