"""Per-source canonicalizers (§11.1).

Canonicalization is where a content-addressed store lives or dies. Volatile
response fields — timestamps, request ids, key ordering — make every fetch a
fresh digest, and the store degenerates into an append-only garbage heap while
``diff()`` reports spurious drift on every pull.

Each source registers a named canonicalizer and the name is recorded in the
manifest entry, so a change to normalization is itself visible as a change.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Final

Canonicalizer = Callable[[bytes], bytes]

_REGISTRY: dict[str, Canonicalizer] = {}


def register(name: str) -> Callable[[Canonicalizer], Canonicalizer]:
    def wrap(fn: Canonicalizer) -> Canonicalizer:
        if name in _REGISTRY:
            raise RuntimeError(f"canonicalizer {name!r} already registered")
        _REGISTRY[name] = fn
        return fn
    return wrap


def get(name: str) -> Canonicalizer:
    if name not in _REGISTRY:
        raise KeyError(f"no canonicalizer {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registered() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


@register("identity")
def _identity(body: bytes) -> bytes:
    """For sources whose bytes are already stable (e.g. parquet frames)."""
    return body


def _strip(obj: object, drop: frozenset[str]) -> object:
    if isinstance(obj, dict):
        return {k: _strip(v, drop) for k, v in obj.items() if k not in drop}
    if isinstance(obj, list):
        return [_strip(v, drop) for v in obj]
    return obj


def json_canonicalizer(name: str, *, drop: frozenset[str]) -> Canonicalizer:
    """Build and register a JSON canonicalizer that drops volatile keys."""

    @register(name)
    def _canon(body: bytes) -> bytes:
        parsed = json.loads(body)
        cleaned = _strip(parsed, drop)
        return json.dumps(cleaned, sort_keys=True, separators=(",", ":")).encode()

    return _canon


#: FFC echoes the request window and a status string; the window moves daily
#: even when the ADP table is unchanged.
FFC_VOLATILE: Final = frozenset({"start_date", "end_date", "status", "total_drafts"})
json_canonicalizer("ffc", drop=FFC_VOLATILE)

#: Yahoo stamps every payload with a request timestamp and a copyright blob.
YAHOO_VOLATILE: Final = frozenset({"time", "copyright", "refresh_rate"})
json_canonicalizer("yahoo", drop=YAHOO_VOLATILE)
