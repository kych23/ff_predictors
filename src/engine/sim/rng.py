"""Counter-based RNG addressing (§15.1, AD-4).

Common Random Numbers falls out of *addressing* rather than out of careful
stream plumbing: replication ``r`` of player ``p`` in week ``w`` resolves to the
same underlying uniform no matter which candidate pick is being evaluated. That
is what makes two candidates comparable at a few thousand replications instead
of the ~27,000 independent draws the paired-variance arithmetic would otherwise
demand.

Two rules make it correct, both of which earlier revisions got wrong:

1. **Each draw type gets its own ``RngKind``.** ``rate``, ``eps`` and
   ``hazard`` are all player-indexed; sharing one kind puts them at colliding
   stream positions.

2. **``c = 0`` on every aleatory stream — the parameter-draw index does not
   enter.** Aleatory draws are held fixed across theta, which is both what
   makes ``epistemic_se`` mean what §15.5 says and a further variance
   reduction. Only ``param`` is indexed by k.

The Philox key is 128 bits (two uint64). Passing the address tuple directly
raises ``ValueError``, so it is mixed through blake2b — which also means the
addressing scheme is stable across platforms and Python versions.
"""
from __future__ import annotations

import hashlib
import struct
from typing import Final

import numpy as np

from src.core.constants import RngKind

#: Domain separator so a future scheme change cannot silently reuse addresses.
_DOMAIN: Final = b"ff_predictors/rng/v2"


def seed_root(snapshot_id: str, model_version: str, strategy_hash: str) -> str:
    """128-bit root as 32 lowercase hex chars.

    Derived from the three things that determine what a simulation means: which
    bytes it read, which model produced its inputs, and which decision config
    it was run under. Any simulation is bit-reproducible from this plus the
    roster.
    """
    payload = "\x1f".join((snapshot_id, model_version, strategy_hash)).encode()
    return hashlib.blake2b(_DOMAIN + payload, digest_size=16).hexdigest()


def _validate_root(root_hex: str) -> bytes:
    """Fail with a readable message rather than an opaque fromhex error.

    The root reaches here from `seed_root`, from config, or from a test. A
    malformed one otherwise surfaces mid-draw as
    "non-hexadecimal number found in fromhex() arg at position 0", which says
    nothing about what to fix.
    """
    try:
        raw = bytes.fromhex(root_hex)
    except ValueError as exc:
        raise ValueError(
            f"seed root must be lowercase hex, got {root_hex!r}. "
            f"Build it with rng.seed_root(snapshot_id, model_version, "
            f"strategy_hash)."
        ) from exc
    if len(raw) != 16:
        raise ValueError(
            f"seed root must be 128 bits (32 hex chars), got {len(raw) * 8} bits"
        )
    return raw


def _key(root_hex: str, kind: RngKind, a: int, b: int, c: int) -> np.ndarray:
    digest = hashlib.blake2b(
        _validate_root(root_hex) + struct.pack("<4q", int(kind), a, b, c),
        digest_size=16,
    ).digest()
    return np.frombuffer(digest, dtype="<u8")


def stream(root_hex: str, kind: RngKind, *, a: int = 0, b: int = 0, c: int = 0
           ) -> np.random.Generator:
    """A generator addressed by ``(kind, a, b, c)``.

    Addressing semantics (§15.1), with ``c = 0`` on every aleatory stream:

    ==========  ==============  =========  ===
    kind        a               b          c
    ==========  ==============  =========  ===
    ``RATE``    player index    0          0
    ``EPS``     player index    week       0
    ``HAZARD``  player index    week       0
    ``WEEK``    NFL team index  week       0
    ``KDST``    player index    week       0
    ``DRAFT``   manager index   pick no    0
    ``PARAM``   draw index      0          0
    ``WAIVER``  week            0          0
    ==========  ==============  =========  ===
    """
    return np.random.Generator(np.random.Philox(key=_key(root_hex, kind, a, b, c)))


def uniforms(root_hex: str, kind: RngKind, *, n: int,
             a: int = 0, b: int = 0, c: int = 0) -> np.ndarray:
    """``n`` uniforms from one address.

    Extending ``n`` reuses the earlier values, which is what lets the allocator
    deepen a candidate without redrawing anything (§17.2).
    """
    return stream(root_hex, kind, a=a, b=b, c=c).random(n)


def normals(root_hex: str, kind: RngKind, *, shape, a: int = 0, b: int = 0,
            c: int = 0) -> np.ndarray:
    """Standard normals by INVERSE CDF, never by Ziggurat.

    ``standard_normal`` consumes a variable number of uniforms per value, so
    extending a draw would not reproduce the earlier values and CRN would break
    exactly where the allocator relies on it. ``ndtri`` on a uniform is ~2.9x
    slower and buys that guarantee.
    """
    from scipy.special import ndtri

    size = int(np.prod(shape))
    u = uniforms(root_hex, kind, n=size, a=a, b=b, c=c)
    # ndtri(0) is -inf; nudge off the boundary
    np.clip(u, 1e-12, 1 - 1e-12, out=u)
    return ndtri(u).reshape(shape)
