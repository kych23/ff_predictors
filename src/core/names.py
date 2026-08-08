"""Player-name normalization — the join key every source is matched on.

Lives in `core` because two layers need it and neither may import the other.
`platform.identity.match` resolves FFC ADP against nflverse with it, and
`models.features.college` needs the same key to match CFBD college production —
but the ADP wall permits `models/**` exactly one platform module
(`platform.asof`), so reaching into `platform.identity` from a feature would
breach the rule that keeps ADP and the DB out of the model layer.

Duplicating the function instead would be worse than either: two spellings of a
join key drift, and when they do, the symptom is a silently lower match rate
rather than an error. Core owns the vocabulary, as it already does for prize
types and calibration buckets.

Dependency-free by construction, which is what `core` requires.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Final

_SUFFIXES: Final = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.IGNORECASE)
_NON_ALNUM: Final = re.compile(r"[^a-z0-9 ]")


def normalize_name(name: str) -> str:
    """Casefold, strip accents and punctuation, fold generational suffixes.

    Suffix folding is what makes 'Marvin Harrison Jr.' and 'Marvin Harrison'
    the same key — the mismatch class the design calls out by name.
    """
    if not isinstance(name, str):
        return ""
    text = unicodedata.normalize("NFKD", name)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace(".", " ").replace("'", "")
    text = _SUFFIXES.sub(" ", text)
    text = _NON_ALNUM.sub(" ", text)
    return " ".join(text.split())
