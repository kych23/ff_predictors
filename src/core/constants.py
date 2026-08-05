"""Constants shared across two or more layers (DraftEngineDesign.md §9.0).

Everything here must be dependency-free — `core` may import nothing else in
`src`, which is exactly the constraint that stops this module becoming a
grab-bag. A helper that cannot be made dependency-free does not belong here,
and duplication is preferable to inverting the layer graph.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Final

# --- positions -------------------------------------------------------------
# The v1 codebase spells the defense slot "DEF"; v2 standardizes on "DST" and
# migrates the `players.position` column with it (§10.7). Every literal is
# replaced by these constants so a future rename is one edit.
POSITION_QB: Final = "QB"
POSITION_RB: Final = "RB"
POSITION_WR: Final = "WR"
POSITION_TE: Final = "TE"
POSITION_K: Final = "K"
POSITION_DST: Final = "DST"

#: Positions that get a projection model. K and DST are draft slots filled at
#: replacement and simulated from empirical distributions instead (§12.5).
MODELED_POSITIONS: Final = (POSITION_QB, POSITION_RB, POSITION_WR, POSITION_TE)

#: Positions drawn from empirical inverse-CDFs rather than the quantile model.
EMPIRICAL_POSITIONS: Final = (POSITION_K, POSITION_DST)

ALL_POSITIONS: Final = MODELED_POSITIONS + EMPIRICAL_POSITIONS

#: Roster slot that is not a position — flex slots are named in league.yaml.
SLOT_BENCH: Final = "BENCH"


class RngKind(IntEnum):
    """Address space for the counter-based RNG (§15.1).

    Each draw type gets its own kind so that player-indexed streams do not
    collide: `rate`, `eps`, and `hazard` are all indexed by player, and sharing
    one kind would place them at the same stream position.

    Lives in `core` because both `engine.sim` and `models.opponents` draw from
    the same addressed streams; putting it in either would invert §9.0.
    """

    RATE = 1
    EPS = 2
    HAZARD = 3
    WEEK = 4
    KDST = 5
    DRAFT = 6
    PARAM = 7
    WAIVER = 8


#: Conformal calibration buckets (§10.2 rule 7), in the calibrator's own
#: priority order. Canonical HERE rather than in the calibrator, because
#: `core.config.league` must validate the configured names and `core` may
#: import nothing above it. When `projection.calibrate` migrates to
#: `models.projection` (phase 6) it imports these — a downward import.
CALIBRATION_BUCKETS: Final = (
    "rookie", "second_year", "team_changed_vet", "established_vet",
)


#: Φ⁻¹(0.9). Used by the split-normal quantile match (§15.2) and by anything
#: converting a P10/P90 pair into a scale parameter.
Z90: Final = 1.2815515655446004
