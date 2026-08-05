"""Exception hierarchy shared across layers (DraftEngineDesign.md §9.0).

One root so callers can catch everything this system raises without also
catching library errors, and so the cockpit's degradation ladder (§7.3) can
distinguish "our invariant tripped" from "numpy blew up".
"""
from __future__ import annotations


class DraftEngineError(Exception):
    """Root of every error this system raises deliberately."""


# --- configuration (§10.5, §10.6) -----------------------------------------
class ConfigError(DraftEngineError):
    """A config file is missing, malformed, or violates a validation rule."""


class PayoutImbalanceError(ConfigError):
    """Expected payouts do not sum to the pot (§10.5 rule 9).

    Carries both figures because the whole point is to show the discrepancy —
    a silently unbalanced payout produces a silently wrong objective.
    """

    def __init__(self, computed: float, pot: float, currency: str = "USD") -> None:
        self.computed = computed
        self.pot = pot
        super().__init__(
            f"payout does not conserve the pot: prizes sum to "
            f"{computed:.2f} {currency}, pot is {pot:.2f} {currency} "
            f"(difference {computed - pot:+.2f})"
        )


class UnknownPrizeTypeError(ConfigError):
    """A prize `type` has no registered reducer (§10.5 rule 10)."""


# --- data access (§11) -----------------------------------------------------
class DataError(DraftEngineError):
    """Something is wrong with the data layer."""


class LeakageError(DataError, AssertionError):
    """Data at or after the as-of cutoff reached feature construction.

    Subclasses AssertionError so existing v1 tests that assert on
    AssertionError keep passing through the migration (§10.7).
    """


class CutoffMismatchError(DataError):
    """Frames drawn from two different `AsOf` instances were joined (§11.2)."""


class EmptySourceError(DataError):
    """A source frame was empty after cutoff filtering (§11.2).

    Silently-empty sources are how a pipeline produces confident garbage, so
    this is an error rather than a warning.
    """


class UnregisteredColumnError(DataError):
    """A column was read that has no availability declaration (§11.3).

    The registry defaults to refusal, not permission.
    """


class ArtifactSnapshotMismatch(DataError):
    """A fitted artifact's snapshot_id does not match the bundle's (§20.3)."""


# --- simulation (§15) ------------------------------------------------------
class SimulationError(DraftEngineError):
    """The simulation kernel could not produce a result."""
