"""Point-in-time data access — the only door for feature code (§11.2)."""
from src.platform.asof.handle import AsOf, asof, assert_same_cutoff
from src.platform.asof.registry import ColumnSpec, availability_of, is_registered

__all__ = ["AsOf", "asof", "assert_same_cutoff", "ColumnSpec",
           "availability_of", "is_registered"]
