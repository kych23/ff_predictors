"""ADP result shape, preserved across the FFC -> Yahoo source swap.

Extracted from the deleted FFC-specific ``adp.py`` ahead of the monte-carlo
strip (docs/superpowers/plans/... strip prompt) so a future Yahoo client has
the same result contract to build against, independent of source.

Eligibility-flag pattern (was inline in ``build_adp_season``): a season's ADP
coverage is measured against the RAW source list size (not just gsis-resolved
rows, so eligibility reflects true market depth), then thresholded against
config:

    ranking_eligible = n_players >= cfg.adp.adp_min_ranking
    sim_eligible     = n_players >= cfg.adp_min_fullsim

A new ``build_adp_season`` (Yahoo-backed) should populate ``AdpSeason``'s
fields the same way.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class AdpSeason:
    season: int
    format: str
    teams: int
    df: pd.DataFrame          # resolved ADP rows (-> adp table)
    n_players: int
    ranking_eligible: bool
    sim_eligible: bool
