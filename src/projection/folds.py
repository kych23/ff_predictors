"""Expanding-by-season cross-validation (DrafterSpec.md §4.6 / §4.6.1).

A model predicting season Y is trained only on seasons < Y. OOF predictions are
emitted only after the ``min_train_seasons`` warmup (default 5) so degenerate early
folds (e.g. train-on-2012-alone) are dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class Fold:
    train_seasons: List[int]
    test_season: int


def make_expanding_folds(seasons: Sequence[int], min_train_seasons: int) -> List[Fold]:
    """One fold per evaluable test season; train = all strictly-earlier seasons.

    A test season is evaluable once at least ``min_train_seasons`` earlier seasons
    exist.
    """
    uniq = sorted(set(int(s) for s in seasons))
    folds: List[Fold] = []
    for i, test_season in enumerate(uniq):
        train = uniq[:i]
        if len(train) >= min_train_seasons:
            folds.append(Fold(train_seasons=train, test_season=test_season))
    return folds
