"""Tests for expanding-season cross-validation folds."""
from src.projection.folds import make_expanding_folds


def test_no_leakage():
    folds = make_expanding_folds([2018, 2019, 2020, 2021, 2022], min_train_seasons=3)
    for f in folds:
        assert f.test_season not in f.train_seasons
        assert all(s < f.test_season for s in f.train_seasons)


def test_warmup_respected():
    folds = make_expanding_folds([2018, 2019, 2020, 2021], min_train_seasons=3)
    assert len(folds) == 1
    assert folds[0].test_season == 2021
    assert folds[0].train_seasons == [2018, 2019, 2020]


def test_expanding_grows():
    folds = make_expanding_folds(range(2012, 2026), min_train_seasons=5)
    for i in range(1, len(folds)):
        assert len(folds[i].train_seasons) > len(folds[i - 1].train_seasons)


def test_empty_when_insufficient_seasons():
    folds = make_expanding_folds([2020, 2021], min_train_seasons=5)
    assert folds == []


def test_deduplicates_seasons():
    folds = make_expanding_folds([2020, 2020, 2021, 2021, 2022, 2023, 2024, 2025],
                                 min_train_seasons=3)
    for f in folds:
        assert len(f.train_seasons) == len(set(f.train_seasons))
