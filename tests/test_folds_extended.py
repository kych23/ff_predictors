"""Black-box extensions for expanding-season CV folds."""
from src.projection.folds import make_expanding_folds


def test_train_is_all_strictly_prior_seasons():
    seasons = [2017, 2018, 2019, 2020, 2021, 2022]
    folds = make_expanding_folds(seasons, min_train_seasons=3)
    for f in folds:
        assert f.train_seasons == [s for s in seasons if s < f.test_season]


def test_fold_count_matches_seasons_minus_warmup():
    seasons = list(range(2015, 2025))  # 10 seasons
    folds = make_expanding_folds(seasons, min_train_seasons=4)
    assert len(folds) == 10 - 4


def test_last_fold_tests_max_season():
    folds = make_expanding_folds([2018, 2019, 2020, 2021, 2022], min_train_seasons=2)
    assert folds[-1].test_season == 2022


def test_first_fold_uses_exactly_warmup_seasons():
    folds = make_expanding_folds([2018, 2019, 2020, 2021, 2022], min_train_seasons=2)
    assert len(folds[0].train_seasons) == 2
    assert folds[0].test_season == 2020


def test_single_fold_when_warmup_is_n_minus_one():
    folds = make_expanding_folds([2019, 2020, 2021, 2022], min_train_seasons=3)
    assert len(folds) == 1


def test_empty_when_warmup_equals_season_count():
    assert make_expanding_folds([2020, 2021, 2022], min_train_seasons=3) == []


def test_empty_input():
    assert make_expanding_folds([], min_train_seasons=3) == []


def test_test_seasons_unique_and_increasing():
    folds = make_expanding_folds(range(2012, 2026), min_train_seasons=5)
    test_seasons = [f.test_season for f in folds]
    assert test_seasons == sorted(set(test_seasons))
