"""The point-in-time handle — the only door to data (§11.2).

Feature code reaches data through ``asof(season)`` and nothing else. The layer
test enforces that structurally: ``models/**`` may import ``platform.asof`` and
no other ``platform`` subpackage, which in one rule gives no-ADP-import
(ADP lives in ``platform.sources``), no-DB-access, and point-in-time discipline.

Each accessor filters by the registered availability of its columns, asserts
nothing at or after the cutoff survived, and asserts the result is non-empty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Mapping

import pandas as pd

from src.core.errors import CutoffMismatchError, DataError
from src.platform.asof import guards
from src.platform.asof.registry import availability_of

#: Loader signature: () -> full frame for the asset, unfiltered.
Loader = Callable[[], pd.DataFrame]


@dataclass
class AsOf:
    """A frozen view of the world before a cutoff.

    ``week=None`` means "before Week 1 of ``season``" — the draft-time view.
    ``week=w`` means "before kickoff of ``(season, w)``" — the in-season view
    the weekly model would use.
    """

    snapshot_id: str
    season: int
    week: int | None = None
    loaders: Mapping[str, Loader] = field(default_factory=dict)
    _kickoffs: dict[tuple[int, int], datetime] = field(default_factory=dict, repr=False)
    as_of_offset_days: int = 1

    # ---------------------------------------------------------------- cutoff
    def register_kickoffs(self, schedules: pd.DataFrame) -> None:
        """Learn kickoff timestamps from the schedule frame.

        ``cutoff`` is defined in terms of kickoffs, and the schedule carries
        them — but the schedule accessor is itself cutoff-filtered, so the
        timestamps are registered from the raw frame before any filtering.
        Schedule *metadata* (gameday, week) is ``preseason``-available; only
        the scores are outcomes.
        """
        df = schedules[["season", "week", "gameday"]].dropna()
        stamps = pd.to_datetime(df["gameday"], errors="coerce", utc=True)
        for (s, w), group in stamps.groupby(
            [pd.to_numeric(df["season"]), pd.to_numeric(df["week"])]
        ):
            if group.notna().any():
                self._kickoffs[(int(s), int(w))] = group.min().to_pydatetime()

    @property
    def cutoff(self) -> datetime:
        target = (self.season, self.week if self.week is not None else 1)
        kickoff = self._kickoffs.get(target)
        if kickoff is None:
            # No schedule registered (unit tests, or a future season): fall back
            # to a conservative Sept 1 boundary rather than silently admitting
            # everything.
            kickoff = datetime(self.season, 9, 1, tzinfo=timezone.utc)
        if self.week is None:
            return kickoff - pd.Timedelta(days=self.as_of_offset_days).to_pytimedelta()
        return kickoff

    # ------------------------------------------------------------- accessors
    def _load(self, asset: str) -> pd.DataFrame:
        loader = self.loaders.get(asset)
        if loader is None:
            raise DataError(
                f"no loader registered for {asset!r} in snapshot "
                f"{self.snapshot_id}; available: {sorted(self.loaders)}"
            )
        return loader()

    def _check_columns(self, df: pd.DataFrame, asset: str) -> None:
        """Every column read must carry an availability declaration."""
        for column in df.columns:
            availability_of(asset, column)

    def _backward(self, asset: str, *, weekly: bool) -> pd.DataFrame:
        df = self._load(asset)
        self._check_columns(df, asset)
        if weekly and self.week is not None:
            out = guards.prior_weeks(df, self.season, self.week)
            guards.assert_no_future_week(out, self.season, self.week, where=asset)
        else:
            out = guards.prior_seasons(df, self.season)
            guards.assert_no_future(out, self.season, where=asset)
        out = guards.assert_non_empty(out, where=f"{asset} @ {self.season}")
        return _tag(out, self)

    def _preseason(self, asset: str) -> pd.DataFrame:
        df = self._load(asset)
        self._check_columns(df, asset)
        past = guards.prior_seasons(df, self.season)
        current = guards.preseason_rows(df, self.season)
        out = pd.concat([past, current], ignore_index=True) if len(current) else past
        out = guards.assert_non_empty(out, where=f"{asset} @ {self.season}")
        return _tag(out, self)

    def player_stats(self) -> pd.DataFrame:
        return self._backward("player_stats", weekly=True)

    def team_stats(self) -> pd.DataFrame:
        return self._backward("team_stats", weekly=True)

    def injuries(self) -> pd.DataFrame:
        return self._backward("injuries", weekly=True)

    def rosters(self) -> pd.DataFrame:
        return self._preseason("rosters")

    def depth_charts(self) -> pd.DataFrame:
        return self._preseason("depth_charts")

    def schedules(self) -> pd.DataFrame:
        """Week-1 lines of the target season plus all prior seasons.

        Later-week lines of the target season are NOT preseason-available; that
        is why §12.5 conditions K/DST on the Week-1 line rather than per week.
        """
        return self._preseason("schedules")

    def combine(self) -> pd.DataFrame:
        return _tag(guards.assert_non_empty(self._load("combine"), where="combine"), self)

    def draft_picks(self) -> pd.DataFrame:
        return _tag(
            guards.assert_non_empty(self._load("draft_picks"), where="draft_picks"), self
        )

    def players(self) -> pd.DataFrame:
        return _tag(
            guards.assert_non_empty(self._load("players"), where="players"), self
        )


_CUTOFF_ATTR = "_ff_asof_key"


def _tag(df: pd.DataFrame, handle: AsOf) -> pd.DataFrame:
    df.attrs[_CUTOFF_ATTR] = (handle.snapshot_id, handle.season, handle.week)
    return df


def assert_same_cutoff(*frames: pd.DataFrame) -> None:
    """Joining frames from two different ``AsOf`` instances is a bug (§11.2)."""
    keys = {f.attrs.get(_CUTOFF_ATTR) for f in frames if f is not None}
    keys.discard(None)
    if len(keys) > 1:
        raise CutoffMismatchError(
            f"frames drawn from different as-of cutoffs: {sorted(keys)}. "
            f"Joining them mixes point-in-time views."
        )


def asof(season: int, *, week: int | None = None, snapshot_id: str,
         loaders: Mapping[str, Loader], as_of_offset_days: int = 1) -> AsOf:
    handle = AsOf(
        snapshot_id=snapshot_id, season=season, week=week,
        loaders=dict(loaders), as_of_offset_days=as_of_offset_days,
    )
    if "schedules" in handle.loaders:
        try:
            handle.register_kickoffs(handle.loaders["schedules"]())
        except Exception:  # noqa: BLE001 - kickoffs are an optimization
            pass
    return handle
