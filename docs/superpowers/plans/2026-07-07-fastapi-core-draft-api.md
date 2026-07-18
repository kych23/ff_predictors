# FastAPI Core Draft API (Manual Mode) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the existing VONA draft recommender as a REST API with DB-backed, resumable draft sessions (manual pick entry), so the Next.js draft room (Plan 2) has a complete backend.

**Architecture:** New `api/` package (FastAPI) imports the existing `src/` engine directly — no logic is ported. Draft sessions are event-sourced exactly like `scripts/draft.py`: a session row holds a JSON `history` (list of commands; each command is a list of events `["pick", player_id, mine]` or `["skip", token]`); state is always rebuilt by pure replay. Board assembly and replay logic are extracted from the CLI into shared modules so CLI and API cannot drift.

**Tech Stack:** Python 3.13, FastAPI, SQLAlchemy 2.x (existing `Base`/`SessionLocal`), pandas engine calls (`recommend`, `RosterState`), pytest + httpx TestClient. Tests use in-memory SQLite (new tables avoid PG-only types).

## Global Constraints

- Python via repo venv: run everything as `venv/bin/pytest`, `venv/bin/python`, `venv/bin/pip` from repo root `/Users/kych2204/cs/personal/ff_predictors`.
- Full suite must pass before every commit: `venv/bin/pytest` (~30s).
- ADP wall: `src/projection/` and `src/features/` must never import `src/ingest/adp.py`. Nothing in this plan touches those directories; if you think you need to, stop — the plan is wrong.
- New product tables (`draft_sessions`) use only portable column types (`String`, `Integer`, `DateTime`, `JSON`) — never `JSONB` — so they can be created on SQLite in tests.
- History format is a shared contract with `scripts/draft.py` save files: `history: list[command]`, `command: list[event]`, `event: ["pick", player_id, mine] | ["skip", token]`. Do not change it.
- Board column contract (from `src/recommender/board.py` after Task 1): `player_id, position, p10, p50, p90, adp, adp_stdev, name, team` (+ `bye_week` when schedules available).
- SQLAlchemy `JSON` columns do not track in-place mutation: always **reassign** (`sess.history = sess.history + [command]`), never `.append()`.
- Commit messages: terse lowercase subject matching `git log` style, with trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- `config/league.yaml` is not modified by this plan.

---

### Task 1: Extract board assembly into `src/recommender/board.py`

The CLI's `_load_board`/`_compute_bye_weeks` become a shared module so the API uses the identical board. Pure refactor — behavior unchanged, existing suite is the test.

**Files:**

- Create: `src/recommender/board.py`
- Modify: `scripts/draft.py` (delete `_compute_bye_weeks` and `_load_board`, import from new module)

**Interfaces:**

- Consumes: `src.db.loaders.load_projections_df/load_adp_df/load_players_df`, `src.db.session.session_scope`, `src.ingest.sources.load_schedules`
- Produces: `load_board(season: int, cfg: LeagueConfig) -> pd.DataFrame` and `compute_bye_weeks(season: int) -> dict` — used by `scripts/draft.py` (now) and `api/draft_service.py` (Task 4). `PROJ_COLS: list[str]` constant.

- [ ] **Step 1: Create `src/recommender/board.py`**

Move the two functions verbatim from `scripts/draft.py:33-71`, renaming `_compute_bye_weeks` → `compute_bye_weeks` and `_load_board` → `load_board`, and add the projection-column constant used at `scripts/draft.py:243-245`:

```python
"""Draft board assembly: projections + ADP + names + bye weeks.

Shared by the draft CLI (scripts/draft.py) and the draft API (api/) so the two
can never disagree about what a board row looks like. Uses ADP only on the
recommender side of the wall (survival/timing), per DrafterSpec §4.0.
"""
from __future__ import annotations

import sys

import pandas as pd

from src.config import LeagueConfig
from src.db.loaders import load_adp_df, load_players_df, load_projections_df
from src.db.session import session_scope

#: Columns recommend() consumes, in preference order; filter to what the board has.
PROJ_COLS = ["player_id", "position", "p10", "p50", "p90",
             "adp", "adp_stdev", "team", "bye_week"]


def compute_bye_weeks(season: int) -> dict:
    """Return {team_abbr: bye_week_number} from nflreadpy schedules. Empty dict on failure."""
    try:
        from src.ingest.sources import load_schedules
        sched = load_schedules([season])
        if sched.empty:
            return {}
        if "game_type" in sched.columns:
            sched = sched[sched["game_type"] == "REG"]
        all_weeks = set(pd.to_numeric(sched["week"], errors="coerce").dropna().astype(int))
        teams = set(sched["home_team"].dropna()) | set(sched["away_team"].dropna())
        bye: dict = {}
        for team in teams:
            played = (
                set(pd.to_numeric(sched.loc[sched["home_team"] == team, "week"],
                                  errors="coerce").dropna().astype(int))
                | set(pd.to_numeric(sched.loc[sched["away_team"] == team, "week"],
                                    errors="coerce").dropna().astype(int))
            )
            missing = sorted(all_weeks - played)
            if missing:
                bye[team] = missing[0]
        return bye
    except Exception as exc:
        print(f"warning: bye weeks unavailable ({exc})", file=sys.stderr)
        return {}


def load_board(season: int, cfg: LeagueConfig) -> pd.DataFrame:
    with session_scope() as session:
        proj = load_projections_df(session, season)[
            ["player_id", "position", "p10", "p50", "p90"]]
        adp = load_adp_df(session, cfg, season)[["player_id", "adp", "adp_stdev"]]
        names = load_players_df(session)
    board = proj.merge(adp, on="player_id", how="left").merge(names, on="player_id", how="left")
    bye_map = compute_bye_weeks(season)
    if bye_map:
        board["bye_week"] = board["team"].map(bye_map).where(board["team"].notna())
    return board
```

- [ ] **Step 2: Update `scripts/draft.py`**

Delete `_compute_bye_weeks` (lines 33-58) and `_load_board` (lines 61-71). Change the imports block to add:

```python
from src.recommender.board import PROJ_COLS, load_board
```

Replace the call site `board = _load_board(args.season, cfg)` with:

```python
    board = load_board(args.season, cfg)
```

Replace the inline column list at lines 243-245:

```python
    _ALL_PROJ_COLS = ["player_id", "position", "p10", "p50", "p90",
                      "adp", "adp_stdev", "team", "bye_week"]
    proj_cols = [c for c in _ALL_PROJ_COLS if c in board.columns]
```

with:

```python
    proj_cols = [c for c in PROJ_COLS if c in board.columns]
```

- [ ] **Step 3: Verify the CLI still imports and the suite passes**

Run: `venv/bin/python -c "import scripts.draft" && venv/bin/pytest`
Expected: import succeeds (argparse only runs under `__main__`), full suite PASSES.

- [ ] **Step 4: Commit**

```bash
git add src/recommender/board.py scripts/draft.py
git commit -m "extract draft board assembly into src/recommender/board.py"
```

---

### Task 2: Shared replay engine in `api/replay.py`

Event replay currently lives in closures inside `scripts/draft.py:176-207`. Extract a pure function the API can use; refactor the CLI onto it.

**Files:**

- Create: `api/__init__.py` (empty), `api/replay.py`
- Modify: `scripts/draft.py:176-207` (`_apply_event`, `_rebuild`)
- Test: `tests/test_api_replay.py`

**Interfaces:**

- Consumes: `RosterState` (`src/recommender/roster_state.py`), board DataFrame per the board column contract
- Produces:
  - `apply_event(state: RosterState, board: pd.DataFrame, ev: list) -> None`
  - `replay_history(history: list, board: pd.DataFrame, cfg: LeagueConfig, draft_position: int) -> RosterState`
  - Used by `api/draft_service.py` (Task 4) and `scripts/draft.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api_replay.py`:

```python
"""Replay is a pure function of (history, board): picks land on the right roster,
skips consume a pick slot, and replay order matches incremental application."""
import pandas as pd

from api.replay import apply_event, replay_history
from src.config import load_config


def make_board():
    rows = []
    pid = 0
    for pos in ["QB", "RB", "WR", "TE"]:
        for i in range(30):
            pid += 1
            p50 = 250.0 - i * 6
            rows.append({"player_id": f"P{pid:04d}", "name": f"{pos} {i+1}",
                         "position": pos, "team": f"T{(pid % 32) + 1}",
                         "p10": p50 * 0.7, "p50": p50, "p90": p50 * 1.3,
                         "adp": float(pid), "adp_stdev": 6.0,
                         "bye_week": (pid % 14) + 1})
    return pd.DataFrame(rows)


def test_replay_records_my_pick_with_metadata():
    cfg = load_config()
    board = make_board()
    history = [[["pick", "P0031", True]]]  # first RB row
    state = replay_history(history, board, cfg, draft_position=1)
    assert "P0031" in state.drafted
    assert len(state.my_roster) == 1
    mine = state.my_roster[0]
    assert mine["position"] == "RB"
    assert mine["team"] == board.loc[board.player_id == "P0031", "team"].iloc[0]
    assert mine["bye_week"] == int(board.loc[board.player_id == "P0031", "bye_week"].iloc[0])


def test_replay_opponent_pick_not_on_my_roster():
    cfg = load_config()
    state = replay_history([[["pick", "P0031", False]]], make_board(), cfg, draft_position=1)
    assert "P0031" in state.drafted
    assert state.my_roster == []


def test_skip_consumes_a_pick_slot():
    cfg = load_config()
    state = replay_history([[["skip", "_skip_1"]]], make_board(), cfg, draft_position=1)
    assert state.current_overall_pick() == 2
    assert state.my_roster == []


def test_unknown_player_id_still_recorded():
    cfg = load_config()
    state = replay_history([[["pick", "GHOST", False]]], make_board(), cfg, draft_position=1)
    assert "GHOST" in state.drafted


def test_replay_equals_incremental_application():
    cfg = load_config()
    board = make_board()
    history = [[["pick", "P0031", False]], [["pick", "P0061", True]], [["skip", "_skip_3"]]]
    replayed = replay_history(history, board, cfg, draft_position=2)
    from src.recommender.roster_state import RosterState
    incremental = RosterState(cfg=cfg, draft_position=2)
    for command in history:
        for ev in command:
            apply_event(incremental, board, ev)
    assert replayed.drafted == incremental.drafted
    assert replayed.my_roster == incremental.my_roster
    assert replayed.slot_fill == incremental.slot_fill
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_api_replay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api'`

- [ ] **Step 3: Implement `api/replay.py`** (and empty `api/__init__.py`)

```python
"""Rebuild draft state by replaying an event history.

The history format is shared with the scripts/draft.py save file:
history = list[command]; command = list[event];
event = ["pick", player_id, mine] | ["skip", token].
State is always a pure function of (history, board, draft_position) — undo is
"pop the last command and replay", never per-action reversal.
"""
from __future__ import annotations

import pandas as pd

from src.config import LeagueConfig
from src.recommender.roster_state import RosterState


def apply_event(state: RosterState, board: pd.DataFrame, ev: list) -> None:
    kind = ev[0]
    if kind == "skip":
        state.drafted.add(ev[1])
        return
    pid, mine = ev[1], ev[2]
    prow = board.loc[board["player_id"] == pid]
    if prow.empty:
        state.record_pick(pid, None, mine=mine)
        return
    prow = prow.iloc[0]
    team_val = prow["team"] if "team" in board.columns and pd.notna(prow.get("team")) else None
    bye_val = (int(prow["bye_week"]) if "bye_week" in board.columns
               and pd.notna(prow.get("bye_week")) else None)
    state.record_pick(pid, prow["position"], mine=mine, team=team_val, bye_week=bye_val)


def replay_history(history: list, board: pd.DataFrame, cfg: LeagueConfig,
                   draft_position: int) -> RosterState:
    state = RosterState(cfg=cfg, draft_position=draft_position)
    for command in history:
        for ev in command:
            apply_event(state, board, ev)
    return state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_api_replay.py -v`
Expected: 5 PASS

- [ ] **Step 5: Refactor `scripts/draft.py` onto the shared engine**

Add import:

```python
from api.replay import apply_event, replay_history
```

Replace the `_apply_event` closure (lines 176-193) with a thin wrapper that also feeds the ManualDraftSource's drafted set:

```python
    def _apply_event(ev: list) -> None:
        token = ev[1]
        src.drafted.add(token)
        apply_event(state, board, ev)
```

Replace `_rebuild` (lines 202-207) with:

```python
    def _rebuild() -> None:
        state.reset()
        src.drafted.clear()
        for command in history:
            for ev in command:
                _apply_event(ev)
```

(`_rebuild` keeps the wrapper so `src.drafted` stays in sync; the state-mutation logic itself now lives only in `api.replay`.)

- [ ] **Step 6: Run the full suite**

Run: `venv/bin/python -c "import scripts.draft" && venv/bin/pytest`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add api/__init__.py api/replay.py scripts/draft.py tests/test_api_replay.py
git commit -m "add shared draft replay engine, wire cli onto it"
```

---

### Task 3: `DraftSession` model + test DB fixtures

DB-backed sessions replace `draft_state_*.json` files. One table, portable types, history as JSON.

**Files:**

- Create: `api/db_models.py`
- Create: `tests/conftest.py`
- Test: `tests/test_api_models.py`

**Interfaces:**

- Consumes: `Base` from `src/db/models.py`
- Produces: `DraftSession` ORM class with columns `session_id (str pk, uuid4-hex default), season (int), draft_position (int), platform (str, default "manual"), status (str, default "active"), snapshot_id (str, nullable), history (JSON, default list), created_at, updated_at`. Fixtures `sqlite_engine`, `db_session` in `tests/conftest.py` used by Tasks 4-5.

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_models.py`:

```python
"""DraftSession persists on SQLite (portable types only) and history round-trips."""
from api.db_models import DraftSession


def test_session_roundtrip_with_history(db_session):
    s = DraftSession(season=2026, draft_position=4)
    db_session.add(s)
    db_session.commit()
    assert s.session_id and len(s.session_id) == 32
    assert s.platform == "manual"
    assert s.status == "active"
    assert s.history == []

    s.history = s.history + [[["pick", "P0001", True]]]
    db_session.commit()
    db_session.expire_all()

    loaded = db_session.get(DraftSession, s.session_id)
    assert loaded.history == [[["pick", "P0001", True]]]
    assert loaded.created_at is not None
```

Create `tests/conftest.py`:

```python
"""Shared API-test fixtures: in-memory SQLite bound to the shared Base metadata.

Only product tables (draft_sessions) are created — research tables use JSONB
and are Postgres-only by design.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.db_models import DraftSession
from src.db.models import Base


@pytest.fixture()
def sqlite_engine():
    eng = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(eng, tables=[DraftSession.__table__])
    yield eng
    eng.dispose()


@pytest.fixture()
def db_session(sqlite_engine):
    factory = sessionmaker(bind=sqlite_engine, autoflush=False, future=True)
    session = factory()
    yield session
    session.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_api_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.db_models'`

- [ ] **Step 3: Implement `api/db_models.py`**

```python
"""Product-side tables (draft sessions).

Kept out of src/db/models.py — that module owns the research schema — but they
share its Base so one metadata covers both. Product tables use only portable
column types (JSON, not JSONB) so tests can create them on SQLite.

snapshot_id records which ingest snapshot's projections served the draft
(provenance, invariant #3); nullable because a session can be created before
any pipeline run in dev environments.
"""
from __future__ import annotations

import uuid

from sqlalchemy import JSON, Column, DateTime, Integer, String
from sqlalchemy.sql import func

from src.db.models import Base


def _new_session_id() -> str:
    return uuid.uuid4().hex


class DraftSession(Base):
    __tablename__ = "draft_sessions"

    session_id = Column(String, primary_key=True, default=_new_session_id)
    season = Column(Integer, nullable=False)
    draft_position = Column(Integer, nullable=False)
    platform = Column(String, nullable=False, default="manual")
    status = Column(String, nullable=False, default="active")
    snapshot_id = Column(String, nullable=True)
    history = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(),
                        onupdate=func.now())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_api_models.py -v && venv/bin/pytest`
Expected: PASS (new test and full suite — conftest must not break existing tests)

- [ ] **Step 5: Commit**

```bash
git add api/db_models.py tests/conftest.py tests/test_api_models.py
git commit -m "add draft session table and sqlite test fixtures"
```

---

### Task 4: Draft service

Orchestration layer: session CRUD, pick/skip/undo via history, state snapshots, recommendations. Framework-free (no FastAPI imports) so it tests without HTTP.

**Files:**

- Create: `api/draft_service.py`
- Test: `tests/test_api_service.py`

**Interfaces:**

- Consumes: `DraftSession` (Task 3), `replay_history` (Task 2), `load_board`/`PROJ_COLS` (Task 1), `recommend`/`build_replacement_from_projections` (`src/recommender/recommend.py`), `load_config`
- Produces (consumed by routers in Task 5):
  - exceptions `DraftNotFound(Exception)`, `InvalidPick(Exception)`
  - `DraftService(db: Session, cfg: LeagueConfig, board_for: Callable[[int], pd.DataFrame])`
    - `create_session(season: int, draft_position: int, snapshot_id: str | None = None) -> DraftSession`
    - `state(session_id: str) -> dict`
    - `record_pick(session_id: str, player_id: str | None = None, skip: bool = False, mine: bool | None = None) -> dict`
    - `undo(session_id: str) -> dict`
    - `recommendations(session_id: str, top_n: int = 10) -> list[dict]`
  - `get_cached_board(season: int) -> pd.DataFrame` (module-level, process cache over `load_board`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api_service.py`:

```python
"""Draft service contracts: session lifecycle, snake-aware mine-defaulting,
pick validation, undo-by-replay, and recommendation shape."""
import pytest

from api.draft_service import DraftNotFound, DraftService, InvalidPick
from src.config import load_config
from tests.test_api_replay import make_board


@pytest.fixture()
def svc(db_session):
    cfg = load_config()
    board = make_board()
    return DraftService(db=db_session, cfg=cfg, board_for=lambda season: board)


def test_create_and_state(svc):
    cfg = load_config()
    s = svc.create_session(season=2026, draft_position=1)
    st = svc.state(s.session_id)
    assert st["session_id"] == s.session_id
    assert st["teams"] == cfg.teams
    assert st["rounds"] == cfg.roster.rounds
    assert st["current_overall_pick"] == 1
    assert st["is_my_turn"] is True          # slot 1 owns pick 1
    assert st["picks"] == []
    assert st["my_roster"] == []


def test_create_rejects_bad_position(svc):
    cfg = load_config()
    with pytest.raises(InvalidPick):
        svc.create_session(season=2026, draft_position=cfg.teams + 1)


def test_state_unknown_session_raises(svc):
    with pytest.raises(DraftNotFound):
        svc.state("nope")


def test_pick_defaults_mine_from_snake(svc):
    s = svc.create_session(season=2026, draft_position=1)
    st = svc.record_pick(s.session_id, player_id="P0031")  # my turn -> mine
    assert st["picks"][0]["mine"] is True
    assert st["my_roster"][0]["player_id"] == "P0031"
    st = svc.record_pick(s.session_id, player_id="P0032")  # opponent turn
    assert st["picks"][1]["mine"] is False
    assert len(st["my_roster"]) == 1


def test_pick_rejects_unknown_and_duplicate(svc):
    s = svc.create_session(season=2026, draft_position=1)
    with pytest.raises(InvalidPick):
        svc.record_pick(s.session_id, player_id="GHOST")
    svc.record_pick(s.session_id, player_id="P0031")
    with pytest.raises(InvalidPick):
        svc.record_pick(s.session_id, player_id="P0031")


def test_skip_advances_without_player(svc):
    s = svc.create_session(season=2026, draft_position=2)
    st = svc.record_pick(s.session_id, skip=True)
    assert st["current_overall_pick"] == 2
    assert st["picks"][0]["skipped"] is True
    assert st["picks"][0]["player_id"] is None


def test_undo_pops_last_command(svc):
    s = svc.create_session(season=2026, draft_position=1)
    svc.record_pick(s.session_id, player_id="P0031")
    st = svc.record_pick(s.session_id, player_id="P0032")
    assert st["current_overall_pick"] == 3
    st = svc.undo(s.session_id)
    assert st["current_overall_pick"] == 2
    assert [p["player_id"] for p in st["picks"]] == ["P0031"]
    # undo on empty history is a no-op
    svc.undo(s.session_id)
    st = svc.undo(s.session_id)
    assert st["picks"] == []


def test_recommendations_shape_and_availability(svc):
    s = svc.create_session(season=2026, draft_position=1)
    svc.record_pick(s.session_id, player_id="P0031")
    recs = svc.recommendations(s.session_id, top_n=5)
    assert 0 < len(recs) <= 5
    drafted_ids = {"P0031"}
    for r in recs:
        assert r["player_id"] not in drafted_ids
        assert set(r) >= {"player_id", "name", "position", "vona_score", "value",
                          "p10", "p50", "p90", "draft_round", "target_quantile",
                          "forced_completion"}
    scores = [r["vona_score"] for r in recs]
    assert scores == sorted(scores, reverse=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_api_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.draft_service'`

- [ ] **Step 3: Implement `api/draft_service.py`**

```python
"""Draft session orchestration: the API-side equivalent of scripts/draft.py's
event loop. State is never stored — only history is; every read replays.

Board/replacement are cached per season at module level: projections change at
most daily (pipeline runs), while a live draft polls every few seconds.
"""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from api.db_models import DraftSession
from api.replay import replay_history
from src.config import LeagueConfig
from src.recommender.board import PROJ_COLS, load_board
from src.recommender.recommend import build_replacement_from_projections, recommend
from src.recommender.roster_state import RosterState


class DraftNotFound(Exception):
    pass


class InvalidPick(Exception):
    pass


_BOARD_CACHE: dict[int, pd.DataFrame] = {}


def get_cached_board(season: int, cfg: Optional[LeagueConfig] = None) -> pd.DataFrame:
    from src.config import load_config
    if season not in _BOARD_CACHE:
        _BOARD_CACHE[season] = load_board(season, cfg or load_config())
    return _BOARD_CACHE[season]


class DraftService:
    def __init__(self, db, cfg: LeagueConfig,
                 board_for: Callable[[int], pd.DataFrame]):
        self.db = db
        self.cfg = cfg
        self.board_for = board_for
        self._replacement_cache: dict[int, object] = {}

    # --- session lifecycle ---

    def create_session(self, season: int, draft_position: int,
                       snapshot_id: Optional[str] = None) -> DraftSession:
        if not (1 <= draft_position <= self.cfg.teams):
            raise InvalidPick(
                f"draft_position must be in 1..{self.cfg.teams}, got {draft_position}")
        sess = DraftSession(season=season, draft_position=draft_position,
                            snapshot_id=snapshot_id)
        self.db.add(sess)
        self.db.commit()
        return sess

    def _get(self, session_id: str) -> DraftSession:
        sess = self.db.get(DraftSession, session_id)
        if sess is None:
            raise DraftNotFound(session_id)
        return sess

    # --- state ---

    def _rebuild(self, sess: DraftSession) -> tuple[RosterState, pd.DataFrame]:
        board = self.board_for(sess.season)
        state = replay_history(sess.history, board, self.cfg, sess.draft_position)
        return state, board

    def state(self, session_id: str) -> dict:
        sess = self._get(session_id)
        state, board = self._rebuild(sess)
        names = dict(zip(board["player_id"], board["name"]))
        picks = []
        n = 0
        for command in sess.history:
            for ev in command:
                n += 1
                if ev[0] == "skip":
                    picks.append({"pick_number": n, "player_id": None, "name": None,
                                  "mine": False, "skipped": True})
                else:
                    picks.append({"pick_number": n, "player_id": ev[1],
                                  "name": names.get(ev[1]), "mine": bool(ev[2]),
                                  "skipped": False})
        cur = state.current_overall_pick()
        return {
            "session_id": sess.session_id,
            "season": sess.season,
            "draft_position": sess.draft_position,
            "platform": sess.platform,
            "status": sess.status,
            "teams": self.cfg.teams,
            "rounds": self.cfg.roster.rounds,
            "my_picks": state.my_picks,
            "current_overall_pick": cur,
            "is_my_turn": cur in state.my_picks,
            "next_my_pick": state.next_my_pick(),
            "remaining_picks": state.remaining_picks(),
            "picks": picks,
            "my_roster": [dict(p, name=names.get(p["player_id"])) for p in state.my_roster],
            "open_starters": state.unfilled_mandatory_slots(),
        }

    # --- mutations (history append + full replay; mirrors the CLI's undo stack) ---

    def record_pick(self, session_id: str, player_id: Optional[str] = None,
                    skip: bool = False, mine: Optional[bool] = None) -> dict:
        sess = self._get(session_id)
        state, board = self._rebuild(sess)
        cur = state.current_overall_pick()
        if skip:
            command = [["skip", f"_skip_{cur}"]]
        else:
            if player_id is None:
                raise InvalidPick("player_id required unless skip=true")
            if player_id not in set(board["player_id"]):
                raise InvalidPick(f"unknown player_id {player_id!r}")
            if player_id in state.drafted:
                raise InvalidPick(f"{player_id!r} already drafted")
            if mine is None:
                mine = cur in state.my_picks
            command = [["pick", player_id, bool(mine)]]
        sess.history = sess.history + [command]   # reassign: JSON col, no mutation tracking
        self.db.commit()
        return self.state(session_id)

    def undo(self, session_id: str) -> dict:
        sess = self._get(session_id)
        if sess.history:
            sess.history = sess.history[:-1]
            self.db.commit()
        return self.state(session_id)

    # --- recommendations ---

    def recommendations(self, session_id: str, top_n: int = 10) -> list[dict]:
        sess = self._get(session_id)
        state, board = self._rebuild(sess)
        if sess.season not in self._replacement_cache:
            self._replacement_cache[sess.season] = \
                build_replacement_from_projections(board, cfg=self.cfg)
        replacement = self._replacement_cache[sess.season]
        proj_cols = [c for c in PROJ_COLS if c in board.columns]
        avail = board[proj_cols][~board["player_id"].isin(state.drafted)].copy()
        recs = recommend(avail, state, replacement, cfg=self.cfg, top_n=top_n)
        if recs.empty:
            return []
        names = dict(zip(board["player_id"], board["name"]))
        out = recs.assign(name=recs["player_id"].map(names))
        out = out.where(pd.notna(out), None)     # NaN -> None for JSON
        return out.to_dict(orient="records")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_api_service.py tests/test_api_replay.py -v && venv/bin/pytest`
Expected: PASS (service tests and full suite)

- [ ] **Step 5: Commit**

```bash
git add api/draft_service.py tests/test_api_service.py
git commit -m "add draft service: sessions, picks, undo, recommendations"
```

---

### Task 5: FastAPI app, schemas, routers

HTTP layer only: pydantic schemas, dependency wiring, domain-exception → status-code mapping.

**Files:**

- Create: `api/main.py`, `api/deps.py`, `api/schemas.py`, `api/routers/__init__.py`, `api/routers/players.py`, `api/routers/draft.py`
- Modify: `requirements.txt` (add fastapi, uvicorn, httpx)
- Test: `tests/test_api_endpoints.py`

**Interfaces:**

- Consumes: `DraftService`, `DraftNotFound`, `InvalidPick`, `get_cached_board` (Task 4); `SessionLocal` (`src/db/session.py`); `latest_snapshot_id` (`src/db/session.py`)
- Produces HTTP API (consumed by Plan 2 frontend):
  - `GET /health` → `{"status": "ok"}`
  - `GET /players?season=` → `list[PlayerOut]`
  - `POST /draft/sessions` (`SessionCreate`) → `StateOut` (201)
  - `GET /draft/sessions/{session_id}` → `StateOut`
  - `POST /draft/sessions/{session_id}/picks` (`PickIn`) → `StateOut`
  - `POST /draft/sessions/{session_id}/undo` → `StateOut`
  - `GET /draft/sessions/{session_id}/recommendations?top_n=` → `list[RecommendationOut]`
  - Errors: 404 unknown session, 400 invalid pick/position, 422 malformed body (FastAPI default)
- Overridable deps for tests: `get_db`, `get_board_for`, `get_snapshot_id`

- [ ] **Step 1: Install HTTP dependencies**

Run: `venv/bin/pip install fastapi 'uvicorn[standard]' httpx`
Append to `requirements.txt`:

```
fastapi
uvicorn[standard]
httpx
```

(`requirements.lock` is a `pip freeze` of the validated env — regenerate it in Task 6, not here.)

- [ ] **Step 2: Write the failing tests**

Create `tests/test_api_endpoints.py`:

```python
"""HTTP contract tests: status codes, response shapes, dependency overrides.
No real DB, no network — SQLite + synthetic board via dependency injection."""
import pytest
from fastapi.testclient import TestClient

from api.deps import get_board_for, get_db, get_snapshot_id
from api.main import create_app
from tests.test_api_replay import make_board


@pytest.fixture()
def client(db_session):
    app = create_app()
    board = make_board()
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_board_for] = lambda: (lambda season: board)
    app.dependency_overrides[get_snapshot_id] = lambda: None
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_players_listing(client):
    r = client.get("/players", params={"season": 2026})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 120
    assert {"player_id", "name", "position", "p10", "p50", "p90"} <= set(rows[0])


def test_session_lifecycle(client):
    r = client.post("/draft/sessions", json={"season": 2026, "draft_position": 1})
    assert r.status_code == 201
    sid = r.json()["session_id"]
    assert r.json()["is_my_turn"] is True

    r = client.post(f"/draft/sessions/{sid}/picks", json={"player_id": "P0031"})
    assert r.status_code == 200
    assert r.json()["picks"][0]["mine"] is True

    r = client.get(f"/draft/sessions/{sid}/recommendations", params={"top_n": 5})
    assert r.status_code == 200
    recs = r.json()
    assert 0 < len(recs) <= 5
    assert "P0031" not in [x["player_id"] for x in recs]

    r = client.post(f"/draft/sessions/{sid}/undo")
    assert r.status_code == 200
    assert r.json()["picks"] == []

    r = client.get(f"/draft/sessions/{sid}")
    assert r.status_code == 200
    assert r.json()["current_overall_pick"] == 1


def test_error_mapping(client):
    assert client.get("/draft/sessions/nope").status_code == 404
    r = client.post("/draft/sessions", json={"season": 2026, "draft_position": 999})
    assert r.status_code == 400
    r = client.post("/draft/sessions", json={"season": 2026, "draft_position": 1})
    sid = r.json()["session_id"]
    assert client.post(f"/draft/sessions/{sid}/picks",
                       json={"player_id": "GHOST"}).status_code == 400
    assert client.post(f"/draft/sessions/{sid}/picks", json={}).status_code == 400
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `venv/bin/pytest tests/test_api_endpoints.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.deps'`

- [ ] **Step 4: Implement the HTTP layer**

`api/deps.py`:

```python
"""FastAPI dependencies — every external resource enters through one of these
so tests can override them individually."""
from __future__ import annotations

from typing import Callable, Iterator, Optional

import pandas as pd
from fastapi import Depends

from api.draft_service import DraftService, get_cached_board
from src.config import LeagueConfig, load_config
from src.db.session import SessionLocal


def get_db() -> Iterator:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_cfg() -> LeagueConfig:
    return load_config()


def get_board_for() -> Callable[[int], pd.DataFrame]:
    return get_cached_board


def get_snapshot_id() -> Optional[str]:
    """Best-effort provenance: which ingest snapshot's projections serve this draft."""
    try:
        from src.db.session import latest_snapshot_id
        return latest_snapshot_id()
    except Exception:
        return None


def get_service(db=Depends(get_db), cfg: LeagueConfig = Depends(get_cfg),
                board_for=Depends(get_board_for)) -> DraftService:
    return DraftService(db=db, cfg=cfg, board_for=board_for)
```

`api/schemas.py`:

```python
"""Pydantic wire schemas. Field sets mirror the dicts DraftService returns —
schemas validate the contract, the service stays framework-free."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class PlayerOut(BaseModel):
    player_id: str
    name: Optional[str] = None
    team: Optional[str] = None
    position: str
    p10: float
    p50: float
    p90: float
    adp: Optional[float] = None
    bye_week: Optional[int] = None


class SessionCreate(BaseModel):
    season: int
    draft_position: int


class PickIn(BaseModel):
    player_id: Optional[str] = None
    skip: bool = False
    mine: Optional[bool] = None


class PickOut(BaseModel):
    pick_number: int
    player_id: Optional[str]
    name: Optional[str]
    mine: bool
    skipped: bool


class RosterEntryOut(BaseModel):
    player_id: str
    name: Optional[str] = None
    position: Optional[str] = None
    team: Optional[str] = None
    bye_week: Optional[int] = None


class StateOut(BaseModel):
    session_id: str
    season: int
    draft_position: int
    platform: str
    status: str
    teams: int
    rounds: int
    my_picks: list[int]
    current_overall_pick: int
    is_my_turn: bool
    next_my_pick: Optional[int]
    remaining_picks: int
    picks: list[PickOut]
    my_roster: list[RosterEntryOut]
    open_starters: dict[str, int]


class RecommendationOut(BaseModel):
    player_id: str
    name: Optional[str]
    position: str
    team: Optional[str] = None
    vona_score: float
    value: float
    p10: float
    p50: float
    p90: float
    adp: Optional[float] = None
    draft_round: int
    target_quantile: float
    forced_completion: bool
```

`api/routers/__init__.py`: empty file.

`api/routers/players.py`:

```python
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.deps import get_board_for
from api.schemas import PlayerOut

router = APIRouter()


@router.get("/players", response_model=list[PlayerOut])
def list_players(season: int, board_for=Depends(get_board_for)):
    import pandas as pd
    board = board_for(season)
    return board.where(pd.notna(board), None).to_dict(orient="records")
```

`api/routers/draft.py`:

```python
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.deps import get_service, get_snapshot_id
from api.draft_service import DraftService
from api.schemas import PickIn, RecommendationOut, SessionCreate, StateOut

router = APIRouter(prefix="/draft/sessions")


@router.post("", response_model=StateOut, status_code=201)
def create_session(body: SessionCreate, svc: DraftService = Depends(get_service),
                   snapshot_id=Depends(get_snapshot_id)):
    sess = svc.create_session(season=body.season, draft_position=body.draft_position,
                              snapshot_id=snapshot_id)
    return svc.state(sess.session_id)


@router.get("/{session_id}", response_model=StateOut)
def get_state(session_id: str, svc: DraftService = Depends(get_service)):
    return svc.state(session_id)


@router.post("/{session_id}/picks", response_model=StateOut)
def record_pick(session_id: str, body: PickIn,
                svc: DraftService = Depends(get_service)):
    return svc.record_pick(session_id, player_id=body.player_id,
                           skip=body.skip, mine=body.mine)


@router.post("/{session_id}/undo", response_model=StateOut)
def undo(session_id: str, svc: DraftService = Depends(get_service)):
    return svc.undo(session_id)


@router.get("/{session_id}/recommendations", response_model=list[RecommendationOut])
def recommendations(session_id: str, top_n: int = 10,
                    svc: DraftService = Depends(get_service)):
    return svc.recommendations(session_id, top_n=top_n)
```

`api/main.py`:

```python
"""App factory. Domain exceptions map to HTTP here and nowhere else."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.draft_service import DraftNotFound, InvalidPick
from api.routers import draft, players


def create_app() -> FastAPI:
    app = FastAPI(title="FantasyForecast API")

    @app.exception_handler(DraftNotFound)
    async def _not_found(request: Request, exc: DraftNotFound):
        return JSONResponse(status_code=404, content={"detail": f"session {exc} not found"})

    @app.exception_handler(InvalidPick)
    async def _invalid(request: Request, exc: InvalidPick):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/health")
    def health():
        return {"status": "ok"}

    app.include_router(players.router)
    app.include_router(draft.router)
    return app


app = create_app()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_api_endpoints.py -v && venv/bin/pytest`
Expected: PASS (endpoint tests and full suite)

- [ ] **Step 6: Commit**

```bash
git add api/ requirements.txt tests/test_api_endpoints.py
git commit -m "add fastapi draft api: players, sessions, picks, recommendations"
```

---

### Task 6: CORS, settings, entrypoint, docs

Make the API runnable against the real DB and documented for the frontend work.

**Files:**

- Create: `api/settings.py`
- Modify: `api/main.py` (CORS middleware), `README.md` (API section), `CLAUDE.md` (api/ note), `requirements.lock` (regenerate)
- Test: `tests/test_api_endpoints.py` (one added test)

**Interfaces:**

- Consumes: env var `ALLOWED_ORIGINS` (comma-separated, default `http://localhost:3000`)
- Produces: `api.settings.allowed_origins() -> list[str]`; CORS-enabled app; `venv/bin/uvicorn api.main:app --reload` run instructions

- [ ] **Step 1: Write the failing test**

Append to `tests/test_api_endpoints.py`:

```python
def test_cors_allows_frontend_origin(client):
    r = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/pytest tests/test_api_endpoints.py::test_cors_allows_frontend_origin -v`
Expected: FAIL — `access-control-allow-origin` header absent (None)

- [ ] **Step 3: Implement settings + CORS**

`api/settings.py`:

```python
"""API runtime settings from environment. Kept tiny on purpose — full
pydantic-settings is not warranted for two values."""
from __future__ import annotations

import os


def allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
    return [o.strip() for o in raw.split(",") if o.strip()]
```

In `api/main.py`, add after `app = FastAPI(...)`:

```python
    from fastapi.middleware.cors import CORSMiddleware

    from api.settings import allowed_origins

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins(),
        allow_methods=["*"],
        allow_headers=["*"],
    )
```

(with the imports moved to the top of the file alongside the existing ones).

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/pytest tests/test_api_endpoints.py -v && venv/bin/pytest`
Expected: PASS

- [ ] **Step 5: Manual smoke against the real DB**

Run: `venv/bin/uvicorn api.main:app --port 8000 &` then `curl -s localhost:8000/health` and `curl -s "localhost:8000/players?season=2026" | head -c 400`; kill the server after.
Expected: `{"status":"ok"}`; players JSON with real projections (requires `.env` `DATABASE_URL` and a completed pipeline run — if `/players` 500s on a fresh DB, that's the missing pipeline, not a code bug).

- [ ] **Step 6: Regenerate the lock and document**

Run: `venv/bin/pip freeze > requirements.lock`

Append to `README.md` under a new `## API` section:

````markdown
## API

FastAPI service exposing the draft recommender (backend for the web draft room).

​`bash
venv/bin/uvicorn api.main:app --reload --port 8000
​`

- `GET /health` — liveness
- `GET /players?season=2026` — draft board (projections + ADP + names)
- `POST /draft/sessions` `{"season": 2026, "draft_position": 4}` — start a session
- `GET /draft/sessions/{id}` — full state (picks, roster, whose turn)
- `POST /draft/sessions/{id}/picks` `{"player_id": "..."}` or `{"skip": true}` — record a pick
- `POST /draft/sessions/{id}/undo` — pop the last command
- `GET /draft/sessions/{id}/recommendations?top_n=10` — ranked VONA board

Sessions are event-sourced in the `draft_sessions` table (same history format as
the CLI save files); state is rebuilt by replay on every read.
CORS origins via `ALLOWED_ORIGINS` (default `http://localhost:3000`).
````

Append to `CLAUDE.md` stack section: `- **API:** FastAPI in api/ (uvicorn api.main:app); draft sessions event-sourced in draft_sessions table`

- [ ] **Step 7: Commit**

```bash
git add api/settings.py api/main.py README.md CLAUDE.md requirements.lock tests/test_api_endpoints.py
git commit -m "add cors, api settings, run docs"
```

---

## Self-Review Notes

- **Spec coverage (Plan 1 scope):** `/players` (Task 5), draft session create/resume-from-DB (Tasks 3-5; "resume" = GET state, no client-side files), manual picks + undo (Tasks 4-5), `/draft/recommend` equivalent (Task 4-5 recommendations), error handling for unknown session/invalid pick (Tasks 4-5), CORS for the Plan 2 frontend (Task 6). Yahoo adapter, `/explain`, SSE/polling cadence are Plans 3-4 by design.
- **Deliberate deviations from spec wording:** spec's `PlatformAdapter` protocol is introduced in Plan 3 when a second platform exists; in manual mode the "adapter" is simply the picks endpoint writing history. Introducing the protocol now would be YAGNI with one implementation. Spec's `/draft/state` and `/draft/recommend` are RESTified under `/draft/sessions/{id}/…`.
- **Type consistency check:** `make_board()` defined in `tests/test_api_replay.py`, imported by later test files; `board_for: Callable[[int], pd.DataFrame]` consistent across deps/service; `history` format identical in replay, service, model default, and CLI.
- **Snapshot invariant:** sessions carry best-effort `snapshot_id` (nullable, documented). Research upsert convention (`upsert_data.py`) intentionally not used — sessions are transactional product state, not snapshot-stamped research data.
