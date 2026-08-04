# Yahoo OAuth + Live Draft Sync (Plan 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **BLOCKED ON EXTERNAL SETUP — do not start before the Week-1 gate below passes.**

**Goal:** Let a user connect their Yahoo league so live draft picks flow into a FantasyForecast session automatically, behind the same `PlatformAdapter` interface the ManualAdapter already implicitly satisfies.

**Architecture:** A `PlatformAdapter` protocol formalizes what the draft service consumes (`get_league_settings`, `get_draft_state`). `ManualAdapter` wraps the existing history-append path (no behavior change). `YahooAdapter` performs OAuth 2.0, stores encrypted tokens in Supabase, and polls Yahoo's `draft_results` resource, translating Yahoo pick rows into the same `["pick", player_id, mine]` events. Picks still land in the event-sourced session — Yahoo is just an *ingest source* for events, not a new state store.

**Tech Stack:** FastAPI (existing `api/`), `httpx` (already a dep) for Yahoo REST, `authlib` for OAuth 2.0, `cryptography` (Fernet) for token-at-rest encryption, pytest + recorded Yahoo fixtures for contract tests.

## Global Constraints

- **Week-1 gate (spec, critical path):** before any code here, verify Yahoo's Fantasy API exposes `draft_results` *during* a live draft, not only after. If it does not, STOP and pivot to Sleeper-first live sync (demote Yahoo to post-draft import) — this plan's architecture is unchanged, only the adapter's polling source differs. The gate is Task 0.
- Secrets via env only: `YAHOO_CLIENT_ID`, `YAHOO_CLIENT_SECRET`, `YAHOO_REDIRECT_URI`, `TOKEN_ENCRYPTION_KEY` (32-byte urlsafe base64 for Fernet). Never commit any of these; add to `.env.local.example` names-only.
- Player-id mapping: Yahoo player ids are NOT nflverse gsis ids. All Yahoo picks map through a crosswalk (`src/ingest/player_ids.py` name+team lookup) to gsis ids before becoming events; unmapped picks record as opponent picks with the raw Yahoo name (never crash the draft).
- New product tables use portable types (`String`, `Integer`, `DateTime`, `JSON`, `LargeBinary` for ciphertext) — SQLite-creatable for tests. Register on the shared `Base` (import in `src/db/init_db.py` — already wired).
- ADP wall untouched: this plan never imports `src/projection` or `src/features`.
- The full Python suite gates every commit: `venv/bin/pytest`.
- Commit style: terse lowercase, trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 0: Yahoo live-draft feasibility spike (GATE)

Prove picks are visible mid-draft. Throwaway script; its output decides whether the rest of the plan proceeds as written.

**Files:**
- Create: `scripts/spikes/yahoo_draft_probe.py` (throwaway, committed for the record)

**Interfaces:**
- Produces: a documented yes/no on `draft_results` mid-draft availability.

- [ ] **Step 1: Register a Yahoo app**

Create a Yahoo Developer app (Fantasy Sports read scope), set redirect URI to `http://localhost:8000/auth/yahoo/callback`. Record client id/secret into local `.env` (names only in `.env.local.example`).

- [ ] **Step 2: Manual OAuth once, dump `draft_results` during a live mock**

Write `scripts/spikes/yahoo_draft_probe.py` that runs the OAuth device/redirect flow once, then polls `https://fantasysports.yahooapis.com/fantasy/v2/league/{league_key}/draftresults` every 10s and prints pick count. Start a Yahoo mock draft in the browser and watch whether counts increase mid-draft.

- [ ] **Step 3: Record the verdict**

Append findings (mid-draft visible? latency? auth quirks?) to this plan under a `## Spike Findings` heading and to memory.
**If picks are NOT visible mid-draft:** stop; open a Sleeper-first variant of Tasks 4-6. Do not implement Yahoo live polling.

- [ ] **Step 4: Commit the probe + findings**

```bash
git add scripts/spikes/yahoo_draft_probe.py docs/superpowers/plans/2026-07-07-yahoo-oauth-live-sync.md
git commit -m "yahoo live-draft feasibility spike + findings"
```

---

### Task 1: `PlatformAdapter` protocol + `ManualAdapter`

Formalize the interface with the one adapter that already exists in spirit. Pure refactor of intent — the manual path keeps working.

**Files:**
- Create: `api/adapters/__init__.py`, `api/adapters/base.py`, `api/adapters/manual.py`
- Test: `tests/test_adapters.py`

**Interfaces:**
- Produces:
  - `LeagueSettings` (dataclass: `teams: int`, `rounds: int`, `roster_slots: dict`).
  - `DraftState` adapter DTO (dataclass: `picks: list[tuple[str|None, bool]]` = ordered `(player_id, mine)`; `None` player = skip).
  - `PlatformAdapter(Protocol)`: `get_league_settings() -> LeagueSettings`, `get_draft_state() -> DraftState`.
  - `ManualAdapter(session_history: list, cfg)` implementing the protocol from stored history.

- [ ] **Step 1: Write the failing test**

```python
from api.adapters.manual import ManualAdapter
from src.config import load_config

def test_manual_adapter_reads_history_as_draft_state():
    cfg = load_config()
    history = [[["pick", "P1", True]], [["skip", "_skip_2"]], [["pick", "P3", False]]]
    adapter = ManualAdapter(history, cfg)
    st = adapter.get_draft_state()
    assert st.picks == [("P1", True), (None, False), ("P3", False)]
    ls = adapter.get_league_settings()
    assert ls.teams == cfg.teams
```

- [ ] **Step 2: Run to confirm failure**

Run: `venv/bin/pytest tests/test_adapters.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `base.py` + `manual.py`**

`base.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol

@dataclass
class LeagueSettings:
    teams: int
    rounds: int
    roster_slots: dict

@dataclass
class DraftState:
    picks: list[tuple[Optional[str], bool]]   # (player_id | None-for-skip, mine)

class PlatformAdapter(Protocol):
    def get_league_settings(self) -> LeagueSettings: ...
    def get_draft_state(self) -> DraftState: ...
```

`manual.py`:

```python
from __future__ import annotations
from api.adapters.base import DraftState, LeagueSettings
from src.config import LeagueConfig

class ManualAdapter:
    def __init__(self, history: list, cfg: LeagueConfig):
        self._history = history
        self._cfg = cfg

    def get_league_settings(self) -> LeagueSettings:
        return LeagueSettings(teams=self._cfg.teams, rounds=self._cfg.roster.rounds,
                              roster_slots=dict(self._cfg.roster.slots))

    def get_draft_state(self) -> DraftState:
        picks: list[tuple[str | None, bool]] = []
        for command in self._history:
            for ev in command:
                if ev[0] == "skip":
                    picks.append((None, False))
                else:
                    picks.append((ev[1], bool(ev[2])))
        return DraftState(picks=picks)
```

- [ ] **Step 4: Run tests to pass**

Run: `venv/bin/pytest tests/test_adapters.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api/adapters tests/test_adapters.py
git commit -m "add platform-adapter protocol and manual adapter"
```

---

### Task 2: Encrypted token store

Yahoo tokens live encrypted at rest in Supabase, keyed by user/connection. Fernet symmetric encryption with a key from env.

**Files:**
- Create: `api/db_models.py` (add `YahooConnection` table), `api/crypto.py`
- Test: `tests/test_crypto.py`, `tests/test_yahoo_connection_model.py`

**Interfaces:**
- Consumes: `TOKEN_ENCRYPTION_KEY` env, shared `Base`.
- Produces:
  - `encrypt(plaintext: str) -> bytes`, `decrypt(token: bytes) -> str` (Fernet).
  - `YahooConnection` table: `connection_id (str pk)`, `session_id (str, fk-ish)`, `yahoo_guid (str)`, `access_token_enc (LargeBinary)`, `refresh_token_enc (LargeBinary)`, `expires_at (DateTime)`, `league_key (str, nullable)`, `created_at`, `updated_at`.

- [ ] **Step 1: Install cryptography + authlib**

Run: `venv/bin/pip install authlib cryptography`

- [ ] **Step 2: Write the failing crypto test**

```python
import os, pytest
from cryptography.fernet import Fernet

def test_encrypt_roundtrip(monkeypatch):
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", Fernet.generate_key().decode())
    from importlib import reload
    import api.crypto as c; reload(c)
    token = "ya29.secret-access-token"
    assert c.decrypt(c.encrypt(token)) == token

def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("TOKEN_ENCRYPTION_KEY", raising=False)
    from importlib import reload
    import api.crypto as c
    with pytest.raises(RuntimeError):
        reload(c).encrypt("x")
```

- [ ] **Step 3: Run to confirm failure**

Run: `venv/bin/pytest tests/test_crypto.py -v`
Expected: FAIL — `api.crypto` missing.

- [ ] **Step 4: Implement `api/crypto.py`**

```python
from __future__ import annotations
import os
from cryptography.fernet import Fernet

def _cipher() -> Fernet:
    key = os.getenv("TOKEN_ENCRYPTION_KEY")
    if not key:
        raise RuntimeError("TOKEN_ENCRYPTION_KEY not set — cannot handle Yahoo tokens")
    return Fernet(key.encode() if isinstance(key, str) else key)

def encrypt(plaintext: str) -> bytes:
    return _cipher().encrypt(plaintext.encode())

def decrypt(token: bytes) -> str:
    return _cipher().decrypt(token).decode()
```

- [ ] **Step 5: Add `YahooConnection` to `api/db_models.py` + model test**

Append the ORM class (portable types, `LargeBinary` for ciphertext). Test round-trips a row on the SQLite fixture.

```python
def test_yahoo_connection_roundtrip(db_session):
    from api.db_models import YahooConnection
    row = YahooConnection(session_id="s", yahoo_guid="g",
                          access_token_enc=b"x", refresh_token_enc=b"y")
    db_session.add(row); db_session.commit()
    assert row.connection_id
```

(Extend the conftest `sqlite_engine` fixture's `tables=[...]` to include `YahooConnection.__table__`.)

- [ ] **Step 6: Run tests to pass**

Run: `venv/bin/pytest tests/test_crypto.py tests/test_yahoo_connection_model.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add api/crypto.py api/db_models.py tests/test_crypto.py tests/test_yahoo_connection_model.py tests/conftest.py requirements.txt
git commit -m "add fernet token crypto and yahoo connection table"
```

---

### Task 3: OAuth routes (`/auth/yahoo/login`, `/auth/yahoo/callback`)

Standard 3-legged OAuth: login redirects to Yahoo; callback exchanges the code, encrypts tokens, stores a `YahooConnection`.

**Files:**
- Create: `api/routers/auth_yahoo.py`, `api/yahoo_oauth.py`
- Modify: `api/main.py` (include router)
- Test: `tests/test_auth_yahoo.py` (authlib client mocked — no live Yahoo)

**Interfaces:**
- Consumes: `authlib`, `api.crypto`, `YahooConnection`, env secrets.
- Produces:
  - `GET /auth/yahoo/login?session_id=` → 307 redirect to Yahoo authorize URL (state carries session_id).
  - `GET /auth/yahoo/callback?code=&state=` → exchanges code, stores connection, redirects to frontend `/draft?session=...&connected=1`.
  - `yahoo_oauth.exchange_code(code) -> dict` and `refresh(refresh_token) -> dict` (thin authlib wrappers, mockable).

- [ ] **Step 1: Write the failing test (mock the token exchange)**

```python
def test_callback_stores_encrypted_connection(client, monkeypatch):
    from api import yahoo_oauth
    monkeypatch.setattr(yahoo_oauth, "exchange_code",
        lambda code: {"access_token": "at", "refresh_token": "rt", "expires_in": 3600,
                      "xoauth_yahoo_guid": "guid"})
    # state encodes the session_id created earlier
    r = client.get("/auth/yahoo/callback?code=abc&state=sess123", follow_redirects=False)
    assert r.status_code in (302, 307)
    # a YahooConnection row now exists for sess123 with non-empty ciphertext
```

- [ ] **Step 2: Run to confirm failure**

Run: `venv/bin/pytest tests/test_auth_yahoo.py -v`
Expected: FAIL — router missing.

- [ ] **Step 3: Implement `yahoo_oauth.py` + `auth_yahoo.py`**

`yahoo_oauth.py` wraps authlib's OAuth2 client for the authorize URL, `exchange_code`, and `refresh`. `auth_yahoo.py` builds the login redirect (state = session_id), and the callback: `tokens = exchange_code(code)`, encrypt access/refresh, upsert `YahooConnection(session_id=state, ...)`, redirect to the frontend.

- [ ] **Step 4: Include router in `api/main.py`, run tests**

Run: `venv/bin/pytest tests/test_auth_yahoo.py && venv/bin/pytest`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api/routers/auth_yahoo.py api/yahoo_oauth.py api/main.py tests/test_auth_yahoo.py
git commit -m "add yahoo oauth login + callback routes"
```

---

### Task 4: `YahooAdapter` + pick crosswalk (contract-tested against fixtures)

Translate Yahoo `draft_results` JSON into `DraftState` events, mapping Yahoo player ids to gsis ids. Contract test uses a recorded fixture — no live calls in CI.

**Files:**
- Create: `api/adapters/yahoo.py`, `tests/fixtures/yahoo_draftresults.json`
- Test: `tests/test_yahoo_adapter.py`

**Interfaces:**
- Consumes: `PlatformAdapter` protocol, `httpx`, crosswalk from `src/ingest/player_ids.py`, decrypted tokens.
- Produces: `YahooAdapter(league_key, token_provider, name_to_gsis)` implementing `get_league_settings`/`get_draft_state`; unmapped picks → `(raw_name, mine=False)` fallback kept as a distinct token so replay never dupes.

- [ ] **Step 1: Record a fixture**

Save one real `draftresults` JSON (from Task 0's probe) to `tests/fixtures/yahoo_draftresults.json` (scrub any tokens).

- [ ] **Step 2: Write the failing contract test**

```python
import json
from api.adapters.yahoo import YahooAdapter

def test_yahoo_draft_results_map_to_events(monkeypatch):
    raw = json.load(open("tests/fixtures/yahoo_draftresults.json"))
    adapter = YahooAdapter(league_key="nfl.l.123",
                           token_provider=lambda: "at",
                           name_to_gsis={"Ja'Marr Chase": "00-0036900"})
    monkeypatch.setattr(adapter, "_fetch_draftresults", lambda: raw)
    st = adapter.get_draft_state()
    assert len(st.picks) > 0
    # a known mapped player resolves to a gsis id
    assert any(pid == "00-0036900" for pid, _ in st.picks)
```

- [ ] **Step 3: Run to confirm failure, then implement `yahoo.py`**

`_fetch_draftresults` does the `httpx` GET with the bearer token; `get_draft_state` walks Yahoo's nested pick list, resolves each Yahoo player to a name→gsis id, and emits ordered `(player_id, mine)` (mine determined by comparing the pick's team to the connected user's team key).

- [ ] **Step 4: Run tests to pass**

Run: `venv/bin/pytest tests/test_yahoo_adapter.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api/adapters/yahoo.py tests/fixtures/yahoo_draftresults.json tests/test_yahoo_adapter.py
git commit -m "add yahoo adapter with pick crosswalk (fixture contract test)"
```

---

### Task 5: Sync endpoint — merge Yahoo picks into the session

`POST /draft/sessions/{id}/sync` pulls the adapter's `DraftState` and appends any *new* picks (beyond what history already has) as events, then returns the state. Idempotent: re-syncing the same Yahoo state is a no-op.

**Files:**
- Modify: `api/draft_service.py` (add `sync_from_adapter`), `api/routers/draft.py` (add `/sync`)
- Test: `tests/test_api_service.py`

**Interfaces:**
- Consumes: `PlatformAdapter`, existing history/replay.
- Produces: `DraftService.sync_from_adapter(session_id, adapter) -> dict`; `POST /draft/sessions/{id}/sync -> StateOut` (resolves the adapter from the session's `YahooConnection`, or a 409 if none).

- [ ] **Step 1: Write the failing test (with a fake adapter)**

```python
def test_sync_appends_only_new_picks(svc):
    s = svc.create_session(season=2026, draft_position=1)
    svc.record_pick(s.session_id, player_id="P0031")  # already 1 pick in history

    class FakeAdapter:
        def get_league_settings(self): ...
        def get_draft_state(self):
            from api.adapters.base import DraftState
            return DraftState(picks=[("P0031", True), ("P0032", False)])

    st = svc.sync_from_adapter(s.session_id, FakeAdapter())
    ids = [p["player_id"] for p in st["picks"]]
    assert ids == ["P0031", "P0032"]           # only the new one appended
    st2 = svc.sync_from_adapter(s.session_id, FakeAdapter())
    assert len([p for p in st2["picks"]]) == 2  # idempotent
```

- [ ] **Step 2: Run to confirm failure, then implement `sync_from_adapter`**

Diff the adapter's `picks` list against the count already in history; append the tail as `["pick", pid, mine]` / `["skip", token]` commands; commit; return `state`.

- [ ] **Step 3: Add `/sync` route + run full suite**

Run: `venv/bin/pytest`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add api/draft_service.py api/routers/draft.py tests/test_api_service.py
git commit -m "add adapter sync endpoint (merge yahoo picks into session)"
```

---

### Task 6: Token refresh + frontend connect button

Silent refresh when access tokens expire; a "Connect Yahoo" button in the draft room that kicks off `/auth/yahoo/login` and, once connected, polls `/sync` alongside `/state`.

**Files:**
- Modify: `api/yahoo_oauth.py` (refresh-on-expiry in the token provider), `web/src/components/DraftRoom.tsx` (Connect button + sync poll), `web/src/lib/api.ts` (`sync`, `yahooLoginUrl`)
- Test: `tests/test_auth_yahoo.py` (refresh path), `web/src/components/DraftRoom.test.tsx` (button visible)

**Interfaces:**
- Produces: a token provider that refreshes when `expires_at` passed and re-encrypts; `api.sync(id)`, `api.yahooLoginUrl(id)`; Connect-Yahoo UI that, when a connection exists, polls `/sync` every 5s.

- [ ] **Step 1: Write the failing refresh test**

Assert that a connection with a past `expires_at` triggers `yahoo_oauth.refresh` and stores new ciphertext before the adapter call.

- [ ] **Step 2: Implement refresh in the token provider; run backend tests**

Run: `venv/bin/pytest tests/test_auth_yahoo.py -v`
Expected: PASS.

- [ ] **Step 3: Frontend Connect button + sync poll**

Add `sync`/`yahooLoginUrl` to `api.ts`; in `DraftRoom`, show "Connect Yahoo" (href `api.yahooLoginUrl(sessionId)`) when `state.platform !== "yahoo"`, and when `?connected=1`, start a `useQuery` that calls `api.sync` every 5s. Component test asserts the button renders.

- [ ] **Step 4: Run frontend + backend suites**

Run: `cd web && npm test` then repo root `venv/bin/pytest`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api/yahoo_oauth.py web/src/components/DraftRoom.tsx web/src/lib/api.ts tests/test_auth_yahoo.py web/src/components/DraftRoom.test.tsx
git commit -m "add yahoo token refresh and frontend connect + sync"
```

---

## Self-Review Notes

- **Spec coverage:** `PlatformAdapter` protocol (Task 1), YahooAdapter + ManualAdapter (Tasks 1,4), OAuth with tokens encrypted at rest (Tasks 2,3), live-sync via polling (Task 5), silent refresh + re-auth only on failure (Task 6), Yahoo-down fallback (ManualAdapter always available; sync is additive, never blocks manual entry). Contract tests against fixtures (Task 4).
- **Gate discipline:** Task 0 is a hard gate; a negative result reroutes Tasks 4-6 to Sleeper without touching Tasks 1-3 (protocol/crypto/oauth-shape are provider-agnostic enough to mostly reuse).
- **Player-id integrity:** every Yahoo pick crosses the name→gsis crosswalk; unmapped picks degrade to opponent picks with the raw name, never crash — matches the spec's "draft continues" error posture.
- **Security:** tokens never logged, never returned to the client, encrypted with Fernet, key from env only.
