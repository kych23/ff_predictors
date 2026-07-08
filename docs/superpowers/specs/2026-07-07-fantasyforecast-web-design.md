# FantasyForecast Web — v1 Draft Assistant Design

**Date:** 2026-07-07
**Status:** Approved
**Target ship:** August 15, 2026 (before fantasy draft season, mid-Aug–early Sept)

## Context & Goals

FantasyForecast today is a Python CLI: pooled LightGBM quantile models (P10/P50/P90), a VONA-based draft recommender, and a weekly start/sit optimizer, backed by Supabase (PostgreSQL). This project turns it into a deployed web product.

**Goals, in priority order:**

1. Ship a usable live-draft assistant before the 2026 draft season and get real users (league mates, Cornell friends, fantasy communities).
2. Add an LLM explanation layer grounded in the model's own outputs — a genuine ML-engineering story, not an API wrapper.
3. Provide a zero-login demo mode so anyone (including recruiters, year-round) can try the product in seconds.

**Non-goals for v1:** weekly start/sit dashboard (v2, mid-season), ESPN/Sleeper sync (post-v1), chat assistant, model changes.

## Decisions Made

| Decision | Choice | Rationale |
|---|---|---|
| Foundation | Evolve existing repo, not new project | Months of ML work banked; draft season timing aligns with recruiting season |
| Platform sync | Yahoo first; ESPN/Sleeper later via adapter interface | User's league is the beachhead; manual entry covers everyone else |
| Architecture | Next.js frontend + FastAPI backend | FastAPI imports existing `src/` engine directly — no logic port, no drift |
| LLM scope | Pick explanations only | Tight scope; grounding design is the interesting part |
| v1 boundary | Draft assistant only | Everything bets on the draft-season window; weekly dashboard is v2 |

## Architecture

```
Browser (Next.js on Vercel)
        │ REST (polling ~5s during draft)
        ▼
FastAPI (Railway or Fly) — new api/ package in this repo
 ├─ /auth/yahoo/*   OAuth 2.0 flow
 ├─ /draft/session  create/resume draft sessions (state in DB)
 ├─ /draft/state    current picks via platform adapter
 ├─ /draft/recommend  VONA top-N + P10/P50/P90 (existing src/ engine)
 ├─ /explain        LLM pick rationale, cached
 └─ /players        projection list/search
        │
        ▼
Supabase (PostgreSQL) — already in place
        ▲
Existing training pipeline (unchanged) writes projections
```

- `api/` is a new package inside `ff_predictors`; it imports `src/` directly. The training pipeline and CLI remain untouched.
- Frontend polls FastAPI every ~5 seconds during an active draft. SSE is a later optimization if polling feels sluggish; at expected scale polling is sufficient and simpler.

## Components

### Platform adapters

```python
class PlatformAdapter(Protocol):
    def get_league_settings(self) -> LeagueSettings: ...
    def get_draft_state(self) -> DraftState: ...
```

- **YahooAdapter (v1):** OAuth 2.0; polls Yahoo Fantasy API `draft_results` during live drafts. Tokens encrypted at rest in Supabase.
- **ManualAdapter (v1):** user enters picks in the UI; same interface. Serves as fallback when Yahoo is down and as the path for ESPN/Yahoo-less users.
- **ESPN/Sleeper (later):** slot in behind the same protocol.
- Contract tests run every adapter against recorded fixture responses.

### Draft sessions

Replaces the current `draft_state_*.json` files. Session state (league settings, picks so far, user roster) lives in Supabase keyed by session id, enabling resume after disconnect and multi-device access.

### Next.js app

- **Landing page** — what it is, live demo button, connect-league CTA.
- **Draft room** — draft board, user's roster, recommendation panel (top VONA picks with P10/P50/P90 bands), explanation cards on click.
- **Mock-draft demo mode** — draft against ADP bots (reusing `scripts/mock_draft.py` logic), zero login. This is the recruiter-facing demo and must stay working year-round.

### LLM explanation service

- Input: structured model outputs only — VONA delta, quantile spread, ADP gap, positional scarcity.
- Output: 2–3 sentence rationale. The LLM never invents numbers; every claim traces to a model output passed in the prompt.
- Cached by (player, round bucket, roster shape) to bound cost and latency.
- Output validated against a schema before display; failures degrade to numbers-only.

## Data Flow (live draft)

1. User connects Yahoo league → OAuth → tokens stored encrypted.
2. User starts a draft session → FastAPI polls Yahoo `draft_results`.
3. Picks merge with projections; VONA recommender scores available players.
4. Top recommendations + quantile bands render in the draft room.
5. User clicks a recommendation → `/explain` returns the cached or fresh LLM rationale.

## Error Handling

- **Yahoo API down / rate-limited:** banner in UI; draft continues via ManualAdapter without losing session state.
- **LLM timeout or schema-invalid output:** show numbers without prose.
- **OAuth token expiry:** silent refresh; re-auth prompt only if refresh fails.
- **Mid-draft disconnect:** session resumes from DB.

## Testing

- Existing engine tests and the ADP-wall leakage test remain enforced (`tests/test_leakage.py`). The `api/` package must respect the ADP wall's spirit: recommendation logic goes through existing `src/` interfaces.
- New: FastAPI endpoint tests (pytest + httpx), adapter contract tests with Yahoo fixtures, LLM output schema validation tests, one Playwright smoke test covering the draft room happy path in manual mode.

## Timeline (~6 weeks to Aug 15)

| Week | Deliverable |
|---|---|
| 1 | **Yahoo API spike: verify picks are visible mid-draft.** FastAPI skeleton + `/players`. |
| 2 | Draft session + recommend endpoints; ManualAdapter end-to-end. |
| 3 | Draft room UI working end-to-end in manual mode. |
| 4 | Yahoo OAuth + live sync. |
| 5 | LLM explanations; deploy to Vercel + Railway/Fly; polish. |
| 6 | Buffer. Demo mode, landing page, README with GIFs. Beta test in league mock drafts. |

**Week-1 gate:** if the Yahoo API cannot expose picks during a live draft (only after), pivot to Sleeper-first live sync and demote Yahoo to post-draft import. The spike is the critical path and is built first.

## Risks

| Risk | Mitigation |
|---|---|
| Yahoo live-draft API infeasible | Week-1 spike + Sleeper pivot plan |
| Timeline slips past draft season | ManualAdapter path ships by week 3 — usable product even if sync slips; demo mode keeps recruiter value regardless |
| LLM cost during drafts | Caching by context bucket; cheap model tier is sufficient for 2–3 sentence rationales |
| Seasonal traction window closes | Demo mode + metrics captured during season stay on the resume year-round |
