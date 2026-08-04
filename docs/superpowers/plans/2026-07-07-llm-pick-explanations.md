# LLM Pick Explanations (Plan 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **BLOCKED ON EXTERNAL SETUP:** requires an LLM API key (`ANTHROPIC_API_KEY`). The grounding + validation design is the substance; the API call is the thin part.

**Goal:** For any recommended pick, return a 2-3 sentence rationale that is *grounded* — every claim traces to a model output passed into the prompt — cached by context bucket, and validated against a schema before display, degrading to numbers-only on any failure.

**Architecture:** A pure `build_explanation_context(rec, state, board)` function assembles the *only* facts the LLM may use: VONA delta, quantile spread, ADP gap, positional scarcity, roster need. The LLM receives those as structured JSON with a strict instruction to use only provided numbers. Output is parsed and *validated against the input* (every number it cites must equal a number we passed, within tolerance) — a grounding check, not just schema validation. Results cache by `(player_id, round_bucket, roster_shape_hash)`. Failures (timeout, schema-invalid, ungrounded) fall back to a deterministic numbers-only template.

**Tech Stack:** FastAPI (existing `api/`), `anthropic` SDK (Claude — use the current default model), pydantic for output schema, existing `recommend`/board for grounding inputs, pytest with a mocked LLM client (no live calls in CI).

## Global Constraints

- **Grounding is the product.** The LLM never introduces a number not in the context payload. The validator (Task 3) rejects any response citing an unprovided/mismatched figure; rejected responses never reach the user.
- `ANTHROPIC_API_KEY` via env only. CI mocks the client — no network, no key needed to run the suite.
- Model: use the current default Claude model id; do not pin a deprecated one. (Consult the claude-api reference at implementation time for the live default.)
- Cache key is `(player_id, round_bucket, roster_shape_hash)` where `round_bucket = ceil(draft_round/3)` and `roster_shape_hash` hashes the sorted position counts of the user's roster. Same bucket → cached prose, bounding cost/latency (spec).
- Explanation table uses portable types (`String`, `JSON`, `DateTime`), registered on the shared `Base`.
- ADP wall untouched; no imports from `src/projection`/`src/features`.
- Full suite gates every commit: `venv/bin/pytest`.
- Commit style: terse lowercase, trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Explanation context builder (pure, grounding source of truth)

The set of facts the LLM is allowed to use. Pure function, no I/O, exhaustively tested — this is the anti-hallucination boundary.

**Files:**
- Create: `api/explain/__init__.py`, `api/explain/context.py`
- Test: `tests/test_explain_context.py`

**Interfaces:**
- Consumes: a `Recommendation`-shaped dict, the `state` dict (`DraftService.state`), the board DataFrame.
- Produces: `build_explanation_context(rec: dict, state: dict, board: pd.DataFrame) -> dict` with keys: `player_name, position, p10, p50, p90, quantile_spread (=p90-p10), vona_score, adp, adp_gap (=adp - current_overall_pick), positional_rank_available, roster_need (bool), draft_round, target_quantile`. All numbers rounded to 1 decimal.

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
from api.explain.context import build_explanation_context

def test_context_only_contains_grounded_facts():
    board = pd.DataFrame([
        {"player_id": "P1", "position": "RB", "p50": 18.0},
        {"player_id": "P2", "position": "RB", "p50": 14.0},
    ])
    rec = {"player_id": "P1", "name": "Star RB", "position": "RB", "p10": 12.0,
           "p50": 18.0, "p90": 26.0, "vona_score": 4.2, "adp": 20.0,
           "draft_round": 2, "target_quantile": 0.3}
    state = {"current_overall_pick": 15, "open_starters": {"RB": 1}}
    ctx = build_explanation_context(rec, state, board)
    assert ctx["quantile_spread"] == 14.0        # 26 - 12
    assert ctx["adp_gap"] == 5.0                  # 20 - 15
    assert ctx["roster_need"] is True             # RB slot open
    assert ctx["positional_rank_available"] == 1  # best available RB by p50
    assert set(ctx) >= {"player_name", "vona_score", "p10", "p50", "p90"}
```

- [ ] **Step 2: Run to confirm failure**

Run: `venv/bin/pytest tests/test_explain_context.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `context.py`**

Compute each field from the inputs only. `positional_rank_available` = 1 + count of same-position players on the board with higher `p50` that are not already drafted (approx via board order; drafted filtering optional in v1). `roster_need` = `state["open_starters"].get(position, 0) > 0`.

- [ ] **Step 4: Run tests to pass**

Run: `venv/bin/pytest tests/test_explain_context.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api/explain tests/test_explain_context.py
git commit -m "add explanation context builder (grounding source of truth)"
```

---

### Task 2: Numbers-only fallback template

The deterministic explanation shown whenever the LLM path fails. Built first so the feature is always safe to ship.

**Files:**
- Create: `api/explain/fallback.py`
- Test: `tests/test_explain_fallback.py`

**Interfaces:**
- Consumes: the context dict (Task 1).
- Produces: `numbers_only(ctx: dict) -> str` — a single deterministic sentence citing P50, the P10-P90 band, and ADP gap.

- [ ] **Step 1: Write the failing test**

```python
from api.explain.fallback import numbers_only

def test_numbers_only_is_deterministic_and_grounded():
    ctx = {"player_name": "Star RB", "position": "RB", "p10": 12.0, "p50": 18.0,
           "p90": 26.0, "adp_gap": 5.0, "vona_score": 4.2}
    out = numbers_only(ctx)
    assert "18.0" in out and "12.0" in out and "26.0" in out
    assert numbers_only(ctx) == out   # deterministic
```

- [ ] **Step 2: Run to confirm failure, then implement `fallback.py`**

```python
def numbers_only(ctx: dict) -> str:
    return (f"{ctx['player_name']} projects to {ctx['p50']:.1f} PPG "
            f"(range {ctx['p10']:.1f}-{ctx['p90']:.1f}), with a VONA edge of "
            f"{ctx['vona_score']:.1f} and an ADP gap of {ctx['adp_gap']:.1f} picks.")
```

- [ ] **Step 3: Run tests to pass**

Run: `venv/bin/pytest tests/test_explain_fallback.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add api/explain/fallback.py tests/test_explain_fallback.py
git commit -m "add numbers-only explanation fallback"
```

---

### Task 3: LLM output schema + grounding validator

The claim-checker. Parses the model's JSON and verifies every number it cites was in the context (within tolerance). This is what makes the feature trustworthy.

**Files:**
- Create: `api/explain/schema.py`, `api/explain/validate.py`
- Test: `tests/test_explain_validate.py`

**Interfaces:**
- Produces:
  - `ExplanationOut(BaseModel)`: `rationale: str` (2-3 sentences), `cited_numbers: list[float]`.
  - `is_grounded(out: ExplanationOut, ctx: dict, tol: float = 0.15) -> bool` — True iff every value in `cited_numbers` matches some numeric context value within `tol`, and `rationale` is 1-3 sentences.

- [ ] **Step 1: Write the failing test**

```python
from api.explain.schema import ExplanationOut
from api.explain.validate import is_grounded

CTX = {"p10": 12.0, "p50": 18.0, "p90": 26.0, "vona_score": 4.2, "adp_gap": 5.0}

def test_grounded_when_all_cited_numbers_present():
    out = ExplanationOut(rationale="Strong P50 of 18 with upside to 26.",
                         cited_numbers=[18.0, 26.0])
    assert is_grounded(out, CTX) is True

def test_ungrounded_when_a_number_is_invented():
    out = ExplanationOut(rationale="Projected for 31 points.", cited_numbers=[31.0])
    assert is_grounded(out, CTX) is False

def test_ungrounded_when_too_long():
    out = ExplanationOut(rationale="A. B. C. D. E.", cited_numbers=[])
    assert is_grounded(out, CTX) is False
```

- [ ] **Step 2: Run to confirm failure, then implement `schema.py` + `validate.py`**

`is_grounded`: for each cited number, require `min(abs(n - v) for v in numeric ctx values) <= tol`; sentence count via split on `.` in `[1,3]`.

- [ ] **Step 3: Run tests to pass**

Run: `venv/bin/pytest tests/test_explain_validate.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add api/explain/schema.py api/explain/validate.py tests/test_explain_validate.py
git commit -m "add explanation schema and grounding validator"
```

---

### Task 4: LLM client wrapper (mockable) + prompt

Thin Claude wrapper that takes a context dict and returns an `ExplanationOut`. The prompt hard-constrains the model to provided numbers. CI mocks it.

**Files:**
- Create: `api/explain/llm.py`, `api/explain/prompt.py`
- Test: `tests/test_explain_llm.py`

**Interfaces:**
- Consumes: `anthropic` SDK, `ExplanationOut`.
- Produces: `explain_with_llm(ctx: dict, *, client=None) -> ExplanationOut` — builds the prompt, calls Claude (JSON output), parses to `ExplanationOut`; `client` injectable for tests.

- [ ] **Step 1: Consult the claude-api reference for the current model id + JSON-output pattern**

(Do not hardcode a deprecated model.)

- [ ] **Step 2: Write the failing test (mock client returns canned JSON)**

```python
from api.explain.llm import explain_with_llm

class FakeClient:
    def complete_json(self, prompt: str) -> dict:
        return {"rationale": "Solid P50 of 18 with room to 26.",
                "cited_numbers": [18.0, 26.0]}

def test_explain_with_llm_parses_to_schema():
    ctx = {"player_name": "Star RB", "p10": 12.0, "p50": 18.0, "p90": 26.0,
           "vona_score": 4.2, "adp_gap": 5.0}
    out = explain_with_llm(ctx, client=FakeClient())
    assert out.rationale.startswith("Solid")
    assert out.cited_numbers == [18.0, 26.0]
```

- [ ] **Step 3: Run to confirm failure, then implement `prompt.py` + `llm.py`**

`prompt.py`: a template instructing "use ONLY these numbers; return JSON `{rationale, cited_numbers}`; 2-3 sentences." `llm.py`: default client wraps the `anthropic` SDK with JSON output; `explain_with_llm` calls `client.complete_json(prompt)` and returns `ExplanationOut(**parsed)`.

- [ ] **Step 4: Run tests to pass**

Run: `venv/bin/pytest tests/test_explain_llm.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api/explain/llm.py api/explain/prompt.py tests/test_explain_llm.py requirements.txt
git commit -m "add mockable llm explanation client + prompt"
```

---

### Task 5: Explanation cache table + service

Ties it together: `explain(session_id, player_id)` builds context, checks the cache, else calls the LLM, validates grounding, stores + returns; on any failure returns the numbers-only fallback (still 200).

**Files:**
- Create: `api/explain/service.py`
- Modify: `api/db_models.py` (add `Explanation` table), `tests/conftest.py` (fixture tables)
- Test: `tests/test_explain_service.py`

**Interfaces:**
- Consumes: Tasks 1-4, `DraftService.state`/board, `Explanation` table.
- Produces:
  - `Explanation` table: `cache_key (str pk)`, `session_id`, `player_id`, `rationale (str)`, `grounded (bool)`, `created_at`.
  - `ExplainService(db, cfg, board_for, llm=explain_with_llm)`: `explain(session_id, player_id) -> dict` = `{player_id, rationale, grounded, cached}`.

- [ ] **Step 1: Write the failing test (fake LLM, in-memory)**

```python
def test_explain_caches_and_falls_back(db_session):
    # grounded fake -> stored + cached; second call cached=True
    # ungrounded fake -> numbers_only fallback, grounded=False
    ...
```

(Full test: one grounded fake LLM asserts `cached` flips True on second call; one ungrounded fake asserts the returned rationale equals `numbers_only(ctx)` and `grounded is False`.)

- [ ] **Step 2: Run to confirm failure, then implement `service.py` + `Explanation` table**

Cache key from `(player_id, ceil(draft_round/3), roster_shape_hash)`. Flow: context → cache hit? return; else LLM → `is_grounded`? store+return : fallback (store `grounded=False` optional). Any exception → fallback, never raise.

- [ ] **Step 3: Run tests + full suite**

Run: `venv/bin/pytest tests/test_explain_service.py && venv/bin/pytest`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add api/explain/service.py api/db_models.py tests/conftest.py tests/test_explain_service.py
git commit -m "add explanation cache table and service with grounded fallback"
```

---

### Task 6: `/explain` endpoint + frontend explanation card

HTTP surface + the draft-room UI: clicking a recommendation reveals its rationale (or numbers-only), lazily fetched and cached.

**Files:**
- Create: `api/routers/explain.py`
- Modify: `api/main.py` (include router), `web/src/lib/api.ts` (`explain`), `web/src/components/RecommendationPanel.tsx` (expandable card)
- Test: `tests/test_explain_endpoint.py`, `web/src/components/RecommendationPanel.test.tsx`

**Interfaces:**
- Produces:
  - `GET /draft/sessions/{id}/explain?player_id=` → `ExplanationResponse {player_id, rationale, grounded, cached}`.
  - `api.explain(id, playerId)`; a RecommendationPanel "Why?" toggle that fetches on first open and shows the rationale.

- [ ] **Step 1: Write the failing backend test**

```python
def test_explain_endpoint_returns_rationale(client):
    # create session, pick, then GET /explain?player_id=... -> 200 with rationale
    ...
```

- [ ] **Step 2: Implement route + include in `api/main.py`; run backend tests**

Run: `venv/bin/pytest tests/test_explain_endpoint.py && venv/bin/pytest`
Expected: PASS.

- [ ] **Step 3: Frontend "Why?" card**

Add `api.explain`; in `RecommendationPanel`, a "Why?" button per rec that lazy-fetches and expands the rationale; a small "grounded" indicator. Component test asserts the button appears and, when clicked with a mocked `api.explain`, the rationale renders.

- [ ] **Step 4: Run frontend + backend suites**

Run: `cd web && npm test` then repo root `venv/bin/pytest`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api/routers/explain.py api/main.py web/src/lib/api.ts web/src/components/RecommendationPanel.tsx tests/test_explain_endpoint.py web/src/components/RecommendationPanel.test.tsx
git commit -m "add explain endpoint and frontend explanation card"
```

---

## Self-Review Notes

- **Spec coverage:** structured-only inputs (Task 1), 2-3 sentence rationale (Tasks 4,3), never invents numbers (Task 3 grounding validator — the core), cache by context bucket (Task 5), schema-validated with numbers-only degradation (Tasks 2,3,5). The `/explain` endpoint and click-to-reveal card (Task 6) match the spec's draft-room interaction.
- **Why grounding-validate, not just schema-validate:** schema validation only proves shape; the spec's hard requirement is "every claim traces to a model output." Task 3 checks the actual cited numbers against the context, which is the defensible ML-engineering story (eval of an LLM's faithfulness), not an API wrapper.
- **Safety posture:** every failure mode (timeout, invalid JSON, ungrounded, missing key) returns the deterministic numbers-only string with `grounded=False` and HTTP 200 — the draft never blocks on the LLM.
- **Cost bound:** cache key collapses same player + round-bucket + roster-shape to one LLM call; a full draft touches each recommended player at most once per bucket.
