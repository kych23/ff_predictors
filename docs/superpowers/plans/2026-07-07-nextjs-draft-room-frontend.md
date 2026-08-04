# Next.js Draft Room Frontend (Plan 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the web draft assistant — a landing page, a manual-mode live draft room (board + roster + VONA recommendations with P10/P50/P90 bands), and a zero-login mock-draft demo — as a Next.js app talking to the existing FastAPI backend (Plan 1).

**Architecture:** New `web/` package (Next.js App Router, TypeScript) alongside the Python repo. It is a pure REST client of the `api/` service — no engine logic crosses over. Draft state lives server-side (Plan 1 sessions); the frontend polls `GET /draft/sessions/{id}` every ~5s during an active draft and renders whatever the backend replays. Demo mode reuses the same session API plus one new backend endpoint (`bot-pick`) so ADP bots draft against the user with zero login.

**Tech Stack:** Next.js 14 (App Router), TypeScript, Tailwind CSS, TanStack Query (polling + cache), Vitest + React Testing Library (component tests), Playwright (one happy-path e2e smoke). Node 20, npm.

## Global Constraints

- Node 20, npm. All frontend commands run from `web/`: `npm run dev|build|test|test:e2e`, `npm run lint`.
- No engine logic in the frontend: the browser only calls the REST API. Anything requiring projections/VONA goes through an `api/` endpoint.
- Backend base URL comes from `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`). Never hardcode.
- The one backend change in this plan (Task 1, `bot-pick`) lives in `api/` and must keep the full Python suite green: `venv/bin/pytest` from repo root.
- Python venv for backend work: `venv/bin/pytest`, `venv/bin/uvicorn`.
- TypeScript strict mode on. No `any` in committed code except at typed API boundaries with an explicit cast + comment.
- Wire types mirror the backend pydantic schemas (`api/schemas.py`) exactly — field names and optionality identical. `web/src/lib/types.ts` is the single source; drift is a bug.
- Commit messages: terse lowercase subject matching `git log` style, trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- `web/node_modules/` is gitignored; commit `package-lock.json`.

---

### Task 1: Backend `bot-pick` endpoint for demo mode

Demo mode needs opponents that draft themselves. Reuse the validated ADP-bot logic (`src/benchmark/draft_sim._bot_pick`) behind a session endpoint so the frontend can advance a bot's turn without any engine code client-side.

**Files:**
- Modify: `api/draft_service.py` (add `bot_pick` method)
- Modify: `api/routers/draft.py` (add `POST /draft/sessions/{id}/bot-pick`)
- Test: `tests/test_api_service.py`, `tests/test_api_endpoints.py`

**Interfaces:**
- Consumes: `_bot_pick` (`src/benchmark/draft_sim.py`), `replay_history`, existing `DraftService._rebuild`/`state`.
- Produces: `DraftService.bot_pick(session_id: str) -> dict` (records the ADP bot's pick as an opponent pick, returns the same state dict as `record_pick`); HTTP `POST /draft/sessions/{id}/bot-pick -> StateOut`.

- [ ] **Step 1: Write the failing service test**

Add to `tests/test_api_service.py`:

```python
def test_bot_pick_records_opponent_by_adp(svc):
    s = svc.create_session(season=2026, draft_position=2)  # pick 1 is an opponent
    st = svc.bot_pick(s.session_id)
    assert len(st["picks"]) == 1
    assert st["picks"][0]["mine"] is False
    assert st["picks"][0]["player_id"] is not None
    assert st["my_roster"] == []
    assert st["current_overall_pick"] == 2


def test_bot_pick_never_duplicates(svc):
    s = svc.create_session(season=2026, draft_position=1)
    svc.record_pick(s.session_id, player_id="P0031")  # my pick
    st = svc.bot_pick(s.session_id)                     # bot takes best available by adp
    ids = [p["player_id"] for p in st["picks"]]
    assert len(ids) == len(set(ids))
    assert "P0031" not in ids[1:]
```

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/pytest tests/test_api_service.py::test_bot_pick_records_opponent_by_adp -v`
Expected: FAIL — `AttributeError: 'DraftService' object has no attribute 'bot_pick'`

- [ ] **Step 3: Implement `bot_pick`**

Add to `api/draft_service.py` (after `record_pick`). The ADP bot needs a `slot_fill` dict; rebuild it from the replayed opponent picks is unnecessary — the demo bot only needs "best available that fits any open slot", and with a full bench every position always fits, so pass an empty fill (bench-open semantics match `_has_open_slot`'s bench fallback):

```python
    def bot_pick(self, session_id: str) -> dict:
        """Advance one opponent turn using the ADP-bot heuristic (demo mode)."""
        from src.benchmark.draft_sim import _bot_pick
        sess = self._get(session_id)
        state, board = self._rebuild(sess)
        slot_fill = {s: 0 for s in self.cfg.roster.slots}
        pid = _bot_pick(board, state.drafted, slot_fill, self.cfg)
        if pid is None:
            raise InvalidPick("no players left for bot to draft")
        command = [["pick", pid, False]]
        sess.history = sess.history + [command]
        self.db.commit()
        return self.state(session_id)
```

- [ ] **Step 4: Run service tests**

Run: `venv/bin/pytest tests/test_api_service.py -v`
Expected: PASS (new + existing).

- [ ] **Step 5: Add the route + endpoint test**

In `api/routers/draft.py` add:

```python
@router.post("/{session_id}/bot-pick", response_model=StateOut)
def bot_pick(session_id: str, svc: DraftService = Depends(get_service)):
    return svc.bot_pick(session_id)
```

Add to `tests/test_api_endpoints.py` (inside `test_session_lifecycle` or a new test):

```python
def test_bot_pick_endpoint(client):
    r = client.post("/draft/sessions", json={"season": 2026, "draft_position": 2})
    sid = r.json()["session_id"]
    r = client.post(f"/draft/sessions/{sid}/bot-pick")
    assert r.status_code == 200
    assert r.json()["picks"][0]["mine"] is False
```

- [ ] **Step 6: Run full Python suite**

Run: `venv/bin/pytest`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add api/draft_service.py api/routers/draft.py tests/test_api_service.py tests/test_api_endpoints.py
git commit -m "add bot-pick endpoint for demo-mode drafts"
```

---

### Task 2: Next.js scaffold + Tailwind + API config

Stand up the app shell so later tasks have a place to render. No product UI yet — just a building, linting, type-checking project with the API base URL wired.

**Files:**
- Create: `web/package.json`, `web/next.config.js`, `web/tsconfig.json`, `web/tailwind.config.ts`, `web/postcss.config.js`, `web/.gitignore`, `web/.env.local.example`, `web/src/app/layout.tsx`, `web/src/app/globals.css`, `web/src/app/page.tsx` (placeholder)
- Modify: repo root `.gitignore` (ignore `web/node_modules`, `web/.next`)

**Interfaces:**
- Produces: a runnable Next.js app (`npm run dev` on :3000), `NEXT_PUBLIC_API_URL` env convention.

- [ ] **Step 1: Scaffold with create-next-app (non-interactive)**

Run from repo root:

```bash
npx create-next-app@14 web --ts --tailwind --eslint --app --src-dir --import-alias "@/*" --no-turbopack --use-npm
```

Expected: `web/` created, `npm install` completed.

- [ ] **Step 2: Add env example + API URL**

Create `web/.env.local.example`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

- [ ] **Step 3: Replace the boilerplate landing placeholder**

Overwrite `web/src/app/page.tsx`:

```tsx
export default function Home() {
  return <main className="p-8"><h1 className="text-2xl font-bold">FantasyForecast</h1></main>;
}
```

- [ ] **Step 4: Verify dev build + typecheck + lint**

Run: `cd web && npm run build && npm run lint`
Expected: build succeeds, no lint errors.

- [ ] **Step 5: Ignore build artifacts in repo root**

Append to repo-root `.gitignore`:

```
web/node_modules/
web/.next/
web/.env.local
```

- [ ] **Step 6: Commit**

```bash
git add web/ .gitignore
git commit -m "scaffold nextjs web app (app router, tailwind, ts)"
```

---

### Task 3: Typed API client + wire types

One typed module for every backend call, mirroring `api/schemas.py`. Everything else imports from here so the wire contract lives in one place.

**Files:**
- Create: `web/src/lib/types.ts`, `web/src/lib/api.ts`
- Test: `web/src/lib/api.test.ts`

**Interfaces:**
- Produces:
  - types `Player`, `Pick`, `RosterEntry`, `DraftState`, `Recommendation` (mirror `PlayerOut`, `PickOut`, `RosterEntryOut`, `StateOut`, `RecommendationOut`).
  - `api.listPlayers(season)`, `api.createSession(season, draftPosition)`, `api.getState(id)`, `api.recordPick(id, body)`, `api.undo(id)`, `api.botPick(id)`, `api.recommendations(id, topN)`.
  - `ApiError` (carries HTTP status + detail).

- [ ] **Step 1: Write the failing test**

Create `web/src/lib/api.test.ts` (Vitest + fetch mock):

```ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import { api, ApiError } from "./api";

const okJson = (body: unknown) =>
  Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) } as Response);

beforeEach(() => vi.restoreAllMocks());

describe("api client", () => {
  it("createSession posts season + draft_position and returns state", async () => {
    const spy = vi.spyOn(globalThis, "fetch").mockReturnValue(
      okJson({ session_id: "abc", current_overall_pick: 1 }) as ReturnType<typeof fetch>);
    const st = await api.createSession(2026, 4);
    expect(st.session_id).toBe("abc");
    const [url, init] = spy.mock.calls[0];
    expect(String(url)).toContain("/draft/sessions");
    expect(JSON.parse(String((init as RequestInit).body))).toEqual(
      { season: 2026, draft_position: 4 });
  });

  it("raises ApiError with status + detail on non-2xx", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false, status: 400, json: () => Promise.resolve({ detail: "bad pick" }),
    } as Response);
    await expect(api.undo("x")).rejects.toMatchObject({
      name: "ApiError", status: 400, detail: "bad pick" });
  });
});
```

- [ ] **Step 2: Install Vitest + run to confirm failure**

Run from `web/`:

```bash
npm install -D vitest @vitejs/plugin-react jsdom @testing-library/react @testing-library/jest-dom @testing-library/user-event
```

Add to `web/package.json` scripts: `"test": "vitest run"`, `"test:watch": "vitest"`.
Create `web/vitest.config.ts`:

```ts
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  test: { environment: "jsdom", globals: true, setupFiles: ["./vitest.setup.ts"] },
  resolve: { alias: { "@": new URL("./src", import.meta.url).pathname } },
});
```

Create `web/vitest.setup.ts`: `import "@testing-library/jest-dom";`

Run: `npm test -- api.test.ts`
Expected: FAIL — cannot resolve `./api`.

- [ ] **Step 3: Implement `types.ts`**

```ts
export interface Player {
  player_id: string; name: string | null; team: string | null; position: string;
  p10: number; p50: number; p90: number; adp: number | null; bye_week: number | null;
}
export interface Pick {
  pick_number: number; player_id: string | null; name: string | null;
  mine: boolean; skipped: boolean;
}
export interface RosterEntry {
  player_id: string; name: string | null; position: string | null;
  team: string | null; bye_week: number | null;
}
export interface DraftState {
  session_id: string; season: number; draft_position: number; platform: string;
  status: string; teams: number; rounds: number; my_picks: number[];
  current_overall_pick: number; is_my_turn: boolean; next_my_pick: number | null;
  remaining_picks: number; picks: Pick[]; my_roster: RosterEntry[];
  open_starters: Record<string, number>;
}
export interface Recommendation {
  player_id: string; name: string | null; position: string; team: string | null;
  vona_score: number; value: number; p10: number; p50: number; p90: number;
  adp: number | null; draft_round: number; target_quantile: number;
  forced_completion: boolean;
}
export interface PickBody { player_id?: string; skip?: boolean; mine?: boolean; }
```

- [ ] **Step 4: Implement `api.ts`**

```ts
import type { DraftState, Player, Recommendation, PickBody } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number; detail: string;
  constructor(status: number, detail: string) {
    super(`API ${status}: ${detail}`);
    this.name = "ApiError"; this.status = status; this.detail = detail;
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* non-json body */ }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  listPlayers: (season: number) => req<Player[]>(`/players?season=${season}`),
  createSession: (season: number, draftPosition: number) =>
    req<DraftState>(`/draft/sessions`, {
      method: "POST", body: JSON.stringify({ season, draft_position: draftPosition }) }),
  getState: (id: string) => req<DraftState>(`/draft/sessions/${id}`),
  recordPick: (id: string, body: PickBody) =>
    req<DraftState>(`/draft/sessions/${id}/picks`, {
      method: "POST", body: JSON.stringify(body) }),
  undo: (id: string) => req<DraftState>(`/draft/sessions/${id}/undo`, { method: "POST" }),
  botPick: (id: string) => req<DraftState>(`/draft/sessions/${id}/bot-pick`, { method: "POST" }),
  recommendations: (id: string, topN = 10) =>
    req<Recommendation[]>(`/draft/sessions/${id}/recommendations?top_n=${topN}`),
};
```

- [ ] **Step 5: Run tests to pass**

Run: `npm test -- api.test.ts`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add web/src/lib web/vitest.config.ts web/vitest.setup.ts web/package.json web/package-lock.json
git commit -m "add typed api client and wire types"
```

---

### Task 4: `QuantileBar` component (P10/P50/P90 band)

The signature visual: a horizontal band from P10 to P90 with a P50 marker, scaled within a position's range. Pure/presentational so it unit-tests without a DOM server.

**Files:**
- Create: `web/src/components/QuantileBar.tsx`
- Test: `web/src/components/QuantileBar.test.tsx`

**Interfaces:**
- Produces: `QuantileBar({ p10, p50, p90, min, max }: { p10: number; p50: number; p90: number; min: number; max: number })` — renders a band; exposes `data-testid="quantile-bar"` and `data-p50-pct` for testing.

- [ ] **Step 1: Write the failing test**

```tsx
import { render, screen } from "@testing-library/react";
import { QuantileBar } from "./QuantileBar";

it("positions p50 marker proportionally within [min,max]", () => {
  render(<QuantileBar p10={5} p50={10} p90={15} min={0} max={20} />);
  const bar = screen.getByTestId("quantile-bar");
  expect(bar.getAttribute("data-p50-pct")).toBe("50"); // 10 of [0,20]
});

it("clamps to [0,100] when values exceed range", () => {
  render(<QuantileBar p10={-5} p50={25} p90={30} min={0} max={20} />);
  const bar = screen.getByTestId("quantile-bar");
  expect(Number(bar.getAttribute("data-p50-pct"))).toBeLessThanOrEqual(100);
});
```

- [ ] **Step 2: Run to confirm failure**

Run: `npm test -- QuantileBar`
Expected: FAIL — cannot resolve component.

- [ ] **Step 3: Implement**

```tsx
"use client";

function pct(v: number, min: number, max: number): number {
  if (max <= min) return 0;
  return Math.max(0, Math.min(100, ((v - min) / (max - min)) * 100));
}

export function QuantileBar({ p10, p50, p90, min, max }: {
  p10: number; p50: number; p90: number; min: number; max: number;
}) {
  const left = pct(p10, min, max);
  const right = pct(p90, min, max);
  const mid = pct(p50, min, max);
  return (
    <div data-testid="quantile-bar" data-p50-pct={String(Math.round(mid))}
         className="relative h-3 w-full rounded bg-slate-200">
      <div className="absolute h-3 rounded bg-emerald-300"
           style={{ left: `${left}%`, width: `${Math.max(0, right - left)}%` }} />
      <div className="absolute top-[-2px] h-4 w-0.5 bg-emerald-700"
           style={{ left: `${mid}%` }} title={`P50 ${p50.toFixed(1)}`} />
    </div>
  );
}
```

- [ ] **Step 4: Run tests to pass**

Run: `npm test -- QuantileBar`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/QuantileBar.tsx web/src/components/QuantileBar.test.tsx
git commit -m "add quantile band component"
```

---

### Task 5: Draft state hook (polling) + start form

`useDraftState` wraps TanStack Query: it holds the session id, polls `getState` every 5s while `status==="active"`, and exposes mutations (pick/skip/undo/botPick) that invalidate the query. A start form creates a session and stores the id in the URL (`?session=`) for resume.

**Files:**
- Create: `web/src/hooks/useDraftState.ts`, `web/src/app/providers.tsx`, `web/src/components/StartDraftForm.tsx`
- Modify: `web/src/app/layout.tsx` (wrap in providers)
- Test: `web/src/hooks/useDraftState.test.tsx`

**Interfaces:**
- Consumes: `api` (Task 3).
- Produces:
  - `Providers` (QueryClientProvider wrapper, `"use client"`).
  - `useDraftState(sessionId: string | null)` → `{ state, recs, isLoading, error, pick, skip, undo, botPick, recsQuery }`.
  - `StartDraftForm({ onCreated }: { onCreated: (id: string) => void })`.

- [ ] **Step 1: Install TanStack Query**

Run from `web/`: `npm install @tanstack/react-query`

- [ ] **Step 2: Write the failing hook test**

```tsx
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { vi, it, expect, beforeEach } from "vitest";
import { useDraftState } from "./useDraftState";
import { api } from "@/lib/api";

function wrapper({ children }: { children: React.ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

beforeEach(() => vi.restoreAllMocks());

it("loads state and recommendations for a session", async () => {
  vi.spyOn(api, "getState").mockResolvedValue({ session_id: "s", status: "active",
    is_my_turn: true, picks: [], my_roster: [], current_overall_pick: 1 } as never);
  vi.spyOn(api, "recommendations").mockResolvedValue([
    { player_id: "P1", vona_score: 1 } as never]);
  const { result } = renderHook(() => useDraftState("s"), { wrapper });
  await waitFor(() => expect(result.current.state?.session_id).toBe("s"));
  await waitFor(() => expect(result.current.recs?.[0].player_id).toBe("P1"));
});
```

- [ ] **Step 3: Run to confirm failure**

Run: `npm test -- useDraftState`
Expected: FAIL — cannot resolve hook.

- [ ] **Step 4: Implement `providers.tsx`**

```tsx
"use client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

export function Providers({ children }: { children: React.ReactNode }) {
  const [qc] = useState(() => new QueryClient());
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}
```

Wrap `web/src/app/layout.tsx` body: `<Providers>{children}</Providers>` (import from `./providers`).

- [ ] **Step 5: Implement `useDraftState.ts`**

```tsx
"use client";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { PickBody } from "@/lib/types";

export function useDraftState(sessionId: string | null) {
  const qc = useQueryClient();
  const enabled = !!sessionId;
  const key = ["draft", sessionId];

  const stateQuery = useQuery({
    queryKey: key, enabled,
    queryFn: () => api.getState(sessionId as string),
    refetchInterval: (q) => (q.state.data?.status === "active" ? 5000 : false),
  });

  const recsQuery = useQuery({
    queryKey: ["recs", sessionId, stateQuery.data?.current_overall_pick], enabled,
    queryFn: () => api.recommendations(sessionId as string, 10),
  });

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: key });
    qc.invalidateQueries({ queryKey: ["recs", sessionId] });
  };
  const mut = (fn: () => Promise<unknown>) =>
    useMutation({ mutationFn: fn, onSuccess: invalidate });

  const pick = useMutation({
    mutationFn: (body: PickBody) => api.recordPick(sessionId as string, body),
    onSuccess: invalidate });
  const skip = mut(() => api.recordPick(sessionId as string, { skip: true }));
  const undo = mut(() => api.undo(sessionId as string));
  const botPick = mut(() => api.botPick(sessionId as string));

  return {
    state: stateQuery.data, recs: recsQuery.data,
    isLoading: stateQuery.isLoading, error: stateQuery.error,
    pick, skip, undo, botPick, recsQuery,
  };
}
```

- [ ] **Step 6: Implement `StartDraftForm.tsx`**

```tsx
"use client";
import { useState } from "react";
import { api } from "@/lib/api";

export function StartDraftForm({ onCreated }: { onCreated: (id: string) => void }) {
  const [season, setSeason] = useState(2026);
  const [pos, setPos] = useState(1);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  async function submit(e: React.FormEvent) {
    e.preventDefault(); setBusy(true); setErr(null);
    try { onCreated((await api.createSession(season, pos)).session_id); }
    catch (e2) { setErr(e2 instanceof Error ? e2.message : "failed"); }
    finally { setBusy(false); }
  }
  return (
    <form onSubmit={submit} className="flex items-end gap-3">
      <label className="flex flex-col text-sm">Season
        <input type="number" value={season} onChange={(e) => setSeason(+e.target.value)}
               className="border rounded px-2 py-1" /></label>
      <label className="flex flex-col text-sm">Draft slot
        <input type="number" min={1} max={12} value={pos}
               onChange={(e) => setPos(+e.target.value)}
               className="border rounded px-2 py-1" /></label>
      <button disabled={busy} className="bg-emerald-600 text-white rounded px-4 py-1.5">
        {busy ? "Starting…" : "Start draft"}</button>
      {err && <span className="text-red-600 text-sm">{err}</span>}
    </form>
  );
}
```

- [ ] **Step 7: Run tests + build**

Run: `npm test -- useDraftState && npm run build`
Expected: PASS, build succeeds.

- [ ] **Step 8: Commit**

```bash
git add web/src/hooks web/src/app/providers.tsx web/src/app/layout.tsx web/src/components/StartDraftForm.tsx web/package.json web/package-lock.json
git commit -m "add draft-state polling hook, providers, start form"
```

---

### Task 6: Recommendation panel + roster panel + pick log

The three read panels of the draft room. All presentational, driven by `DraftState`/`Recommendation`.

**Files:**
- Create: `web/src/components/RecommendationPanel.tsx`, `web/src/components/RosterPanel.tsx`, `web/src/components/PickLog.tsx`
- Test: `web/src/components/RecommendationPanel.test.tsx`

**Interfaces:**
- Consumes: `Recommendation`, `DraftState`, `QuantileBar` (Task 4).
- Produces:
  - `RecommendationPanel({ recs, min, max, onDraft }: { recs: Recommendation[]; min: number; max: number; onDraft: (id: string) => void })`.
  - `RosterPanel({ state }: { state: DraftState })`.
  - `PickLog({ state }: { state: DraftState })`.

- [ ] **Step 1: Write the failing test**

```tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi, it, expect } from "vitest";
import { RecommendationPanel } from "./RecommendationPanel";

const rec = (over: object) => ({ player_id: "P1", name: "Star WR", position: "WR",
  team: "SF", vona_score: 4.2, value: 18, p10: 12, p50: 18, p90: 24, adp: 5,
  draft_round: 1, target_quantile: 0.25, forced_completion: false, ...over });

it("renders recs and fires onDraft with player_id", async () => {
  const onDraft = vi.fn();
  render(<RecommendationPanel recs={[rec({})]} min={0} max={30} onDraft={onDraft} />);
  expect(screen.getByText("Star WR")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: /draft/i }));
  expect(onDraft).toHaveBeenCalledWith("P1");
});

it("badges forced_completion picks", () => {
  render(<RecommendationPanel recs={[rec({ forced_completion: true })]}
                              min={0} max={30} onDraft={() => {}} />);
  expect(screen.getByText(/must fill/i)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run to confirm failure**

Run: `npm test -- RecommendationPanel`
Expected: FAIL.

- [ ] **Step 3: Implement `RecommendationPanel.tsx`**

```tsx
"use client";
import type { Recommendation } from "@/lib/types";
import { QuantileBar } from "./QuantileBar";

export function RecommendationPanel({ recs, min, max, onDraft }: {
  recs: Recommendation[]; min: number; max: number; onDraft: (id: string) => void;
}) {
  if (!recs.length) return <p className="text-slate-500">No recommendations.</p>;
  return (
    <ul className="space-y-2">
      {recs.map((r) => (
        <li key={r.player_id} className="border rounded p-3 flex flex-col gap-1">
          <div className="flex items-center justify-between">
            <div>
              <span className="font-semibold">{r.name ?? r.player_id}</span>
              <span className="ml-2 text-xs text-slate-500">{r.position} · {r.team ?? "—"}</span>
              {r.forced_completion &&
                <span className="ml-2 text-xs bg-amber-200 rounded px-1">must fill slot</span>}
            </div>
            <button onClick={() => onDraft(r.player_id)}
                    className="bg-emerald-600 text-white text-sm rounded px-3 py-1">Draft</button>
          </div>
          <QuantileBar p10={r.p10} p50={r.p50} p90={r.p90} min={min} max={max} />
          <div className="flex justify-between text-xs text-slate-500">
            <span>VONA {r.vona_score.toFixed(1)}</span>
            <span>P10 {r.p10.toFixed(0)} · P50 {r.p50.toFixed(0)} · P90 {r.p90.toFixed(0)}</span>
            <span>ADP {r.adp?.toFixed(0) ?? "—"}</span>
          </div>
        </li>
      ))}
    </ul>
  );
}
```

- [ ] **Step 4: Implement `RosterPanel.tsx` + `PickLog.tsx`**

`RosterPanel.tsx`:

```tsx
import type { DraftState } from "@/lib/types";

export function RosterPanel({ state }: { state: DraftState }) {
  return (
    <div>
      <h3 className="font-semibold mb-2">My roster ({state.my_roster.length})</h3>
      <ul className="text-sm space-y-1">
        {state.my_roster.map((p, i) => (
          <li key={`${p.player_id}-${i}`} className="flex justify-between">
            <span>{p.name ?? p.player_id}</span>
            <span className="text-slate-500">{p.position ?? "—"}{p.bye_week ? ` · bye ${p.bye_week}` : ""}</span>
          </li>
        ))}
      </ul>
      <h4 className="font-medium mt-3 text-sm">Open starters</h4>
      <ul className="text-xs text-slate-600">
        {Object.entries(state.open_starters).map(([slot, n]) =>
          <li key={slot}>{slot}: {n}</li>)}
      </ul>
    </div>
  );
}
```

`PickLog.tsx`:

```tsx
import type { DraftState } from "@/lib/types";

export function PickLog({ state }: { state: DraftState }) {
  return (
    <ol className="text-sm space-y-1">
      {state.picks.map((p) => (
        <li key={p.pick_number} className={p.mine ? "font-semibold text-emerald-700" : ""}>
          {p.pick_number}. {p.skipped ? "— skipped —" : (p.name ?? p.player_id)}
          {p.mine && " (you)"}
        </li>
      ))}
    </ol>
  );
}
```

- [ ] **Step 5: Run tests to pass**

Run: `npm test -- RecommendationPanel`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add web/src/components/RecommendationPanel.tsx web/src/components/RosterPanel.tsx web/src/components/PickLog.tsx web/src/components/RecommendationPanel.test.tsx
git commit -m "add recommendation, roster, and pick-log panels"
```

---

### Task 7: Draft room page (manual mode)

Compose the panels into `/draft`. Reads `?session=` from the URL (resume), renders start form when absent, wires pick/skip/undo, shows whose turn it is, and computes the QuantileBar min/max from the current recs.

**Files:**
- Create: `web/src/app/draft/page.tsx`, `web/src/components/DraftRoom.tsx`
- Test: `web/src/components/DraftRoom.test.tsx`

**Interfaces:**
- Consumes: `useDraftState`, `StartDraftForm`, `RecommendationPanel`, `RosterPanel`, `PickLog`.
- Produces: `DraftRoom({ sessionId, onSession }: { sessionId: string | null; onSession: (id: string) => void })`; route `/draft`.

- [ ] **Step 1: Write the failing test**

```tsx
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { vi, it, expect, beforeEach } from "vitest";
import { DraftRoom } from "./DraftRoom";
import { api } from "@/lib/api";

const wrap = (ui: React.ReactNode) => {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{ui}</QueryClientProvider>;
};
beforeEach(() => vi.restoreAllMocks());

it("shows the start form when no session", () => {
  render(wrap(<DraftRoom sessionId={null} onSession={() => {}} />));
  expect(screen.getByRole("button", { name: /start draft/i })).toBeInTheDocument();
});

it("renders recs and turn banner for an active session", async () => {
  vi.spyOn(api, "getState").mockResolvedValue({ session_id: "s", status: "active",
    is_my_turn: true, current_overall_pick: 1, picks: [], my_roster: [],
    open_starters: {}, teams: 12, rounds: 15 } as never);
  vi.spyOn(api, "recommendations").mockResolvedValue([{ player_id: "P1", name: "Star",
    position: "WR", team: "SF", vona_score: 4, value: 18, p10: 12, p50: 18, p90: 24,
    adp: 5, draft_round: 1, target_quantile: 0.25, forced_completion: false } as never]);
  render(wrap(<DraftRoom sessionId="s" onSession={() => {}} />));
  await waitFor(() => expect(screen.getByText("Star")).toBeInTheDocument());
  expect(screen.getByText(/your pick/i)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run to confirm failure**

Run: `npm test -- DraftRoom`
Expected: FAIL.

- [ ] **Step 3: Implement `DraftRoom.tsx`**

```tsx
"use client";
import { useMemo } from "react";
import { useDraftState } from "@/hooks/useDraftState";
import { StartDraftForm } from "./StartDraftForm";
import { RecommendationPanel } from "./RecommendationPanel";
import { RosterPanel } from "./RosterPanel";
import { PickLog } from "./PickLog";

export function DraftRoom({ sessionId, onSession }: {
  sessionId: string | null; onSession: (id: string) => void;
}) {
  const d = useDraftState(sessionId);
  const [min, max] = useMemo(() => {
    const ps = (d.recs ?? []).flatMap((r) => [r.p10, r.p90]);
    return ps.length ? [Math.min(...ps), Math.max(...ps)] : [0, 30];
  }, [d.recs]);

  if (!sessionId) {
    return <div className="p-8"><h1 className="text-xl font-bold mb-4">Live draft</h1>
      <StartDraftForm onCreated={onSession} /></div>;
  }
  if (d.isLoading || !d.state) return <p className="p-8">Loading draft…</p>;
  const s = d.state;
  return (
    <div className="p-6 grid grid-cols-1 lg:grid-cols-[1fr_20rem] gap-6">
      <section>
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-xl font-bold">
            Pick {s.current_overall_pick} ·{" "}
            <span className={s.is_my_turn ? "text-emerald-700" : "text-slate-500"}>
              {s.is_my_turn ? "Your pick" : "Waiting on other teams"}</span>
          </h1>
          <div className="flex gap-2">
            <button onClick={() => d.skip.mutate()} className="border rounded px-3 py-1 text-sm">Skip</button>
            <button onClick={() => d.undo.mutate()} className="border rounded px-3 py-1 text-sm">Undo</button>
          </div>
        </div>
        <RecommendationPanel recs={d.recs ?? []} min={min} max={max}
          onDraft={(id) => d.pick.mutate({ player_id: id })} />
      </section>
      <aside className="space-y-6">
        <RosterPanel state={s} />
        <div><h3 className="font-semibold mb-2">Pick log</h3><PickLog state={s} /></div>
      </aside>
    </div>
  );
}
```

- [ ] **Step 4: Implement the route `web/src/app/draft/page.tsx`**

```tsx
"use client";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { DraftRoom } from "@/components/DraftRoom";

function DraftPageInner() {
  const params = useSearchParams();
  const router = useRouter();
  const sessionId = params.get("session");
  return <DraftRoom sessionId={sessionId}
    onSession={(id) => router.replace(`/draft?session=${id}`)} />;
}

export default function DraftPage() {
  return <Suspense><DraftPageInner /></Suspense>;
}
```

- [ ] **Step 5: Run tests + build**

Run: `npm test -- DraftRoom && npm run build`
Expected: PASS, build succeeds.

- [ ] **Step 6: Commit**

```bash
git add web/src/app/draft web/src/components/DraftRoom.tsx web/src/components/DraftRoom.test.tsx
git commit -m "add manual-mode draft room page"
```

---

### Task 8: Mock-draft demo mode (zero login)

`/demo` runs a session where, after the user picks, the frontend auto-advances every opponent turn via `bot-pick` until it's the user's turn again. Same panels; no start form (auto-creates a session on mount at a fixed slot).

**Files:**
- Create: `web/src/app/demo/page.tsx`, `web/src/components/DemoRoom.tsx`
- Test: `web/src/components/DemoRoom.test.tsx`

**Interfaces:**
- Consumes: `useDraftState`, `api.createSession`, `RecommendationPanel`, `RosterPanel`.
- Produces: `DemoRoom()`; route `/demo`.

- [ ] **Step 1: Write the failing test**

```tsx
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { vi, it, expect, beforeEach } from "vitest";
import { DemoRoom } from "./DemoRoom";
import { api } from "@/lib/api";

const wrap = (ui: React.ReactNode) => {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={qc}>{ui}</QueryClientProvider>;
};
beforeEach(() => vi.restoreAllMocks());

it("auto-creates a session and shows recs on the user's turn", async () => {
  vi.spyOn(api, "createSession").mockResolvedValue({ session_id: "demo" } as never);
  vi.spyOn(api, "getState").mockResolvedValue({ session_id: "demo", status: "active",
    is_my_turn: true, current_overall_pick: 1, picks: [], my_roster: [],
    open_starters: {}, teams: 12, rounds: 15 } as never);
  vi.spyOn(api, "recommendations").mockResolvedValue([{ player_id: "P1", name: "Star",
    position: "WR", team: "SF", vona_score: 4, value: 18, p10: 12, p50: 18, p90: 24,
    adp: 5, draft_round: 1, target_quantile: 0.25, forced_completion: false } as never]);
  render(wrap(<DemoRoom />));
  await waitFor(() => expect(screen.getByText("Star")).toBeInTheDocument());
});
```

- [ ] **Step 2: Run to confirm failure**

Run: `npm test -- DemoRoom`
Expected: FAIL.

- [ ] **Step 3: Implement `DemoRoom.tsx`**

Auto-create a session on mount (slot 1), then an effect advances bots whenever it is not the user's turn and the draft is active:

```tsx
"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "@/lib/api";
import { useDraftState } from "@/hooks/useDraftState";
import { RecommendationPanel } from "./RecommendationPanel";
import { RosterPanel } from "./RosterPanel";

const DEMO_SEASON = 2024;   // a completed season always has projections + ADP

export function DemoRoom() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const creating = useRef(false);
  useEffect(() => {
    if (sessionId || creating.current) return;
    creating.current = true;
    api.createSession(DEMO_SEASON, 1).then((s) => setSessionId(s.session_id));
  }, [sessionId]);

  const d = useDraftState(sessionId);
  const s = d.state;
  const advancing = d.botPick.isPending;

  useEffect(() => {
    if (!s || s.status !== "active" || s.is_my_turn || advancing) return;
    if (s.remaining_picks <= 0) return;
    d.botPick.mutate();
  }, [s?.current_overall_pick, s?.is_my_turn, s?.status, advancing]); // eslint-disable-line

  const [min, max] = useMemo(() => {
    const ps = (d.recs ?? []).flatMap((r) => [r.p10, r.p90]);
    return ps.length ? [Math.min(...ps), Math.max(...ps)] : [0, 30];
  }, [d.recs]);

  if (!s) return <p className="p-8">Setting up a mock draft…</p>;
  return (
    <div className="p-6 grid grid-cols-1 lg:grid-cols-[1fr_20rem] gap-6">
      <section>
        <h1 className="text-xl font-bold mb-1">Mock draft demo</h1>
        <p className="text-sm text-slate-500 mb-4">
          You draft from slot {s.draft_position}; the other {s.teams - 1} teams pick by ADP.
        </p>
        <div className="mb-3 text-sm">
          Pick {s.current_overall_pick} —{" "}
          {s.is_my_turn ? <b className="text-emerald-700">your pick</b> : "bots drafting…"}
        </div>
        <RecommendationPanel recs={d.recs ?? []} min={min} max={max}
          onDraft={(id) => d.pick.mutate({ player_id: id })} />
      </section>
      <aside><RosterPanel state={s} /></aside>
    </div>
  );
}
```

- [ ] **Step 4: Implement the route `web/src/app/demo/page.tsx`**

```tsx
import { DemoRoom } from "@/components/DemoRoom";
export default function DemoPage() { return <DemoRoom />; }
```

- [ ] **Step 5: Run tests + build**

Run: `npm test -- DemoRoom && npm run build`
Expected: PASS, build succeeds.

- [ ] **Step 6: Commit**

```bash
git add web/src/app/demo web/src/components/DemoRoom.tsx web/src/components/DemoRoom.test.tsx
git commit -m "add zero-login mock-draft demo mode"
```

---

### Task 9: Landing page

The public entry: what it is, a "Try the demo" button (→ `/demo`), a "Start a live draft" button (→ `/draft`). Recruiter-facing, must render without a backend call.

**Files:**
- Modify: `web/src/app/page.tsx`
- Test: `web/src/app/page.test.tsx`

**Interfaces:**
- Produces: route `/` with links to `/demo` and `/draft`.

- [ ] **Step 1: Write the failing test**

```tsx
import { render, screen } from "@testing-library/react";
import Home from "./page";

it("links to the demo and the live draft", () => {
  render(<Home />);
  expect(screen.getByRole("link", { name: /try the demo/i })).toHaveAttribute("href", "/demo");
  expect(screen.getByRole("link", { name: /live draft/i })).toHaveAttribute("href", "/draft");
});
```

- [ ] **Step 2: Run to confirm failure**

Run: `npm test -- page.test`
Expected: FAIL.

- [ ] **Step 3: Implement `page.tsx`**

```tsx
import Link from "next/link";

export default function Home() {
  return (
    <main className="max-w-3xl mx-auto p-10">
      <h1 className="text-4xl font-bold mb-3">FantasyForecast</h1>
      <p className="text-lg text-slate-600 mb-8">
        A draft assistant powered by quantile projection models (P10/P50/P90) and a
        VONA recommender — it tells you who to draft and how risky each pick is.
      </p>
      <div className="flex gap-4">
        <Link href="/demo"
          className="bg-emerald-600 text-white rounded px-5 py-2.5 font-medium">
          Try the demo</Link>
        <Link href="/draft"
          className="border border-slate-300 rounded px-5 py-2.5 font-medium">
          Start a live draft</Link>
      </div>
    </main>
  );
}
```

- [ ] **Step 4: Run tests to pass**

Run: `npm test -- page.test`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add web/src/app/page.tsx web/src/app/page.test.tsx
git commit -m "add landing page"
```

---

### Task 10: Playwright e2e smoke (demo happy path) + README

One browser test covering the spec's required smoke: load `/demo`, wait for the board, make a pick, see the roster grow. Runs against a live backend + dev server. Plus a `web/README.md`.

**Files:**
- Create: `web/playwright.config.ts`, `web/tests/e2e/demo.spec.ts`, `web/README.md`
- Modify: `web/package.json` (scripts), `README.md` (repo root, link to web)

**Interfaces:**
- Consumes: running FastAPI (`:8000`) + Next dev server (`:3000`).
- Produces: `npm run test:e2e`.

- [ ] **Step 1: Install Playwright**

Run from `web/`: `npm install -D @playwright/test && npx playwright install chromium`

- [ ] **Step 2: Configure Playwright**

Create `web/playwright.config.ts`:

```ts
import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests/e2e",
  use: { baseURL: "http://localhost:3000" },
  webServer: {
    command: "npm run dev",
    url: "http://localhost:3000",
    reuseExistingServer: true,
    timeout: 120_000,
  },
});
```

Add script to `web/package.json`: `"test:e2e": "playwright test"`.

- [ ] **Step 3: Write the smoke test**

Create `web/tests/e2e/demo.spec.ts`:

```ts
import { test, expect } from "@playwright/test";

test("demo: pick a player and see the roster grow", async ({ page }) => {
  await page.goto("/demo");
  // wait until it is our turn and recs render (bots may pick first)
  const draftButtons = page.getByRole("button", { name: "Draft" });
  await expect(draftButtons.first()).toBeVisible({ timeout: 30_000 });
  await draftButtons.first().click();
  await expect(page.getByText(/My roster \(1\)/)).toBeVisible({ timeout: 30_000 });
});
```

- [ ] **Step 4: Run the smoke (backend must be up)**

Run from repo root: `venv/bin/uvicorn api.main:app --port 8000 &` then from `web/`: `NEXT_PUBLIC_API_URL=http://localhost:8000 npm run test:e2e`
Expected: 1 test PASS. (Requires a completed pipeline so `2024` has projections + ADP; if `/players?season=2024` is empty, seed first.) Kill the server after.

- [ ] **Step 5: Write `web/README.md`**

````markdown
# FantasyForecast Web

Next.js draft assistant. Talks to the FastAPI backend (`../api`).

## Dev

```bash
cp .env.local.example .env.local     # point NEXT_PUBLIC_API_URL at the API
npm install
npm run dev                          # http://localhost:3000
```

Backend (from repo root): `venv/bin/uvicorn api.main:app --port 8000`

- `/` landing · `/demo` zero-login mock draft · `/draft` live manual draft
- `npm test` unit/component (Vitest) · `npm run test:e2e` Playwright smoke
````

Append to repo-root `README.md` under the API section: a `## Web` note linking to `web/README.md`.

- [ ] **Step 6: Commit**

```bash
git add web/playwright.config.ts web/tests/e2e web/README.md web/package.json web/package-lock.json README.md
git commit -m "add playwright demo smoke and web readme"
```

---

## Self-Review Notes

- **Spec coverage:** landing page (Task 9), draft room with recommendation panel + P10/P50/P90 bands (Tasks 4,6,7), mock-draft demo zero-login (Task 8 + Task 1 backend), session resume via `?session=` (Task 7), polling ~5s (Task 5). Yahoo sync, `/explain` cards, deploy are Plans 3-4 + deploy step by design.
- **Deliberate deviations:** explanation cards (spec's draft-room click → LLM) are stubbed out of this plan — they arrive in Plan 4; the recommendation card leaves room for them. Demo uses season 2024 (a completed season with ADP) because FFC has no ADP for the just-finished season (documented repo gotcha).
- **Type consistency:** `DraftState`/`Recommendation`/`Player` in `types.ts` mirror `api/schemas.py` field-for-field; `useDraftState` returns `{state, recs, pick, skip, undo, botPick}` used identically by `DraftRoom` and `DemoRoom`.
- **Backend touch:** only Task 1 (`bot-pick`) modifies `api/`; it reuses `_bot_pick` and keeps the ADP wall intact (recommender-side ADP use). Full `venv/bin/pytest` gates it.
