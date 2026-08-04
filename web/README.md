# FantasyForecast Web

Next.js draft assistant. A pure REST client of the FastAPI backend (`../api`) — no
engine logic runs in the browser.

## Dev

```bash
cp .env.local.example .env.local      # point NEXT_PUBLIC_API_URL at the API
npm install
npm run dev                           # http://localhost:3000
```

Backend (from repo root): `venv/bin/uvicorn api.main:app --port 8000`

## Routes

- `/` — landing page (renders with no backend)
- `/draft` — no `?session=`: connect form (season, draft slot, Yahoo league
  key); with `?session=<id>`: read-only live view (board, roster, VONA
  recommendations), polling `/sync` every 5s. Not a draft client — you
  draft on Yahoo's own app/site; this mirrors it.

Yahoo sync is currently gated on Yahoo's manual API-access approval — see
the repo root README's "Connect Yahoo" section.

## Tests

```bash
npm test          # unit + component (Vitest)
```

No Playwright e2e currently — the prior smoke drove the now-removed demo
mode. A connect→sync e2e would need a mocked Yahoo OAuth + draftresults
server (flagged as a follow-up in `notes/yahoo-live-sync-frontend.md`).
