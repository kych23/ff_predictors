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
- `/demo` — zero-login mock draft: you pick, ADP bots fill the rest (season 2024)
- `/draft` — live manual draft; resume via `?session=<id>`

## Tests

```bash
npm test          # unit + component (Vitest)
npm run test:e2e  # Playwright demo smoke — needs the backend up on :8000
```

The e2e smoke drives the real demo flow (load → bots pick → draft a player → roster
grows), so it requires a completed pipeline (2024 projections + ADP present).
