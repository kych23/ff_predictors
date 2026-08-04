# FantasyForecast

PPR fantasy football draft assistant. Pooled LightGBM quantile model (P10/P50/P90) + VONA-based draft recommender, plus a weekly start/sit optimizer for the season itself.

## Stack

- Python — ETL, feature engineering, ML
- LightGBM — pooled quantile GBM across QB/RB/WR/TE
- PostgreSQL — snapshot-stamped storage
- nflverse (`nflreadpy`) — play-by-play, rosters, combine data
- CFBD REST API — college stats (for rookies)

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`.env` in project root:

```
DATABASE_URL=postgresql://user:password@host:5432/dbname          # serving (Supabase)
RESEARCH_DATABASE_URL=postgresql://localhost:5432/ff_research      # research (local PG)
CFBD_API_KEY=your_key_here   # optional — enables college features
```

League settings live in `config/league.yaml`.

### Two databases

Storage is split so the hosted DB stays under Supabase's free-tier 500 MB cap:

- **Research** (`RESEARCH_DATABASE_URL`, local Postgres) — full raw box scores,
  features, labels, and backtest rows. The pipeline, training, and benchmarks
  read and write this. ~640 MB.
- **Serving** (`DATABASE_URL`, Supabase) — only what the API and live draft/weekly
  serving read: players, adp, projections, weekly_projections, current-season
  weekly_features, draft_sessions, latest snapshot. ~43 MB.

`scripts/publish_serving.py` (the last pipeline step) copies the serving subset
research → serving. If `RESEARCH_DATABASE_URL` is unset, both roles resolve to the
same DB — the original single-DB behavior (used by the whole test suite). Start
local Postgres with
`/opt/homebrew/opt/postgresql@15/bin/pg_ctl -D /opt/homebrew/var/postgresql@15 start`.

## Pipeline

Each season: wipe the DB, then run the pipeline script.

**Step 0 — wipe Supabase** (run in Supabase SQL Editor):

```sql
DROP SCHEMA public CASCADE;
CREATE SCHEMA public;
```

**Step 1 — run full pipeline** (defaults to current year):

```bash
bash scripts/run_pipeline.sh
```

Or manually, step by step:

```bash
python scripts/seed_db.py --start 2012 --end 2026
python scripts/build_labels.py
python scripts/build_features.py --start 2012 --end 2026
python scripts/train_projection.py
python scripts/run_benchmark.py --with-sim
# weekly grain (start/sit)
python scripts/build_weekly_data.py --start 2017 --end 2026
python scripts/train_weekly_projection.py
python scripts/run_weekly_benchmark.py --start 2017
```

## Live Draft

```bash
python scripts/draft.py --season 2026 --position 4   # --position = your draft slot (1..teams)
python scripts/draft.py --season 2026 --position 4 --resume   # continue an in-progress draft
```

Commands during draft:

- `go` — auto-advance opponents by ADP (falls back to projection order when ADP is exhausted in late rounds) up to your next pick; blocked if it's your own turn
- `me <player name>` — record your pick (fuzzy match; same-name players prompt to disambiguate by position/team)
- `<player name>` — record an opponent's pick (on your turn, a bare name is recorded as **your** pick)
- `skip` — advance the pick counter by one opponent pick without naming the player (useful when ADP is fully exhausted)
- `undo` — revert the last command (a pick, a `go` batch, or a skip); state is rebuilt by replaying the pick log
- `roster` — show your current team, open starter slots, and any bye-week stacks
- `board` — reprint the recommendation board
- `quit` — exit

The header shows `** YOUR PICK (round N) **` when it's your turn, so you know to use `me <name>` instead of `go`.

**Crash-safe**: every pick is written to `draft_state_<season>_slot<pos>.json` as it happens. If the CLI dies mid-draft (network blip, accidental Ctrl-C), relaunch with `--resume` to pick up exactly where you left off. The save file is the same event log that powers `undo`.

At each of your picks it shows the **top 10** candidates with a **FIT** score and the gap to the top pick, so you decide rather than blindly taking #1:

```
  Round 1  (risk quantile 0.25)
   #  PLAYER             POS    FIT     Δ#1   proj    ADP   wait
   1  Justin Jefferson   WR    2.97     —    12.4   10.1   0.44
   2  Puka Nacua         WR    1.06   -1.91  15.3    3.0   0.44
   3  Ja'Marr Chase      WR    0.91   -2.06  15.6    3.1   0.44
   4  A.J. Brown         WR    0.58   -2.39  13.1   18.4   0.44
   5  Travis Kelce       TE    0.17   -2.80   8.1   42.0   1.23
   ... (top 10 shown)
   read: CLEAR #1 — leads #2 by 1.91
   runs (picks 1→13, 12 picks): QB 1! | RB 4 | WR 6 | TE 2
```

- **FIT** — the VONA fit-for-your-roster score (higher = better pick now).
- **Δ#1** — gap to the top pick; small gaps across the board mean the choices are about equal.
- **proj** — the risk-adjusted projection at this round's target quantile; **ADP** — the market's average draft position, so you can see where FIT disagrees with the market.
- **wait** — the VONA wait term: the expected VOR of the best same-position player surviving to your next pick. Low wait = scarce position (grab now); high wait = deep position (safe to wait). Same value for all players at the same position.
- **read** — auto-classifies the spread: _CLEAR #1_ (leads #2 by ≥1.0), _TOSS-UP_ (top 10 within 0.5 → pick your preference), or a _slight edge_ in between.
- **runs** — how many players at each position the market expects to go before your next pick (`!` = 0–1 left, i.e. that position is running out).

## Weekly Start/Sit ("Who Should I Start?")

Once the season starts, the weekly model answers the other hard question: who goes in the lineup this week. Two commands per week:

```bash
python scripts/project_week.py --season 2026 --week 5          # 1. project the upcoming week
python scripts/start.py --season 2026 --week 5 --roster my_team.yaml   # 2. optimize the lineup
```

Or refresh data + project in one shot (run Tuesdays, after the previous week's stats land):

```bash
bash scripts/update_week.sh 2026 5    # re-seed stats → rebuild weekly grain → project week 5
```

`project_week.py` builds as-of-kickoff features for the target week (rolling in-season stats, opponent DvP, Vegas lines, season-P50 prior), trains on all strictly-prior weeks, and writes forward projections (`weekly_fwd_v1.*`). `start.py` then solves the lineup as a small ILP — exact slot assignment, FLEX included — and prints START/BENCH verdicts with matchup grades.

Your roster lives in a YAML file:

```yaml
season: 2026
roster:
  - { name: Josh Allen, position: QB, team: BUF }
  - { name: Bijan Robinson, position: RB, team: ATL }
  - { name: Amon-Ra St. Brown, position: WR, team: DET }
  - { name: Travis Kelce, position: TE, team: KC, status: questionable }
  # ... status IR/OUT excludes a player from the lineup
```

Options:

- `--strategy safe|balanced|upside` — maximize P10 (floor), P50 (median, default), or P90 (ceiling). Chasing a comeback? `upside`. Protecting a lead? `safe`.
- `compare "Player A" vs "Player B"` — head-to-head: quantiles, matchup grade, and a verdict (< 1 pt median gap is called a TOSS-UP rather than a fake edge).
- Falls back to season projections (flagged as such) if weekly projections don't exist for that week.

**Two-layer model**: the season P50 projection enters the weekly model as a prior feature; the weekly GBM learns how far to adjust it given in-season context. Same pooled quantile architecture, same rearrangement monotonicity, split-conformal calibration by position bucket. All backward-looking features pass through `leakage_guard.prior_weeks()` — nothing from the target week or later ever enters the matrix.

## Architecture

**ADP wall**: `src/projection/` and `src/features/` never import the `adp` module — the **draft market** (ADP) stays quarantined so the engine is measured against it, never trained on it. Only `src/ingest/adp.py`, `src/recommender/`, and `src/benchmark/` may touch ADP.

**Team scoring context** (deliberate market signal, *not* an ADP-wall violation): `src/features/team_context.py` adds a forward-looking view of each player's offense from the **Week-1 betting line** (nflverse `total_line`/`spread_line`) — `team_ctx_implied_pts = total_line/2 ± spread_line/2`, plus the game total and the team's spread. Week-1 lines are posted pre-kickoff, so it's a preseason-safe artifact (like the Week-1 depth chart), attached to each player by their season-Y team. This intentionally puts *game-line* market data into the projection features (the ADP/draft-market wall is untouched); it prices the offseason roster/coaching changes lagged production can't see. Uses `total_line`/`spread_line` only — never `total` (the final score, which would leak).

**Model**: one pooled `QuantileGBM` (position as categorical feature) — not per-position models. Quantile monotonicity via rearrangement (Chernozhukov 2010), not clipping.

**Recommender**: VONA score = marginal VOR − E[best same-position surviving to next pick]. ADP survival via log-normal distribution. Round-shifted quantile (P25 early → P85 late).

## Benchmark Results

Pre-registered gate: **NDCG@k**, paired per season, with Wilcoxon signed-rank / sign tests and a bootstrap 95% CI (see [Glossary](#glossary)). Ground-truth relevance is **actual-season FPPG** (availability-neutral, so injuries don't confound the ranking). Reported on the data-complete era — test seasons **2017–2024** (8 evaluable seasons, after the 5-season cross-validation warmup). Current snapshot: **3,386 projections**.

### Tier-1 — projection ranking vs ADP (per season, k = 84)

| Season | n   | NDCG@84 engine | NDCG@84 ADP | Spearman engine | Spearman ADP | Hit@84 engine | Hit@84 ADP |
| ------ | --- | -------------- | ----------- | --------------- | ------------ | ------------- | ---------- |
| 2017   | 136 | 0.889          | 0.851       | 0.632           | 0.425        | 0.762         | 0.714      |
| 2018   | 144 | 0.904          | 0.866       | 0.645           | 0.412        | 0.738         | 0.655      |
| 2019   | 148 | 0.946          | 0.866       | 0.734           | 0.470        | 0.810         | 0.702      |
| 2020   | 141 | 0.931          | 0.855       | 0.759           | 0.477        | 0.798         | 0.762      |
| 2021   | 150 | 0.943          | 0.860       | 0.739           | 0.538        | 0.821         | 0.702      |
| 2022   | 131 | 0.925          | 0.882       | 0.773           | 0.551        | 0.810         | 0.821      |
| 2023   | 156 | 0.923          | 0.881       | 0.693           | 0.530        | 0.821         | 0.762      |
| 2024   | 160 | 0.918          | 0.839       | 0.673           | 0.471        | 0.750         | 0.631      |

**Tier-1 gate (paired across 8 seasons):**

| Metric                 | mean(engine − ADP) | Wilcoxon p | sign-test p | bootstrap 95% CI | Result   |
| ---------------------- | ------------------ | ---------- | ----------- | ---------------- | -------- |
| NDCG@84 (full board)   | **+0.062**         | 0.0078     | 0.0078      | [+0.048, +0.075] | **PASS** |
| NDCG@36 (early rounds) | **+0.083**         | 0.0078     | 0.0078      | [+0.064, +0.102] | **PASS** |

Engine beats ADP in **8 of 8** seasons. Honest ceiling: ~8–9 evaluable seasons is the hard limit of available NFL history — not "many" — so significance is taken across season × player cross-sections.

### Tier-3 — risk calibration (interval honesty)

| Bucket           | n    | P10–P90 coverage | nominal |
| ---------------- | ---- | ---------------- | ------- |
| overall          | 3386 | 0.804            | 0.80    |
| established_vet  | 1646 | 0.801            | 0.80    |
| rookie           | 479  | 0.804            | 0.80    |
| second_year      | 566  | 0.802            | 0.80    |
| team_changed_vet | 695  | 0.816            | 0.80    |

Coverage is from the **split-conformal per-bucket** calibrator (`src/projection/calibrate.py`), validated on the 2026-06-27 run (with the `team_context` feature). The width scale per side is a quantile of the normalized nonconformity score (residual ÷ predicted half-width), which targets the empirical coverage fraction directly — unlike the earlier mean-matching scale, which left intervals narrow (~0.77, rookies worst at 0.756) because matching *mean* half-widths under-covers whenever per-row widths are heterogeneous (exactly the rookie case). Every bucket now sits on the 0.80 nominal; the rookie bucket — the whole reason for per-type calibration — went 0.756 → 0.804. The calibrator only ever widens (scale floored at 1.0), so well-calibrated buckets are left untouched.

### Tier-2 — recommender vs ADP bots

Draft simulation: our recommender vs 11 bots drafting by ADP, on `SIM_ELIGIBLE` seasons, rosters scored availability-neutral (starting-lineup FPPG). Across **35 (season × slot) drafts**:

| Metric              | mean(ours − bots) | Wilcoxon p | bootstrap 95% CI | wins    | Result   |
| ------------------- | ----------------- | ---------- | ---------------- | ------- | -------- |
| Starting-lineup PPG | **+4.74**         | 0.000277   | [+2.64, +6.94]   | 28 / 35 | **PASS** |

The recommender — not just the engine — beats the market. Adding the `team_context` market feature lifted this tier (mean edge +3.97 → +4.74, win rate 25 → 28 of 35, p 0.0014 → 0.000277) while leaving Tier-1 flat within its CI. Down years like 2024 still pull the mean, but the paired edge is robust.

### Weekly — lineup benchmark (points left on bench)

Simulated rosters (snake-draft, averaged over early/mid/late draft slots), lineups set each week from OOF weekly projections, then scored against what actually happened:

| Metric                    | Result | Target | Gate     |
| ------------------------- | ------ | ------ | -------- |
| Mean pts left on bench    | 18.38  | < 20.0 | **PASS** |
| Optimal starter hit rate  | 67.5%  | —      | —        |

Weekly OOF interval coverage sits at 0.798–0.812 per position bucket against the 0.80 nominal, under leave-one-fold-out calibration (each season calibrated only on the *other* seasons, so reported coverage can't flatter itself).

## API

FastAPI service exposing the draft recommender (backend for the web draft room).

```bash
venv/bin/uvicorn api.main:app --reload --port 8000
```

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
`POST /draft/sessions/{id}/bot-pick` advances one ADP-bot opponent (demo mode).

## Web

Next.js draft room + zero-login mock-draft demo + landing page in [`web/`](web/README.md).
It is a pure REST client of the API above.

```bash
cd web && npm install && npm run dev   # http://localhost:3000  (needs the API on :8000)
```

## Data Sources

- [nflverse](https://www.nflverse.com/) — play-by-play, rosters, combine, draft picks, **Week-1 betting lines** (`total_line`/`spread_line`, 100% coverage 2012–2025) for team scoring context
- [Fantasy Football Calculator](https://fantasyfootballcalculator.com/) — full-PPR ADP
- [CFBD](https://collegefootballdata.com/) — college stats (optional)

## Glossary

**Scoring & fantasy terms**

- **PPR** — Points Per Reception: scoring that awards 1 point per catch.
- **ADP** — Average Draft Position: where a player is drafted on average across thousands of public drafts. The market baseline this project aims to beat.
- **FPPG** — Fantasy Points Per Game: the model's prediction target (a per-game _rate_, not a season total — so it isn't distorted by missed games).
- **VOR** — Value Over Replacement: a player's projected value minus a freely-available "replacement" player at his position (the first one who doesn't earn a starting slot).
- **VONA** — Value Of Not Available: VOR _now_ minus the expected value of the best same-position player still available at your _next_ pick. Encodes "take the scarce position now, wait on the deep one."
- **FIT** — the draft UI's per-player number: a player's VONA score at your current pick. Higher = better fit for your roster right now; the spread across the top 10 shows whether one pick clearly dominates or the choices are about equal.
- **DvP** — Defense vs Position: how many fantasy points a defense has allowed to each position so far this season. Drives the weekly matchup grade (A/B/C).
- **ILP** — Integer Linear Program: the exact optimizer behind start/sit. Assigns each rostered player to a starting slot (or bench) to maximize the chosen quantile's total, with FLEX eligibility as a constraint.

**Model terms**

- **GBM** — Gradient-Boosted (decision) trees Model; an ensemble of trees built sequentially.
- **LightGBM** — a fast, widely-used GBM implementation (the model used here).
- **P10 / P50 / P90** — the 10th / 50th (median) / 90th percentile of a player's predicted outcome: the low / middle / high cases. The model predicts a _range_, not one number.
- **Snapshot** — a frozen extraction of source data (tagged with a `snapshot_id`) so results stay reproducible even though nflverse retroactively corrects past stats.

**Metrics**

- **NDCG@k** — Normalized Discounted Cumulative Gain over the top _k_ players: a 0–1 ranking-quality score that weights the top of the list most heavily (getting pick #1 right matters more than #95). The primary gate metric; `k=84` ≈ the startable universe, `k=36` ≈ the early rounds.
- **Spearman** — Spearman rank correlation between the predicted order and the actual finish (−1 to +1; higher is better).
- **Hit@k** — Hit rate: the fraction of the true top-_k_ players that the ranking also placed in its top _k_.
- **Pinball loss** — the standard quantile-regression loss; lower means a better-calibrated quantile (P10/P50/P90).
- **Coverage** — the fraction of actual outcomes that fell inside the predicted P10–P90 interval (target = 0.80).
- **Bootstrap 95% CI** — a confidence interval estimated by resampling; if it excludes 0, the measured edge is unlikely to be noise.
- **Wilcoxon signed-rank / sign test** — paired non-parametric significance tests comparing engine vs ADP across seasons.

**Data sources**

- **CFBD** — College Football Data API (college statistics for rookies).
- **nflverse** — open NFL data project, accessed via the `nflreadpy` Python library.
- **Fantasy Football Calculator (FFC)** - full PPR ADP.
