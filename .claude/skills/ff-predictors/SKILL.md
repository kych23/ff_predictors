---
name: ff-predictors
description: Work on FantasyForecast (ff_predictors) — a PPR fantasy football draft assistant with a pooled LightGBM quantile projection engine, VONA draft recommender, and weekly start/sit optimizer on Supabase/PostgreSQL. Use when editing projection/feature/recommender code, running the pipeline, debugging leakage or calibration, adding tests, or serving draft/weekly recommendations.
---

# FantasyForecast (ff_predictors)

Season projections (P10/P50/P90 per player) feed two consumers: a VONA-based
snake-draft recommender and a weekly start/sit ILP optimizer. Everything is
snapshot-reproducible in PostgreSQL (Supabase). Python 3.13, pip + venv.

## Architecture map

| Path | Owns |
|---|---|
| `src/ingest/` | nflreadpy (nflverse), CFBD REST, FFC ADP. `adp.py` is quarantined — see ADP wall |
| `src/labels/` | Fantasy-point scoring (`scoring.py`), season + weekly labels |
| `src/features/` | Season assembly (`assemble.py`), weekly assembly (`weekly_assemble.py`), `leakage_guard.py` |
| `src/projection/` | `quantile_model.py` (pooled 3-quantile GBM), `train.py` / `weekly_train.py`, `calibrate.py` (split-conformal), `folds.py` (expanding CV), `eval.py` |
| `src/recommender/` | `vona.py`, `replacement.py`, `survival.py`, `roster_state.py`, `recommend.py` — draft-time only |
| `src/lineup/` | `optimizer.py` — weekly ILP (strategies: safe=p10, balanced=p50, upside=p90) |
| `src/benchmark/` | Tier-1/2/3 season gates, `weekly_bench.py` (pts left on bench) |
| `src/db/` | SQLAlchemy models, upserts, `session_scope()`, `loaders.py`, `resolve_snapshot` |
| `config/league.yaml` | Scoring, roster, ADP thresholds, training params. `version_hash` embedded in every model_version |

## Invariants (violations are fatal research bugs)

1. **ADP wall.** `src/projection/` and `src/features/` must NEVER import
   `src/ingest/adp.py`. The market is the benchmark, never a feature.
   `tests/test_leakage.py` enforces it — run it after ANY change to those two
   directories. (`team_ctx_*` Vegas features are allowed: the wall bans
   draft-market data only, not betting markets.)
2. **Temporal leakage.** Season features: pre-season data only (strictly before
   Week 1 of target season). Weekly features: as-of-kickoff only — filter
   through `leakage_guard.prior_weeks()` / `prior_seasons()` and assert with
   `assert_no_future_week()`. Forward serving trains via `forward_train_mask`
   (strictly before target week).
3. **Snapshot reproducibility.** Every DB write carries `snapshot_id`
   (`resolve_snapshot()` for CLIs).
4. **Quantile monotonicity.** P10 ≤ P50 ≤ P90, enforced by rearrangement
   (sorting, never clipping) in `quantile_model.py` and re-asserted after
   calibration. Calibration only widens intervals (scales floored at 1.0).
5. **Config coupling.** Any `config/league.yaml` change → full pipeline re-run
   + benchmark re-evaluation (version_hash changes).

## Commands

```sh
source venv/bin/activate            # or prefix commands with venv/bin/
venv/bin/pytest                     # full suite (~30s); pythonpath via pyproject.toml

bash scripts/run_pipeline.sh [YEAR] # full rebuild: seed → labels → features →
                                    # train → bench → weekly data → weekly train → weekly bench
python scripts/draft.py --season 2026 --position 4 [--resume]   # live draft
python scripts/project_week.py --season 2026 --week 1           # forward weekly projections
python scripts/start.py --season 2026 --week 1 --roster r.yaml  # start/sit (after project_week)
python scripts/start.py ... compare "Player A" vs "Player B"
```

`.env` in repo root: `DATABASE_URL=postgresql://...` (Supabase **Session pooler**
URI, port 5432 — direct connection is IPv6-only and fails on IPv4 Macs) and
`CFBD_API_KEY=...`. DB engine is lazy: imports never need DATABASE_URL.

## Model versions in `projections` / `weekly_projections`

- `draft_v1.<hash>` — season projections (forward + OOF backtest rows)
- `weekly_v1.<hash>` — weekly OOF (backtest/benchmark rows)
- `weekly_fwd_v1.<hash>` — forward-serving weekly rows; `start.py` prefers
  these over OOF when both exist for a week

## Testing conventions

- Poison-sentinel pattern for leakage tests: inject absurd values
  (e.g. `99999.0` or 999-point labels) into forbidden rows, assert they never
  reach features/predictions.
- Black-box style: assert contracts (bounds, monotonicity, additivity,
  partition), not implementation details.
- Gates: `tests/test_leakage.py` (ADP wall), `tests/test_weekly_leakage.py`,
  `tests/test_quantile_monotonic.py`. Run after touching `src/features/`,
  `src/projection/`, or `src/ingest/`.
- Synthetic weekly data generators live in `tests/test_weekly_model.py` and
  `tests/test_weekly_forward.py` — reuse, don't reinvent. Pass small LightGBM
  params (`n_estimators≈30`) to keep model tests fast.

## Gotchas

- CFBD via REST (`src/ingest/http.py`), NOT the `cfbd` pip client (pins
  pydantic v1, breaks nflreadpy).
- FFC publishes NO ADP for the just-finished season — use the prior year for
  mocks/benchmarks.
- Player universe per season = nflverse `rosters_y` strictly (keeps cut/FA
  players off the board).
- `pd.DataFrame([{}])` is shape (1,0) and `.empty` is True — guard row counts
  with `len(df)`, not `df.empty`.
- `CLAUDE.md`, `.claude/`, `notes/`, `DrafterSpec.md` are gitignored and exist
  in NO remote — never rely on them surviving; back up separately.
- K/DEF are draft slots filled at replacement level, never modeled
  (`modeled_positions: [QB, RB, WR, TE]`).
- Known accepted skew: weekly `season_p50` feature is OOF at train time but
  full-model at serve time (documented in `weekly_train.py`).

## Change checklists

- Touched `src/features/` or `src/projection/` → `venv/bin/pytest
  tests/test_leakage.py tests/test_weekly_leakage.py` at minimum; full suite
  before commit.
- Changed `config/league.yaml` → full `run_pipeline.sh` + check benchmark
  gates (Tier-1 NDCG, weekly <20 pts left on bench).
- New DB write path → include `snapshot_id`, add upsert to
  `src/db/upsert_data.py` with explicit conflict key.
- Commits: terse lowercase subject, match `git log` style; commit only when
  the user explicitly authorizes.
