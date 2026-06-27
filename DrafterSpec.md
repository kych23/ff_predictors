# Who Should I Draft — Source of Truth & Guardrails

> This document is the single source of truth for the "Who Should I Draft" feature. Code must conform to it. If code and this document disagree, the document wins until the document is deliberately changed.

## 1. The Problem

Fantasy football managers win or lose their season at the draft, where they take turns selecting real NFL players and the manager who assembles the highest-scoring roster comes out ahead. The goal of this project is a personal tool that recommends which player to take at each of my picks, tuned to my specific league's scoring rules and roster requirements. Unlike free public draft rankings — which are generic, one-size-fits-all, and ignore the players I have already drafted — this tool projects how many points each available player will score, accounts for how scarce each position is and when players are likely to be taken, and recommends the pick that does the most for my particular roster. It also weighs the uncertainty in each projection, favoring safer picks early and higher-upside picks late. Success means consistently building better rosters than I would by following the best free public resources.

## 2. The Technical Plan

At the highest level the system is two machines with a wall between them: a **Projection Engine** that decides *what each player is worth*, and a **Recommender** that decides *who to actually draft right now*. Keeping these separate is the most important design choice in the whole project, and the reason is explained at the end of this section.

```
        ┌──────────────────────────────────────────────────────────────┐
        │ 1. DATA LAYER                                                  │
        │  NFL stats, depth charts, snap counts, draft picks, combine,   │
        │  college stats, player-ID crosswalk   →  stored in a database  │
        └───────────────┬──────────────────────────────┬────────────────┘
                        │ (player & game data)          │ (market data only)
                        ▼                                ▼
        ┌──────────────────────────────┐     ┌─────────────────────────┐
        │ 2. SCORING CONFIG            │     │  ADP DATA               │
        │  my league's scoring rules   │     │  "where players are      │
        │  + roster slots, in one file │     │   actually drafted"      │
        └──────────┬───────────────────┘     └───────────┬─────────────┘
                   │                                      │
                   ▼                                      │
        ┌──────────────────────────────┐                 │
        │ 3. PROJECTION ENGINE         │                 │
        │  predicts each player's       │                 │
        │  per-game points for the      │                 │
        │  season — as a RANGE, not a   │                 │
        │  single number (low / mid /   │                 │
        │  high outcome)                │                 │
        │  *** NEVER SEES ADP ***       │                 │
        └──────────┬───────────────────┘                 │
                   │ (projections)                        │
                   ▼                                      ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ 4. RECOMMENDER                                                 │
        │  knows my roster so far, who's left, the scoring/roster rules, │
        │  and the ADP market → ranks the best pick for ME, right now    │
        └───────────────┬──────────────────────────────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ 5. BENCHMARK HARNESS                                           │
        │  replays past seasons to prove the system would have beaten    │
        │  the public draft market — the scoreboard for the whole thing  │
        └──────────────────────────────────────────────────────────────┘
```

**1. Data layer.** Everything starts with public NFL data: each player's historical game-by-game production, plus the signals that explain *why* production changes — where a player sits on his team's depth chart, how many snaps he plays, his draft pedigree, his athletic testing, and (for rookies) his college stats. A separate feed supplies **ADP** ("average draft position"), which is simply where each player is being drafted on average across thousands of real drafts — the crowd's collective opinion. All of it lands in a database, with a crosswalk that keeps a single consistent ID for every player across these different sources.

**2. Scoring config.** My league's exact rules — how many points a touchdown is worth, how many of each position I start, how many teams are in the league — live in one small configuration file. This file is deliberately central: it defines both how player performance is converted into fantasy points *and* how the recommender judges scarcity. Change the file, and the whole system re-tunes to a different league.

**3. Projection engine.** This is the core forecasting brain. For every player it predicts how many fantasy points per game they'll score over the coming season. Crucially, it predicts a **range, not a single number** — a low, middle, and high outcome — because a steady veteran and an unproven rookie can have the same expected output but wildly different certainty, and that difference matters at the draft. The engine is built only from on-field and player-background data. It is forbidden from ever seeing ADP or any public projection, so that when we later claim to beat the public, the claim is honest rather than circular.

**4. Recommender.** This is what I actually interact with during a draft. It takes the engine's projections and combines them with three things the engine never sees: the players I've already drafted, who is still available, and the ADP market. From these it answers the real question — not "who is the best player," but "who is the best *pick for me, at this exact moment*." It accounts for **positional scarcity** (a position that drops off a cliff is worth grabbing before one that stays deep) and for **draft timing** (no point reaching for a player who will still be on the board at my next pick). It also shifts its appetite for risk over the course of the draft: safe, high-floor picks early to protect my starters, higher-upside swings late.

**5. Benchmark harness.** This is the scoreboard. It replays historical seasons and asks two questions: does the engine's ranking of players predict the actual season better than the public's draft order did, and — when we simulate a full draft against opponents who pick by public ADP — does our roster end up scoring more? Because a single season is mostly luck, it measures across many seasons and many draft positions before trusting any result.

**Why the wall between the engine and the recommender.** ADP is genuinely useful information, so the tempting shortcut is to feed it into the projection engine to sharpen its forecasts. We deliberately do not. If the engine's projections were partly built from ADP, then "our projections beat the public market" would be measuring the market against a copy of itself — meaningless. By keeping the engine completely ignorant of the market and letting ADP influence only the live drafting decision, we preserve at least one clean, honest measurement of whether this system actually knows something the public doesn't. That honesty is the entire point of the project, so the wall is non-negotiable.

## 3. Alternatives Considered and Rejected

This section records ideas that were genuinely considered during planning and deliberately ruled out, with the reasoning. It exists to prevent re-litigating settled decisions and to guard against a future contributor (human or AI) "helpfully" reintroducing a rejected approach. Reversing any of these requires a deliberate, documented change to this file — not a silent code decision.

### Framing & goal

- **Reusing the existing weekly model.** The current codebase predicts next-week points from in-season rolling averages. Rejected for drafting: at draft time it is the preseason, so there are zero current-season games to average, and drafting cares about full-season value, not one week. The draft needs a new, season-long model. The weekly model is being retired, not extended.
- **Goal = beat public point projections.** Rejected. Public consensus projections are an aggregate of many experts and are extremely hard to beat on raw accuracy, and historical projections are mostly paywalled or unavailable, so we could not even assemble a fair baseline. The goal was reframed to **beat ADP** (the public draft market), which is free, available historically, and an even stronger consensus.
- **Recommend the "best player" by raw projected points.** Rejected. The best pick is not the highest scorer — it is the highest **value over replacement** given positional scarcity and draft cost. A lower-scoring player at a thin position can be the correct pick over a higher-scoring player at a deep one.
- **The 1-to-1 pairwise draft comparison** (notes.md line 53). Rejected as the target feature in favor of the full roster-aware recommender (line 55). Pairwise comparison throws away roster state and scarcity, which is exactly where the real decision lives. Pairwise falls out for free as a trivial case of the roster-aware tool.

### Target & risk

- **Predicting season *total* points.** Rejected in favor of **per-game points**, which cleanly separates production from availability (the v1 scope deliberately ignores injury/games-played). Accepted cost: ranking on per-game alone will over-value injury-prone players; this is a known v1 limitation, fixed later by a games-played model.
- **Using the data provider's pre-computed PPR points as the label.** Rejected. The label must be computed from raw box-score stats through *my* scoring config, or the custom-scoring edge — a core reason this beats generic tools — would be fake.
- **Modeling week-to-week volatility (boom/bust) as the risk axis.** Rejected for drafting. Weekly variance largely averages out across a full roster over a full season, so it barely affects season-long roster value; it matters for *start/sit* decisions, a later feature. The draft's risk axis is **projection uncertainty** (how sure we are of a player's expected output), not weekly volatility.
- **Predicting a single point estimate.** Rejected in favor of a predicted **range (low/middle/high outcome)**, because the uncertainty itself drives draft strategy (safe early, upside late) and is where rookies and role-changers are correctly distinguished from known quantities.

### Model

- **Exotic model architectures (deep learning, etc.).** Rejected for v1. For this kind of structured, tabular, modest-sized data, gradient-boosted trees remain state of the art; complex architectures add cost and risk without expected gain. "Extensive research into the best techniques" concluded in favor of the well-understood tool, not the novel one.
- **Letting ADP or public projections into the projection engine as features.** Rejected — this is the hard wall. ADP is predictive and would lower projection error, but using it would make the "we beat the public" claim circular and unprovable. The engine never sees market data; ADP influences only the live drafting decision and the benchmark.
- **Separate per-position models (four independent GBMs).** Initially planned, then rejected. TE (~10–15 productive/season, ~195 rows total) and QB (~32/season) are too small-N for an independent model — it would overfit. Replaced by a **single pooled model with `position` as a categorical feature** (§4.6.1), which borrows strength across positions, regularizes the thin ones, handles position-changers natively, and is consistent with the shared feature schema (§4.0). Per-position rankings come from grouping pooled predictions.

### Recommender

- **Static value-over-replacement (a fixed global ranking).** Rejected in favor of a **dynamic** approach that, at each pick, weighs taking a scarce position now against waiting — because real drafts are sequential and the board drains between your picks.
- **Risk-neutral recommender (use the middle projection only).** Rejected in favor of shifting risk appetite by round (favor the floor early, the ceiling late). Using only the middle estimate would discard the uncertainty information the engine works hard to produce.
- **Recomputing replacement level during the draft.** Rejected in favor of a **fixed** preseason replacement baseline. Letting the replacement bar float *and* using the dynamic wait-or-take logic would count intra-draft scarcity twice and cause the recommender to over-react to positional runs. Fixed replacement owns structural scarcity; the dynamic logic owns draft-flow scarcity — one effect, one place. The cost (under-reaction to positional runs, e.g. zero-RB) is real and treated in full in §4.8.1: VONA already captures run *direction*, the Tier-2 sim measures the residual cost, and a damped dynamic replacement is the planned v1.5 fix.
- **Full Monte-Carlo / game-tree draft optimization.** Rejected for v1 as intractable and over-engineered for a one-month timeline. A greedy "value of not waiting" approach with draft-survival probabilities is the pragmatic standard. Multi-step lookahead is a later upgrade that will reuse the benchmark's draft simulator.

### Data sources

- **Sleeper API for ADP / a custom ADP proxy.** Rejected, including on re-examination. ADP is *revealed human draft behavior*, and there is no ADP that matches our exact scoring — because no population of humans drafts under our exact rules; drafters everywhere use generic full-PPR consensus. Reconstructing ADP from raw Sleeper drafts would reproduce that *same* generic signal at far higher cost, not recover a league-exact one. We use Fantasy Football Calculator's **full-PPR** ADP, the closest available format: our league is full PPR (1.0 per reception, the dominant scoring lever), so reception value matches exactly, and the only deviations from generic full PPR (interception value, return-TD credit) touch QBs slightly and rare events negligibly — they do not move draft order. Half-PPR would be the *wrong* choice (it mis-prices every receiver at the biggest lever).
- **Scoring-format mismatch in ADP — why it is faithful, not a defect.** Generic ADP is the *correct* input for every job it serves. The recommender's survival model must predict how my **opponents** draft, and they draft to public consensus, not my scoring — so generic ADP models their behavior correctly; using a hypothetical scoring-exact ADP would be wrong. The benchmark deliberately pits my exact-scoring value (outcomes scored under `league.yaml`) against opponents using generic ADP; that gap between their vanilla signal and my exact value *is the inefficiency this project exploits*, not contamination. (The earlier wording "format-matched ADP directly" was an overstatement; corrected to "closest format, full PPR, with negligible scoring deviations." A one-time sanity check — player value under `league.yaml` vs generic full PPR — bounds the gap and confirms it does not distort rankings.)
- **A secondary ADP site (beatADP) as the primary source.** Kept only as an optional cross-check, not the foundation; FFC is the primary feed.
- **Lagged production alone as the opportunity signal.** Rejected as insufficient. Past box-score stats describe a player's *old* role and are blind to change (new team, new depth-chart rank, vacated touches). Depth charts, snap counts, and opportunity data are added specifically to see role *as it is now*.
- **Flat penalty on all rookies.** Rejected as the user's initial instinct after review. It is both wrong and lazy: rookie outcomes depend heavily on draft capital and landing spot (a high pick into an open role often produces immediately). The model learns the rookie mapping from history instead of applying a blanket discount.
- **Modeling kickers and defenses.** Rejected. They are the noise floor of fantasy, nearly unpredictable year to year, and conventionally streamed; projecting them is high effort for ~zero edge. They are filled at replacement level in the final rounds and never modeled.

### Evaluation & process

- **Judging success on a single season.** Rejected. One season is dominated by luck; a one-year win could be noise. Evaluation runs across many seasons and many draft positions and reports a distribution with significance, never a single number.
- **Running a full 15-round draft simulation on the oldest seasons.** Rejected: a 12×15 draft needs ADP for 180 picks, but early seasons list only ~100–150 players. Full-draft simulation (Tier 2) runs **only on `SIM_ELIGIBLE` seasons** (ADP coverage ≥ `adp_min_fullsim` = teams × rounds = 180; §4.0), and thinner seasons are **excluded rather than tail-backfilled** — backfilling late picks with our own projection-rank would leak our signal into opponent behavior and corrupt the gate. The lighter ranking benchmark (Tier 1) still runs on every `RANKING_ELIGIBLE` season (≥100), i.e. effectively the full usable history.
- **Building the full feature set before validating.** Rejected in favor of a thin end-to-end slice that must clear the benchmark *before* heavy feature work. If a stripped-down engine cannot beat the public market, more features will not rescue it, and we want to learn that early.
- **Building the frontend now.** Deferred. A user interface comes only after the model and recommender are demonstrably good; building UI first would polish something not yet worth using.

## 4. Detailed Implementation

This is the binding implementation plan. **Every file to be created, changed, or deleted is enumerated below with its rationale.** An implementer (human or AI) must not create files outside this inventory or leave listed files unbuilt without amending this document first. New scope = new entry here, first.

### 4.0 Conventions that every file must obey

- **Grain.** The unit of modeling is one **(player, season)** row. The old system's player-week grain is retired.
- **The leakage rule (most important correctness property).** For any row keyed `(player p, season Y)`, features may read **only** information available before season `Y`'s Week 1 kickoff: production/snaps/opportunity from seasons `≤ Y−1`, and preseason-`Y` artifacts (depth chart effective entering the season, roster as of preseason, age at season start, draft capital). The label is season `Y`. Cross-validation is expanding-by-season so a model predicting `Y` is trained only on seasons `< Y`. Enforcement is at **pipeline-construction time, not query time** — features are stored as opaque JSONB and cannot be checked by a database assertion. The feature builder takes a hard `as_of` cutoff and filters every source frame to rows strictly before it *before* aggregation; `features/leakage_guard.py` asserts this at the data-access boundary, and `tests/test_leakage.py` poisons a future value with a sentinel in synthetic data and asserts it never reaches any output feature. This is a code-correctness problem, treated as such — not a schema constraint.
- **As-of cutoff & depth-chart snapshot.** The "before Week 1 kickoff" rule is enforced against a concrete date, `as_of(Y)` = the day before season-`Y` Week 1 kickoff (configurable), stored in `season_features.as_of_date` and pinned by `snapshot_id`. Depth charts (nflverse = weekly granularity) use the **Week-1-effective chart**, never an in-season (≥ Week 2) one. Late-preseason information (e.g. an August depth-chart slide) is *allowed* — known before kickoff, so rule-compliant, not leakage. **Known train/serve skew:** the live draft is late July, before final cuts, so the draft-day chart is less settled than the Week-1 charts used in training; v1 mitigates by leaning on coarse, July-stable role signals (projected starter / committee membership, not fine ordering) and by absorbing residual role uncertainty into wide quantile intervals. Monitored, not fully closed in v1.
- **Position is as-of season Y.** A row's position = the player's position entering season `Y` (preseason roster/depth chart, known pre-kickoff), and the row is scored by **that** position's model — we predict `Y`, so `Y`'s position governs. To make this safe, all positions share **one feature schema** (the union of positional stats, NaN where inapplicable — see missing-data rule), so a player's real historical production flows into the same physical columns regardless of a position switch (a converted RB→WR's prior carries are informative, not corrupting). A `position_changed` indicator (prior-season position ≠ current) is set so the model prices the added uncertainty (and widens the interval). Ultra-rare multi-position hybrids (e.g. Taysom Hill) take their listed fantasy/depth-chart primary position; the flag + wide interval absorb the imprecision — not worth special-casing a handful of players.
- **Canonical ID.** `gsis_id` is the one true player key everywhere. All other source IDs are mapped to it via the crosswalk.
- **The wall.** Only three packages may import ADP/market data: `ingest` (to fetch it), `recommender` (to use it), `benchmark` (to score against it). The `projection` and `features` packages must never import the `adp` module. This is a reviewable, testable boundary.
- **Config version propagation.** The league/scoring config carries a version hash. It is embedded in `model_version` on every projection and prediction so outputs are always traceable to the rules that produced them.
- **ADP coverage thresholds & season eligibility (benchmark only).** Historical ADP coverage is a gradient — early seasons list far fewer players than recent ones (≈100–150 in 2012–2014, ≈387 by 2024) — so each season is tagged for which benchmark it can support, against thresholds from the league config:
  - `adp_min_ranking` (default **100** players with ADP): season is `RANKING_ELIGIBLE` → usable for the **Tier-1 ranking** benchmark and **Tier-3 calibration**. 100 covers all modeled starters (12 × 7 non-K/DEF = 84) plus depth, enough for a meaningful rank correlation.
  - `adp_min_fullsim` = **teams × rounds** (here 12 × 15 = **180**): season is `SIM_ELIGIBLE` → usable for the **Tier-2 full-draft simulation**, which needs ADP for every pick. Seasons below 180 are **excluded** from the sim — **no tail backfill**, because filling late picks with our own projection-rank would leak our signal into "opponent" behavior and flatter the gate.
  - Seasons below `adp_min_ranking` are excluded from all ADP-based benchmarks.
  - **Projection training is unaffected by ADP depth.** Training uses labels + features only and never touches ADP, so the engine trains on the full 2012–2025 history regardless; ADP thresholds shrink only the *benchmark* window, never the *training* set.
  Per-season ADP coverage counts are recorded at ingest (§4.2); eligibility flags derive from them, and the benchmark consumes the flags rather than hard-coding years.
- **External data access & resilience.** Third-party API keys live in `.env` (`DATABASE_URL`, `CFBD_API_KEY`), never hard-coded. All outbound HTTP to CFBD and FFC goes through one client (`ingest/http.py`) providing exponential-backoff retries, per-source rate-limit throttling (CFBD free tier = 60 req/min), an on-disk cache, and **graceful degradation**: on failure or rate-limit it serves the last good cached response rather than crashing. Draft day is a hard deadline — stale data beats no tool. (`nflreadpy` manages its own fetching and is exempt.)
- **Missing-data representation.** Never drop a player and never zero-fill a missing feature (zero is a real value and therefore a wrong signal). Represent missing as NaN plus an explicit `has_*` / indicator flag (`has_college_stats`, `has_depth_data`, `is_udfa`); the gradient-boosted trees split on NaN natively and can isolate the missing-data class. This one rule covers unmatched college stats (§4.5), undrafted players (§4.5), and sparse early-year depth charts (§4.7).
- **Data snapshots & reproducibility.** `nflreadpy` retroactively corrects past-season stats, so a live re-pull silently changes history and invalidates cross-season comparisons — fatal for a "we beat ADP across many seasons" claim. Therefore: raw ingested tables are stamped with a `snapshot_id` + `extracted_at`; training and benchmarking read the frozen DB snapshot and **never live-pull**, and every projection/benchmark output records the `snapshot_id` it used. Re-ingesting creates a *new* snapshot; results are comparable only within one snapshot.

### 4.1 Target repository layout

```
config/
  league.yaml                     league scoring + roster + settings (load-bearing)

src/
  config/        league_config.py, __init__.py
  db/            session.py·KEEP  init_db.py·KEEP  models.py·REWRITE
                 upsert_data.py·REWRITE  __init__.py·KEEP
  ingest/        sources.py  college.py  player_ids.py  adp.py
                 run_ingest.py  __init__.py
  labels/        scoring.py  build_labels.py  __init__.py
  features/      prior_production.py  role_change.py  rookies.py
                 assemble.py  leakage_guard.py  __init__.py
  projection/    dataset.py  folds.py  quantile_model.py  calibrate.py
                 train.py  eval.py  __init__.py
  recommender/   replacement.py  survival.py  quantile_schedule.py
                 roster_state.py  vona.py  recommend.py
                 draft_state_source.py  __init__.py
  benchmark/     ranking.py  draft_sim.py  calibration.py  report.py  __init__.py

scripts/         seed_db.py·REWRITE  build_labels.py  build_features.py
                 train_projection.py  run_benchmark.py  draft.py

tests/           test_scoring.py  test_leakage.py  test_id_crosswalk.py
                 test_replacement.py  test_survival.py  test_vona.py
                 test_quantile_monotonic.py  test_draft_layout.py

requirements.txt·CHANGE   DrafterSpec.md·THIS FILE (repo root)
```

Everything under `src/api/`, `src/ml/`, and the old weekly tables is **deleted** (see 4.9).

### 4.2 Milestone M0 — Infrastructure & data ingestion

Stand up the fresh database and pull every raw source into it. *(Context: the old Supabase DB was paused past recovery and must be rebuilt from scratch; all source data is re-pullable from public APIs, so nothing is lost but compute.)*

| File | Action | Rationale |
|---|---|---|
| `config/league.yaml` | NEW | The load-bearing config — full enumerated structure in §4.2.1: scoring values, **roster slots + flex eligibility**, team count, benchmark ADP thresholds (`adp_min_ranking`; `adp_min_fullsim` derives from teams × roster_rounds), and training params (`min_games_train`=8, `ewm_halflife_seasons`=1.0, `min_train_seasons`=5, `min_bucket_n`=30). Defines labels, recommender scarcity, and benchmark eligibility (§4.0). |
| `src/config/league_config.py` | NEW | Loads + validates `league.yaml` into a typed object; computes the version hash used in `model_version`. Central so every module reads settings one way. |
| `src/config/__init__.py` | NEW | Package marker. |
| `src/db/session.py` | KEEP | Existing engine/`SessionLocal` is correct and reused unchanged. |
| `src/db/init_db.py` | KEEP | Idempotent `create_all` is exactly what we need for a fresh DB. |
| `src/db/models.py` | REWRITE | Replace weekly tables with season-grain schema (4.3). |
| `src/db/upsert_data.py` | REWRITE | Keep the proven `_chunk_iter` + `on_conflict_do_update` pattern; one upsert per new table. **Removes the old precomputed-`fantasy_points_ppr` label path** — `weekly_stats_raw` stores only raw component stats, and labels are computed in M1 (§4.4.1). This is the explicit owner of that transition. |
| `src/ingest/sources.py` | NEW | Thin, testable wrappers around `nflreadpy` loaders (players, player_stats, schedules, depth_charts, snap_counts, ff_opportunity, draft_picks, combine) returning pandas. Isolates the external API behind one module. |
| `src/ingest/http.py` | NEW | Shared resilient HTTP client for CFBD and FFC: exponential-backoff retries, per-source rate-limit throttle (CFBD 60/min), on-disk cache, and fallback to last-good cache on failure (§4.0). The one place all third-party HTTP resilience lives. |
| `src/ingest/player_ids.py` | NEW | Loads `load_ff_playerids`, builds the `gsis_id` crosswalk, and owns the **college→NFL bridge** with an explicit matching strategy: (1) join on `load_draft_picks` keys (pfr/college ids) where present; (2) else fuzzy-match on normalized name + college + class year — normalization strips suffixes (Jr/Sr/II/III), punctuation, and applies a nickname map. **Fallback for unmatched rookies:** keep the player, set college features to NaN with `has_college_stats=0` (never drop, never zero-fill, §4.0). Records and exposes the match rate. The single place player identity is reconciled. |
| `src/ingest/college.py` | NEW | Pulls college production from **CFBD** (College Football Data — the data behind the R package `cfbfastR`), via its Python client/API. Requires `CFBD_API_KEY` in `.env` (CFBD is not anonymous); all requests route through `ingest/http.py` to honor the 60 req/min free-tier limit. College seasons are historical and static, so pulls are cached to local parquet and fetched once. Isolated here so the dependency is contained. |
| `src/ingest/adp.py` | NEW | Fantasy Football Calculator client (via `ingest/http.py`): fetch **full-PPR** ADP by teams/year (endpoint + response schema in §4.2.2; closest format to `league.yaml`; deviations limited to INT/return-TD, immaterial to draft order — no scoring-exact ADP exists, see §3), persist to the `adp` table (incl. `adp_stdev` from the `stdev` field) + a local parquet cache keyed by (format, teams, year). Cache TTL = 24h for the current season (FFC updates daily), immutable for past seasons. **On draft day, if FFC is unreachable it serves the last cached ADP with a loud staleness warning (timestamp shown) rather than failing.** Records **per-season coverage count** so benchmark eligibility (§4.0) is derivable. **Quarantined** — part of the ADP wall. |
| `src/ingest/run_ingest.py` | NEW | Orchestrates the full pull into the DB. Replaces the old `load_initial_data.py` and the ingestion half of `seed_db.py`. |
| `src/ingest/__init__.py` | NEW | Package marker. |
| `scripts/seed_db.py` | REWRITE | Becomes a thin CLI over `ingest.run_ingest`. |

**Acceptance:**
- NFL/player source tables (stats, depth charts, snaps, opportunity, draft picks, combine, college) populate for **all of 2012–2025** — these have no coverage gradient.
- The **ADP table populates per-season as available**, recording each season's player-coverage count. Coverage is a gradient (early seasons may list only ~100–150 players) and is **not** required to reach the full-draft count of 180 in early years. Each season is tagged `RANKING_ELIGIBLE` / `SIM_ELIGIBLE` per the §4.0 thresholds.
- Every `season_labels`/`projections`-eligible player resolves to a `gsis_id`.

*(This corrects an earlier blanket "all source tables populate for 2012–2025": true for NFL data, false for ADP, whose historical depth is inherently limited. The two are now stated separately.)*

#### 4.2.1 Canonical `league.yaml` (authoritative config contract)

Roster slots are **enumerated here**, not left to assumption — `replacement.py` and the greedy lineup-fill (§4.8.1) depend on the exact counts and flex eligibility. The user's league:

```yaml
league:
  teams: 12
  type: snake
  kind: redraft
roster:
  slots: {QB: 1, RB: 2, WR: 2, TE: 1, FLEX: 1, K: 1, DEF: 1, BENCH: 6}
  flex_eligible: [RB, WR, TE]        # which positions the FLEX slot can take
  modeled_positions: [QB, RB, WR, TE] # K/DEF are draft slots but filled at replacement, never modeled
scoring:                              # full map in §4.4.1
  pass_yd: 0.04   pass_td: 4   int: -1
  rush_yd: 0.1    rush_td: 6
  rec: 1          rec_yd: 0.1  rec_td: 6
  two_pt: 2       fumble_lost: -2     return_td: 6
adp:
  format: ppr
  adp_min_ranking: 100                # adp_min_fullsim derives = teams × roster_rounds = 12 × 15 = 180
training:
  min_games_train: 8
  ewm_halflife_seasons: 1.0
  min_train_seasons: 5
  min_bucket_n: 30
```

- **Roster size = 15** (9 starters + 6 bench), so draft rounds = 15 and the full draft = 12 × 15 = 180 picks — the basis of `adp_min_fullsim` (§4.0).
- **Pure starter slots** (for replacement, §4.8.1): QB 12, RB 24, WR 24, TE 12; **FLEX** 12 (RB/WR/TE); K 12, DEF 12.
- `flex_eligible` is read by replacement/VONA — never hard-coded. A different league (e.g. superflex adding QB to flex) changes only this file.
- K/DEF appear in the draft (bots pick them; the recommender fills them at replacement in the last rounds) but are excluded from the projection engine.

#### 4.2.2 FFC ADP endpoint (verified)

- **URL:** `GET https://fantasyfootballcalculator.com/api/v1/adp/{format}`, `format` ∈ {`standard`, `ppr`, `half-ppr`, `2qb`, `dynasty`, `rookie`}; we use **`ppr`**.
- **Query params:** `teams` (=12), `year` (season; omit for the current live season).
- **Response:** `{status, meta:{type, teams, rounds, total_drafts, start_date, end_date}, players:[…]}`. Each player object (verified live): `player_id, name, position, team, adp, adp_formatted, times_drafted, high, low, stdev, bye`. `high`/`low` are pick numbers (low number = earlier).
- **`adp_stdev` sourcing:** taken directly from the player object's **`stdev`** field — FFC provides it, so no estimation is normally needed. Fallback if ever missing (thin sample): `stdev ≈ (low − high) / 4` (treat the high–low range as ≈ ±2σ); last resort, a position/round default.
- Player names join to `gsis_id` via the `ff_playerids` crosswalk (§4.2); responses cached per (format, teams, year) (§4.0).

### 4.3 Database schema (defined here, implemented in `models.py`)

| Table | Action | Purpose |
|---|---|---|
| `players` | EXTEND | gsis_id PK + name, position, college, rookie_year, draft_round/pick/team. |
| `player_id_map` | NEW | gsis_id ↔ fantasypros/sleeper/espn/pfr ids, from the crosswalk. |
| `weekly_stats_raw` | NEW | Raw weekly box score; staging input for label computation. |
| `season_labels` | NEW | Per (player, season): games_played, fantasy_points_total, fppg. |
| `season_features` | NEW | Per (player, season): feature JSONB + is_rookie + as_of_date. JSONB keeps the feature set fluid, mirroring the old design's strength. |
| `projections` | NEW | Per (player, season, model_version): p10, p50, p90, pos_rank. |
| `adp` | NEW | Per (source, season, format, teams, player): adp, adp_stdev, n_drafts + `snapshot_id`, `extracted_at`. Separate table so the wall is structural. |
| `ingest_snapshots` | NEW | One row per ingest run: `snapshot_id`, `extracted_at`, source/library versions. Raw tables (`weekly_stats_raw`, `adp`, and the depth/snaps/opportunity stages) carry `snapshot_id` + `extracted_at`; downstream runs pin to one snapshot for reproducibility against later nflverse corrections (§4.0). |
| weekly `features`/`labels`/`predictions` | DELETE | Player-week grain retired. |

### 4.4 Milestone M1 — Scoring config & per-game labels

| File | Action | Rationale |
|---|---|---|
| `src/labels/scoring.py` | NEW | The scoring function: applies `league.yaml` to raw `load_player_stats` columns → fantasy points, per the authoritative column map in §4.4.1 (fumbles and 2-pt each summed across three columns; return TD from `special_teams_tds`). **Never** uses the pre-computed `fantasy_points_ppr`. The one place scoring rules become numbers. |
| `src/labels/build_labels.py` | NEW | Aggregates weekly fantasy points → season `games_played`, `total`, `fppg`; applies the `min_games_train` floor (default 8, §4.4.2) as a label-quality filter only. Writes `season_labels`. |
| `src/labels/__init__.py` | NEW | Package marker. |
| `scripts/build_labels.py` | NEW | CLI to run label construction. |
| `tests/test_scoring.py` | NEW | Asserts the scoring function reproduces known point totals **and each component** (fumbles/2-pt summed correctly across all three columns) for sample stat lines — protects the custom-scoring edge and catches a missing column. |

#### 4.4.1 Scoring column map (authoritative contract for `scoring.py`)

`scoring.py` computes points from raw nflreadpy `load_player_stats` columns — **never** the pre-computed `fantasy_points_ppr` (which the old `upsert_data.py` used and which the M0 REWRITE removes). `weekly_stats_raw` stores the raw component columns below; nothing downstream reads a pre-scored total.

| `league.yaml` rule | nflreadpy column(s) — summed | value |
|---|---|---|
| passing yards | `passing_yards` | × 0.04 (1 per 25) |
| passing TD | `passing_tds` | × 4 |
| interception | `passing_interceptions` | × −1 |
| rushing yards | `rushing_yards` | × 0.1 (1 per 10) |
| rushing TD | `rushing_tds` | × 6 |
| reception | `receptions` | × 1 |
| receiving yards | `receiving_yards` | × 0.1 |
| receiving TD | `receiving_tds` | × 6 |
| 2-pt conversion | `passing_2pt_conversions` + `rushing_2pt_conversions` + `receiving_2pt_conversions` | × 2 |
| fumble lost | `rushing_fumbles_lost` + `receiving_fumbles_lost` + `sack_fumbles_lost` | × −2 |
| return TD | `special_teams_tds` | × 6 |

- **Fumbles and 2-pt are each split across three columns and MUST be summed** — missing any under-counts.
- **Return TD:** `special_teams_tds` bundles kick/punt-return and other ST scores, but for our scope (QB/RB/WR/TE) a skill player's ST touchdowns *are* return TDs, so the bundle is exact in practice. Non-return/defensive ST scores belong to DST, which we do not model. Finer granularity (from play-by-play) is a future-league concern, not v1.
- `offensive_fumble_return_td` (notes.md, 6 pts) has no clean dedicated column and is vanishingly rare for skill players → **omitted in v1**; revisit from PBP only if it ever matters.

#### 4.4.2 Min-games floor

- `min_games_train` is an explicit, tunable hyperparameter in `league.yaml` (training section), **default 8** (≈ half a season — enough games that the per-game rate is a stable estimate without over-excluding). Ablate it; do not assume it optimal.
- It is a **label-quality filter only**: it decides which past-season rows are reliable enough to *train on*. It does **not** define replacement level — replacement (§4.7) is computed from projected P50 rank over the full draftable universe, independent of this filter, or the floor would silently redefine scarcity.
- A raw games count cannot distinguish a 4-game part-timer from a 4-game injury victim — true, but irrelevant in v1: availability is ignored, so the floor exists purely to control *per-game-rate sample noise* (few games → unreliable rate), not to judge why games were missed. Injury-vs-role separation is the deferred v2 availability model.

#### 4.4.3 Season-length era — 16 vs 17 games

- The label is a **per-game rate** (`fppg`), so it is era-robust at first order: a rate does not inflate because a season has one more game.
- **Features must be rates, never season totals** (per-game / per-opportunity), so cross-season EWMA aggregates (§4.5) mix 2012–2020 (16-game) and 2021+ (17-game) production without distortion. Mandated in `prior_production.py`.
- `season_length` (16/17) is known preseason and may be used as a context feature (not leakage). The second-order effect the user notes (a 17th low-usage game slightly dilutes a per-game average) is inherent to per-game scoring in every era and is left to the model, not corrected.
- Era effects on injury exposure / roster churn reach replacement level only through availability, which v1 ignores; deferred with the games-played model.

**Acceptance:** `fppg` for a sample of known players matches manual computation under `league.yaml`; `test_scoring.py` passes at the per-component level.

### 4.5 Milestone M2 — Preseason-safe feature builder

This is where the leakage rule and the project's edge both live.

| File | Action | Rationale |
|---|---|---|
| `src/features/prior_production.py` | NEW | Cross-season weighted/EWMA aggregates of opportunity, efficiency, and fppg from seasons `< Y`, expressed as **rates (per-game / per-opportunity), never season totals**, so 16- and 17-game eras mix without distortion (§4.4.3). Cross-season decay = `ewm_halflife_seasons` (config, default 1.0, **tuned on CV folds, not hand-set**; §4.6.1). Exact column list in §4.5.1. The stable production baseline. |
| `src/features/role_change.py` | NEW | **Week-1-effective** depth-chart rank (per the `as_of` cutoff, §4.0), team-change flag, and vacated-opportunity on the new team — the signals that see role *as it is now*. Prefers coarse, draft-day-stable role signals (projected starter / committee) over fine ordering, to limit the late-July train/serve skew (§4.0). Columns + the precise **vacated-opportunity** definition in §4.5.1. The core edge over public tools. |
| `src/features/rookies.py` | NEW | Draft-capital, combine, and college features for players with no NFL history; sets `is_rookie`. Implements "no flat penalty" by learning the rookie mapping. **UDFAs** (null draft capital) get a synthetic late draft slot (configurable, just past the last real pick) for monotonic ordering **plus** an `is_udfa` indicator, so the model isolates the undrafted class instead of forcing them onto a round-8 prior (some UDFAs become stars). **Missing college stats** → NaN + `has_college_stats=0` (§4.0), never dropped or zeroed. College/combine column list in §4.5.1. |
| `src/features/assemble.py` | NEW | Joins all feature blocks for `(player, Y)`, attaches `age_at_season_start` (explicit feature — see Age note below), sets the row's position **as of season Y** and a `position_changed` flag (§4.0), enforces the as-of rule, writes `season_features`. The single assembly point. |
| `src/features/leakage_guard.py` | NEW | Enforces the leakage rule at **construction time** (opaque JSONB output can't be SQL-checked): wraps the data-access boundary so every source frame handed to a feature function contains only rows strictly before the row's `as_of` cutoff. A code guard, not a DB constraint. |
| `src/features/__init__.py` | NEW | Package marker. |
| `scripts/build_features.py` | NEW | CLI to build features for a season range. |
| `tests/test_leakage.py` | NEW | Runs the feature pipeline on **synthetic fixtures** with a future-season value set to a detectable sentinel and asserts it never appears in any output feature — verifies the *code path*, not stored data. Non-negotiable gate. |
| `tests/test_id_crosswalk.py` | NEW | Verifies the college→NFL bridge: asserts a minimum match rate **and** that unmatched rookies still produce valid rows via the NaN + `has_college_stats=0` fallback (no silent bad rows, no drops). |
| `scripts/check_adp_format_gap.py` | NEW | One-time sanity check (§3): computes player value under `league.yaml` vs generic full PPR and reports rank shift, to bound the ADP scoring-format gap and confirm it does not distort draft order. |

**Age & aging curves (explicit feature).** `age_at_season_start` is an **explicit feature** (attached in `assemble.py`), not left for the model to infer from lagged production. Aging curves are among the most validated FF signals — RB production falls sharply after ~28, WR peaks ~26–28, QB is flatter. Because `position` is a model feature in the pooled GBM (§4.6.1), the trees learn these **position-specific** curves directly from age × position interactions. Birthdate is preseason-known, so age is leakage-safe.

#### 4.5.1 Feature column enumeration (v1 authoritative list)

This is the v1 feature set. Additions are M7 feature-lift and must be added **here** first. All production features are **rates** (per-game or per-opportunity, §4.4.3), cross-season EWMA-aggregated from seasons `< Y`, in **one shared schema** (NaN where a stat doesn't apply to a position, §4.0).

**`prior_production.py` — volume / opportunity** (the stable, most-predictive block; sources `load_player_stats`, `load_ff_opportunity`, `load_snap_counts`):
- `snap_share` (offensive snap %); `routes_per_game` / route participation where available
- `target_share` (of team targets), `targets_per_game`
- `carry_share` (of team carries), `carries_per_game`, `touches_per_game` (carries + receptions)
- `rz_targets_per_game`, `rz_carries_per_game` (red-zone opportunity)
- `expected_fp_per_game` (xFP from `load_ff_opportunity`)
- QB: `dropbacks_per_game`, `pass_attempts_per_game`, `rush_attempts_per_game` (QB rushing is a major fantasy lever)

**`prior_production.py` — efficiency** (noisier; regresses harder, but kept):
- receiving: `yards_per_route_run`, `yards_per_target`, `catch_rate` (rec/targets), `yards_per_reception`, `rec_td_rate` (TD/target)
- rushing: `yards_per_carry`, `rush_td_rate` (TD/carry)
- `fp_over_expected` (actual FP ÷ xFP — efficiency over expectation)
- `prior_fppg` (the lagged label)
- QB: `completion_pct`, `yards_per_attempt`, `pass_td_rate`, `int_rate`, `sack_rate`, `rush_yards_per_game`

**`role_change.py` — role & change** (Week-1-effective, §4.0):
- `depth_chart_rank` (positional rank on the season-`Y` team), `is_projected_starter` (rank == 1)
- `team_changed` (team_`Y` ≠ team_`Y−1`; rookies always new-team), `same_position_competition` (count of same-position teammates ahead on the depth chart)
- **Vacated opportunity — precise definition:** for player `p` on team `T` in season `Y`, `vacated_X = Σ` over players `q` who were on `T` in `Y−1` but are **not** on `T` in `Y` (departed via FA / trade / retirement), of `q`'s `Y−1` `X`. Computed for `X ∈ {targets, carries, air_yards}` and expressed **both** absolute (`vacated_targets`, …) **and** as a share of `T`'s `Y−1` team total (`vacated_target_share`, `vacated_carry_share`, …). "On the team" is resolved from `load_rosters` roster diff. This is the central change signal — a player inheriting a large vacated share is poised to break out regardless of his own prior role.

**`rookies.py` — college (CFBD) + athletic + capital** (only when `is_rookie`; `has_college_stats=0` + NaN when the bridge fails, §4.0):
- **`college_dominator`** (share of team receiving yards + TDs; rush+rec share for RB) — top WR/TE predictor
- **`breakout_age`** (age at first dominant college season) — strong WR signal
- final-college-season rates: `college_rec_per_game`, `college_rec_yards_per_game`, `college_ypr`, `college_rush_yards_per_game`, `college_ypc`, `college_td_per_game`; QB: `college_completion_pct`, `college_ypa`, `college_td_rate`
- `college_target_share` (market share of team passing), `competition_level` (Power-5 / G5 / FCS from CFBD conference)
- athletic (`load_combine`): `forty_time`, `vertical`, `broad_jump`, `agility` (3-cone/shuttle), `weight`, `bmi`, `speed_score`; missing combine → NaN + indicator
- draft capital: `draft_round`, `draft_pick` (UDFA → synthetic slot + `is_udfa`, §4.5)

**Shared (all rows):** `age_at_season_start`, `position` (categorical), `position_changed`, `season_length`, `is_rookie`, and the indicator flags (`has_college_stats`, `has_depth_data`, `is_udfa`).

**Acceptance:** leakage test passes; rookie rows populate from college/draft data; `age_at_season_start` present on every row; the §4.5.1 columns are all produced (NaN-filled where inapplicable); spot-checked role-change features reflect known offseason moves.

### 4.6 Milestone M3 — Projection engine

| File | Action | Rationale |
|---|---|---|
| `src/projection/dataset.py` | NEW | Reads `season_features` + `season_labels`, flattens JSONB into **one pooled matrix** with `position` (as-of season Y, §4.0) as a categorical feature — not four per-position matrices (§4.6.1) — using one shared feature schema (NaN where a stat doesn't apply) so position-changers keep correct semantics. Tags each row's position for per-position output/eval. Replaces the old `train.py` data-loading half. |
| `src/projection/folds.py` | NEW | Expanding-by-season CV, ported from `make_expanding_folds`; produces OOF only after the `min_train_seasons` warmup (default 5, §4.6.1) so degenerate early folds are dropped. |
| `src/projection/quantile_model.py` | NEW | **One pooled LightGBM per quantile** (P10/P50/P90) across all positions with `position` as a categorical feature (§4.6.1); guarantees P10 ≤ P50 ≤ P90 by **rearrangement (sorting), not clipping**. The forecasting core. |
| `src/projection/calibrate.py` | NEW | Adjusts interval widths per player-type bucket using OOF residuals (so a rookie's interval is honestly wide). Buckets assigned by the **priority rookie > 2nd-year > team-changed vet > established vet**, with a `min_bucket_n` fallback to coarser grouping. |
| `src/projection/train.py` | NEW | Orchestrates dataset → folds → fit → out-of-fold predictions → calibrate → write `projections`, stamping `model_version` with the config hash (also fixes the old version-suffix bug). |
| `src/projection/eval.py` | RESHAPE | Ports the old `eval.py` regression metrics and adds ranking metrics; shared with the benchmark. |
| `src/projection/__init__.py` | NEW | Package marker. |
| `scripts/train_projection.py` | NEW | CLI to train and write projections. |
| `tests/test_quantile_monotonic.py` | NEW | Asserts quantiles never cross. |

#### 4.6.1 Model architecture & training methodology

- **Single pooled model, not four (consistent with §4.0's shared schema).** TE has only ~10–15 productive players/season (~195 rows over 13 years) and QB ~32 starters/season — a *separate* GBM per position overfits. Train **one pooled model across QB/RB/WR/TE with `position` as a categorical feature** (three pooled models, one per quantile level). Trees split on `position` to learn position-specific behavior where data supports it and borrow strength across the pool where it doesn't — the standard small-N remedy. Per-position outputs (`pos_rank`) come from grouping predictions by position after inference. **This supersedes the earlier 'separate per-position models' design (see §3).**
- **Quantile monotonicity by rearrangement, not clipping.** The three quantile models are fit independently and can cross. Guarantee P10 ≤ P50 ≤ P90 by **rearrangement** — sorting the three predicted values per row and reassigning in order (Chernozhukov et al. 2010), which weakly *reduces* estimation error vs the crossing curve. Preferred over naive clipping (`P10 = min(P10,P50)`), which is asymmetric and biases the interval — and the interval feeds the recommender's risk math. `tests/test_quantile_monotonic.py` verifies the post-rearrangement output.
- **EWMA cross-season decay is a *feature-construction* parameter, tuned by an outer loop.** `ewm_halflife_seasons` (how fast season `Y−2` is down-weighted vs `Y−1`) lives in `league.yaml`, **default 1.0**. Unlike a model hyperparameter, changing it requires **rebuilding the features**, so tuning is an **outer loop**: for each candidate halflife in a *small* coarse grid (e.g. {0.5, 1.0, 2.0}), rebuild `prior_production` features on the training folds, refit the model, and score the held-out fold by the §4.7.1 ranking metric — using only training-fold data, never the held-out season. This is expensive (feature rebuild × refit per grid point), so the grid stays small and the search runs once, not nested per CV fold beyond the expanding split. If compute-bound, ship the default 1.0 and defer tuning — it is a coarse knob with few prior seasons per player.
- **Minimum training window.** Expanding folds emit OOF only after `min_train_seasons` (config, **default 5**) — early folds (e.g. train-on-2012-alone) are too thin. With 2012–2025 that yields ~8–9 benchmark seasons. **Honest ceiling:** 13 years of NFL history is the hard limit and ~8–9 evaluable seasons is *not* "many" at the season level. Significance is therefore computed across **season × player cross-sections** (each season ranks a few hundred players), reported with confidence intervals, and the season-count limit is stated in every report — no overclaiming. Pooling (above) also makes early folds usable by giving them the full multi-position row count rather than ~32 QBs.
- **Calibration bucket priority.** The overlapping buckets are made mutually exclusive by the priority **rookie > 2nd-year > team-changed vet > established vet** (history depth drives interval width more than situational change, so thin-history classes win: a rookie-on-a-new-team is a rookie; a traded 2nd-year is 2nd-year). A `min_bucket_n` fallback drops to the next coarser grouping (ultimately a global calibration) so calibration never runs on a degenerate sample.

**Acceptance:** out-of-fold projections written for all seasons **after the `min_train_seasons` warmup**; quantiles monotonic **after rearrangement**; a single pooled model with a `position` feature (not four separate models).

### 4.7 Milestone M4 — Benchmark (engine gate) + risk calibration

Built *before* the recommender, because the engine must prove itself first.

| File | Action | Rationale |
|---|---|---|
| `src/benchmark/ranking.py` | NEW | Tier 1: scores out-of-fold projection rankings vs ADP rankings against actual outcomes. **Primary gate metric = NDCG@k (§4.7.1)**, relevance = actual-season FPPG, with Spearman/top-N as diagnostics, over `RANKING_ELIGIBLE` seasons (§4.0). The honest, ADP-free "do we beat the market" test. |
| `src/benchmark/calibration.py` | NEW | Tier 3: interval coverage by player-type + pinball loss. Validates the risk claim. |
| `src/benchmark/report.py` | NEW | Aggregates results across seasons and draft slots into distributions with significance. Reports metrics **both overall and on the data-complete era** (≈2016+, where depth-chart/role coverage is solid): pre-2015 role-change data is sparse, so whole-history numbers *understate* the edge the engine will have deployed forward on complete data — the complete-era split is the honest forward estimate. Prevents single-season conclusions; significance is computed across season × player cross-sections with CIs, and the ~8–9 evaluable-season ceiling (§4.6.1) is stated, never overclaimed as "many." |
| `src/benchmark/__init__.py` | NEW | Package marker. |
| `scripts/run_benchmark.py` | NEW | CLI to produce the scoreboard. |

**Gate (criteria in §4.7.1):** the engine's NDCG@k must beat ADP's, significantly, before any recommender work begins.

#### 4.7.1 Gate criteria (pre-registered — the most important control point)

The pass/fail metric is fixed **in advance** to prevent post-hoc metric-shopping. Ground-truth relevance is **actual season FPPG** (per-game, availability-neutral — consistent with the per-game target, so injuries do not confound the comparison).

**Tier-1 / M4 gate (projection engine vs ADP):**
- **Primary metric = NDCG@k** per season, relevance = actual-season FPPG, `k` = the startable universe (≈ 84 modeled starters; also reported at k = 36 for early-round emphasis). NDCG is chosen over plain Spearman because draft value is dominated by getting the *top* of the board right.
- **Pass = the engine's NDCG exceeds ADP's NDCG**, evaluated per `RANKING_ELIGIBLE` season, with the paired per-season difference (engine − ADP) **positive and significant**: Wilcoxon signed-rank / sign test across seasons at **p < 0.05**, and a bootstrap CI on the mean difference excluding 0.
- **Secondary diagnostics (not gate conditions):** Spearman, top-N hit rate, per-position NDCG, Tier-3 calibration coverage. They explain *where* the edge comes from; they do not pass/fail the gate.

**Tier-2 / M6 gate (recommender vs ADP-bots):**
- **Primary metric = realized roster value, scored availability-neutral**: each starter contributes `actual-season FPPG` (equivalently FPPG × a fixed games constant), **not** raw season totals — because v1 deliberately does not model availability, so the gate must not punish the recommender for opponents' injury luck (a stud who plays 5 games would otherwise tank the roster despite a correct projection).
- **Reported secondary = realized total points** (raw real-world outcome, what actually wins leagues) — informative but not pass/fail, since it confounds projection skill with injury luck.
- **Pass = our roster's availability-neutral value beats the ADP-bots'**, paired per (season, slot) across many draft slots × `SIM_ELIGIBLE` seasons, significant at **p < 0.05** with the bootstrap CI excluding 0.

Both gates also report the **data-complete-era split** (§4.6.1); the complete-era result is the forward-looking pass/fail.

### 4.8 Milestone M5–M6 — Recommender + draft-simulation gate

| File | Action | Rationale |
|---|---|---|
| `src/recommender/replacement.py` | NEW | Computes **fixed** preseason replacement level per position via the **greedy lineup-fill** algorithm that resolves flex endogenously (§4.8.1) — the recommender's most important calculation. Owns structural scarcity. |
| `src/recommender/survival.py` | NEW | ADP-survival probability that a player lasts to my next pick, using a **right-skewed, lower-bounded (log-normal-in-pick) distribution**, not a symmetric normal (§4.8.1). The only market input in the live loop. |
| `src/recommender/quantile_schedule.py` | NEW | Maps draft round → risk appetite (floor early, ceiling late). |
| `src/recommender/roster_state.py` | NEW | Tracks my roster, open starter slots, flex eligibility, and the **snake pick layout** (my next overall pick from `n_teams` + `draft_position`, §4.8.1). |
| `src/recommender/vona.py` | NEW | The core scoring loop: marginal starting-lineup value minus expected best surviving same-position player. Owns draft-flow scarcity. |
| `src/recommender/recommend.py` | NEW | Top-level entry: draft state → ranked board. |
| `src/recommender/draft_state_source.py` | NEW | Pluggable live draft-state reader. **v1 primary = fast manual CLI entry** (fuzzy name match + ADP auto-advance for deviations-only input) since the league is on **Yahoo** (no free live feed); also ships a Sleeper live-poll mode for reuse/testing. Yahoo OAuth live adapter = v1.5. Feeds picks into `roster_state`; not a wall violation. |
| `src/recommender/__init__.py` | NEW | Package marker. |
| `src/benchmark/draft_sim.py` | NEW | Tier 2: snake draft of ADP-bots vs the recommender, rosters scored **availability-neutral (starters' actual-season FPPG), with raw totals reported secondary (§4.7.1)**, over `SIM_ELIGIBLE` seasons only (ADP ≥ 180; §4.0) — no tail backfill. Doubles as the future Monte-Carlo lookahead engine. |
| `scripts/draft.py` | NEW | The live draft-day CLI: reads draft state via `draft_state_source.py` (Sleeper live poll or manual entry, §4.8.1), prints the recommended board. |
| `tests/test_replacement.py` | NEW | Verifies the greedy lineup-fill replacement levels and flex allocation (§4.8.1). |
| `tests/test_survival.py` | NEW | Verifies survival monotonicity vs pick distance and the extreme-pick floors of the skewed distribution (§4.8.1). |
| `tests/test_vona.py` | NEW | Verifies the take-now-vs-wait logic on constructed scenarios, including no double-counting with fixed replacement, the last-pick (no-next-pick → 0) edge case, and the hard roster-completion constraint (§4.8.1). |
| `tests/test_draft_layout.py` | NEW | Validates the snake pick sequence + draft-position bounds. |

**Gate (criteria in §4.7.1):** our rosters must beat ADP-bot rosters on availability-neutral realized value, significantly, across seasons/slots.

#### 4.8.1 Recommender mechanics (live draft)

**Flex-aware replacement level — the most important calculation.** Replacement is computed by a **greedy lineup-fill** over the projected ranking, which resolves flex endogenously:
1. Rank all skill players (RB/WR/TE) by projected value (the round-appropriate quantile).
2. Walk the ranking top-down. Each player fills an open **pure** slot at his position (12 teams × {2 RB, 2 WR, 1 TE}); if his pure slots are full, he fills an open **FLEX** slot (12 × 1, eligibility from `roster.flex_eligible` — RB/WR/TE here, §4.2.1, never hard-coded); otherwise he is a non-starter.
3. `replacement[P]` = the projected value of the **highest-ranked player at P who earned no starting slot** (the first non-starter at P). QB has no flex: `replacement[QB]` = the (12 × QB_slots + 1)-th QB. K/DEF are filled at replacement in the last rounds, never modeled.

The flex share of each position is thus emergent (a deeper position wins more flex slots, pushing its replacement deeper) rather than hand-assigned. Replacement is **fixed preseason** (§3); see the run limitation below.

**ADP survival distribution.** A symmetric normal mis-models ADP: real distributions are right-skewed and bounded at pick 1, so a normal **over**estimates survival of elite (top-5) picks (which almost never slip) and **under**estimates late-round fliers (fatter right tail). v1 models the pick as a **right-skewed, lower-bounded distribution (log-normal in pick-number, parameterized by ADP mean + `adp_stdev`)**, flooring elite-pick survival near 0 — fixing both biases with no extra data. Residual: the true per-player pick distribution is only approximated; the v1.5 upgrade replaces the parametric form with **empirical survival curves** from realized historical draft picks.

**Snake draft layout.** Given `n_teams` (`league.yaml`) and my `draft_position` p (validated 1 ≤ p ≤ n_teams; a per-draft CLI input), my overall pick in round r (1-indexed) is `(r−1)·n_teams + p` for odd r and `(r−1)·n_teams + (n_teams − p + 1)` for even r (the snake turn). "Picks until my next turn" = (next overall pick) − (current overall pick), and feeds the survival term. Example: position 6 of 12 → picks 6, 19, 30, 43, …

**Draft-state input — the actual user interface.** The league drafts on **Yahoo**, which has no free no-auth live feed (unlike Sleeper). **v1 primary mode = fast manual CLI entry**: the board is pre-loaded and each pick is entered by fuzzy name match (removing that player); to keep a 180-pick draft tractable, an **ADP auto-advance** shortcut consumes the expected top-ADP available players up to the next relevant pick, so the user types only *deviations* from ADP. `draft_state_source.py` stays pluggable and also ships a **Sleeper live-poll** mode (`/draft/{id}/picks`) for reuse/testing, but that is not this user's path. Draft state is availability, never a projection feature — not a wall violation. **v1.5:** a **Yahoo Fantasy API (OAuth2) live adapter** for real-time auto-tracking, deferred because OAuth setup + polling is heavier than the one-month v1 budget.

**Fixed replacement vs positional runs — known limitation, expanded.** Fixed preseason replacement does not rebase when a run drains a position: after, say, 7 of the top 8 picks are RBs, the *effective* replacement RB is ~10–15 players deeper than the preseason bar, so static VOR **under-values** the RBs you'd grab post-run and, symmetrically, in a zero-RB environment **over-values** RBs still on the board. Zero-RB / hero-RB are mainstream strategies, so this is not a corner case.
- **Why still fixed in v1:** recomputing replacement *and* running VONA double-counts intra-draft scarcity (§3). And crucially, **VONA already captures the run's *direction*** — its survival term sees RBs draining and pushes you to take one now; the static bar only stales the *magnitude* of VOR, not the pick direction.
- **Bounded + measured:** the Tier-2 draft-sim is run-heavy by construction (ADP bots produce real runs), so it **measures** the cost of static replacement rather than assuming it away. A material cost shows up at the gate.
- **v1.5 fix:** a **damped dynamic replacement** — recompute the bar from the live remaining pool but heavily damped, with an explicit correction term against the VONA double-count. Deferred, not silently ignored.

**VONA at the last pick / no future pick.** VONA subtracts the expected value of the best same-position player surviving to my *next* pick. When there is no next pick (the final pick, or the last chance to use a slot), that term is **defined as 0**, so the score reduces to pure marginal lineup value — take the best available player that improves the lineup at the current risk quantile. `recommend.py` must treat "no next pick" as 0, never awaiting or dividing by a nonexistent pick, or the last round produces garbage / crashes.

**Roster completion is a hard constraint.** VONA alone could keep recommending high-VOR RB/WR and never fill QB/TE/K/DEF, leaving an illegal lineup. The recommender enforces completion: it tracks unfilled **mandatory** slots and remaining picks, and whenever `remaining_picks ≤ unfilled_mandatory_slots` it **restricts recommendations to players that fill a still-needed mandatory slot** (most-constrained slot first if several compete). Before that threshold, an unowned required starter already carries high marginal VOR so needs are pressured naturally; the hard constraint only guarantees a legal roster regardless of how VONA would otherwise rank. K/DEF are mandatory slots filled at replacement in the final rounds under this same rule.

### 4.9 Files deleted (weekly system retired)

All deleted because they implement the player-week model this spec replaces; leaving them would create dead code that contradicts the source of truth. They remain in git history if ever needed.

```
src/ml/features/load_initial_data.py   src/ml/features/clean_data.py
src/ml/features/calc_features.py        src/ml/features/persist.py
src/ml/models/train.py                  src/ml/models/eval.py  (metrics ported to projection/eval.py)
src/api/main.py  src/api/schemas.py
src/api/services/model_loader.py        src/api/services/prediction_service.py
src/api/routes/games.py  players.py  predict.py  leaderboard.py
scripts/test_train_pipeline.py
```

The HTTP API and a frontend are **out of scope for v1** and will be rebuilt against the new feature once the model is proven; until then the only interface is `scripts/draft.py`.

### 4.10 Dependency changes (`requirements.txt`)

- **Add:** `pyyaml` (config), `lightgbm` (already present — confirm quantile-objective version), an HTTP client (`httpx` or `requests`) backing the resilient `ingest/http.py`, and the `cfbd` Python client for college data. New `.env` key: `CFBD_API_KEY` (CFBD is not anonymous), alongside the existing `DATABASE_URL`.
- **Remove:** `fastapi`, `uvicorn` (API retired for v1) — keep available but optional, since they return with the frontend.
- **Keep:** pandas, numpy, scipy, scikit-learn, sqlalchemy, psycopg2, nflreadpy, joblib.

### 4.11 Milestone M7 — Feature lift & polish

No new files by default. M7 *extends* `features/role_change.py` and `features/rookies.py` with richer signals and re-runs M4/M6 to measure incremental lift. Any genuinely new module here requires adding it to this inventory first.

### 4.12 Build sequence & gates (summary)

```
M0 ingest+schema → M1 labels → M2 features(+leakage gate)
   → M3 projection → M4 benchmark  ◄ GATE: beat ADP ranking
   → M5 recommender → M6 draft-sim ◄ GATE: beat ADP bots
   → M7 feature lift → (frontend later)
```

Vertical-slice discipline: get the thinnest engine (production + age + draft capital only) through the M4 gate before investing in role/opportunity/rookie depth. If the thin engine cannot beat ADP, more features will not save it — and we learn that with weeks to spare before the draft.
