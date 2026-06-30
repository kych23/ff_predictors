# PRD: Who Should I Start?

## Executive summary

A weekly CLI lineup optimizer that recommends optimal starting lineups and head-to-head player comparisons using quantile projections (P10/P50/P90). Leverages the existing season-level LightGBM model as a prior, augmented with weekly context features (Vegas game lines, opponent defense rank, rolling player stats, snap share trends). Targets a single-league PPR manager who re-runs multiple times per week as injury news breaks. No mainstream tool exposes distributional projections for weekly start/sit decisions — this is genuine whitespace.

## Problem statement

Fantasy managers face 3-5 start/sit decisions each week across 17 regular-season weeks. Current tools are expert-consensus aggregators (FantasyPros) or heuristic-based (ESPN, Yahoo). None provide distributional projections that let the user make risk-aware decisions (play the floor when favored, play the ceiling when underdog). The existing FantasyForecast engine already produces calibrated quantile projections at the season level — extending to weekly granularity unlocks 17x more touchpoints per season than the draft assistant.

## Target users

**Persona:** Single-league PPR manager — same user as the draft assistant. Technically comfortable (runs CLI tools, edits YAML files). Manages one roster through a full NFL season.

**Job-to-be-done:** Before each game slate (primarily Sunday, but also Thursday and Monday), set the optimal starting lineup from their roster, accounting for matchups, injuries, and bye weeks.

**Trigger:** Injury reports drop (Wednesday initial, Friday final). User re-runs the optimizer after each report update. May run 2-3 times between Wednesday and Sunday kickoff.

## Current state

**Reusable:**
- `WeeklyStatsRaw` table — weekly box scores already stored (player_id, season, week, stats JSONB)
- `score_dataframe()` — modular scoring function, works on any row with raw stat components
- `optimal_lineup_value()` — greedy lineup solver in `draft_sim.py` (needs upgrade to ILP)
- `RosterState` — slot tracking (pure, flex, bench), bye-week counting, position counting
- `load_schedules()` — nflverse schedules with per-game Vegas lines (total_line, spread_line) for every week
- `load_player_stats()` — weekly stats from nflverse
- `team_context.py` — pattern for extracting implied team totals from Vegas lines (currently Week-1 only, generalizes to all weeks)
- Season-level quantile GBM (P10/P50/P90) — becomes the prior/anchor for the weekly model

**Missing:**
- Weekly projection model (no in-season feature assembly, no weekly labels table, no weekly GBM)
- Opponent defense-vs-position (DvP) features
- Rolling in-season player features (snap share, targets, carries trends)
- Weekly projection DB tables
- Lineup optimizer CLI
- K/DEF scoring models

## Proposed solution

A two-layer weekly projection system:

1. **Layer 1 (prior):** The existing season-level P50 projection serves as a baseline anchor. Strong early-season when in-season sample is thin; decays in influence as weekly data accumulates.

2. **Layer 2 (weekly adjustment):** A weekly quantile GBM trained on (player, week) observations. Features include the season P50 prior, per-game Vegas line, opponent DvP rank, rolling 3-5 week EWMA stats (snap share, target share, carries, xFP), and bye/injury context. Produces weekly P10/P50/P90.

3. **Lineup optimizer:** ILP solver (PuLP or scipy.optimize.milp) that assigns players to slots optimally, with a `--strategy` flag that shifts weighting between P10 (safe), P50 (balanced), and P90 (upside).

4. **CLI:** `scripts/start.py` — reads a YAML roster file, runs projections, outputs slot-based optimal lineup + optional `compare X vs Y` command.

## Scope

**In:**
- Weekly quantile projections for QB, RB, WR, TE, K, DEF
- ILP-based optimal lineup solver with strategy modes (safe/balanced/upside)
- CLI with slot-based lineup display, matchup grades (color-coded), head-to-head comparison
- YAML roster file (name, position, team, status)
- Auto-detect bye weeks from nflverse schedules
- Manual injury marking in roster file (active/Q/D/O/IR)
- Rolling in-season features: snap share, target share, carries, xFP (EWMA, 3-5 week window)
- Per-game Vegas line features (implied team total, spread, game total)
- Opponent defense-vs-position (DvP) ranking
- "Points left on bench" benchmark across historical seasons
- New DB tables: WeeklyProjection, WeeklyFeature

**Out:**
- Platform sync (Sleeper, ESPN, Yahoo API) — user manages YAML manually
- Multi-league support
- Waiver wire / trade recommendations (separate future PRDs)
- Live in-game updates or real-time scoring
- Weather features (low signal-to-noise for fantasy)
- Daily fantasy (DFS) salary-cap optimization
- Push notifications or alerts
- Web UI

## User stories

- As a fantasy manager, I want to see my optimal starting lineup for the upcoming week, so I can set my roster with confidence.
- As a fantasy manager, I want to compare two players head-to-head for a specific slot, so I can resolve close start/sit decisions.
- As a fantasy manager, I want to choose between a "safe" and "upside" lineup strategy, so I can adapt to whether I'm favored or underdog in my matchup.
- As a fantasy manager, I want to re-run the optimizer after injury reports update, so my lineup reflects the latest information.
- As a fantasy manager, I want to see matchup grades (favorable/neutral/tough) alongside projections, so I can weigh context beyond raw numbers.

## Functional requirements

**F1 — Roster file:** YAML format with fields: name, position, team, status (active/Q/D/O/IR). Loaded by CLI. Example:
```yaml
season: 2026
roster:
  - name: "Josh Allen"
    position: QB
    team: BUF
    status: active
  - name: "Saquon Barkley"
    position: RB
    team: PHI
    status: IR
```

**F2 — Weekly feature assembly:** For a target (season, week), compute features using only data available before that week's kickoff:
- Season P50 projection (from existing model) as prior
- Per-game Vegas line: implied team total, spread, game total (from nflverse schedules)
- Opponent DvP: rolling fantasy points allowed by opponent defense to each position, EWMA-smoothed, with early-season blending (lean on prior-season DvP for weeks 1-3, ramp to current-season by week 7+)
- Rolling player stats: EWMA over weeks 1..(N-1) for snap share, target share, carries per game, xFP, actual FPPG
- Bye week flag (auto-detected from schedules)
- Player status from roster file (OUT/IR players excluded from lineup, Q/D included with flag)

**F3 — Weekly quantile model:** Pooled LightGBM quantile GBM (P10/P50/P90) trained on (player, week) observations across historical seasons. Position as categorical feature. Same monotonicity enforcement (rearrangement) and per-bucket conformal calibration as the season model.

**F4 — ILP lineup optimizer:** Given weekly projections for all roster players, solve for optimal slot assignment (QB:1, RB:2, WR:2, TE:1, FLEX:1, K:1, DEF:1). FLEX eligible: RB, WR, TE. Exclude players with status OUT or IR or on bye. Strategy modes:
- `safe` — optimize on P10 (maximize floor)
- `balanced` — optimize on P50 (default)
- `upside` — optimize on P90 (maximize ceiling)

**F5 — CLI output (lineup):** Slot-based table with columns: SLOT, PLAYER, OPP, GRADE, P10, P50, P90, verdict. Matchup grade color-coded (green = top-10 DvP, yellow = 11-22, red = 23-32). Verdict: START (in optimal lineup), BENCH (not in lineup), FLEX? (marginal FLEX candidate). Bench section below starters showing what's left. Include a "read" interpretation line (same pattern as draft.py).

**F6 — CLI output (compare):** `compare <A> vs <B>` command renders two-column table with: weekly P10/P50/P90, matchup grade, opponent, last 3 game scores, season P50, and a verdict line.

**F7 — CLI invocation:**
```bash
python scripts/start.py --season 2026 --week 5 --roster roster.yaml
python scripts/start.py --season 2026 --week 5 --roster roster.yaml --strategy upside
python scripts/start.py --season 2026 --week 5 --roster roster.yaml compare "Josh Allen" vs "Jalen Hurts"
```

**F8 — Benchmark pipeline:** Backtest across historical seasons (2017-2025). For each (season, week), project all rostered players, solve optimal lineup, compare to hindsight-optimal lineup. Primary metric: mean points left on bench per week.

**F9 — New DB tables:**
- `WeeklyFeature(player_id, season, week, snapshot_id, features JSONB)` — weekly feature vectors
- `WeeklyProjection(player_id, season, week, model_version, snapshot_id, p10, p50, p90, position)` — weekly quantile projections

**F10 — Leakage enforcement:** New guard: `as_of_kickoff(df, target_season, target_week)` — filters to data strictly before the target game's kickoff. Extends the existing `prior_seasons()` / `preseason_rows()` pattern. Test coverage in `test_leakage.py`.

**F11 — K/DEF scoring models:** Extend the quantile GBM to include K and DEF positions. K features: team implied total, indoor/outdoor (if available), opponent FG allowed. DEF features: opponent implied total, opponent turnovers trend, sacks trend. These positions have lower predictability but users want full-lineup coverage.

## Non-functional requirements

- **Performance:** Weekly projections for a 15-player roster must complete in <5 seconds. Full-season backtest (17 weeks x 8 seasons) in <10 minutes.
- **Data freshness:** Vegas lines from nflverse update mid-week. User re-runs CLI to pick up latest schedules data.
- **Reproducibility:** All DB writes include `snapshot_id`. Weekly projections are deterministic given the same snapshot + model_version + week.
- **Config coupling:** Weekly model inherits scoring rules from `league.yaml`. Changing scoring requires full re-train + re-benchmark.

## Success metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Mean points left on bench (per week) | <5.0 FPPG | Backtest across 2017-2025, all weeks |
| Weekly P50 MAE | <6.0 FPPG | Per-player weekly prediction error |
| Weekly P10-P90 coverage | 0.78-0.82 | Fraction of actual scores within interval |
| Optimal starter hit rate | >70% | % of recommended starters matching hindsight optimal |

## Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Weekly model overfits to small in-season sample (weeks 1-3) | High | Medium | Two-layer architecture: season P50 prior dominates early, weekly features ramp up via blending weight |
| K/DEF projections are low-signal noise | High | Low | Accept wider intervals; K/DEF projections are "better than random" bar, not "beat experts" |
| Opponent DvP is noisy early-season | High | Medium | SAFPA blending: prior-season DvP weighted 90% at week 1, linearly ramp to 0% by week 7 |
| nflverse Vegas lines not available for future weeks | Low | High | Fallback to season-level team_context (Week-1 line) when per-week line unavailable |
| YAML roster file is error-prone (typos, stale data) | Medium | Medium | Fuzzy name matching (existing pattern from draft.py); warn on unrecognized names |
| ILP solver adds dependency (PuLP) | Low | Low | PuLP is pure Python, well-maintained; scipy.optimize.milp as fallback (no new dep) |

## Implementation hints

**Phase 1 — Weekly data pipeline:**
- New module: `src/features/weekly_features.py` — rolling EWMA stats, opponent DvP, per-week Vegas context
- New module: `src/labels/weekly_labels.py` — per-(player, week) fantasy points from existing `WeeklyStatsRaw` + `score_dataframe()`
- New DB models: `WeeklyFeature`, `WeeklyProjection`
- Leakage guard extension: `as_of_kickoff()` filter

**Phase 2 — Weekly model:**
- Extend `QuantileGBM` to train on weekly observations
- Two-layer: season P50 as a feature column in the weekly model
- Per-bucket conformal calibration at weekly level
- K/DEF position models (new feature engineering)

**Phase 3 — Lineup optimizer:**
- ILP solver using PuLP or scipy milp
- Strategy modes (safe/balanced/upside) via quantile selection
- Replace existing greedy `optimal_lineup_value()` in benchmarks too

**Phase 4 — CLI:**
- `scripts/start.py` — roster loading, projection, optimization, display
- Color-coded matchup grades (consider Rich library or ANSI escapes)
- `compare` subcommand
- YAML roster file format

**Phase 5 — Benchmark:**
- Backtest harness: simulate setting lineup each week across 2017-2025
- Points left on bench metric
- Weekly MAE and quantile calibration metrics

**Key dependencies:**
- Season-level model must be trained first (existing pipeline)
- nflverse schedules must include Vegas lines for the target week
- Roster file must be populated by user

## Open questions

- **Model approach (G4):** Deferred. Should the weekly model be a standalone GBM trained on weekly data, or an adjustment layer on the season P50? Research suggests the two-layer approach (season prior + weekly adjustment) outperforms early-season, but a standalone weekly GBM may be better late-season. Recommendation: start with two-layer, benchmark against standalone, pick the winner.
- **PuLP vs scipy.optimize.milp:** PuLP has cleaner API for constraint modeling but adds a dependency. scipy.optimize.milp is already available (scipy is in requirements.txt). Test both.
- **Rich library for CLI rendering:** Adds a dependency but provides color-coded tables, proper column alignment, and terminal-width fitting. The existing draft.py uses raw print(). Decide whether to adopt Rich for both tools or keep raw print().
- **Weekly model training window:** Should the model train on all historical weeks (2012-2025, ~40K observations) or use a rolling window (last 3-4 seasons)?
- **How to handle Thursday/Monday games:** Players on TNF have already played by Sunday. Exclude from Sunday optimizer? Show actual score?

## Sources & references

**Market & Prior Art:**
- [FantasyPros Start/Sit](https://www.fantasypros.com/nfl/start/) — expert consensus aggregator, category leader
- [DraftSharks Who to Start](https://www.draftsharks.com/who-should-i-start) — floor/median/ceiling + strategy slider
- [PFN Start/Sit Optimizer](https://www.profootballnetwork.com/fantasy-hq/start-sit-optimizer) — free 12-player comparison
- [Fantasy Projection Lab](https://fantasyprojectionlab.com/nfl-fantasy-projections) — only tool with 90% confidence intervals
- [ffopportunity](https://github.com/ffverse/ffopportunity) — nflverse expected fantasy points (post-game, not pre-game)
- [KeepTradeCut Start/Sit](https://keeptradecut.com/fantasy/start-sit-tool) — community-vote-based

**Technical Patterns:**
- [SAFPA — Schedule-Adjusted FPA](https://www.fantasypoints.com/nfl/stats/points-allowed/schedule-adjusted) — opponent quality normalization
- [PuLP DFS Optimizer](https://zwlevonian.medium.com/integer-linear-programming-with-pulp-optimizing-a-draftkings-nfl-lineup-5e7524dd42d3) — ILP for lineup optimization
- [arXiv 2309.15253](https://arxiv.org/abs/2309.15253) — ML + LP for DFS lineup optimization
- [nflreadpy](https://github.com/nflverse/nflreadpy) — Python interface to nflverse data

**UX Precedent:**
- [Yahoo Matchup Ratings](https://help.yahoo.com/kb/SLN9034.html) — 5-star + color grading system
- [Rich Library](https://github.com/textualize/rich) — Python CLI table rendering with color
- [RotoGrinders LineupHQ](https://rotogrinders.com/lineuphq) — slot-based lineup display pattern

## Out-of-scope follow-ups

- **"Will I Win My Matchup?"** — given both teams' starting rosters, simulate win probability using quantile distributions (Monte Carlo over P10-P90 intervals)
- **"Who Should I Trade?"** — 1-to-1 trade value comparison using rest-of-season projections
- **Trade Generator** — find optimal trade targets given roster needs and opponent surplus
- **Platform sync** — Sleeper/ESPN/Yahoo API integration to auto-populate roster file
- **Waiver wire recommendations** — rank available free agents by marginal value to user's roster
- **In-game live updates** — real-time scoring and projection adjustments during games
