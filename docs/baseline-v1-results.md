# Baseline: v1 VONA-Validated Results

**Snapshot commit:** `56aabbc` (tagged `v1-vona-validated`)

These are the benchmark numbers the validated pooled-LightGBM-quantile +
VONA-recommender system achieved before the Monte Carlo decision-layer
rebuild (`strip/monte-carlo-rebuild` branch). Copied verbatim from
`README.md` prior to any deletions — **not reproducible after the strip**,
since `src/benchmark/draft_sim.py` (Tier-2) and the FFC-based
`src/ingest/adp.py` are removed as part of it. The new system must beat
these numbers under its own objective (expected dollars) before it's
considered a real improvement, not just a different metric.

## Benchmark Results (as of `56aabbc`)

Pre-registered gate: **NDCG@k**, paired per season, with Wilcoxon
signed-rank / sign tests and a bootstrap 95% CI. Ground-truth relevance is
**actual-season FPPG** (availability-neutral, so injuries don't confound
the ranking). Reported on the data-complete era — test seasons **2017–2024**
(8 evaluable seasons, after the 5-season cross-validation warmup). Current
snapshot: **3,386 projections**.

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

Engine beats ADP in **8 of 8** seasons. Honest ceiling: ~8–9 evaluable
seasons is the hard limit of available NFL history — not "many" — so
significance is taken across season × player cross-sections.

### Tier-3 — risk calibration (interval honesty)

| Bucket           | n    | P10–P90 coverage | nominal |
| ---------------- | ---- | ---------------- | ------- |
| overall          | 3386 | 0.804            | 0.80    |
| established_vet  | 1646 | 0.801            | 0.80    |
| rookie           | 479  | 0.804            | 0.80    |
| second_year      | 566  | 0.802            | 0.80    |
| team_changed_vet | 695  | 0.816            | 0.80    |

Coverage is from the **split-conformal per-bucket** calibrator
(`src/projection/calibrate.py`), validated on the 2026-06-27 run (with the
`team_context` feature). The width scale per side is a quantile of the
normalized nonconformity score (residual ÷ predicted half-width), which
targets the empirical coverage fraction directly — unlike the earlier
mean-matching scale, which left intervals narrow (~0.77, rookies worst at
0.756) because matching *mean* half-widths under-covers whenever per-row
widths are heterogeneous (exactly the rookie case). Every bucket now sits
on the 0.80 nominal; the rookie bucket — the whole reason for per-type
calibration — went 0.756 → 0.804. The calibrator only ever widens (scale
floored at 1.0), so well-calibrated buckets are left untouched.

### Tier-2 — recommender vs ADP bots

Draft simulation: our recommender vs 11 bots drafting by ADP, on
`SIM_ELIGIBLE` seasons, rosters scored availability-neutral
(starting-lineup FPPG). Across **35 (season × slot) drafts**:

| Metric              | mean(ours − bots) | Wilcoxon p | bootstrap 95% CI | wins    | Result   |
| ------------------- | ----------------- | ---------- | ----------------- | ------- | -------- |
| Starting-lineup PPG | **+4.74**         | 0.000277   | [+2.64, +6.94]     | 28 / 35 | **PASS** |

The recommender — not just the engine — beats the market. Adding the
`team_context` market feature lifted this tier (mean edge +3.97 → +4.74,
win rate 25 → 28 of 35, p 0.0014 → 0.000277) while leaving Tier-1 flat
within its CI. Down years like 2024 still pull the mean, but the paired
edge is robust. **This tier is not reproducible after the strip** —
`src/benchmark/draft_sim.py` is deleted (superseded by the Monte Carlo
layer); `optimal_lineup` is superseded by `src/lineup/optimizer.py`, and
`_seat_on_clock`/`_fill_slot` are preserved in `src/recommender/snake.py`.

### Weekly — lineup benchmark (points left on bench)

Simulated rosters (snake-draft, averaged over early/mid/late draft slots),
lineups set each week from OOF weekly projections, then scored against
what actually happened:

| Metric                   | Result | Target | Gate     |
| ------------------------ | ------ | ------ | -------- |
| Mean pts left on bench   | 18.38  | < 20.0 | **PASS** |
| Optimal starter hit rate | 67.5%  | —      | —        |

Weekly OOF interval coverage sits at 0.798–0.812 per position bucket
against the 0.80 nominal, under leave-one-fold-out calibration (each
season calibrated only on the *other* seasons, so reported coverage can't
flatter itself). This tier's machinery (`src/projection/weekly_dataset.py`,
`weekly_train.py`, `src/features/weekly_features.py`, `weekly_assemble.py`,
`scripts/build_weekly_data.py`, `train_weekly_projection.py`,
`run_weekly_benchmark.py`) is **kept** through the strip — it's the
in-season belief-updating component the Monte Carlo simulator needs to set
lineups non-clairvoyantly.
