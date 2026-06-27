# FantasyForecast

PPR fantasy football draft assistant. Pooled LightGBM quantile model (P10/P50/P90) + VONA-based draft recommender.

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
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
CFBD_API_KEY=your_key_here   # optional — enables college features
```

League settings live in `config/league.yaml`.

## Pipeline

Run in order once you have a `DATABASE_URL`:

```bash
# 1. Pull NFL data + ADP into DB
python scripts/seed_db.py --start 2012 --end 2025

# 2. Compute per-season labels (FPPG)
python scripts/build_labels.py --snapshot-id <id>

# 3. Build model features
python scripts/build_features.py --start 2012 --end 2025 --snapshot-id <id>

# 4. Train pooled quantile model
python scripts/train_projection.py --snapshot-id <id>

# 5. Evaluate vs ADP baseline
python scripts/run_benchmark.py --snapshot-id <id>
```

## Live Draft

```bash
python scripts/draft.py --season 2025 --position 4   # --position = your draft slot (1..teams)
```

Commands during draft:

- `go` — auto-advance opponents by ADP up to your next pick
- `me <player name>` — record your pick (fuzzy match; same-name players prompt to disambiguate by position/team)
- `<player name>` — record an opponent's pick
- `board` — reprint the recommendation board
- `quit` — exit

## Architecture

**ADP wall**: `src/projection/` and `src/features/` never import ADP or market data. Only `src/ingest/adp.py`, `src/recommender/`, and `src/benchmark/` may touch ADP.

**Model**: one pooled `QuantileGBM` (position as categorical feature) — not per-position models. Quantile monotonicity via rearrangement (Chernozhukov 2010), not clipping.

**Recommender**: VONA score = marginal VOR − E[best same-position surviving to next pick]. ADP survival via log-normal distribution. Round-shifted quantile (P25 early → P85 late).

## Benchmark Results

Pre-registered gate: **NDCG@k**, paired per season, with Wilcoxon signed-rank / sign tests and a bootstrap 95% CI (see [Glossary](#glossary)). Ground-truth relevance is **actual-season FPPG** (availability-neutral, so injuries don't confound the ranking). Reported on the data-complete era — test seasons **2017–2024** (8 evaluable seasons, after the 5-season cross-validation warmup). Current snapshot: **3,386 projections**.

### Tier-1 — projection ranking vs ADP (per season, k = 84)

| Season | n | NDCG@84 engine | NDCG@84 ADP | Spearman engine | Spearman ADP | Hit@84 engine | Hit@84 ADP |
|---|---|---|---|---|---|---|---|
| 2017 | 136 | 0.889 | 0.851 | 0.632 | 0.425 | 0.762 | 0.714 |
| 2018 | 144 | 0.904 | 0.866 | 0.645 | 0.412 | 0.738 | 0.655 |
| 2019 | 148 | 0.946 | 0.866 | 0.734 | 0.470 | 0.810 | 0.702 |
| 2020 | 141 | 0.931 | 0.855 | 0.759 | 0.477 | 0.798 | 0.762 |
| 2021 | 150 | 0.943 | 0.860 | 0.739 | 0.538 | 0.821 | 0.702 |
| 2022 | 131 | 0.925 | 0.882 | 0.773 | 0.551 | 0.810 | 0.821 |
| 2023 | 156 | 0.923 | 0.881 | 0.693 | 0.530 | 0.821 | 0.762 |
| 2024 | 160 | 0.918 | 0.839 | 0.673 | 0.471 | 0.750 | 0.631 |

**Tier-1 gate (paired across 8 seasons):**

| Metric | mean(engine − ADP) | Wilcoxon p | sign-test p | bootstrap 95% CI | Result |
|---|---|---|---|---|---|
| NDCG@84 (full board) | **+0.062** | 0.0078 | 0.0078 | [+0.048, +0.075] | **PASS** |
| NDCG@36 (early rounds) | **+0.083** | 0.0078 | 0.0078 | [+0.064, +0.102] | **PASS** |

Engine beats ADP in **8 of 8** seasons. Honest ceiling: ~8–9 evaluable seasons is the hard limit of available NFL history — not "many" — so significance is taken across season × player cross-sections.

### Tier-3 — risk calibration (interval honesty)

| Bucket | n | P10–P90 coverage | nominal | pinball P10 | pinball P50 | pinball P90 |
|---|---|---|---|---|---|---|
| overall | 3386 | 0.771 | 0.80 | 0.487 | 1.265 | 0.657 |
| established_vet | 1646 | 0.765 | 0.80 | 0.481 | 1.201 | 0.637 |
| rookie | 479 | 0.756 | 0.80 | 0.549 | 1.477 | 0.791 |
| second_year | 566 | 0.769 | 0.80 | 0.475 | 1.317 | 0.691 |
| team_changed_vet | 695 | 0.796 | 0.80 | 0.468 | 1.230 | 0.584 |

Per-type calibration is active: **rookies are the most under-covered group** (0.756) — exactly the overconfidence the per-type widening targets. Intervals still run slightly narrow overall (~0.77 vs 0.80 target); a conformal per-bucket widening is the planned refinement.

### Tier-2 — recommender vs ADP bots

Draft simulation: our recommender vs 11 bots drafting by ADP, on `SIM_ELIGIBLE` seasons, rosters scored availability-neutral (starting-lineup FPPG). Across **35 (season × slot) drafts**:

| Metric | mean(ours − bots) | Wilcoxon p | bootstrap 95% CI | wins | Result |
|---|---|---|---|---|---|
| Starting-lineup PPG | **+4.62** | 0.0008 | [+2.16, +6.97] | 28 / 35 | **PASS** |

The recommender — not just the engine — beats the market. (The margin is a touch lower than an earlier run because the ADP namesake-join fix now lets the bots draft the previously-unmatched stars too — a fairer comparison; win rate rose 25→28 of 35.) Down years like 2024 still pull the mean, but the paired edge is robust.

## Data Sources

- [nflverse](https://www.nflverse.com/) — play-by-play, rosters, combine, draft picks
- [Fantasy Football Calculator](https://fantasyfootballcalculator.com/) — full-PPR ADP
- [CFBD](https://collegefootballdata.com/) — college stats (optional)

## Glossary

**Scoring & fantasy terms**

- **PPR** — Points Per Reception: scoring that awards 1 point per catch.
- **ADP** — Average Draft Position: where a player is drafted on average across thousands of public drafts. The market baseline this project aims to beat.
- **FPPG** — Fantasy Points Per Game: the model's prediction target (a per-game *rate*, not a season total — so it isn't distorted by missed games).
- **VOR** — Value Over Replacement: a player's projected value minus a freely-available "replacement" player at his position (the first one who doesn't earn a starting slot).
- **VONA** — Value Of Not Available: VOR *now* minus the expected value of the best same-position player still available at your *next* pick. Encodes "take the scarce position now, wait on the deep one."

**Model terms**

- **GBM** — Gradient-Boosted (decision) trees Model; an ensemble of trees built sequentially.
- **LightGBM** — a fast, widely-used GBM implementation (the model used here).
- **P10 / P50 / P90** — the 10th / 50th (median) / 90th percentile of a player's predicted outcome: the low / middle / high cases. The model predicts a *range*, not one number.
- **Snapshot** — a frozen extraction of source data (tagged with a `snapshot_id`) so results stay reproducible even though nflverse retroactively corrects past stats.

**Metrics**

- **NDCG@k** — Normalized Discounted Cumulative Gain over the top *k* players: a 0–1 ranking-quality score that weights the top of the list most heavily (getting pick #1 right matters more than #95). The primary gate metric; `k=84` ≈ the startable universe, `k=36` ≈ the early rounds.
- **Spearman** — Spearman rank correlation between the predicted order and the actual finish (−1 to +1; higher is better).
- **Hit@k** — Hit rate: the fraction of the true top-*k* players that the ranking also placed in its top *k*.
- **Pinball loss** — the standard quantile-regression loss; lower means a better-calibrated quantile (P10/P50/P90).
- **Coverage** — the fraction of actual outcomes that fell inside the predicted P10–P90 interval (target = 0.80).
- **Bootstrap 95% CI** — a confidence interval estimated by resampling; if it excludes 0, the measured edge is unlikely to be noise.
- **Wilcoxon signed-rank / sign test** — paired non-parametric significance tests comparing engine vs ADP across seasons.

**Data sources**

- **CFBD** — College Football Data API (college statistics for rookies).
- **nflverse** — open NFL data project, accessed via the `nflreadpy` Python library.
