# Design Doc — Phase 1: Weekly Data Pipeline (rev 3)

## Why

Foundation for "Who Should I Start?" (PRD `notes/prds/who-should-i-start-2026-06-28.md`). Weekly projections need weekly labels, weekly features, and a leakage guard extension. Without this layer, no weekly model can train.

## Scope

**In:** Weekly labels table + builder, weekly features table + assembler, weekly leakage guard (`as_of_kickoff`), opponent DvP computation, per-week Vegas context (generalizing `team_context.py`), rolling in-season EWMA features, upsert helpers, build script, tests.

**Out:** Weekly quantile model (Phase 2), ILP optimizer (Phase 3), CLI (Phase 4), benchmark (Phase 5), K/DEF models (Phase 2).

## Surfaces touched

| File | Change |
|------|--------|
| `src/db/models.py` | Add `WeeklyLabel`, `WeeklyFeature` tables |
| `src/db/upsert_data.py` | Add `upsert_weekly_labels`, `upsert_weekly_features` |
| `src/labels/build_labels.py` | Add `compute_weekly_labels()`, `build_weekly_labels()`, extend `_load_weekly_raw()` |
| `src/features/leakage_guard.py` | Add `prior_weeks()`, `assert_no_future_week()` |
| `src/features/weekly_features.py` | **NEW** — rolling stats, opponent DvP, per-week Vegas |
| `src/features/weekly_assemble.py` | **NEW** — orchestrator analogous to `assemble.py` |
| `scripts/build_weekly_data.py` | **NEW** — CLI script for weekly pipeline |
| `tests/test_weekly_leakage.py` | **NEW** — poison test for as-of-kickoff rule |
| `tests/test_weekly_labels.py` | **NEW** — scoring correctness at weekly grain |

## Interfaces

### DB models (`src/db/models.py`)

```python
class WeeklyLabel(Base):
    __tablename__ = "weekly_labels"
    player_id = Column(String, primary_key=True)
    season = Column(Integer, primary_key=True)
    week = Column(Integer, primary_key=True)
    position = Column(String, nullable=True)   # carried from WeeklyStatsRaw
    team = Column(String, nullable=True)        # carried from WeeklyStatsRaw
    fantasy_points = Column(Float, nullable=False)
    snapshot_id = Column(String, nullable=False, index=True)
    __table_args__ = (Index("ix_weekly_labels_season", "season"),)

class WeeklyFeature(Base):
    __tablename__ = "weekly_features"
    player_id = Column(String, primary_key=True)
    season = Column(Integer, primary_key=True)
    week = Column(Integer, primary_key=True)
    position = Column(String, nullable=True)
    features = Column(JSONB, nullable=False)
    snapshot_id = Column(String, nullable=False, index=True)
    __table_args__ = (Index("ix_weekly_features_season", "season"),)
```

`WeeklyFeature` intentionally omits `is_rookie` and `as_of_date` (present on `SeasonFeature`). Player type metadata lives inside the JSONB blob if needed by the weekly model. `as_of_date` is implicit from (season, week).

### Upserts (`src/db/upsert_data.py`)

```python
def upsert_weekly_labels(df: pd.DataFrame, session: Session) -> int:
    cols = ["player_id", "season", "week", "position", "team", "fantasy_points", "snapshot_id"]
    idx = [WeeklyLabel.player_id, WeeklyLabel.season, WeeklyLabel.week]
    update = ["position", "team", "fantasy_points", "snapshot_id"]

def upsert_weekly_features(df: pd.DataFrame, session: Session) -> int:
    cols = ["player_id", "season", "week", "position", "features", "snapshot_id"]
    idx = [WeeklyFeature.player_id, WeeklyFeature.season, WeeklyFeature.week]
    update = ["position", "features", "snapshot_id"]
```

Both use the existing `_upsert()` helper pattern.

### Leakage guard (`src/features/leakage_guard.py`)

```python
def prior_weeks(df, target_season, target_week, season_col="season", week_col="week"):
    """Return rows where:
      (season < target_season) OR (season == target_season AND week < target_week).
    Standalone function — does NOT call prior_seasons() internally.
    Does NOT filter by season_type (caller responsibility)."""

def assert_no_future_week(df, target_season, target_week, *,
                          season_col="season", week_col="week", where=""):
    """Hard assert no rows where:
      (season > target_season) OR (season == target_season AND week >= target_week).
    Raises LeakageError with count + location."""
```

### Labels (`src/labels/build_labels.py`)

**Extend `_load_weekly_raw`:** Add `r.position`, `r.team`, `r.season_type` from the ORM row. Set ORM values AFTER `rec.update(r.stats or {})` so ORM values win if there's a key collision in the JSONB (defensive ordering).

```python
def _load_weekly_raw(session, snapshot_id=None):
    # ... existing query logic ...
    for r in rows:
        rec = {"player_id": r.player_id, "season": r.season, "week": r.week}
        rec.update(r.stats or {})
        # ORM columns set AFTER stats update so they win on collision
        rec["position"] = r.position
        rec["team"] = r.team
        rec["season_type"] = r.season_type
        recs.append(rec)
    return pd.DataFrame(recs)

def compute_weekly_labels(weekly: pd.DataFrame, snapshot_id: str) -> pd.DataFrame:
    """Score each week individually. No min_games filter (that's a season-label
    concern; weekly labels have exactly 1 game per row).
    Filters to season_type == 'REG' only (postseason excluded).
    Returns: (player_id, season, week, position, team, fantasy_points, snapshot_id)."""
    if "season_type" in weekly.columns:
        weekly = weekly[weekly["season_type"] == "REG"].copy()
    else:
        weekly = weekly.copy()
    weekly["fantasy_points"] = score_dataframe(weekly)
    return weekly[["player_id", "season", "week", "position", "team",
                   "fantasy_points"]].assign(snapshot_id=snapshot_id)

def build_weekly_labels(snapshot_id=None) -> int:
    """Load WeeklyStatsRaw via _load_weekly_raw(), compute, upsert to weekly_labels."""
```

### Weekly features (`src/features/weekly_features.py`)

```python
def build_opponent_dvp(weekly_labels: pd.DataFrame, schedules: pd.DataFrame,
                       target_season: int, target_week: int,
                       prior_season_labels: pd.DataFrame,
                       halflife_games: float = 3.0) -> pd.DataFrame:
    """Fantasy points allowed by each defense to each position, EWMA-smoothed.

    Data used:
    - `weekly_labels`: already filtered to prior_weeks (season Y weeks < W)
    - `prior_season_labels`: full prior season (Y-1) weekly labels for baseline
    - `schedules`: full schedules for seasons Y-1 and Y (for opponent resolution)

    Step 1: Join opponent to each label row. For each (season, week, team) in
    weekly_labels, look up the game in schedules to find the opponent:
      - If team == home_team -> opponent = away_team
      - If team == away_team -> opponent = home_team
    
    Step 2: Same for prior_season_labels.
    
    Step 3: Concatenate prior_season + current-season labels (sorted chronologically).
    
    Step 4: Group by (opponent, position). For each group, sort by (season, week),
    compute sum of fantasy_points per game-week, then EWMA(halflife=halflife_games).
    Take the final EWMA value as dvp_fpa.

    halflife_games: EWMA halflife in units of GAMES (observations), not calendar
    weeks. Since teams play once per week, 3.0 games ≈ 3 weeks. Bye weeks create
    gaps in the observation sequence but EWMA operates on observation index, so
    byes simply mean no observation (the prior EWMA value carries forward).

    Returns: DataFrame with columns (team, position, dvp_fpa).
    'team' is the DEFENSE (the team that allowed the points)."""

def build_weekly_vegas(schedules: pd.DataFrame, target_season: int,
                       target_week: int) -> pd.DataFrame:
    """Per-team Vegas context for a specific week.

    Filters schedules internally to (season == target_season, week == target_week,
    game_type == 'REG'). Same formula as team_context.py:
      home_implied = total_line/2 + spread_line/2
      away_implied = total_line/2 - spread_line/2

    Returns: DataFrame with columns (team, opponent, wk_implied_pts, wk_game_total,
    wk_spread). One row per team (two rows per game: one home, one away).
    
    The `opponent` column IS used downstream by the assembler to:
    (a) determine each player's opponent for the target week
    (b) join DvP on (opponent, position)
    
    Empty DataFrame if no lines available for that week."""

def build_rolling_player_stats(player_stats: pd.DataFrame,
                                snap_counts: pd.DataFrame,
                                pfr_to_gsis: dict,
                                halflife_games: float = 3.0) -> pd.DataFrame:
    """Rolling EWMA over recent games for each player.

    IMPORTANT: caller must pre-filter both player_stats and snap_counts via
    prior_weeks() AND filter to season_type == 'REG' (postseason games excluded
    to avoid playoff selection bias and usage pattern distortion).

    Data sources (all pre-filtered by caller):
    - `player_stats`: nflreadpy load_player_stats() output. Has target_share,
      carries, receiving_yards, etc. Keyed by (player_id, season, week).
      NOT WeeklyStatsRaw.stats JSONB (which only has scoring components).
    - `snap_counts`: nflreadpy load_snap_counts(). Keyed by
      (pfr_player_id, season, week). Joined via pfr_to_gsis mapping.
    - `pfr_to_gsis`: dict mapping pfr_player_id -> gsis_id (player_id).

    halflife_games: EWMA halflife in GAMES (observations), not calendar weeks.

    Computation: for each player, sort games by (season, week), compute EWMA:
      - rolling_fppg (score raw stats via score_dataframe)
      - rolling_target_share (from player_stats.target_share)
      - rolling_snap_share (from snap_counts.offense_pct, joined via pfr_to_gsis)
      - rolling_carries_pg (carries per game from player_stats)
    Take the final EWMA value per player.

    Returns: one row per player_id with rolling_* columns only (no season/week
    columns — the assembler knows the target (season, week) context and adds it)."""
```

### Weekly assembler (`src/features/weekly_assemble.py`)

```python
def assemble_weekly_features(target_season: int, target_week: int,
                              frames: dict, *, cfg=None,
                              snapshot_id=None) -> pd.DataFrame:
    """Assemble all weekly feature blocks for one (season, week).

    Required `frames` keys:
    - "player_stats": pd.DataFrame — nflreadpy load_player_stats() (all seasons)
    - "snap_counts": pd.DataFrame — nflreadpy load_snap_counts() (all seasons, >= 2012)
    - "schedules": pd.DataFrame — nflreadpy load_schedules() (all seasons)
    - "weekly_labels": pd.DataFrame — from WeeklyLabel table (all seasons+weeks)
    - "projections": pd.DataFrame — from Projection table, columns:
        player_id, season, position, p50. Loaded via:
        session.execute(select(Projection)).scalars().all()
        then take the latest model_version per (player_id, season) and keep
        only the p50 column (renamed to season_p50).
    - "rosters": pd.DataFrame — nflreadpy load_rosters() (all seasons)
    - "pfr_to_gsis": dict — from build_id_map(), same as season assembler
        (assemble.py:228-229)

    Assembly steps:

    1. LEAKAGE FILTERING
       Filter player_stats + snap_counts to prior_weeks(target_season, target_week).
       Additionally filter to season_type == 'REG' (exclude postseason).
       Assert with assert_no_future_week().
       Filter weekly_labels to prior_weeks for DvP + rolling stats.
       Prepare prior_season_labels = weekly_labels where season == target_season - 1.

    2. ROLLING PLAYER STATS
       build_rolling_player_stats(filtered_stats, filtered_snaps, pfr_to_gsis)
       -> rolling_* columns per player_id.

    3. OPPONENT DvP
       build_opponent_dvp(filtered_labels, schedules, target_season, target_week,
                          prior_season_labels)
       -> dvp_fpa per (defense_team, position).

    4. WEEKLY VEGAS
       build_weekly_vegas(schedules, target_season, target_week)
       -> (team, opponent, wk_implied_pts, wk_game_total, wk_spread) per team.
       This also resolves each team's opponent for the target week.

    5. PLAYER UNIVERSE
       Determine which players get a feature row for this (season, week):
       - From weekly_vegas: get the set of teams playing in target_week.
       - For week 1 of a season (no prior weekly_labels in this season):
         use rosters for target_season to get player_id -> team mapping.
       - For week 2+: use the most recent weekly_labels appearance for each
         player in the current season (any week < target_week, not just the
         immediately prior week). This handles injury absences: a player who
         missed week 4 but played weeks 1-3 is still in the universe for week 5.
       - Filter to players whose team is in the set of teams playing this week
         (excludes bye-week players).
       - Position: use position from most recent weekly_label appearance in
         the current season; fallback to Projection.position; fallback to
         Player table. Priority: label > projection > player table (mirrors
         the season assembler's roster > depth > player priority).

    6. SEASON P50 PRIOR
       Join season P50 from projections DataFrame on (player_id, season).
       If no projection exists, season_p50 = NaN.

    7. OPPONENT JOIN
       From step 4 (weekly_vegas), each player's team maps to an opponent.
       Join DvP on (opponent, player_position) -> opp_dvp_fpa feature.

    8. PACK FEATURES
       Combine: rolling_*, season_p50, wk_implied_pts, wk_game_total,
       wk_spread, opp_dvp_fpa into JSONB features dict.
       Output: (player_id, season, week, position, features, snapshot_id).

    Players whose team has a bye in target_week are EXCLUDED (no game = no row)."""

def build_weekly_features_range(start_season, end_season, *,
                                 snapshot_id=None) -> int:
    """Build weekly features for all (season, week) in range.

    Source loading (done ONCE, then shared across all weeks):

    1. nflreadpy sources:
       load_seasons = range(start_season - 1, end_season + 1)
       - player_stats: load_player_stats(load_seasons)
       - snap_counts: load_snap_counts([s for s in load_seasons if s >= 2012])
       - schedules: load_schedules(load_seasons)
       - rosters: load_rosters(load_seasons)

    2. ID mapping:
       - pfr_to_gsis: from build_id_map() (same as season assembler,
         assemble.py:228-229: build_id_map() -> dict(zip(pfr_id, gsis_id)))

    3. DB reads:
       - weekly_labels: query all WeeklyLabel rows via SQLAlchemy
         select(WeeklyLabel). Build DataFrame with columns
         (player_id, season, week, position, team, fantasy_points).
       - projections: query all Projection rows. For each (player_id, season),
         keep the row with the latest model_version (max lexicographic).
         Build DataFrame with (player_id, season, position, p50).
         Rename p50 -> season_p50.

    4. Construct frames dict with all the above.

    Week discovery: for each season in [start_season, end_season], get distinct
    weeks from weekly_labels for that season. This auto-handles 16 vs 17 game
    seasons and excludes postseason (weekly_labels is REG-only).

    Loop: for each (season, week), call assemble_weekly_features(),
    upsert result to WeeklyFeature table, commit per-season.

    Ordering: weekly_labels MUST be populated first (via build_weekly_labels).
    Season-level projections SHOULD exist (via train_projection.py); if the
    Projection table is empty, log a warning and continue with season_p50=NaN."""
```

### Build script (`scripts/build_weekly_data.py`)

```python
"""CLI: build weekly labels + weekly features.

ORDERING: must run AFTER:
  1. scripts/seed_db.py (populates WeeklyStatsRaw + creates all tables via
     Base.metadata.create_all in init_db)
  2. scripts/train_projection.py (populates Projection table for season_p50)

This script is NOT part of run_pipeline.sh (it serves the weekly "Who Should I
Start?" feature, not the season-level draft pipeline). Standalone script.

Usage:
  python scripts/build_weekly_data.py --start 2017 --end 2025
  python scripts/build_weekly_data.py --start 2017 --end 2025 --snapshot-id <id>

Steps:
  1. resolve_snapshot(snapshot_id) — validate snapshot exists
  2. build_weekly_labels(snapshot_id) — score + upsert weekly_labels
  3. Check Projection table: if empty, warn "season_p50 will be NaN"
  4. build_weekly_features_range(start, end, snapshot_id) — assemble + upsert
  5. Print row counts for both tables

Table creation: handled by seed_db.py (Base.metadata.create_all is idempotent).
If tables don't exist, the script will fail with a clear SQLAlchemy error
pointing the user to run seed_db.py first.
"""
```

### Tests

**`tests/test_weekly_leakage.py`:**

```python
"""Poison test for the as-of-kickoff leakage rule.

Test 1 — prior_weeks guard:
  Create synthetic DataFrame: season=2024 weeks 1-4 + season=2023 weeks 1-17.
  Call prior_weeks(df, target_season=2024, target_week=3).
  Assert result contains: all of season 2023, plus season 2024 weeks 1-2 only.
  Assert season 2024 week 3+ excluded.

Test 2 — assert_no_future_week:
  Call assert_no_future_week on unfiltered data with target (2024, 3).
  Assert raises LeakageError.
  Call on properly filtered data. Assert no error.

Test 3 — build_rolling_player_stats poison:
  Create synthetic player_stats: player A has week 1-4 in season 2024.
  Inject SENTINEL=99999.0 into player A's week 3 rushing_yards.
  Filter to prior_weeks(2024, 3) + season_type=='REG'.
  Call build_rolling_player_stats().
  Assert SENTINEL absent from all rolling_* output columns.

Test 4 — build_opponent_dvp poison:
  Create synthetic weekly_labels: teams X, Y play each other weeks 1-4.
  Inject SENTINEL=99999.0 into week 3 label for a player on team X.
  Filter labels to prior_weeks(2024, 3).
  Call build_opponent_dvp() for target (2024, 3).
  Assert SENTINEL absent from dvp_fpa.

Test 5 — assemble_weekly_features end-to-end:
  Build full synthetic frames dict with SENTINEL in week 3 data.
  Call assemble_weekly_features(2024, 3, frames).
  Assert SENTINEL absent from ALL JSONB feature values in output.

All tests use synthetic DataFrames. No DB connection needed.
"""

# test functions:
# test_prior_weeks_filters_correctly()
# test_assert_no_future_week_raises()
# test_rolling_stats_no_leakage()
# test_dvp_no_leakage()
# test_assembler_end_to_end_no_leakage()
```

**`tests/test_weekly_labels.py`:**

```python
"""Weekly label scoring correctness.

test_weekly_scoring_matches_stat_line():
  Create synthetic weekly_raw with known stat lines (reuse test_scoring fixtures).
  Call compute_weekly_labels().
  Assert fantasy_points == score_stat_line() for each row.

test_postseason_excluded():
  Create rows with season_type='REG' and season_type='POST'.
  Call compute_weekly_labels().
  Assert only REG rows in output.

test_position_team_carried():
  Create rows with specific position and team values.
  Assert output DataFrame has matching position and team columns.

test_snapshot_id_set():
  Assert snapshot_id column is set on all output rows.
"""
```

## Data flow

1. `WeeklyStatsRaw` → `_load_weekly_raw()` (extended: position, team, season_type; ORM values set AFTER stats update for defensive ordering) → `compute_weekly_labels()` (REG filter, score) → `upsert_weekly_labels()` → `WeeklyLabel`

2. Source loading (once): player_stats, snap_counts (>= 2012 floor), schedules, rosters from nflreadpy; pfr_to_gsis from build_id_map(); weekly_labels + projections from DB.

3. For each (season, week) discovered from weekly_labels:
   - Filter player_stats + snap_counts to prior_weeks + REG only
   - assert_no_future_week() validates
   - build_rolling_player_stats() → rolling_* per player
   - build_opponent_dvp() (with schedules for opponent resolution) → dvp_fpa per (team, position)
   - build_weekly_vegas() → wk_implied_pts, wk_game_total, wk_spread, opponent per team
   - Player universe from teams-playing-this-week × (rosters for wk1, or most-recent label appearance for wk2+)
   - Season P50 joined from projections
   - Opponent joined from weekly_vegas; DvP joined on (opponent, position)
   - Features packed into JSONB → upsert_weekly_features() → WeeklyFeature

## ADP wall compliance

No ADP imports. Weekly features use only nflverse data (stats, schedules, snap counts) and the season-level P50 projection from `src/db/models.Projection`.

## Data leakage

`as_of_kickoff` rule. For (season Y, week W):
- Prior seasons (< Y): all weeks allowed
- Current season Y: only weeks < W (via `prior_weeks()`)
- Postseason games: excluded everywhere (REG filter on player_stats, snap_counts, weekly_labels)
- Vegas lines for week W: allowed (pre-kickoff artifact)
- Schedules for week W: allowed (opponent known pre-kickoff)

## Snapshot reproducibility

Both `WeeklyLabel` and `WeeklyFeature` carry `snapshot_id`.

## Config coupling

No `league.yaml` changes. Scoring uses existing `score_dataframe()`. EWMA halflife is a function parameter with defaults.

## Failure modes

- Empty `WeeklyStatsRaw` for a season/week → skip, log warning
- No Vegas lines for a game week → NaN for Vegas features
- No snap count data for a player → NaN for rolling_snap_share
- Season P50 missing (Projection table empty) → warn at startup, NaN for season_p50
- Player's team has bye → excluded (no game = no label = no feature row)
- Postseason rows in source data → filtered out (REG only)
- Key collision in WeeklyStatsRaw.stats JSONB → ORM values win (set after update)
- seed_db.py not run → tables don't exist → clear SQLAlchemy error

## Verification criteria

- `pytest tests/test_weekly_leakage.py` — all 5 tests pass (guard, assert, rolling, dvp, e2e assembler)
- `pytest tests/test_weekly_labels.py` — scoring, REG filter, position/team carried, snapshot_id
- `python scripts/build_weekly_data.py --start 2017 --end 2025` completes
- Row count sanity: weekly_labels ~40-50K (not a hard gate)
- Spot-check: known player/week label matches nflverse scored through scoring.py
- DvP for week 1 uses only prior-season data
- All existing tests (`pytest`) still pass

## Out-of-scope follow-ups

Weekly quantile model (Phase 2), ILP optimizer (Phase 3), CLI (Phase 4), Benchmark (Phase 5).
