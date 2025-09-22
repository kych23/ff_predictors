-- Register Parquet datasets
CREATE OR REPLACE VIEW player_stats AS SELECT * FROM read_parquet('data/raw/player_stats.parquet');
CREATE OR REPLACE VIEW pbp          AS SELECT * FROM read_parquet('data/raw/pbp.parquet');
CREATE OR REPLACE VIEW snap_counts  AS SELECT * FROM read_parquet('data/raw/snap_counts.parquet');
CREATE OR REPLACE VIEW schedules    AS SELECT * FROM read_parquet('data/raw/schedules.parquet');
CREATE OR REPLACE VIEW players      AS SELECT * FROM read_parquet('data/raw/players.parquet');
CREATE OR REPLACE VIEW injuries     AS SELECT * FROM read_parquet('data/raw/injuries.parquet');

-- 1) Fantasy points per player-week (offense; extend for K/DST later)
CREATE OR REPLACE TABLE player_week AS
SELECT
  player_id,
  player_name,
  position,
  team,
  opponent_team         AS opp,
  season,
  week,
  passing_yards,
  passing_tds,
  passing_interceptions,
  rushing_yards,
  rushing_tds,
  receiving_fumbles_lost,
  sack_fumbles_lost,
  receptions,
  receiving_yards,
  receiving_tds,
  /* PPR scoring aligned with your config for offense */
  (passing_yards / 25.0)
  + (passing_tds * 4)
  + (passing_interceptions * -1)
  + (rushing_yards / 10.0)
  + (rushing_tds * 6)
  + ((COALESCE(receiving_fumbles_lost, 0) + COALESCE(sack_fumbles_lost, 0)) * -2)
  + (receptions * 1.0)
  + (receiving_yards / 10.0)
  + (receiving_tds * 6) AS fantasy_points
FROM player_stats;

-- 2) Recent form (rolling last-3 incl. current)
CREATE OR REPLACE TABLE player_week_form AS
SELECT
  player_id,
  season,
  week,
  position,
  team,
  opp,
  fantasy_points,
  AVG(fantasy_points) OVER (
    PARTITION BY player_id ORDER BY season, week
    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
  ) AS fp_avg_3,
  STDDEV_SAMP(fantasy_points) OVER (
    PARTITION BY player_id ORDER BY season, week
    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
  ) AS fp_std_3
FROM player_week;

-- 3) Snap share (usage)
CREATE OR REPLACE TABLE snaps_enriched AS
WITH snaps_base AS (
  SELECT
    s.pfr_player_id,
    s.season,
    s.week,
    s.team,
    s.offense_snaps,
    CAST(s.offense_snaps AS DOUBLE)
      / NULLIF(SUM(s.offense_snaps) OVER (PARTITION BY s.season, s.week, s.team), 0)
      AS snap_share
  FROM snap_counts s
)
SELECT
  p.player_id,
  b.season,
  b.week,
  b.team,
  b.offense_snaps,
  b.snap_share
FROM snaps_base b
LEFT JOIN players p
  ON p.pfr_player_id = b.pfr_player_id;

-- 4) Red-zone & goal-line features from PBP
CREATE OR REPLACE TABLE rz_targets AS
SELECT
  receiver_id AS player_id,
  season,
  week,
  COUNT(*) FILTER (WHERE yardline_100 <= 20) AS rz_targets,
  COUNT(*) FILTER (WHERE yardline_100 <= 5)  AS gl_targets,
  AVG(air_yards) FILTER (WHERE air_yards IS NOT NULL) AS avg_air_yards
FROM pbp
WHERE pass = 1 AND receiver_id IS NOT NULL
GROUP BY receiver_id, season, week;

CREATE OR REPLACE TABLE rz_rush AS
SELECT
  rusher_id AS player_id,
  season,
  week,
  COUNT(*) FILTER (WHERE yardline_100 <= 20) AS rz_carries,
  COUNT(*) FILTER (WHERE yardline_100 <= 5)  AS gl_carries
FROM pbp
WHERE rush = 1 AND rusher_id IS NOT NULL
GROUP BY rusher_id, season, week;

-- 5) Team points (realized)
CREATE OR REPLACE TABLE team_points AS
SELECT season, week, home_team AS team, home_score AS team_points FROM schedules
UNION ALL
SELECT season, week, away_team AS team, away_score AS team_points FROM schedules;

-- 6) Weather / stadium (aggregate by game → map to team)
CREATE OR REPLACE TABLE game_weather AS
SELECT
  season,
  week,
  home_team,
  away_team,
  any_value(roof) AS roof,
  AVG(temp)       AS avg_temp,
  AVG(wind)       AS avg_wind
FROM pbp
GROUP BY season, week, home_team, away_team;

CREATE OR REPLACE TABLE team_weather AS
SELECT season, week, home_team AS team, roof, avg_temp, avg_wind
FROM game_weather
UNION ALL
SELECT season, week, away_team AS team, roof, avg_temp, avg_wind
FROM game_weather;

-- 7) Opponent FP allowed vs position (leak-free last-3 prior weeks)
CREATE OR REPLACE TABLE def_fp_allowed AS
SELECT
  opp AS def_team,
  position,
  season,
  week,
  SUM(fantasy_points) AS fp_allowed
FROM player_week
GROUP BY def_team, position, season, week;

CREATE OR REPLACE TABLE def_vs_pos AS
SELECT
  def_team,
  position,
  season,
  week,
  AVG(fp_allowed) OVER (
    PARTITION BY def_team, position
    ORDER BY season, week
    ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING
  ) AS def_fp_allowed_last3
FROM def_fp_allowed;

-- 8) Final GOLD table
CREATE OR REPLACE TABLE player_week_features AS
WITH base AS (
  SELECT
    w.player_id, w.player_name, w.position, w.team, w.opp, w.season, w.week, w.fantasy_points
  FROM player_week w
)
SELECT
  b.*,
  f.fp_avg_3,
  f.fp_std_3,
  s.snap_share,
  rz.rz_targets,
  rz.gl_targets,
  rr.rz_carries,
  rr.gl_carries,
  d.def_fp_allowed_last3,
  tp.team_points,
  tw.roof,
  tw.avg_temp,
  tw.avg_wind,
  rz.avg_air_yards
FROM base b
LEFT JOIN player_week_form f
  ON f.player_id = b.player_id AND f.season = b.season AND f.week = b.week
LEFT JOIN snaps_enriched s
  ON s.player_id = b.player_id AND s.season = b.season AND s.week = b.week
LEFT JOIN rz_targets rz
  ON rz.player_id = b.player_id AND rz.season = b.season AND rz.week = b.week
LEFT JOIN rz_rush rr
  ON rr.player_id = b.player_id AND rr.season = b.season AND rr.week = b.week
LEFT JOIN def_vs_pos d
  ON d.def_team = b.opp AND d.position = b.position AND d.season = b.season AND d.week = b.week
LEFT JOIN team_points tp
  ON tp.team = b.team AND tp.season = b.season AND tp.week = b.week
LEFT JOIN team_weather tw
  ON tw.team = b.team AND tw.season = b.season AND tw.week = b.week;

-- 9) Persist GOLD
COPY (SELECT * FROM player_week_features)
TO 'data/processed/player_week_features.parquet' (FORMAT PARQUET);
