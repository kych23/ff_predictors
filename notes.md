Cursor Prompt:

```
I am creating a machine learning project that uses machine learning models to predict specific tasks for fantasy football. My current goal is to predict which NFL player to start in my lineup
between two players and calculate the projected points based on past data. I need to scrape data online, formalize datasets, train models, create a decision function, and display results. The point calculation will be as follows:

{
  "offense": {
    "passing_yards": {"points": 1, "per": 25},
    "passing_td": 4,
    "interception": -1,
    "rushing_yards": {"points": 1, "per": 10},
    "rushing_td": 6,
    "reception": 1,
    "receiving_yards": {"points": 1, "per": 10},
    "receiving_td": 6,
    "return_td": 6,
    "two_pt_conversion": 2,
    "fumble_lost": -2,
    "offensive_fumble_return_td": 6
  },
  "kickers": {
    "fg_0_19": 3,
    "fg_20_29": 3,
    "fg_30_39": 3,
    "fg_40_49": 4,
    "fg_50_plus": 5,
    "pat_made": 1
  },
  "defense_special_teams": {
    "sack": 1,
    "interception": 2,
    "fumble_recovery": 2,
    "td": 6,
    "safety": 2,
    "block_kick": 2,
    "kick_punt_return_td": 6,
    "points_allowed": {
      "0": 10,
      "1_6": 7,
      "7_13": 4,
      "14_20": 1,
      "21_27": 0,
      "28_34": -1,
      "35_plus": -4
    },
    "extra_point_returned": 2
  }
}

Generate the initial file scaffold for this project. Keep in my these features I plan on adding in the future:
- Will I Win my Matchup?
- Who Should I Trade?
- Who Should I Draft? (1 - 1 player comparison)
- Who Should I Start? (Full starting roster)
- Who Should I Draft? (Given current roster)
```

Pipeline:

1. Find the data

- What statistics / features do I need?
- Where am I getting this data?
- How to store the data? --> Parquet and DuckDB

2. Construct data into organized dataset

- DuckDB and Parquet
- current team and opposing team features should be converted to integer

3. Model Selection

- supervised learning approach
- for predicting exact projected fantasy points, use continuous models (regression - linear, lasso, ridge)
- Random Forest regression, Gradient Boosting regression

4. Validate with current season results

Who Should I Start? (1 - 1 player comparison)

Useful Citations:

https://github.com/FantasyFootballAnalytics/ffanalytics

Statistics I should collect:

Shared features:

- Weather
- Recent Total Fantasy Point Average
- Fantasy Points Standard Deviation
- Snap share
- Percentage of snaps in red zone
- Percentage of snaps in goal line situations
- Opponent Fantasy Points Allowed vs Position
- Team's point total

QB:

- QB Rushing Yards / Attempts
- Carries per game
- Yards per Carry
- Pass attempts per game
- Red zone carries + targets
- Goal-Line Rush Attempts
- Interception Rate / Turnover Rate

RB:

- Carries per game
- Yards per Carry
- Red zone carries + targets
- Goal-Line Rush Attempts
- Routes Run
- Yards per Target
- Catch Rate
- Average Depth of Target

WR:

- Targets per game
- Red zone carries + targets
- Routes Run
- Route participation rate
- Yards per Target
- Catch Rate
- Average Depth of Target
- Target share percentage (targets / team targets)

TE:

- Targets per game
- Red zone carries + targets
- Routes Run
- Route participation rate
- Yards per Target
- Catch Rate
- Average Depth of Target
- Target share percentage (targets / team targets)

K:

- Field Goal Attempt Volume
- Field Goal accuracy

DST:

- Defensive / Special Teams Turnovers + Sacks + Defensive TDs
- Defense's DVOA (Defense-adjusted Value Over Average)

Data Sources:

- nflreadpy
  - https://github.com/nflverse/nfldata/tree/master for schedules and game data

Columns in dataset (_ is excluded from features):
player_id_\*
player_name\*
position --> encoded
team\*
opp\*
season\*
week\*
fantasy_points
fp_avg_3
fp_std_3
snap_share
rz_targets
gl_targets
rz_carries
gl_carries
def_fp_allowed_last3
team_points
roof --> encoded
avg_temp
avg_wind
avg_air_yards
