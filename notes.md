```
I am creating a machine learning project that uses machine learning models to predict specific tasks for fantasy football. My current goal is to predict which NFL player to start in my lineup
between two players and calculate the projected points based on past data. I want to also calculate the 95% confidence interval. I need to scrape data online, formalize datasets, train models, create a decision function, and display results. The point calculation will be as follows:

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
- Trade Generator
```

**Pipeline:**

1. Find the data

- use nflreadpy API to extract data
- insert in postgresql database

2. Feature Engineering

- given exact data of each game, we must generate priors to train the model on

3.  Model Selection

- supervised learning approach
- for predicting exact projected fantasy points, use continuous models (regression - linear, lasso, ridge)
- Random Forest regression, Gradient Boosting regression

4. Validate with current season results

Who Should I Start? (1 - 1 player comparison)

**Data Sources:**

- nflreadpy
  - https://github.com/nflverse/nflreadpy?tab=readme-ov-file for extracting data

Columns in dataset ( is excluded from features):
player_id\*
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
