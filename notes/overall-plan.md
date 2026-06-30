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
- Who should I start? (1 to 1 comparison)
- Will I Win my Matchup? (given startings rosters of both teams)
- Who Should I Trade? (1 to 1 comparison)
- Who Should I Draft? (1 - 1 player comparison)
- Who Should I Start? (Gives you the full roster you should start given all of your players)
- Who Should I Draft? (Given your current drafted roster, and who's top N players available)
- Trade Generator (finds best teams and trade targets given your current needs and opposing team's needs)
```
