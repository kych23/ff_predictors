"""Did the engine's disagreements with the market actually pay?

Every "we beat ADP" claim this project has made so far is **self-graded**: the
engine simulates a season under its own model, prices it under its own payout,
and reports that its own picks won. That is a consistency check, not evidence.
A model that is confidently wrong scores just as well.

This grades against what happened. Train on seasons strictly before the test
year, project it, pull the ADP the market published *that* preseason, and
compare both rankings against realized points per game.

Two questions, and the second is the one that matters:

1. Does the engine rank players better than ADP does? (Spearman against
   realized fppg.) If not, the projection has no edge and every reach is a
   donation.

2. **Do its REACHES pay?** A player the engine likes far more than the market
   is exactly what it tells you to draft early. Split by disagreement and ask
   whether the ones it reached on outscored the ones the market preferred.
   This is where a bias toward past production would show: aging producers
   have high prior-season numbers, the market has moved on, and the model has
   not.

    venv/bin/python scripts/backtest_vs_adp.py --season 2024
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.names import normalize_name  # noqa: E402
from src.models.artifacts import newest  # noqa: E402
from src.models.projection.quantile_model import QuantileGBM  # noqa: E402
from src.platform.sources import ffc, nflverse  # noqa: E402

META = {"player_id", "season", "position", "fppg", "games", "is_rookie"}

#: A player needs a real market price to be part of a market comparison.
#: Beyond this the ADP pool is thin and the ranking is mostly noise.
MAX_ADP = 200.0


def _features(matrix: pd.DataFrame) -> list[str]:
    return [c for c in matrix.columns
            if c not in META and pd.api.types.is_numeric_dtype(matrix[c])]


def run(season: int) -> None:
    matrix = pd.read_parquet(newest("training_matrix_*.parquet"))
    cols = _features(matrix)

    train = matrix[matrix["season"] < season]
    test = matrix[matrix["season"] == season].copy()
    if test.empty or test["fppg"].isna().all():
        raise SystemExit(
            f"no realized labels for {season}; pick a completed season")
    print(f"train {len(train):,} rows (<{season})   test {len(test):,} rows")

    model = QuantileGBM().fit(train[cols], train["fppg"])
    test["projected"] = model.predict(test[cols])["p50"].to_numpy()

    # Names, so the market can be joined at all.
    players = nflverse.fetch("players").frame.rename(
        columns={"gsis_id": "player_id", "display_name": "name"})
    test = test.merge(players[["player_id", "name"]], on="player_id", how="left")
    test["key"] = test["name"].map(normalize_name)

    adp = ffc.fetch_adp(fmt="ppr", teams=12, season=season).frame
    adp["key"] = adp["player_name"].map(normalize_name)
    adp = adp.drop_duplicates("key")[["key", "adp"]]

    both = test.merge(adp, on="key", how="inner").dropna(
        subset=["adp", "fppg", "projected"])
    both = both[both["adp"] <= MAX_ADP]
    print(f"matched to {season} preseason ADP: {len(both)} players\n")

    # ---- 1. who ranks better against what actually happened ----------------
    engine = spearmanr(both["projected"], both["fppg"]).statistic
    market = spearmanr(-both["adp"], both["fppg"]).statistic
    print("RANKING SKILL vs realized points per game")
    print(f"  engine projection   spearman {engine:+.4f}")
    print(f"  market ADP          spearman {market:+.4f}")
    print(f"  edge                         {engine - market:+.4f}"
          f"   -> {'ENGINE' if engine > market else 'MARKET'} ranks better\n")

    # ---- 2. do the reaches pay ---------------------------------------------
    both["eng_rank"] = both["projected"].rank(ascending=False)
    both["mkt_rank"] = both["adp"].rank()
    both["disagree"] = both["mkt_rank"] - both["eng_rank"]

    edge = both["disagree"].quantile(0.85)
    reaches = both[both["disagree"] >= edge]
    fades = both[both["disagree"] <= both["disagree"].quantile(0.15)]

    print(f"REACHES — the engine's top 15% disagreements ({len(reaches)} players)")
    print(f"  their realized fppg      {reaches['fppg'].mean():6.2f}")
    print(f"  everyone else            {both.loc[~both.index.isin(reaches.index), 'fppg'].mean():6.2f}")
    print(f"  players it FADED         {fades['fppg'].mean():6.2f}")

    # The fair comparison: a reach costs you the player the market preferred at
    # that slot. Pair each reach with the market's choice at the same rank.
    paired = []
    market_order = both.sort_values("mkt_rank")
    for _, row in reaches.iterrows():
        alternative = market_order[market_order["mkt_rank"] >= row["eng_rank"]]
        if not alternative.empty:
            paired.append(row["fppg"] - alternative.iloc[0]["fppg"])
    if paired:
        delta = float(np.mean(paired))
        wins = float(np.mean([p > 0 for p in paired]))
        print("\n  HEAD TO HEAD, reach vs the market's pick at that slot:")
        print(f"    mean fppg difference   {delta:+.3f}")
        print(f"    reach won              {100 * wins:.0f}% of {len(paired)} pairs")
        print(f"    -> reaching {'PAID' if delta > 0 else 'COST'} "
              f"{abs(delta):.2f} points per game per pick")

    # ---- 3. is the bias about age and past production? ---------------------
    if "age_at_season_start" in both.columns:
        print("\nWHERE THE DISAGREEMENT COMES FROM")
        for column in ("age_at_season_start", "prior_fppg",
                       "seasons_of_history"):
            if column in both.columns:
                r = both[["disagree", column]].corr().iloc[0, 1]
                print(f"  corr(disagreement, {column:22s}) {r:+.3f}")
        old = both[both["age_at_season_start"] >= 29]
        if len(old) > 5:
            resid = (old["fppg"].mean()
                     - both["fppg"].mean())
            print(f"\n  players 29+: engine disagreement {old['disagree'].mean():+.1f} "
                  f"ranks, realized fppg {resid:+.2f} vs the pool average")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2024,
                    help="completed season to grade against")
    run(ap.parse_args().season)


if __name__ == "__main__":
    main()
