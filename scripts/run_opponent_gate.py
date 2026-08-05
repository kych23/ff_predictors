"""Run the pre-registered opponent-model gate (§14.2, AD-9).

    venv/bin/python scripts/run_opponent_gate.py [--permutations 200]

Decides whether per-manager draft policies are worth building. A null is a
deliverable: on failure the system ships league-mean + slot covariate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_league, load_strategy  # noqa: E402
from src.models.opponents.diagnostics import run_gate  # noqa: E402
from src.models.opponents.fit import CHOICE_SET_SIZE, Choice  # noqa: E402
from src.platform.identity.match import normalize_name  # noqa: E402
from src.platform.sources import ffc, league_history  # noqa: E402

#: FFC serves no 2025 ADP (status="Error"), so 2025 picks cannot be fitted:
#: the conditional logit needs the choice set the manager actually faced.
ADP_SEASONS = (2021, 2022, 2023, 2024)


def build_choices(picks: pd.DataFrame, adp_by_season: dict[int, pd.DataFrame],
                  cfg, exclude: set[str]) -> tuple[list[Choice], tuple[str, ...]]:
    positions = tuple(cfg.roster.modeled_positions)
    pos_index = {p: i for i, p in enumerate(positions)}
    choices: list[Choice] = []
    skipped = {"no_adp_match": 0, "off_position": 0, "not_discriminating": 0}

    rounds = cfg.roster.rounds
    for season, season_picks in picks.groupby("season"):
        if season not in adp_by_season:
            continue
        adp = adp_by_season[season].copy()
        adp["key"] = adp["player_name"].map(normalize_name) + "|" + adp["position"]
        adp = adp.dropna(subset=["adp"]).drop_duplicates("key")
        pool = adp.set_index("key")

        taken: set[str] = set()
        ordered = season_picks.sort_values("overall")
        for _, pick in ordered.iterrows():
            key = normalize_name(pick["player_name"]) + "|" + pick["position"]
            if pick["position"] not in pos_index:
                taken.add(key)
                skipped["off_position"] += 1
                continue
            # "Discriminating" picks only: rounds 1-2 are near-best-available
            # for everyone and the last two are forced K/DST, so neither
            # carries manager signal (§14.2).
            if not (3 <= pick["round"] <= rounds - 2):
                taken.add(key)
                skipped["not_discriminating"] += 1
                continue
            if key not in pool.index or pick["manager"] in exclude:
                taken.add(key)
                skipped["no_adp_match"] += 1
                continue

            available = pool[~pool.index.isin(taken)]
            available = available[available["position"].isin(positions)]
            ranked = available.nsmallest(CHOICE_SET_SIZE, "adp")
            if key not in ranked.index:      # always include the actual pick
                ranked = pd.concat([ranked, pool.loc[[key]]])
            if len(ranked) < 5:
                taken.add(key)
                continue

            slot_z = (pick["slot"] - (cfg.teams + 1) / 2) / ((cfg.teams - 1) / 2)
            choices.append(Choice(
                manager=pick["manager"], season=int(season),
                overall=int(pick["overall"]), slot_z=float(slot_z),
                adp=ranked["adp"].to_numpy(dtype="float64"),
                position_idx=np.array([pos_index[p] for p in ranked["position"]]),
                chosen=int(list(ranked.index).index(key)),
            ))
            taken.add(key)

    return choices, positions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--history", type=Path, default=Path("notes/league-history.md"))
    args = ap.parse_args()

    cfg = load_league()
    strategy = load_strategy(cfg)

    print("[1/4] loading league history...")
    load = league_history.load_history(args.history, teams=cfg.teams)
    if load.snake_errors:
        print(f"      {len(load.snake_errors)} snake mismatches:")
        for err in load.snake_errors[:5]:
            print(f"        {err}")
    else:
        print(f"      {len(load.picks)} picks, all consistent with their slots")

    roster_2026 = {m.display_name for m in strategy.opponents.managers}
    everyone = set(load.picks["manager"].unique())
    excluded = everyone - roster_2026
    print(f"      2026 cohort: {len(roster_2026)} opponents")
    print(f"      excluded   : {sorted(excluded)} (departed, or my own seat)")

    print(f"[2/4] pulling contemporaneous ADP {ADP_SEASONS}...")
    adp_by_season = {}
    for season in ADP_SEASONS:
        pull = ffc.fetch_adp(fmt=strategy.adp.format, teams=strategy.adp.teams,
                             season=season)
        adp_by_season[season] = pull.frame
        print(f"      {season}: {pull.n_players} players")
    print("      2025: SKIPPED — FFC returns no data, so those picks are "
          "unfittable (§11.5.1)")

    print("[3/4] assembling choice sets...")
    choices, positions = build_choices(load.picks, adp_by_season, cfg, excluded)
    per_manager = pd.Series([c.manager for c in choices]).value_counts()
    print(f"      {len(choices)} discriminating choices over "
          f"{per_manager.size} managers")
    print(f"      per manager: min {per_manager.min()}, "
          f"median {int(per_manager.median())}, max {per_manager.max()}")
    missing = sorted(roster_2026 - set(per_manager.index))
    if missing:
        print(f"      NO FITTABLE PICKS: {missing}")

    print(f"[4/4] running gate with {args.permutations} permutations...")
    gate = run_gate(
        choices, positions, n_permutations=args.permutations,
        permutation_threshold=strategy.opponents.gate.require_permutation_p,
        max_statistic_threshold=strategy.opponents.gate.require_max_statistic_p,
        prior_tau_sd=float(strategy.opponents.shrinkage["prior_tau_sd"]),
        prior_beta_sd=float(strategy.opponents.shrinkage["prior_beta_sd"]),
    )
    print()
    print(gate.describe())
    print()
    if gate.passed:
        print("  -> per-manager policies are justified; fit and ship them.")
    else:
        print("  -> NULL RESULT. Ship league-mean + slot covariate "
              f"({strategy.opponents.gate.on_fail}).")
        print("     This is a deliverable, not a failure: it retires the")
        print("     highest-cost, lowest-data-support layer in the design")
        print("     before it is built, and the null is reportable with its")
        print("     own power statement.")


if __name__ == "__main__":
    main()
