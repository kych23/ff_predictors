"""Build the draft-night bundle (DraftEngineDesign.md §12.6, §19, §22.1).

Day-4 deliverable: a WORKING TOOL at tier 2. Per the build order, value comes
from last season's actual points per game — not from ADP. That distinction is
the ADP wall: the market is what the engine is measured against, so it may
price availability (survival curves) but never supply value.

Phase 6 replaces `value_source: prior_season_ppg` with `quantile_model`; the
bundle schema does not change, only the column's provenance.

Usage:
    venv/bin/python scripts/build_bundle.py [--season 2026] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_league, load_strategy  # noqa: E402
from src.domain.scoring.engine import score_offense  # noqa: E402
from src.engine.decision.tiers import build_tier_table, write_tiers  # noqa: E402
from src.platform import bundle as bundle_mod  # noqa: E402
from src.platform.identity.match import match_by_name  # noqa: E402
from src.platform.sources import ffc, nflverse  # noqa: E402
from src.platform.store.manifest import build_snapshot, utc_now, write_manifest  # noqa: E402

DEFAULT_OUT = Path("data/bundles/draft_night_bundle.parquet")


def load_projections(season: int) -> tuple[pd.DataFrame, str] | tuple[None, None]:
    """Newest calibrated projection artifact for `season`, if one exists.

    Falls back to prior-season points when absent, so the tool never stops
    producing a recommendation because a model has not been fitted yet — the
    build order's hard rule (§22.1).
    """
    art = Path("data/artifacts")
    metas = sorted(art.glob("projections_*.meta.json"))
    for meta_path in reversed(metas):
        meta = json.loads(meta_path.read_text())
        if int(meta.get("target_season", -1)) != season:
            continue
        frame = pd.read_parquet(art / f"projections_{meta['snapshot_id']}.parquet")
        return frame, meta["snapshot_id"]
    return None, None


def prior_season_value(season: int, league, entries: list) -> pd.DataFrame:
    """Points per game from the PRIOR season, through our own scoring config.

    Never uses `fantasy_points_ppr`: that column encodes a different league's
    rules (measured: it scores interceptions at -2 where this league uses -1,
    and awards no offensive fumble-return TD).
    """
    prior = season - 1
    pull = nflverse.fetch("player_stats", seasons=[prior])
    entries.append(pull.entry)
    df = pull.frame
    df = df[(df["season_type"] == "REG")
            & df["position"].isin(league.roster.modeled_positions)].copy()
    df["points"] = score_offense(df, league.scoring.offense)

    agg = (df.groupby(["player_id", "player_display_name", "position"], as_index=False)
             .agg(points=("points", "sum"), games=("week", "nunique")))
    agg = agg[agg["games"] >= 1]
    agg["value"] = agg["points"] / agg["games"]
    return agg.rename(columns={"player_display_name": "nfl_name"})


def build(season: int, out: Path) -> bundle_mod.Bundle:
    league = load_league()
    strategy = load_strategy(league)
    entries: list = []

    projections, proj_snapshot = load_projections(season)
    if projections is not None:
        print(f"[1/4] calibrated projections ({proj_snapshot})...")
        nfl_names = prior_season_value(season, league, entries)[
            ["player_id", "nfl_name"]].drop_duplicates("player_id")
        values = projections.merge(nfl_names, on="player_id", how="left")
        # rookies have no prior-season row and therefore no name from that
        # source; fall back to the players table via the projection index
        values = values.rename(columns={"p50": "value"})
        values = values.dropna(subset=["nfl_name"])
        value_source = "quantile_model"
        print(f"      {len(values)} players with calibrated P10/P50/P90")
    else:
        print(f"[1/4] prior-season value ({season - 1}) through league scoring...")
        values = prior_season_value(season, league, entries)
        value_source = "prior_season_ppg"
        print(f"      {len(values)} players with >=1 game "
              f"(no projection artifact; run train_projection_v2.py)")

    print(f"[2/4] {strategy.adp.source.upper()} ADP for {season}...")
    if strategy.adp.source != "ffc":
        raise SystemExit(
            f"adp.source={strategy.adp.source!r} is not available; Yahoo is "
            f"blocked pending Fantasy Sports API permission (§11.5). "
            f"Set adp.source: ffc in config/strategy.yaml."
        )
    adp_pull = ffc.fetch_adp(fmt=strategy.adp.format, teams=strategy.adp.teams,
                             season=season)
    entries.append(adp_pull.entry)
    adp = adp_pull.frame
    print(f"      {len(adp)} players priced")

    print("[3/4] resolving ADP names against nflverse...")
    report = match_by_name(adp, values, left_name="player_name",
                           right_name="nfl_name")
    # Coverage is only meaningful over MODELED positions: K and DST have no
    # prior-season line by construction (they are not in modeled_positions), so
    # counting them as misses understates match quality and hides a real
    # regression behind a permanently-bad number.
    modeled = set(league.roster.modeled_positions)
    skill_missed = report.unresolved_left[
        report.unresolved_left["position"].isin(modeled)]
    skill_matched = report.matched[report.matched["position"].isin(modeled)]
    denom = len(skill_matched) + len(skill_missed)
    skill_cov = len(skill_matched) / denom if denom else 0.0
    print(f"      {report.describe()}")
    print(f"      modeled-position coverage: {skill_cov:.1%} "
          f"({len(skill_matched)}/{denom}) — misses are rookies and players "
          f"who did not appear in {season - 1}")
    if skill_cov < 0.85:
        print("      WARNING: modeled coverage below 85%; check name matching")

    # EVERY priced player enters the bundle, matched or not. Rookies, K and DST
    # have ADP but no prior-season line; dropping them would make them invisible
    # to the board, and a player the room can draft must be rankable even if
    # only at replacement (§19). `coverage` records why each row is what it is.
    def _rows(src: pd.DataFrame, matched: bool) -> pd.DataFrame:
        return pd.DataFrame({
            "player_id": (src.get("player_id_r") if matched and "player_id_r" in src
                          else src.get("player_id")).astype(str),
            "player_name": src["player_name"],
            "position": src["position"],
            "team": src.get("team"),
            "bye_week": pd.to_numeric(src.get("bye"), errors="coerce"),
            "value": (pd.to_numeric(src["value"], errors="coerce").fillna(0.0)
                      if matched else 0.0),
            "value_source": value_source if matched else "none",
            "coverage": "full" if matched else "no_prior_season",
            "adp": pd.to_numeric(src["adp"], errors="coerce"),
            "adp_stdev": pd.to_numeric(src["adp_stdev"], errors="coerce"),
        })

    frame = pd.concat(
        [_rows(report.matched, True), _rows(report.unresolved_left, False)],
        ignore_index=True,
    ).sort_values("adp").reset_index(drop=True)

    by_cov = frame.groupby(["coverage", "position"]).size()
    print("      bundle composition:")
    for (cov, pos), n in by_cov.items():
        print(f"        {cov:16s} {pos:4s} {n}")

    snapshot_id = build_snapshot(entries)
    write_manifest(snapshot_id, entries)
    print(f"[4/4] snapshot {snapshot_id}")

    b = bundle_mod.Bundle(
        players=frame, snapshot_id=snapshot_id,
        model_version=league.model_version,
        decision_version=strategy.decision_version,
        value_source=value_source, built_at=utc_now(),
    )
    bundle_mod.write(b, out)
    print(f"      wrote {out} ({b.n_players} players)")

    # Tier-3 artifact (§7.4). Written here because this is the only place that
    # holds the finished board, and tier 3 must not depend on anything that
    # could fail at draft time — a parquet file is the whole contract.
    tiers = build_tier_table(frame)
    tier_path = write_tiers(tiers, snapshot_id)
    counts = tiers.groupby("position")["tier"].nunique()
    print(f"      tier-3 fallback: {tier_path}")
    print("      tiers per position: "
          + ", ".join(f"{pos} {int(n)}" for pos, n in counts.items()))
    return b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    build(args.season, args.out)


if __name__ == "__main__":
    main()
