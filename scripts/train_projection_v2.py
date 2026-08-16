"""Train the pooled quantile projection model and project the target season.

    venv/bin/python scripts/train_projection_v2.py [--target 2026]

Replaces the bundle's `prior_season_ppg` placeholder with `quantile_model`
values: calibrated P10/P50/P90 of season points per game.

This script owns the orchestration the model layer may not do — pulling
sources and writing artifacts — so `models/` stays frame-in/frame-out and
reaches only `platform.asof` (§9.0).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_league  # noqa: E402
from src.domain.scoring.engine import score_offense  # noqa: E402
from src.models.artifacts import code_digest  # noqa: E402
from src.models.features.assemble import assemble_season  # noqa: E402
from src.models.features.role_change import (  # noqa: E402
    normalize_depth_charts,
)
from src.models.projection.calibrate import (  # noqa: E402
    IntervalCalibrator,
    assign_bucket,
    calibrate_oof_lofo,
    coverage_by_bucket,
)
from src.models.projection.folds import make_expanding_folds  # noqa: E402
from src.models.projection.quantile_model import QuantileGBM  # noqa: E402
from src.platform.sources import cfbd, nflverse  # noqa: E402
from src.platform.store.manifest import build_snapshot, write_manifest  # noqa: E402

ARTIFACTS = Path("data/artifacts")
META_COLS = {"player_id", "season", "position", "as_of_date", "name", "team"}
TARGET = "fppg"


def _pbp_aggregates(seasons: list[int], entries: list, cfg) -> dict:
    """Red zone, routes, team pace and defence, aggregated per season.

    Pulled once and reduced immediately. A decade of full play-by-play is ~370
    columns and half a million rows; nothing downstream needs a single play, so
    only the fifteen columns these features read are kept and the rest is
    dropped before it is ever concatenated.

    Failures degrade to empty rather than stopping the build. Play-by-play is a
    large pull and a board that cannot be built is worse than one without
    red-zone features — `prior_production` already emits NaN for them.

    **NOT WIRED INTO `load_frames`, on purpose.** These thirteen features were
    built, trained and then measured against a model without them, held out,
    trained on prior seasons only:

        season    n   spearman with   without    MAE with   without
          2023  377          0.7529    0.7532       2.681     2.679
          2024  375          0.7969    0.7971       2.568     2.558
          2025  383          0.8396    0.8347       2.234     2.283

    Mean Spearman delta +0.0015 — noise, and negative in two of three seasons.
    By position only TE moved (+0.0163); RB/WR/QB were flat. The likely reason
    is that the model already had the signal: `targets_per_game`,
    `carries_per_game`, `snap_share` and `expected_fp_per_game` are present for
    98%+ of the board and red-zone usage correlates strongly with total usage,
    so these largely re-describe what it could already see. Coverage does not
    help either — 57% for the red-zone rates, 66% for routes (`participation`
    is 2018+ only), 35% for red-zone carry share.

    Compare `_college_stats`, which moved rookie Spearman +0.121/+0.022/+0.025
    and won on both metrics in all three seasons. That is what a feature that
    works looks like.

    Kept rather than deleted because the pull and the reductions are the
    expensive part to rewrite, and there are two live reasons to come back:
    routes gain a season of coverage every year, and `team_environment` /
    `defence_allowed` belong in the WEEKLY draw rather than as season features,
    where averaging destroys exactly the week-to-week variation that makes a
    matchup worth knowing.
    """
    from src.models.features import pbp as pbp_mod

    player, team, defence = [], [], []
    for season in seasons:
        try:
            pull = nflverse.fetch("pbp", seasons=[season])
            entries.append(pull.entry)
            frame = pull.frame
        except Exception as exc:                      # noqa: BLE001
            print(f"      pbp {season}: unavailable ({type(exc).__name__})")
            continue
        keep = [c for c in pbp_mod.PBP_COLUMNS if c in frame.columns]
        frame = frame[keep]

        player.append(pbp_mod.red_zone_opportunity(frame))
        team.append(pbp_mod.team_environment(frame))
        defence.append(pbp_mod.defence_allowed(frame, cfg.scoring.offense))

        try:
            part = nflverse.fetch("participation", seasons=[season])
            entries.append(part.entry)
            routes = pbp_mod.routes_run(part.frame, frame)
            if not routes.empty:
                player.append(routes)
        except Exception:                             # noqa: BLE001
            pass                                       # 2018+ only

    def _stack(parts: list, keys: list[str]):
        usable = [p for p in parts if p is not None and not p.empty]
        if not usable:
            return pd.DataFrame()
        out = usable[0]
        for extra in usable[1:]:
            shared = set(out.columns) & set(extra.columns) - set(keys)
            if shared:                                 # same shape, new season
                out = pd.concat([out, extra], ignore_index=True)
            else:                                      # new columns, same key
                out = out.merge(extra, on=keys, how="outer")
        return out

    player_frame = _stack(player, ["player_id", "season"])
    team_frame = _stack(team, ["team", "season"])
    defence_frame = _stack(defence, ["team", "season"])
    print(f"      play-by-play: {len(player_frame):,} player-seasons, "
          f"{len(team_frame):,} team-seasons")
    return {"pbp_player": player_frame, "pbp_team": team_frame,
            "pbp_defence": defence_frame}


def _college_stats(seasons: list[int], entries: list) -> pd.DataFrame:
    """CFBD player-season production, or an empty frame if the key is absent.

    Degrading to empty rather than raising is deliberate: CFBD is the only
    source behind an API key, and a board that cannot be built is worse than
    one without college features. `build_college_features` emits nulls and the
    model treats them as missing.
    """
    pull = cfbd.season_stats(seasons)
    entries.append(pull.entry)
    if pull.frame.empty:
        print("      college stats: NONE (no CFBD_API_KEY, or the API is down)")
    else:
        print(f"      college stats: {len(pull.frame):,} player-seasons "
              f"over {pull.frame['season'].nunique()} seasons")
    return pull.frame


def _depth_charts(pull, seasons: list[int]) -> pd.DataFrame:
    """Depth charts across the legacy and current nflverse assets.

    **A single multi-season pull silently loses the recent years.** nflverse's
    legacy table stops at 2024; 2025 onward live in a differently-shaped feed
    (``dt``/``pos_abb``/``pos_rank``, no ``season``). Asking for 2011-2026 in
    one call returns 1.48M rows covering 2011-2024 and nothing raises — which
    is why `depth_chart_rank` read 93.6% populated in 2024 and 0.0% in 2025 and
    2026, taking the whole role block with it for exactly the players who have
    no prior production to fall back on.

    Fetched per era and normalized to the legacy column names, so everything
    downstream — including the as-of guard — sees one shape.
    """
    LEGACY_LAST = 2024
    frames = []
    legacy_years = [s for s in seasons if s <= LEGACY_LAST]
    modern_years = [s for s in seasons if s > LEGACY_LAST]
    if legacy_years:
        frames.append(pull("depth_charts", seasons=legacy_years))
    for season in modern_years:
        # One call per season: the feed is a per-season asset and a combined
        # request silently returns only what the legacy table holds.
        frames.append(normalize_depth_charts(
            pull("depth_charts", seasons=[season])))
    usable = [f for f in frames if f is not None and not f.empty]
    if not usable:
        return pd.DataFrame()
    combined = pd.concat(usable, ignore_index=True)
    print(f"      depth charts: {len(combined):,} rows over "
          f"{combined['season'].nunique()} seasons")
    return combined


def load_frames(seasons: list[int], target: int, entries: list,
                cfg) -> tuple[dict, dict]:
    """Pull every source assemble_season reads, recording provenance.

    Backward-looking sources stop at ``target - 1``: the target season has not
    been played, so nflverse has no stats file for it and requesting one 404s.
    Preseason artifacts (depth charts, rosters) DO include the target season —
    they are known before kickoff and are the only target-season rows any
    feature may read (§11.3).
    """
    played = [s for s in seasons if s < target]
    def pull(asset, **kw):
        p = nflverse.fetch(asset, blob_root=None, **kw)
        entries.append(p.entry)
        return p.frame

    players = pull("players").rename(columns={
        "gsis_id": "player_id", "display_name": "name",
        "college_name": "college", "rookie_season": "rookie_year"})
    id_map = pull("ff_playerids")
    pfr_to_gsis = {}
    if {"pfr_id", "gsis_id"} <= set(id_map.columns):
        pairs = id_map[["pfr_id", "gsis_id"]].dropna()
        pfr_to_gsis = dict(zip(pairs["pfr_id"].astype(str),
                               pairs["gsis_id"].astype(str), strict=False))

    frames = {
        "players": players,
        "id_map": id_map,
        "stats": pull("player_stats", seasons=played),
        "snaps": pull("snap_counts", seasons=[s for s in played if s >= 2012]),
        "opp": pull("ff_opportunity", seasons=[s for s in played if s >= 2006]),
        "depth": _depth_charts(pull, seasons),
        "rosters": pull("rosters", seasons=seasons),
        "schedules": pull("schedules"),
        # A decade back: a 2016 rookie's final college season is 2015, and
        # veterans on the board still carry the college row from their own
        # draft year. Three seasons covered only 24% of the board.
        "cfb_stats": _college_stats(
            list(range(target - 11, target)), entries),
        # `_pbp_aggregates(played, entries, cfg)` is DELIBERATELY not spread in
        # here — see its docstring. Re-wiring it is this one line plus the
        # `pbp_player=` argument in `assemble_season`.
    }
    return frames, pfr_to_gsis


def build_labels(stats: pd.DataFrame, cfg) -> pd.DataFrame:
    """Season points per game, scored through league config.

    Per-game rather than season totals so injury luck does not confound the
    ranking — the ground-truth relevance §10 specifies.
    """
    df = stats[(stats["season_type"] == "REG")
               & stats["position"].isin(cfg.roster.modeled_positions)].copy()
    df["points"] = score_offense(df, cfg.scoring.offense)
    agg = (df.groupby(["player_id", "season"], as_index=False)
             .agg(points=("points", "sum"), games=("week", "nunique")))
    agg = agg[agg["games"] >= cfg.training.min_games_train]
    agg[TARGET] = agg["points"] / agg["games"]
    return agg[["player_id", "season", TARGET, "games"]]


def flatten(feature_rows: pd.DataFrame) -> pd.DataFrame:
    """Explode the packed ``features`` dict column into wide columns.

    assemble_season packs features into a single dict column because v1 stored
    them as opaque JSONB — which is also why leakage cannot be a database
    constraint and has to be enforced at the data-access boundary (§8.11).
    """
    exploded = pd.json_normalize(feature_rows["features"]).reset_index(drop=True)
    base = feature_rows[["player_id", "season", "position", "is_rookie"]] \
        .reset_index(drop=True)
    return pd.concat([base, exploded], axis=1)


def matrix(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    merged = features.merge(labels, on=["player_id", "season"], how="inner")
    return merged.dropna(subset=[TARGET])


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in META_COLS | {TARGET, "games", "is_rookie"}
            and pd.api.types.is_numeric_dtype(df[c])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=2026)
    ap.add_argument("--start", type=int, default=2012)
    args = ap.parse_args()

    cfg = load_league()
    entries: list = []
    seasons = list(range(args.start - 1, args.target + 1))

    print(f"[1/6] pulling sources {seasons[0]}-{seasons[-1]}...")
    frames, pfr_to_gsis = load_frames(seasons, args.target, entries, cfg)
    print(f"      {len(frames['stats']):,} player-week rows")

    print("[2/6] assembling features per season...")
    feature_seasons = list(range(args.start, args.target + 1))
    blocks = []
    for season in feature_seasons:
        rows = assemble_season(season, frames, cfg=cfg,
                               as_of_date=dt.date(season, 9, 1),
                               pfr_to_gsis=pfr_to_gsis)
        blocks.append(rows)
        print(f"      {season}: {len(rows):,} rows", end="\r")
    features = flatten(pd.concat(blocks, ignore_index=True))
    print(f"      {len(features):,} feature rows over {len(feature_seasons)} seasons")

    print("[3/6] building labels...")
    labels = build_labels(frames["stats"], cfg)
    data = matrix(features, labels)
    cols = feature_columns(data)
    print(f"      {len(data):,} labeled rows, {len(cols)} numeric features")

    print("[4/6] expanding-fold training...")
    train_seasons = sorted(s for s in data["season"].unique() if s < args.target)
    folds = make_expanding_folds(train_seasons, cfg.training.min_train_seasons)
    print(f"      {len(folds)} evaluable folds "
          f"({folds[0].test_season}..{folds[-1].test_season})")

    oof = []
    for fold in folds:
        tr = data[data["season"].isin(fold.train_seasons)]
        te = data[data["season"] == fold.test_season]
        if te.empty:
            continue
        model = QuantileGBM().fit(tr[cols], tr[TARGET])
        pred = model.predict(te[cols])
        # the calibrator's contract names the realized label `y`
        block = te[["player_id", "season", "position", TARGET]].reset_index(drop=True)
        block = block.rename(columns={TARGET: "y"})
        for q in ("p10", "p50", "p90"):
            block[q] = pred[q].to_numpy()
        for extra in ("is_rookie", "seasons_of_history", "team_changed"):
            if extra in te.columns:
                block[extra] = te[extra].to_numpy()
        oof.append(block)
    oof = pd.concat(oof, ignore_index=True)
    # per-bucket conformal calibration needs the bucket label (§8.1): rookies
    # and role-changers are overconfident in different ways than veterans, so
    # one global scale would under-widen the classes that need it most.
    oof["bucket"] = oof.apply(assign_bucket, axis=1)
    print(f"      {len(oof):,} out-of-fold predictions")

    print("[5/6] conformal calibration...")
    calibrated = calibrate_oof_lofo(oof, cfg.training.min_bucket_n)
    before = coverage_by_bucket(oof)
    after = coverage_by_bucket(calibrated)
    print("      P10-P90 coverage vs 0.80 nominal:")
    merged_cov = before.merge(after, on="bucket", suffixes=("_raw", "_cal"))
    for _, row in merged_cov.iterrows():
        print(f"        {row['bucket']:18s} raw {row['coverage_raw']:.3f} "
              f"-> calibrated {row['coverage_cal']:.3f}  (n={int(row['n_raw'])})")

    print(f"[6/6] projecting {args.target}...")
    train = data[data["season"] < args.target]
    final = QuantileGBM().fit(train[cols], train[TARGET])
    target_rows = features[features["season"] == args.target].copy()
    if target_rows.empty:
        raise SystemExit(f"no feature rows for {args.target}")
    pred = final.predict(target_rows[cols])
    # RAW oof, not `calibrated`. The LOFO frame has already had its intervals
    # widened, so its nonconformity scores are ~1 by construction and fitting
    # on it returns a no-op calibrator — the shipped artifact would then carry
    # the model's own overconfident quantiles while this script printed a
    # coverage table implying otherwise. See test_projection_port.py::
    # test_fitting_on_calibrated_rows_collapses_the_scales.
    calibrator = IntervalCalibrator(cfg.training.min_bucket_n).fit(oof)
    print("      width scales (1.0 = no widening):")
    for bucket in sorted(calibrator.scales):
        ls, us = calibrator.scales[bucket]
        print(f"        {bucket:18s} lower x{ls:.3f}  upper x{us:.3f}")
    out = target_rows[["player_id", "season", "position"]].reset_index(drop=True)
    for q in ("p10", "p50", "p90"):
        out[q] = pred[q].to_numpy()
    for extra in ("is_rookie", "seasons_of_history", "team_changed"):
        if extra in target_rows.columns:
            out[extra] = target_rows[extra].to_numpy()
    out["bucket"] = out.apply(assign_bucket, axis=1)
    out = calibrator.transform(out, out["bucket"])

    # monotonicity is enforced by rearrangement, never by clipping
    assert (out["p10"] <= out["p50"]).all() and (out["p50"] <= out["p90"]).all(), \
        "quantile crossing survived rearrangement"

    snapshot_id = build_snapshot(entries)
    write_manifest(snapshot_id, entries)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / f"projections_{snapshot_id}.parquet"
    meta_path = ARTIFACTS / f"projections_{snapshot_id}.meta.json"
    digest = code_digest()

    # An artifact is identified by its data AND by the code that fitted it.
    # Same sources under changed feature code yields the same snapshot_id, so
    # without this the overwrite is silent and nothing downstream can tell the
    # two apart — including the parity golden, whose staleness check is keyed
    # on snapshot_id alone.
    if meta_path.exists():
        previous = json.loads(meta_path.read_text()).get("code_digest")
        if previous and previous != digest:
            print(f"      ** overwriting {snapshot_id} fitted by DIFFERENT "
                  f"code ({previous} -> {digest}). The projections may move "
                  f"under an unchanged snapshot id; regenerate the parity "
                  f"golden. **")

    out.to_parquet(path, index=False)

    # Persist the training matrix and the target-season features. The §21.4
    # reliability study refits this model on resampled data and must resample
    # THE SAME matrix — rebuilding it from sources would confound estimation
    # variance with any drift in the sources between runs, which is the one
    # thing that study is trying to isolate.
    matrix_path = ARTIFACTS / f"training_matrix_{snapshot_id}.parquet"
    keep = ["player_id", "season", "position", TARGET] + cols
    pd.concat(
        [data[keep], target_rows.reindex(columns=keep)], ignore_index=True,
    ).to_parquet(matrix_path, index=False)
    print(f"      training matrix -> {matrix_path}")
    meta_path.write_text(json.dumps({
        "snapshot_id": snapshot_id, "model_version": cfg.model_version,
        "code_digest": digest,
        "target_season": args.target, "n_players": len(out),
        "n_features": len(cols), "folds": len(folds),
    }, indent=2))
    print(f"      {len(out):,} projections -> {path}")
    print(f"      snapshot {snapshot_id}  code {digest}")
    print()
    top = out.nlargest(10, "p50").merge(
        frames["players"][["player_id", "name"]], on="player_id", how="left")
    print(top[["name", "position", "p10", "p50", "p90"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
