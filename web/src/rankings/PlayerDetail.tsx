/**
 * The player card.
 *
 * What this shows and does not show is the honest part. The engine projects
 * **points per game only** — one number plus a calibrated P10/P90 band. There
 * are no projected per-stat lines anywhere in this system, so a "PYDS / PTDS /
 * INT" grid would have to be invented. Instead: the projection band, what the
 * market thinks, and what the player actually did last season.
 *
 * Every K and DST has no training-matrix row at all — 49 of 260 players on the
 * live board — so the production, role and capital cards are absent rather
 * than showing a column of zeroes that would read as "he did nothing".
 */
import { useEffect, useState } from "react";
import { api } from "../api";
import { POSITION_CHIP } from "../positions";
import type { PlayerDetail as Detail } from "../types";

const LABELS: Record<string, string> = {
  prior_fppg: "Points / game",
  prior_targets_per_game: "Targets / game",
  prior_target_share: "Target share",
  prior_carries_per_game: "Carries / game",
  prior_touches_per_game: "Touches / game",
  prior_snap_share: "Snap share",
  prior_catch_rate: "Catch rate",
  prior_yards_per_target: "Yards / target",
  prior_yards_per_reception: "Yards / catch",
  prior_rec_td_rate: "Rec TD rate",
  prior_yards_per_carry: "Yards / carry",
  prior_rush_td_rate: "Rush TD rate",
  prior_rush_yards_per_game: "Rush yards / game",
  prior_pass_attempts_per_game: "Attempts / game",
  prior_completion_pct: "Completion %",
  prior_yards_per_attempt: "Yards / attempt",
  prior_pass_td_rate: "Pass TD rate",
  prior_int_rate: "INT rate",
  prior_sack_rate: "Sack rate",
  depth_chart_rank: "Depth chart",
  is_projected_starter: "Projected starter",
  same_position_competition: "Same-position bodies",
  vacated_targets: "Vacated targets",
  vacated_targets_share: "Vacated target share",
  vacated_carries: "Vacated carries",
  vacated_carries_share: "Vacated carry share",
  draft_round: "Draft round",
  draft_pick_overall: "Draft pick",
  is_undrafted: "Undrafted",
  draft_capital_score: "Draft capital",
  seasons_of_history: "Seasons of history",
  team_changed: "Changed teams",
  age_at_season_start: "Age",
  team_ctx_implied_pts: "Team implied points",
  team_ctx_game_total: "Week 1 total",
  team_ctx_spread: "Week 1 spread",
  college_rec_yds_z: "College rec yards (z)",
  college_rec_td_z: "College rec TD (z)",
  college_rush_yds_z: "College rush yards (z)",
  college_rush_td_z: "College rush TD (z)",
  college_scrimmage_yds_z: "College scrimmage (z)",
  college_conference_tier: "Conference tier",
};

const show = (value: number | boolean | null): string => {
  if (value === null) return "—";
  if (typeof value === "boolean") return value ? "yes" : "no";
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
};

function Stats({
  title,
  rows,
}: {
  title: string;
  rows: Record<string, number | boolean | null> | null;
}) {
  if (!rows) return null;
  const entries = Object.entries(rows).filter(([, v]) => v !== null);
  if (entries.length === 0) return null;
  return (
    <section className="mt-5">
      <h4 className="mb-2 font-sans text-xs uppercase tracking-wide text-muted">
        {title}
      </h4>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-1">
        {entries.map(([key, value]) => (
          <div key={key} className="flex justify-between gap-2 text-xs">
            <dt className="truncate text-muted">{LABELS[key] ?? key}</dt>
            <dd className="font-mono text-ink">{show(value)}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

export function PlayerDetail({ playerId }: { playerId: string | null }) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cache] = useState(() => new Map<string, Detail>());

  useEffect(() => {
    if (!playerId) {
      setDetail(null);
      return;
    }
    const cached = cache.get(playerId);
    if (cached) {
      setDetail(cached);
      return;
    }
    let live = true;
    void api
      .playerDetail(playerId)
      .then((d) => {
        cache.set(playerId, d);
        if (live) {
          setDetail(d);
          setError(null);
        }
      })
      .catch((e) => live && setError(String(e)));
    return () => {
      live = false;
    };
  }, [playerId, cache]);

  if (!playerId) {
    return (
      <aside className="p-6 text-sm text-muted">
        Click a player to see projection, market and prior production.
      </aside>
    );
  }
  if (error) return <aside className="p-6 text-sm text-warn">{error}</aside>;
  if (!detail) return <aside className="p-6 text-sm text-muted">Loading…</aside>;

  const { projection: p, market: m } = detail;

  return (
    <aside className="overflow-y-auto p-5">
      <header className="flex items-center gap-2">
        <span
          className={`rounded border px-1.5 py-0.5 font-mono text-[10px]
                      ${POSITION_CHIP[detail.position] ?? ""}`}
        >
          {detail.position}
        </span>
        <h3 className="font-sans text-lg text-ink">{detail.name}</h3>
      </header>
      <p className="mt-1 font-mono text-xs text-muted">
        {detail.team ?? "free agent"} · bye {detail.bye_week ?? "—"}
        {detail.engine_position_tier !== null &&
          ` · engine ${detail.position} tier ${detail.engine_position_tier}`}
      </p>

      <section className="mt-5 rounded border border-line p-3">
        <h4 className="mb-2 font-sans text-xs uppercase tracking-wide text-muted">
          Projection
        </h4>
        <p className="font-mono text-2xl text-ink">
          {p.value === null ? "—" : p.value.toFixed(1)}
          <span className="ml-1 text-xs text-muted">pts/g</span>
        </p>
        <p className="mt-1 font-mono text-xs text-muted">
          {p.p10 === null || p.p90 === null
            ? "no calibrated band (K and DST are fitted empirically)"
            : `P10 ${p.p10.toFixed(1)} — P90 ${p.p90.toFixed(1)}`}
        </p>
        {p.vor !== null && (
          <p className="mt-1 font-mono text-xs text-muted">
            {p.vor.toFixed(1)} over replacement
          </p>
        )}
      </section>

      <section className="mt-4 rounded border border-line p-3">
        <h4 className="mb-2 font-sans text-xs uppercase tracking-wide text-muted">
          Market
        </h4>
        <p className="font-mono text-sm text-ink">
          ADP {m.adp === null ? "—" : m.adp.toFixed(1)}
          {m.adp_stdev !== null && (
            <span className="text-muted"> ± {m.adp_stdev.toFixed(1)}</span>
          )}
        </p>
        {m.matched ? (
          <>
            <div className="mt-2 grid grid-cols-3 gap-x-3 gap-y-1">
              {Object.entries(m.ranks).map(([platform, rank]) => (
                <div key={platform} className="flex justify-between text-xs">
                  <span className="text-muted">{platform}</span>
                  <span className="font-mono text-ink">{rank}</span>
                </div>
              ))}
            </div>
            {m.rank_spread !== null && (
              <p className="mt-2 text-xs text-muted">
                platforms disagree by {m.rank_spread} ranks
              </p>
            )}
          </>
        ) : (
          <p className="mt-1 text-xs text-muted">
            no platform ranks matched this name
          </p>
        )}
      </section>

      {!detail.has_matrix_row && (
        <p className="mt-4 text-xs text-muted">
          No prior-production data — kickers and defences are projected from a
          fitted distribution, not from a per-player feature row.
        </p>
      )}

      <Stats title="Last season" rows={detail.production} />
      <Stats title="Role" rows={detail.role} />
      <Stats title="Draft capital" rows={detail.capital} />
      <Stats title="College" rows={detail.college} />
      <Stats title="Team context" rows={detail.team_context} />
    </aside>
  );
}
