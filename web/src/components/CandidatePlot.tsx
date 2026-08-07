import type { Candidate } from "../types";
import { makeScale } from "../scale";

/**
 * A dot-plot with error-bar whiskers, NOT a bar chart.
 *
 * A bar's length implies a precision this estimate does not have. The entire
 * point of splitting `aleatory_se` from `epistemic_se` is that two candidates
 * whose intervals overlap are *not* ranked, and a bar chart says the opposite
 * as loudly as it can.
 *
 * The leader is marked explicitly rather than by position. `candidates` is
 * sorted by point estimate while the leader accounts for uncertainty, so the
 * recommended player can legitimately sit second — observed live: a rival at
 * $44.81 on 2 draws above the pick at $44.64 on 50. Any UI that renders row 0
 * as "the pick" is wrong.
 *
 * Indifference-set membership is carried by a bracket and a label, never by
 * colour alone.
 */
export function CandidatePlot({
  candidates,
  leader,
}: {
  candidates: Candidate[];
  leader: string | null;
}) {
  const priced = candidates.filter((c) => c.e_dollars !== null);
  if (priced.length === 0) {
    return (
      <p className="text-muted text-sm">
        No dollar estimates on this tier — the ladder demoted before the
        simulator ran.
      </p>
    );
  }

  const bounds = priced.flatMap((c) => [
    (c.e_dollars ?? 0) - (c.total_se ?? 0),
    (c.e_dollars ?? 0) + (c.total_se ?? 0),
  ]);
  const scale = makeScale(bounds);
  const indifferent = priced.filter((c) => c.in_indifference_set).length;

  return (
    <div>
      <ul className="space-y-3" aria-hidden="true">
        {priced.map((c) => {
          const value = c.e_dollars ?? 0;
          const se = c.total_se ?? 0;
          const isLeader = c.player_id === leader;
          const left = scale.at(value - se) * 100;
          const right = scale.at(value + se) * 100;
          return (
            <li key={c.player_id} className="grid grid-cols-[10rem_1fr_5rem] items-center gap-3">
              <span
                className={`truncate text-sm ${isLeader ? "text-ink font-semibold" : "text-muted"}`}
              >
                {isLeader && <span className="text-accent mr-1">▸</span>}
                {c.name}
              </span>
              <span className="relative h-5">
                <span className="absolute inset-x-0 top-1/2 h-px bg-line" />
                <span
                  className={`absolute top-1/2 h-1 -translate-y-1/2 rounded ${
                    isLeader ? "bg-primary/60" : "bg-line"
                  }`}
                  style={{ left: `${left}%`, width: `${Math.max(right - left, 0.5)}%` }}
                />
                <span
                  className={`absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full ${
                    isLeader ? "bg-primary" : "bg-muted"
                  }`}
                  style={{ left: `${scale.at(value) * 100}%` }}
                />
              </span>
              <span className="tnum text-right text-sm">
                ${value.toFixed(2)}
                <span className="block text-[11px] text-muted">
                  {c.draws !== null ? `${c.draws} draws` : "—"}
                </span>
              </span>
            </li>
          );
        })}
      </ul>

      {indifferent > 1 && (
        <p className="mt-3 text-xs text-muted">
          ⟨ {indifferent} within noise — the engine does not claim to separate
          these ⟩
        </p>
      )}

      {/* The same numbers, reachable by screen reader. A chart is not an
          excuse to make data unavailable. */}
      <table className="sr-only">
        <caption>Candidates by expected winnings</caption>
        <thead>
          <tr>
            <th>Player</th><th>Expected dollars</th><th>Standard error</th>
            <th>Draws</th><th>Recommended</th>
          </tr>
        </thead>
        <tbody>
          {priced.map((c) => (
            <tr key={c.player_id}>
              <td>{c.name}</td>
              <td>{(c.e_dollars ?? 0).toFixed(2)}</td>
              <td>{(c.total_se ?? 0).toFixed(2)}</td>
              <td>{c.draws ?? "unknown"}</td>
              <td>{c.player_id === leader ? "yes" : "no"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
