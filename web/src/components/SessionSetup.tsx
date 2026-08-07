import { useState } from "react";
import type { League } from "../types";

/**
 * Start a draft.
 *
 * This screen exists because without it the app is unusable: the cockpit
 * cannot begin a draft at all, and the placeholder it replaced told the
 * operator to POST to the API by hand.
 *
 * Seats are **0-indexed everywhere in the API** and 1-indexed nowhere except
 * the label a human reads. That conversion happens here, once, and is the only
 * place in the frontend allowed to do it.
 */
export function SessionSetup({
  league,
  onStart,
  error,
}: {
  league: League;
  onStart: (seat: number, source: string, resume: boolean) => void;
  error: string | null;
}) {
  const [seat, setSeat] = useState(0);
  const [source, setSource] = useState(league.default_source);
  const busy = false;

  return (
    <main className="mx-auto grid min-h-screen max-w-lg place-items-center p-8">
      <div className="w-full space-y-6">
        <header>
          <h1 className="text-2xl font-semibold">Draft cockpit</h1>
          <p className="tnum mt-1 text-sm text-muted">
            {league.teams} teams × {league.rounds} rounds ·{" "}
            {league.players} players · {league.snapshot_id}
          </p>
        </header>

        {league.session_exists && (
          <div className="rounded border border-warn/40 bg-surface p-4">
            <p className="text-sm">
              A draft is already saved on this machine.
            </p>
            <button
              type="button"
              onClick={() => onStart(seat, source, true)}
              className="focusable mt-3 cursor-pointer rounded bg-accent px-4 py-2 text-sm font-medium text-bg transition-colors duration-200 hover:bg-accent/80"
            >
              Resume it
            </button>
            <p className="mt-2 text-xs text-muted">
              Starting a new one instead will refuse with a conflict until you
              archive the old draft.
            </p>
          </div>
        )}

        <div className="space-y-2">
          <label htmlFor="seat" className="block text-xs uppercase tracking-wider text-muted">
            Your seat
          </label>
          <select
            id="seat"
            value={seat}
            onChange={(e) => setSeat(Number(e.target.value))}
            className="focusable tnum w-full cursor-pointer rounded border border-line bg-surface px-3 py-2 text-sm"
          >
            {Array.from({ length: league.teams }, (_, i) => (
              <option key={i} value={i}>
                Seat {i + 1}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label htmlFor="source" className="block text-xs uppercase tracking-wider text-muted">
            Pick source
          </label>
          <select
            id="source"
            value={source}
            onChange={(e) => setSource(e.target.value)}
            className="focusable w-full cursor-pointer rounded border border-line bg-surface px-3 py-2 text-sm"
          >
            {league.sources.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          {source === "yahoo" && (
            <p className="text-xs text-warn">
              Yahoo has no credentials configured — it will report failed and
              you will type picks yourself. That still works.
            </p>
          )}
        </div>

        <button
          type="button"
          disabled={busy}
          onClick={() => onStart(seat, source, false)}
          className="focusable w-full cursor-pointer rounded bg-primary px-4 py-3 text-sm font-medium transition-colors duration-200 hover:bg-primary/80 disabled:opacity-50"
        >
          Start draft
        </button>

        {error && <p className="text-sm text-warn">{error}</p>}
      </div>
    </main>
  );
}
