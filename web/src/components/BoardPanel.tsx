import { useState } from "react";
import { POSITION_CHIP, POSITION_TEXT, POSITIONS } from "../positions";
import type { BoardPlayer } from "../types";

/**
 * Available players, filtered live by whatever is in the pick input.
 *
 * Filtering happens CLIENT-SIDE over the whole available board. A server
 * round-trip per keystroke would put network latency between a typed letter
 * and the list — on a pick clock that is the difference between the list being
 * useful and being ignored.
 *
 * Clicking a row records that player directly, which is also the fastest way
 * out of an ambiguous name: pick the row instead of retyping.
 *
 * Position filters sit ALONGSIDE the text query rather than replacing it. On
 * the clock the question is usually "who is the best back left", which is a
 * filter, not a search — but typing a name has to keep working while a filter
 * is on, or the filter becomes a thing you must remember to clear.
 */
export function BoardPanel({
  players,
  query,
  onPick,
  visible = 10,
}: {
  players: BoardPlayer[];
  query: string;
  onPick: (playerId: string) => void;
  visible?: number;
}) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const q = query.trim().toLowerCase();

  // Positions present on the board, in canonical order — never a hardcoded
  // list, so a league without kickers shows no K chip.
  const available = POSITIONS.filter((pos) =>
    players.some((p) => p.position === pos),
  );

  const byPosition =
    selected.size === 0
      ? players
      : players.filter((p) => selected.has(p.position));

  const matches = q
    ? byPosition.filter(
        (p) =>
          p.name.toLowerCase().includes(q) ||
          p.position.toLowerCase() === q ||
          (p.team ?? "").toLowerCase() === q,
      )
    : byPosition;

  const toggle = (pos: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(pos)) next.delete(pos);
      else next.add(pos);
      return next;
    });
  // Scroll rather than truncate: cutting the list hides players that ARE
  // available, and on a clock the operator cannot tell a short list from
  // an exhausted one.
  const rowRem = 1.9;

  return (
    <section className="mt-8 max-w-2xl">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <h2 className="text-xs uppercase tracking-wider text-muted">Board</h2>
        <div className="flex flex-wrap gap-1" role="group" aria-label="Filter by position">
          {available.map((pos) => {
            const on = selected.has(pos);
            return (
              <button
                key={pos}
                type="button"
                aria-pressed={on}
                onClick={() => toggle(pos)}
                className={`focusable cursor-pointer rounded border px-2 py-0.5 text-[11px] font-medium transition-colors duration-200 ${
                  on
                    ? POSITION_CHIP[pos]
                    : "border-line text-muted hover:border-muted hover:text-fg"
                }`}
              >
                {pos}
              </button>
            );
          })}
          {selected.size > 0 && (
            <button
              type="button"
              onClick={() => setSelected(new Set())}
              className="focusable cursor-pointer rounded border border-line px-2 py-0.5 text-[11px] text-muted transition-colors duration-200 hover:border-muted hover:text-fg"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Column headers — `adp` is a bare float otherwise, and an unlabelled
          number next to a team abbreviation reads as anything. */}
      <div className="flex items-baseline gap-3 px-3 pb-1 text-[11px] uppercase tracking-wider text-muted">
        <span className="w-9 shrink-0">Pos</span>
        <span className="flex-1">Player</span>
        <span className="w-10 shrink-0 text-right">Team</span>
        <span className="w-12 shrink-0 text-right">ADP</span>
      </div>

      {matches.length === 0 ? (
        <p className="text-sm text-muted">
          {q
            ? `nothing on the board matches “${query.trim()}”`
            : "no available players at that position"}
        </p>
      ) : (
        <ul
          className="divide-y divide-line/60 overflow-y-auto rounded border border-line"
          style={{ maxHeight: `${visible * rowRem}rem` }}
        >
          {matches.map((p) => (
            <li key={p.player_id}>
              <button
                type="button"
                onClick={() => onPick(p.player_id)}
                className="focusable flex w-full cursor-pointer items-baseline gap-3 px-3 py-1.5 text-left text-sm transition-colors duration-200 hover:bg-surface"
              >
                <span
                  className={`tnum w-9 shrink-0 text-xs font-medium ${
                    POSITION_TEXT[p.position] ?? "text-muted"
                  }`}
                >
                  {p.position}
                </span>
                <span className="flex-1 truncate">{p.name}</span>
                <span className="w-10 shrink-0 text-right text-xs text-muted">
                  {p.team ?? "—"}
                </span>
                <span className="tnum w-12 shrink-0 text-right text-muted">
                  {p.adp === null ? "—" : p.adp.toFixed(1)}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}

    </section>
  );
}
