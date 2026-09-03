import { useState } from "react";
import { POSITION_CHIP, POSITION_TEXT, POSITIONS } from "../positions";
import { TIER_HEADER } from "../rankings/palette";
import type { Ranked } from "../rankings/useBoardOrdering";
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
  ranking = null,
}: {
  players: BoardPlayer[];
  query: string;
  onPick: (playerId: string) => void;
  visible?: number;
  /** Your own board's order. Null (the default) means order by ADP. */
  ranking?: Map<string, Ranked> | null;
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

  const filtered = q
    ? byPosition.filter(
        (p) =>
          p.name.toLowerCase().includes(q) ||
          p.position.toLowerCase() === q ||
          (p.team ?? "").toLowerCase() === q,
      )
    : byPosition;

  // With a board selected, YOUR order wins. Players you never ranked sort
  // after everyone you did, among themselves by ADP — dropping them would
  // hide players who are genuinely available, which on a clock is worse than
  // showing them last.
  const matches = ranking
    ? [...filtered].sort((a, b) => {
        const ra = ranking.get(a.player_id)?.rank ?? Infinity;
        const rb = ranking.get(b.player_id)?.rank ?? Infinity;
        if (ra !== rb) return ra - rb;
        return (a.adp ?? Infinity) - (b.adp ?? Infinity);
      })
    : filtered;

  const toggle = (pos: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(pos)) next.delete(pos);
      else next.add(pos);
      return next;
    });
  // With a board selected the list is already in tier order, so a run of
  // equal labels IS a tier. Grouping the FILTERED list means a tier with no
  // visible player contributes no header — filter to backs and you see only
  // the tiers that contain one.
  const groups: Array<{ label: string; color: string; rows: BoardPlayer[] }> =
    [];
  if (ranking) {
    for (const player of matches) {
      const ranked = ranking.get(player.player_id);
      const label = ranked?.tierLabel ?? "Unranked";
      const color = ranked?.tierColor ?? "t8";
      const last = groups[groups.length - 1];
      if (last && last.label === label) last.rows.push(player);
      else groups.push({ label, color, rows: [player] });
    }
  }

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
          {ranking
            ? groups.map((group) => (
                <li key={group.label}>
                  {/* The tier owns everything below it until the next header,
                      which is how a tiered board is read on paper. */}
                  {/* NOT sticky. A pinned header sits ON the first row of
                      the tier below it, and a half-covered player name on a
                      pick clock is a misread. Tiers are small enough that the
                      header is nearly always on screen anyway. */}
                  <div
                    className={`flex items-baseline
                                justify-between border-b px-3 py-1
                                text-[11px] font-medium uppercase
                                tracking-wider ${
                                  TIER_HEADER[group.color] ?? ""
                                }`}
                  >
                    <span>{group.label}</span>
                    <span className="tnum opacity-70">
                      {group.rows.length}
                    </span>
                  </div>
                  <ul className="divide-y divide-line/60">
                    {group.rows.map((p) => (
                      <li key={p.player_id}>
                        <PlayerButton player={p} onPick={onPick} />
                      </li>
                    ))}
                  </ul>
                </li>
              ))
            : matches.map((p) => (
                <li key={p.player_id}>
                  <PlayerButton player={p} onPick={onPick} />
                </li>
              ))}
        </ul>
      )}

    </section>
  );
}


/** One pickable player. Identical in both orderings — the tier lives in the
 * header above, not in a column, so the row keeps showing ADP either way. */
function PlayerButton({
  player,
  onPick,
}: {
  player: BoardPlayer;
  onPick: (playerId: string) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onPick(player.player_id)}
      className="focusable flex w-full cursor-pointer items-baseline gap-3 px-3 py-1.5 text-left text-sm transition-colors duration-200 hover:bg-surface"
    >
      <span
        className={`tnum w-9 shrink-0 text-xs font-medium ${
          POSITION_TEXT[player.position] ?? "text-muted"
        }`}
      >
        {player.position}
      </span>
      <span className="flex-1 truncate">{player.name}</span>
      <span className="w-10 shrink-0 text-right text-xs text-muted">
        {player.team ?? "—"}
      </span>
      <span
        className="tnum w-12 shrink-0 text-right text-muted"
        title="consensus ADP across all platforms in the export"
      >
        {(() => {
          const shown = player.adp_consensus ?? player.adp;
          return shown === null || shown === undefined
            ? "—"
            : shown.toFixed(1);
        })()}
      </span>
    </button>
  );
}
