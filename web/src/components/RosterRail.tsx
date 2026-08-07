import type { BoardPlayer, RosterEntry } from "../types";

export function RosterRail({
  roster,
  board,
  onPick,
}: {
  roster: RosterEntry[];
  board: BoardPlayer[];
  onPick: (playerId: string) => void;
}) {
  return (
    <aside className="w-64 shrink-0 space-y-6 border-r border-line p-4">
      <section>
        <h2 className="mb-2 text-xs uppercase tracking-wider text-muted">
          My roster
        </h2>
        {roster.length === 0 ? (
          <p className="text-sm text-muted">empty</p>
        ) : (
          <ul className="space-y-1 text-sm">
            {roster.map((p) => (
              <li key={p.player_id} className="flex justify-between gap-2">
                <span className="truncate">{p.name}</span>
                <span className="tnum text-muted">{p.position}</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section>
        <h2 className="mb-2 text-xs uppercase tracking-wider text-muted">
          Board
        </h2>
        <ul className="space-y-1 text-sm">
          {board.map((p) => (
            <li key={p.player_id}>
              <button
                type="button"
                onClick={() => onPick(p.player_id)}
                className="focusable flex w-full cursor-pointer justify-between gap-2 rounded px-1 py-0.5 text-left transition-colors duration-200 hover:bg-surface"
              >
                <span className="truncate">{p.name}</span>
                <span className="tnum text-muted">
                  {p.adp === null ? "—" : p.adp.toFixed(1)}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </section>
    </aside>
  );
}
