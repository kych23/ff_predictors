/**
 * One draggable player.
 *
 * The position chip reuses `POSITION_CHIP` from `positions.ts` rather than
 * inventing a second position palette — the whole point of that file is that a
 * colour means the same thing everywhere.
 *
 * Keyboard drag is dnd-kit's default sensor (space to lift, arrows to move,
 * space to drop) and works across tier boundaries, not just within one. That
 * is most of why the dependency is here: hand-rolled HTML5 drag events give
 * mouse-only reordering and no screen-reader announcements.
 */
import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { POSITION_CHIP } from "../positions";
import type { CataloguePlayer } from "../types";

const fmt = (value: number | null | undefined, digits = 1) =>
  value === null || value === undefined ? "—" : value.toFixed(digits);

export function PlayerRow({
  playerId,
  rank,
  player,
  stale,
  selected,
  draggable,
  onSelect,
}: {
  playerId: string;
  rank: number;
  player: CataloguePlayer | undefined;
  stale: boolean;
  selected: boolean;
  draggable: boolean;
  onSelect: (id: string) => void;
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
    useSortable({ id: playerId, disabled: !draggable });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.4 : 1,
  };

  // A player the current bundle no longer carries is greyed, never dropped:
  // silently deleting a row from saved research is the one thing this feature
  // must not do.
  if (stale || !player) {
    return (
      <div
        ref={setNodeRef}
        style={style}
        {...attributes}
        {...listeners}
        className="flex items-center gap-3 border-b border-line/50 px-3 py-2
                   text-muted/60"
      >
        <span className="w-8 font-mono text-xs">{rank}</span>
        <span className="flex-1 truncate text-sm line-through">{playerId}</span>
        <span className="text-xs">not on current board</span>
      </div>
    );
  }

  return (
    <div
      ref={setNodeRef}
      style={style}
      {...attributes}
      {...listeners}
      onClick={() => onSelect(playerId)}
      className={`flex cursor-pointer items-center gap-3 border-b
                  border-line/50 px-3 py-2 transition-colors
                  hover:bg-white/5 ${selected ? "bg-white/10" : ""}`}
    >
      <span className="w-8 font-mono text-xs text-muted">{rank}</span>
      <span
        className={`rounded border px-1.5 py-0.5 font-mono text-[10px]
                    ${POSITION_CHIP[player.position] ?? ""}`}
      >
        {player.position}
      </span>
      <span className="flex-1 truncate text-sm text-ink">{player.name}</span>
      <span className="w-10 font-mono text-xs text-muted">
        {player.team ?? "FA"}
      </span>
      <span className="w-10 font-mono text-xs text-muted" title="bye week">
        {player.bye_week ?? "—"}
      </span>
      <span className="w-12 text-right font-mono text-xs text-ink"
            title="projected points per game">
        {fmt(player.value)}
      </span>
      <span className="w-12 text-right font-mono text-xs text-muted"
            title="average draft position">
        {fmt(player.adp)}
      </span>
    </div>
  );
}
