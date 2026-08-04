import type { DraftState } from "@/lib/types";

export function PickLog({ state }: { state: DraftState }) {
  return (
    <ol className="text-sm space-y-1">
      {state.picks.map((p) => (
        <li key={p.pick_number} className={p.mine ? "font-semibold text-emerald-700" : ""}>
          {p.pick_number}. {p.skipped ? "— skipped —" : p.name ?? p.player_id}
          {p.mine && " (you)"}
        </li>
      ))}
    </ol>
  );
}
