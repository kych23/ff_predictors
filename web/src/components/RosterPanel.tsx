import type { DraftState } from "@/lib/types";

export function RosterPanel({ state }: { state: DraftState }) {
  return (
    <div>
      <h3 className="font-semibold mb-2">My roster ({state.my_roster.length})</h3>
      <ul className="text-sm space-y-1">
        {state.my_roster.map((p, i) => (
          <li key={`${p.player_id}-${i}`} className="flex justify-between">
            <span>{p.name ?? p.player_id}</span>
            <span className="text-slate-500">
              {p.position ?? "—"}
              {p.bye_week ? ` · bye ${p.bye_week}` : ""}
            </span>
          </li>
        ))}
      </ul>
      <h4 className="font-medium mt-3 text-sm">Open starters</h4>
      <ul className="text-xs text-slate-600">
        {Object.entries(state.open_starters).map(([slot, n]) => (
          <li key={slot}>
            {slot}: {n}
          </li>
        ))}
      </ul>
    </div>
  );
}
