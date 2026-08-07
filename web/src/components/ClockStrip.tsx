import { Wifi, WifiOff, AlertTriangle } from "lucide-react";
import type { SessionState } from "../types";

const CHIP: Record<string, { cls: string; Icon: typeof Wifi }> = {
  ok: { cls: "text-good", Icon: Wifi },
  degraded: { cls: "text-warn", Icon: AlertTriangle },
  failed: { cls: "text-red-400", Icon: WifiOff },
};

export function ClockStrip({ state }: { state: SessionState }) {
  const chip = CHIP[state.source.state] ?? CHIP.failed;
  const { Icon } = chip;
  return (
    <header className="flex items-center justify-between border-b border-line bg-surface px-4 py-3">
      <div className="tnum text-sm">
        <span className="text-muted">R{state.round}</span>
        <span className="mx-2 text-line">·</span>
        <span>PICK {state.pick_number} OF {state.teams * state.rounds}</span>
        <span className="mx-2 text-line">·</span>
        {state.is_my_turn ? (
          <span className="font-semibold text-accent">YOU ARE UP</span>
        ) : (
          <span className="text-muted">
            seat {state.on_the_clock + 1}
            {state.picks_until_my_turn !== null &&
              ` — ${state.picks_until_my_turn} until you`}
          </span>
        )}
      </div>

      <div className="flex items-center gap-4 text-xs">
        {state.unresolved.length > 0 && (
          /* Persistent, because a live feed records these automatically and an
             unnoticed one is only correctable by undoing real picks. */
          <span className="text-warn" title={state.unresolved.join(", ")}>
            {state.unresolved.length} unresolved
          </span>
        )}
        <span className={`flex items-center gap-1.5 ${chip.cls}`}
              title={state.source.detail || state.source.state}>
          <Icon size={14} aria-hidden="true" />
          {state.source.name}
        </span>
      </div>
    </header>
  );
}
