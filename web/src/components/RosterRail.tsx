import type { RosterSlot } from "../types";

/**
 * My roster as a SHAPE, not a list.
 *
 * Showing only drafted players hides the thing that actually drives a late
 * pick: which starting slots are still open. An empty TE in round 12 is the
 * most useful item on the screen, and a list of what you already have cannot
 * express it. Every slot renders; unfilled ones say so in italics.
 *
 * Bench slots render whether or not they are filled, for the same reason the
 * starters do: "two bench spots left" is a real constraint in the last rounds,
 * and overflow lands there — a third WR fills FLEX, the fourth benches.
 */
export function RosterRail({ slots }: { slots: RosterSlot[] }) {
  const starters = slots.filter((s) => s.slot !== "BENCH");
  const bench = slots.filter((s) => s.slot === "BENCH");

  return (
    <aside className="w-64 shrink-0 overflow-y-auto border-r border-line p-4">
      <h2 className="mb-3 text-xs uppercase tracking-wider text-muted">
        My roster
      </h2>

      <ul className="space-y-1.5 text-sm">
        {starters.map((s, i) => (
          <li key={`${s.slot}-${i}`} className="flex items-baseline gap-2">
            <span className="tnum w-11 shrink-0 text-xs text-muted">
              {s.slot}
            </span>
            {s.name ? (
              <span className="truncate">{s.name}</span>
            ) : (
              <span className="italic text-muted/60">empty</span>
            )}
          </li>
        ))}
      </ul>

      {bench.length > 0 && (
        <>
          <h3 className="mb-2 mt-5 text-xs uppercase tracking-wider text-muted">
            Bench
          </h3>
          <ul className="space-y-1.5 text-sm">
            {bench.map((s, i) => (
              <li key={`bench-${i}`} className="flex items-baseline gap-2">
                {/* Position for a filled seat, the slot label for an empty one
                    — an empty bench row has no position to show. */}
                <span className="tnum w-11 shrink-0 text-xs text-muted">
                  {s.position ?? "BN"}
                </span>
                {s.name ? (
                  <span className="truncate">{s.name}</span>
                ) : (
                  <span className="italic text-muted/60">empty</span>
                )}
              </li>
            ))}
          </ul>
        </>
      )}
    </aside>
  );
}
