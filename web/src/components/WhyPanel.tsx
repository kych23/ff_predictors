import type { Narration } from "../types";

export function WhyPanel({
  narration,
  separatingAxis,
  staleFlags,
}: {
  narration: Narration | null;
  separatingAxis: string;
  staleFlags: string[];
}) {
  return (
    <aside className="w-72 shrink-0 space-y-4 border-l border-line p-4">
      <h2 className="text-xs uppercase tracking-wider text-muted">Why</h2>

      {narration ? (
        <p className="text-sm leading-relaxed">{narration.text}</p>
      ) : (
        /* Narration arrives as its own event so a slow local model can never
           sit on the pick clock. Absent is a normal state, not an error. */
        <p className="text-sm text-muted">
          waiting for the explanation — the recommendation above is already
          final
        </p>
      )}

      {narration && !narration.verified && (
        <p className="text-xs text-warn">
          unverified against the simulation record
        </p>
      )}

      {separatingAxis && (
        <p className="text-xs text-muted">
          separating axis:{" "}
          <span className="text-ink">{separatingAxis}</span>
        </p>
      )}

      {staleFlags.length > 0 && (
        <ul className="space-y-1 text-xs text-warn">
          {staleFlags.map((f) => (
            <li key={f}>⚠ {f}</li>
          ))}
        </ul>
      )}
    </aside>
  );
}
