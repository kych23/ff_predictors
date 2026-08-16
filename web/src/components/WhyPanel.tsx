import type { Narration } from "../types";

export function WhyPanel({
  narration,
  separatingAxis,
  embedded = false,
}: {
  narration: Narration | null;
  separatingAxis: string;
  /** Rendered inside the right rail rather than being the rail itself, so the
   *  notes pad can share the column without a nested scroll region. */
  embedded?: boolean;
}) {
  const Tag = embedded ? "section" : "aside";
  return (
    <Tag
      className={
        embedded
          ? "space-y-4 p-4"
          : "w-72 shrink-0 space-y-4 overflow-y-auto border-l border-line p-4"
      }
    >
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

      {/* `stale_flags` is deliberately NOT shown here. It still travels in the
          payload and into the hash-chained ledger, where the provenance of a
          recommendation is worth having in November — it is just not what the
          operator needs in the two seconds before a pick. */}
    </Tag>
  );
}
