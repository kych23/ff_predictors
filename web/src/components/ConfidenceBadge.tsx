import type { ConfidenceScore } from "../types";

/**
 * How much the engine backs this pick, 0-100.
 *
 * It answers one question — the chance this recommendation beats the next
 * candidate — and it is anchored on `p_best`, a real bootstrap over the paired
 * CRN difference, not a vibe.
 *
 * **Uncertainty pulls the number toward 50, never toward 0.** A pick the
 * engine cannot support is a coin flip, not a mistake, and the colour scale
 * says the same thing: neutral rather than alarming.
 *
 * The drivers are shown rather than hidden behind a tooltip. A bare score on a
 * screen where everything else is measured would read as authority it has not
 * earned; "50 — 20 picks earlier than the market" is a number you can argue
 * with, which is the point.
 */
const TONE: Record<ConfidenceScore["label"], string> = {
  strong: "border-green-500/50 bg-green-600/15 text-green-200",
  moderate: "border-sky-500/50 bg-sky-600/15 text-sky-200",
  slight: "border-amber-500/50 bg-amber-600/15 text-amber-200",
  "coin flip": "border-line bg-surface text-muted",
};

export function ConfidenceBadge({
  confidence,
}: {
  confidence: ConfidenceScore | null;
}) {
  if (!confidence) return null;
  return (
    <div className="mt-3">
      <div className="flex items-center gap-2">
        <span
          className={`tnum rounded border px-2 py-0.5 text-xs font-medium ${
            TONE[confidence.label] ?? TONE["coin flip"]
          }`}
          title="Chance this beats the next candidate"
        >
          {confidence.score} / 100
        </span>
        <span className="text-xs uppercase tracking-wider text-muted">
          {confidence.label}
        </span>
      </div>

      {confidence.drivers.length > 0 && (
        <ul className="mt-1.5 space-y-0.5">
          {confidence.drivers.map((d) => (
            <li key={d} className="text-[11px] leading-snug text-muted">
              — {d}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
