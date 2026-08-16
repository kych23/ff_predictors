/**
 * What produced this recommendation, in words.
 *
 * The engine numbers its ladder 0-3 and lower is better, which reads backwards
 * on a draft screen and looks like a player tier — the one thing it is not.
 * The operator does not need the rung number; they need to know whether a
 * simulation stands behind the dollar figure.
 */
export const TIER_LABEL: Record<number, string> = {
  0: "full sim",
  1: "partial sim",
  2: "heuristic",
  3: "ADP only",
};

/** Longer form, for the title attribute. */
export const TIER_DETAIL: Record<number, string> = {
  0: "Full simulation — 50 parameter draws x 2,048 seasons, priced in dollars",
  1: "Partial simulation — the parameter draws that finished before the clock",
  2: "VONA heuristic and survival curves. No simulation behind this number",
  3: "Static ADP list from the bundle. No projection, no simulation",
};

/** Tiers 2 and 3 have no simulation behind them and should read as a warning. */
export function tierIsFallback(tier: number): boolean {
  return tier >= 2;
}
