/**
 * Position colours, defined ONCE.
 *
 * The draft grid and the board filters both key off position, and a colour
 * that means "running back" in one place and something else two components
 * over is worse than no colour at all — the whole point is that a wall of
 * green in round 2 is readable at a glance without reading a single name.
 */
export const POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"] as const;

export type Position = (typeof POSITIONS)[number];

/** Filled cell on the draft grid. */
export const POSITION_STYLE: Record<string, string> = {
  RB: "bg-green-700/40 border-green-500/40",
  WR: "bg-orange-600/40 border-orange-500/40",
  QB: "bg-blue-700/40 border-blue-500/40",
  TE: "bg-pink-600/40 border-pink-500/40",
  K: "bg-slate-700/40 border-slate-500/40",
  DST: "bg-slate-600/40 border-slate-500/40",
};

/** Compact chip, used by the board's position filters. */
export const POSITION_CHIP: Record<string, string> = {
  RB: "bg-green-600/25 text-green-200 border-green-500/50",
  WR: "bg-orange-600/25 text-orange-200 border-orange-500/50",
  QB: "bg-blue-600/25 text-blue-200 border-blue-500/50",
  TE: "bg-pink-600/25 text-pink-200 border-pink-500/50",
  K: "bg-slate-600/25 text-slate-200 border-slate-500/50",
  DST: "bg-slate-600/25 text-slate-200 border-slate-500/50",
};

/** Text-only tint for a position label in a list row. */
export const POSITION_TEXT: Record<string, string> = {
  RB: "text-green-300",
  WR: "text-orange-300",
  QB: "text-blue-300",
  TE: "text-pink-300",
  K: "text-slate-400",
  DST: "text-slate-400",
};
