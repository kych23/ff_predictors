/**
 * Tier colours: an intensity ramp on ONE hue, not eight different hues.
 *
 * Two constraints shaped this.
 *
 * **It must not collide with position colour.** `positions.ts` already spends
 * green on RB, orange on WR, blue on QB, pink on TE and slate on K/DST, and a
 * `PlayerRow` renders a position chip *inside* a tier. That file's own comment
 * is the rule: "a colour that means 'running back' in one place and something
 * else two components over is worse than no colour at all." Indigo is free.
 *
 * **The classes must be literal strings.** Tailwind's JIT scans
 * `./src/**\/*.{ts,tsx}` for whole class names and purges anything built at
 * runtime, so `bg-${color}-600` compiles to nothing and ships colourless
 * tiers — a bug that looks like a CSS problem and is a build-tool one.
 * `positions.ts` spells its classes out for the same reason.
 *
 * A ramp rather than a rainbow because tiers are ORDERED. Eight arbitrary hues
 * would say "these are eight categories"; fading intensity says "these are
 * ranked", which is what a tier is.
 */
import { TierColor } from "./model";

/** Header bar for a tier. */
export const TIER_HEADER: Record<string, string> = {
  t1: "bg-indigo-500/80 border-indigo-300/60 text-white",
  t2: "bg-indigo-500/65 border-indigo-300/50 text-white",
  t3: "bg-indigo-500/50 border-indigo-400/45 text-indigo-50",
  t4: "bg-indigo-600/40 border-indigo-400/40 text-indigo-50",
  t5: "bg-indigo-600/30 border-indigo-400/35 text-indigo-100",
  t6: "bg-indigo-700/25 border-indigo-500/30 text-indigo-100",
  t7: "bg-indigo-800/20 border-indigo-600/25 text-indigo-200",
  t8: "bg-indigo-900/20 border-indigo-700/25 text-indigo-200",
};

/** The body a tier's rows sit on. Deliberately quieter than the header — the
 * player names are the content, the tier is the frame. */
export const TIER_BODY: Record<string, string> = {
  t1: "border-indigo-400/30",
  t2: "border-indigo-400/25",
  t3: "border-indigo-400/20",
  t4: "border-indigo-500/20",
  t5: "border-indigo-500/15",
  t6: "border-indigo-600/15",
  t7: "border-indigo-700/15",
  t8: "border-indigo-800/15",
};

/** Swatch for the colour picker. */
export const TIER_SWATCH: Record<string, string> = {
  t1: "bg-indigo-500/80",
  t2: "bg-indigo-500/65",
  t3: "bg-indigo-500/50",
  t4: "bg-indigo-600/40",
  t5: "bg-indigo-600/30",
  t6: "bg-indigo-700/25",
  t7: "bg-indigo-800/20",
  t8: "bg-indigo-900/20",
};

export const TIER_COLORS: TierColor[] = [
  "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8",
];

export const headerClass = (color: string): string =>
  TIER_HEADER[color] ?? TIER_HEADER.t1;

export const bodyClass = (color: string): string =>
  TIER_BODY[color] ?? TIER_BODY.t1;
