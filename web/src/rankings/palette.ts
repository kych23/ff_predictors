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

/**
 * SOLID shades, not opacity steps.
 *
 * The first version faded one colour by opacity — `bg-indigo-500/80` down to
 * `bg-indigo-900/20`. Two things broke. Over a `#0D1117` page the bottom of
 * that ramp is nearly black, so tiers 5-8 were indistinguishable from each
 * other and from the background. And the colour-picker swatch for a tier sat
 * ON that tier's own header in the same colour, so the control rendered as an
 * empty outline — you could not see what you were about to pick, or that there
 * was a button there at all.
 *
 * Solid shades give eight steps that are actually different, and colour drains
 * toward neutral as the tiers deepen, which is the right signal: tier 1 is a
 * claim, tier 8 is a shrug. Zinc rather than slate for the last two, because
 * `positions.ts` owns slate for K and DST.
 */
export const TIER_HEADER: Record<string, string> = {
  t1: "bg-indigo-400 border-indigo-300 text-indigo-950",
  t2: "bg-indigo-500 border-indigo-400 text-white",
  t3: "bg-indigo-600 border-indigo-500 text-white",
  t4: "bg-indigo-700 border-indigo-600 text-indigo-50",
  t5: "bg-indigo-800 border-indigo-700 text-indigo-100",
  t6: "bg-indigo-900 border-indigo-800 text-indigo-100",
  t7: "bg-zinc-700 border-zinc-600 text-zinc-100",
  t8: "bg-zinc-800 border-zinc-700 text-zinc-300",
};

/** The body a tier's rows sit on. Deliberately quieter than the header — the
 * player names are the content, the tier is the frame. */
export const TIER_BODY: Record<string, string> = {
  t1: "border-indigo-400/40",
  t2: "border-indigo-500/40",
  t3: "border-indigo-600/40",
  t4: "border-indigo-700/40",
  t5: "border-indigo-800/40",
  t6: "border-indigo-900/40",
  t7: "border-zinc-700/50",
  t8: "border-zinc-800/50",
};

/** Swatch for the colour picker — the same solid fill the header will take, so
 * the picker shows you the actual outcome. */
export const TIER_SWATCH: Record<string, string> = {
  t1: "bg-indigo-400",
  t2: "bg-indigo-500",
  t3: "bg-indigo-600",
  t4: "bg-indigo-700",
  t5: "bg-indigo-800",
  t6: "bg-indigo-900",
  t7: "bg-zinc-700",
  t8: "bg-zinc-800",
};

export const TIER_COLORS: TierColor[] = [
  "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8",
];

export const headerClass = (color: string): string =>
  TIER_HEADER[color] ?? TIER_HEADER.t1;

export const bodyClass = (color: string): string =>
  TIER_BODY[color] ?? TIER_BODY.t1;
