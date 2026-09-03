/**
 * Board mutations, as pure functions.
 *
 * Every one takes a board and returns a NEW board. Nothing here mutates its
 * input, touches the network, reads a clock, or holds module state — which is
 * what lets it be the one part of this feature with real unit tests.
 * `web/vite.config.ts` has no `test` block and no jsdom, so a component test is
 * not possible without new tooling; keeping the logic out of the components is
 * what makes that acceptable rather than a gap.
 *
 * The invariant every function preserves: **a player appears at most once per
 * scope.** The server refuses a save that breaks it, so a bug here would show
 * up as a 400 the user cannot act on.
 */

export type ScopeName =
  | "overall" | "QB" | "RB" | "WR" | "TE" | "K" | "DST";

/** Mirrors `TIER_COLORS` in `src/app/rankings/schema.py`; the server refuses
 * any other token. `tests/test_rankings_palette.py` pins the two together. */
export type TierColor =
  | "t1" | "t2" | "t3" | "t4" | "t5" | "t6" | "t7" | "t8";

export const SCOPES: ScopeName[] = [
  "overall", "QB", "RB", "WR", "TE", "K", "DST",
];

export interface Tier {
  id: string;
  label: string;
  color: string;
  player_ids: string[];
}

export interface Scope {
  tiers: Tier[];
}

export interface Board {
  schema_version: number;
  rev: number;
  board_id: string;
  name: string;
  created_at: string;
  updated_at: string;
  seeded_from: Record<string, unknown>;
  scopes: Record<ScopeName, Scope>;
  stale_player_ids?: string[];
}

const clone = (board: Board): Board => ({
  ...board,
  scopes: Object.fromEntries(
    Object.entries(board.scopes).map(([name, scope]) => [
      name,
      { tiers: scope.tiers.map((t) => ({ ...t, player_ids: [...t.player_ids] })) },
    ]),
  ) as Record<ScopeName, Scope>,
});

/** A label this module generated, as opposed to one the user typed. */
const AUTO_LABEL = /^Tier \d+$/;

/**
 * Restore `Tier 1..n` after the tier list changes shape.
 *
 * Deleting the third of five tiers leaves "Tier 1, Tier 2, Tier 4, Tier 5",
 * which reads as data loss even though nothing was lost.
 *
 * **Only auto-generated labels are touched.** A tier you renamed to "Elite" or
 * "Bench fodder" keeps that name wherever it moves — renumbering is a cleanup
 * of this module's own placeholder text, not a licence to overwrite something
 * you wrote. `id` is identity and never changes; only the display label does.
 */
function renumber(tiers: Tier[]): Tier[] {
  return tiers.map((tier, i) =>
    AUTO_LABEL.test(tier.label) ? { ...tier, label: `Tier ${i + 1}` } : tier,
  );
}

const withScope = (
  board: Board,
  scope: ScopeName,
  tiers: Tier[],
  { resequence = true }: { resequence?: boolean } = {},
): Board => {
  const next = clone(board);
  next.scopes[scope] = { tiers: resequence ? renumber(tiers) : tiers };
  return next;
};

/** Tier ids are `t-<n>`, and the next one is max+1 WITHIN the scope.
 *
 * Derived from the board rather than a counter or a clock, so `addTier` stays
 * pure and two identical boards produce identical ids. */
export function nextTierId(board: Board, scope: ScopeName): string {
  const numbers = board.scopes[scope].tiers.map((tier) => {
    const match = /^t-(\d+)$/.exec(tier.id);
    return match ? Number(match[1]) : 0;
  });
  return `t-${Math.max(0, ...numbers) + 1}`;
}

export function movePlayer(
  board: Board,
  scope: ScopeName,
  playerId: string,
  toTierId: string,
  toIndex: number,
): Board {
  // Remove first, everywhere in the scope, so a move within one tier cannot
  // leave a duplicate behind and a cross-tier move cannot double-insert.
  const tiers = board.scopes[scope].tiers.map((tier) => ({
    ...tier,
    player_ids: tier.player_ids.filter((id) => id !== playerId),
  }));

  const target = tiers.find((tier) => tier.id === toTierId);
  if (!target) return board;

  const index = Math.max(0, Math.min(toIndex, target.player_ids.length));
  target.player_ids.splice(index, 0, playerId);
  return withScope(board, scope, tiers);
}

export function moveTier(
  board: Board,
  scope: ScopeName,
  tierId: string,
  toIndex: number,
): Board {
  const tiers = board.scopes[scope].tiers.map((t) => ({ ...t }));
  const from = tiers.findIndex((tier) => tier.id === tierId);
  if (from < 0) return board;

  const [moved] = tiers.splice(from, 1);
  tiers.splice(Math.max(0, Math.min(toIndex, tiers.length)), 0, moved);
  return withScope(board, scope, tiers);
}

export function addTier(
  board: Board,
  scope: ScopeName,
  atIndex: number,
  color: string,
): Board {
  const tiers = board.scopes[scope].tiers.map((t) => ({ ...t }));
  const index = Math.max(0, Math.min(atIndex, tiers.length));
  tiers.splice(index, 0, {
    id: nextTierId(board, scope),
    // Auto-shaped so `renumber` in `withScope` gives it the number its
    // POSITION earns, not one derived from how many tiers happen to exist.
    label: `Tier ${index + 1}`,
    color,
    player_ids: [],
  });
  return withScope(board, scope, tiers);
}

export function renameTier(
  board: Board,
  scope: ScopeName,
  tierId: string,
  label: string,
): Board {
  // `resequence: false` — otherwise typing "Tier 7" into the third slot would
  // be rewritten to "Tier 3" the instant you pressed Enter, which looks like
  // the field rejecting your input.
  return withScope(
    board,
    scope,
    board.scopes[scope].tiers.map((tier) =>
      tier.id === tierId ? { ...tier, label } : { ...tier },
    ),
    { resequence: false },
  );
}

export function recolorTier(
  board: Board,
  scope: ScopeName,
  tierId: string,
  color: string,
): Board {
  return withScope(
    board,
    scope,
    board.scopes[scope].tiers.map((tier) =>
      tier.id === tierId ? { ...tier, color } : { ...tier },
    ),
  );
}

/**
 * Delete a tier, relocating its players rather than dropping them.
 *
 * They go to the tier BELOW, or above when this is the last one. Deleting the
 * only tier throws: there is nowhere to put the players, and silently taking
 * them off the board would destroy work the user cannot get back.
 */
export function deleteTier(
  board: Board,
  scope: ScopeName,
  tierId: string,
): Board {
  const tiers = board.scopes[scope].tiers.map((t) => ({
    ...t,
    player_ids: [...t.player_ids],
  }));
  const index = tiers.findIndex((tier) => tier.id === tierId);
  if (index < 0) return board;
  if (tiers.length === 1) {
    throw new Error("cannot delete the only tier; its players would be lost");
  }

  const [removed] = tiers.splice(index, 1);
  const receiver = tiers[index] ?? tiers[index - 1];
  receiver.player_ids = [...receiver.player_ids, ...removed.player_ids];
  // withScope renumbers, so the survivors close the gap the delete opened.
  return withScope(board, scope, tiers);
}

/** Flat scope order — what a save sends and what a rank column counts. */
export function flatten(board: Board, scope: ScopeName): string[] {
  return board.scopes[scope].tiers.flatMap((tier) => tier.player_ids);
}

/** 1-based rank of each player within a scope, for the leading column. */
export function ranks(board: Board, scope: ScopeName): Map<string, number> {
  const out = new Map<string, number>();
  flatten(board, scope).forEach((id, i) => out.set(id, i + 1));
  return out;
}
