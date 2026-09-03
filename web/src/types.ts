export interface Candidate {
  player_id: string;
  name: string;
  position: string;
  team: string | null;
  adp: number | null;
  e_dollars: number | null;
  aleatory_se: number | null;
  epistemic_se: number | null;
  total_se: number | null;
  draws: number | null;
  vona_score: number | null;
  in_indifference_set: boolean;
}

export interface Narration {
  text: string;
  source: string;
  verified: boolean;
  reason: string;
  model: string;
  latency_ms: number | null;
}

export interface Recommendation {
  status?: "idle" | "running" | "ready" | "error";
  snapshot_id: string;
  generation: number;
  tier: number;
  confidence: ConfidenceScore | null;
  leader: string | null;
  leader_name: string | null;
  elapsed_s: number | null;
  p_best: number | null;
  draws_used: number;
  stopped_because: string;
  separating_axis: string;
  stale_flags: string[];
  indifference_set: string[];
  engine: { reps: number; shortlist: number; budget_seconds: number };
  candidates: Candidate[];
  narration: Narration | null;
}

export interface RosterEntry {
  player_id: string;
  name: string;
  position: string;
}

/** One starting slot, filled or not. Empty slots are the useful part. */
export interface RosterSlot {
  slot: string;
  player_id: string | null;
  name: string | null;
  position: string | null;
}

export interface SessionState {
  /** This draft's identity — distinct from `snapshot_id`, which names the
   *  bundle and is shared by every draft run against it. */
  session_id: string;
  seat: number;
  teams: number;
  rounds: number;
  snapshot_id: string;
  pick_number: number;
  round: number;
  on_the_clock: number;
  is_my_turn: boolean;
  picks_until_my_turn: number | null;
  is_complete: boolean;
  generation: number;
  drafted_count: number;
  unresolved: string[];
  pick_clock_seconds: number;
  my_roster: RosterEntry[];
  roster_slots: RosterSlot[];
  picks: DraftPick[];
  team_names: string[];
  source: { name: string; state: string; detail: string };
}

export interface BoardPlayer {
  /** Consensus across every platform in the export — what the board column
   * shows. Distinct from `adp`, which is the single-platform ordering the
   * recommender models. */
  adp_consensus?: number | null;
  player_id: string;
  name: string;
  position: string;
  team: string | null;
  adp: number | null;
  bye_week: number | null;
}

export interface League {
  teams: number;
  rounds: number;
  snapshot_id: string;
  players: number;
  sources: string[];
  default_source: string;
  session_exists: boolean;
  active: boolean;
}

/** One archived draft, as offered by the Replay picker. */
export interface ReplayOption {
  id: string;
  /** When the draft STARTED — the question the picker answers is "which
   *  draft was this", and a long draft starts well before it is filed. */
  started_at: string | null;
  archived_at: string;
  picks: number;
  seat: number | null;
  snapshot_id: string | null;
  session_id: string | null;
  readable: boolean;
}

/** How much the engine backs this pick, 0-100, plus why it is not higher. */
export interface ConfidenceScore {
  score: number;
  label: "strong" | "moderate" | "slight" | "coin flip";
  drivers: string[];
}

export interface DraftPick {
  pick_number: number;
  round: number;
  seat: number;
  player_id: string | null;
  name: string;
  position: string | null;
  resolved: boolean;
  is_mine: boolean;
}

// ------------------------------------------------------------- My Board
// The board document mirrors `src/app/rankings/schema.py`. `Tier`, `Scope`
// and `Board` are re-exported from `rankings/model.ts`, which owns the
// mutations; these are the API-shaped types the client sends and receives.
export type { Board, Scope, ScopeName, Tier, TierColor } from "./rankings/model";

export interface BoardSummary {
  board_id: string;
  name: string;
  updated_at: string;
  rev: number;
  counts: Record<string, number>;
}

/** One row of `/api/rankings/catalogue` — what `PlayerRow` renders.
 *
 * A board stores only player ids, and `/api/board` cannot supply this: it
 * omits `value` and it filters out drafted players, so rows would vanish from
 * a research board mid-draft. */
export interface CataloguePlayer {
  player_id: string;
  name: string;
  position: string;
  team: string | null;
  bye_week: number | null;
  value: number | null;
  vor: number | null;
  adp: number | null;
}

export interface PlayerDetail {
  player_id: string;
  name: string;
  position: string;
  team: string | null;
  bye_week: number | null;
  projection: {
    value: number | null;
    vor: number | null;
    p10: number | null;
    p90: number | null;
    source: string;
    coverage: string;
  };
  market: {
    adp: number | null;
    adp_stdev: number | null;
    matched: boolean;
    consensus_rank: number | null;
    rank_spread: number | null;
    ranks: Record<string, number>;
  };
  // Null as a whole for every K and DST — they have no training-matrix row.
  production: Record<string, number | null> | null;
  role: Record<string, number | boolean | null> | null;
  capital: Record<string, number | boolean | null> | null;
  college: Record<string, number | boolean | null> | null;
  team_context: Record<string, number | null> | null;
  engine_position_tier: number | null;
  has_matrix_row: boolean;
}
