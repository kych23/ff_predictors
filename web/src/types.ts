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
