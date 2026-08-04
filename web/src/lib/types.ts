export interface Player {
  player_id: string;
  name: string | null;
  team: string | null;
  position: string;
  p10: number;
  p50: number;
  p90: number;
  adp: number | null;
  bye_week: number | null;
}

export interface Pick {
  pick_number: number;
  player_id: string | null;
  name: string | null;
  mine: boolean;
  skipped: boolean;
}

export interface RosterEntry {
  player_id: string;
  name: string | null;
  position: string | null;
  team: string | null;
  bye_week: number | null;
}

export interface DraftState {
  session_id: string;
  season: number;
  draft_position: number;
  platform: string;
  status: string;
  teams: number;
  rounds: number;
  my_picks: number[];
  current_overall_pick: number;
  is_my_turn: boolean;
  next_my_pick: number | null;
  remaining_picks: number;
  picks: Pick[];
  my_roster: RosterEntry[];
  open_starters: Record<string, number>;
}

export interface Recommendation {
  player_id: string;
  name: string | null;
  position: string;
  team: string | null;
  vona_score: number;
  value: number;
  p10: number;
  p50: number;
  p90: number;
  adp: number | null;
  draft_round: number;
  target_quantile: number;
  forced_completion: boolean;
}

export interface PickBody {
  player_id?: string;
  skip?: boolean;
  mine?: boolean;
}
