// API client for FastAPI backend

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface Player {
  player_id: string;
  name: string;
  position: string;
  team_current: string | null;
}

export interface Prediction {
  player_id: string;
  name: string;
  position: string;
  team: string | null;
  season: number;
  week: number;
  opponent_team: string;
  y_pred: number;
  y_std: number | null;
  ci_lower: number | null;
  ci_upper: number | null;
  model_version: string;
}

export interface PredictRequest {
  player_id: string;
  season: number;
  week: number;
  opponent_team: string;
}

export interface PredictResponse {
  y_pred: number;
  y_std: number | null;
  model_version: string;
  ci_lower: number | null;
  ci_upper: number | null;
}

export interface CurrentWeek {
  season: number;
  week: number;
}

// API Client
export const api = {
  async getCurrentWeek(): Promise<CurrentWeek> {
    const response = await fetch(`${API_BASE_URL}/current`);
    if (!response.ok) throw new Error('Failed to fetch current week');
    return response.json();
  },

  async getPredictions(
    season: number,
    week: number,
    position?: string,
    limit: number = 50
  ): Promise<Prediction[]> {
    const params = new URLSearchParams({
      season: season.toString(),
      week: week.toString(),
      limit: limit.toString(),
    });
    if (position) params.append('position', position);
    
    const response = await fetch(`${API_BASE_URL}/predictions?${params}`);
    if (!response.ok) throw new Error('Failed to fetch predictions');
    return response.json();
  },

  async getPlayers(position?: string, team?: string): Promise<Player[]> {
    const params = new URLSearchParams();
    if (position) params.append('position', position);
    if (team) params.append('team', team);
    
    const url = params.toString() 
      ? `${API_BASE_URL}/players?${params}`
      : `${API_BASE_URL}/players`;
    
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch players');
    return response.json();
  },

  async getWeeks(season: number): Promise<number[]> {
    const response = await fetch(`${API_BASE_URL}/weeks?season=${season}`);
    if (!response.ok) throw new Error('Failed to fetch weeks');
    return response.json();
  },

  async getOpponents(season: number, week: number): Promise<string[]> {
    const response = await fetch(`${API_BASE_URL}/opponents?season=${season}&week=${week}`);
    if (!response.ok) throw new Error('Failed to fetch opponents');
    return response.json();
  },

  async predict(req: PredictRequest): Promise<PredictResponse> {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    if (!response.ok) throw new Error('Failed to get prediction');
    return response.json();
  },

  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      return response.ok;
    } catch {
      return false;
    }
  },
};

