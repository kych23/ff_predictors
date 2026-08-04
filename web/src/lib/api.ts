import type { DraftState, Player, Recommendation, PickBody } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(`API ${status}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* non-json error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  listPlayers: (season: number) => req<Player[]>(`/players?season=${season}`),
  createSession: (season: number, draftPosition: number) =>
    req<DraftState>(`/draft/sessions`, {
      method: "POST",
      body: JSON.stringify({ season, draft_position: draftPosition }),
    }),
  getState: (id: string) => req<DraftState>(`/draft/sessions/${id}`),
  recordPick: (id: string, body: PickBody) =>
    req<DraftState>(`/draft/sessions/${id}/picks`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  undo: (id: string) =>
    req<DraftState>(`/draft/sessions/${id}/undo`, { method: "POST" }),
  botPick: (id: string) =>
    req<DraftState>(`/draft/sessions/${id}/bot-pick`, { method: "POST" }),
  recommendations: (id: string, topN = 10) =>
    req<Recommendation[]>(`/draft/sessions/${id}/recommendations?top_n=${topN}`),
};
