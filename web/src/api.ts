import type { BoardPlayer, League, Recommendation, SessionState } from "./types";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status}: ${detail}`);
  }
  return (await res.json()) as T;
}

export const api = {
  league: () => fetch("/api/league").then(json<League>),
  session: () => fetch("/api/session").then(json<SessionState>),
  createSession: (seat: number, source: string, resume: boolean) =>
    fetch("/api/session", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ seat, source, resume }),
    }).then(json<SessionState>),
  board: (limit = 12) =>
    fetch(`/api/board?limit=${limit}`).then(json<{ players: BoardPlayer[] }>),
  pick: (body: Record<string, unknown>) =>
    fetch("/api/picks", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }).then(json<Record<string, unknown>>),
  undo: () => fetch("/api/undo", { method: "POST" }).then(json<SessionState>),
  recommend: () =>
    fetch("/api/recommendation", { method: "POST" }).then(json<{ status: string }>),
  recommendation: () => fetch("/api/recommendation").then(json<Recommendation>),
};
