import type {
  BoardPlayer,
  League,
  Recommendation,
  ReplayOption,
  SessionState,
} from "./types";

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
  createSession: (
    seat: number,
    source: string,
    resume: boolean,
    archiveId?: string | null,
  ) =>
    fetch("/api/session", {
      method: "POST",
      headers: { "content-type": "application/json" },
      // Only sent for replay; omitted, the server falls back to the fixture.
      body: JSON.stringify({
        seat,
        source,
        resume,
        archive_id: archiveId ?? null,
      }),
    }).then(json<SessionState>),
  replays: () =>
    fetch("/api/replays").then(
      json<{ replays: ReplayOption[]; fixture: string | null; count: number }>,
    ),
  board: (limit = 12) =>
    fetch(`/api/board?limit=${limit}`).then(json<{ players: BoardPlayer[] }>),
  pick: (body: Record<string, unknown>) =>
    fetch("/api/picks", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }).then(json<Record<string, unknown>>),
  undo: () => fetch("/api/undo", { method: "POST" }).then(json<SessionState>),
  clearSession: (purge: boolean) =>
    fetch(`/api/session?purge=${purge}`, { method: "DELETE" }).then(
      json<{ status: string; path: string | null }>,
    ),
  recommend: () =>
    fetch("/api/recommendation", { method: "POST" }).then(json<{ status: string }>),
  recommendation: () => fetch("/api/recommendation").then(json<Recommendation>),
};
