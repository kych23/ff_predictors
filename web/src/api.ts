import type {
  Board,
  BoardPlayer,
  BoardSummary,
  CataloguePlayer,
  PlayerDetail,
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

  // ------------------------------------------------------------- My Board
  boards: () => fetch("/api/rankings").then(json<{ boards: BoardSummary[] }>),
  createBoard: (name: string, seedMethod: string | null) =>
    fetch("/api/rankings", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name, seed_method: seedMethod }),
    }).then(json<Board>),
  // NOT `board` — that name is already the cockpit's `/api/board`.
  rankingBoard: (id: string) => fetch(`/api/rankings/${id}`).then(json<Board>),
  deleteBoard: (id: string) =>
    fetch(`/api/rankings/${id}`, { method: "DELETE" }).then(
      json<{ deleted: boolean }>,
    ),
  seedScope: (id: string, body: Record<string, unknown>) =>
    fetch(`/api/rankings/${id}/seed`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }).then(json<Board>),
  catalogue: () =>
    fetch("/api/rankings/catalogue").then(
      json<{ players: CataloguePlayer[]; target_season: number }>,
    ),
  playerDetail: (id: string) =>
    fetch(`/api/players/${id}/detail`).then(json<PlayerDetail>),
};

/**
 * Saving a board, with the conflict path the generic helper cannot express.
 *
 * `json()` above throws on any non-2xx and discards the body, so a 409 would
 * reach the caller as an opaque `Error`. But a rev conflict is the one failure
 * whose BODY is the whole point — it carries the board the client lost to, and
 * without it the view can only say "something went wrong" and strand the user.
 */
export class RevConflictError extends Error {
  constructor(readonly current: Board) {
    super("this board changed in another tab");
    this.name = "RevConflictError";
  }
}

export async function saveBoard(
  id: string,
  body: { expected_rev: number; scopes: unknown; name?: string },
  options: { flush?: boolean } = {},
): Promise<Board> {
  // `beforeunload` needs POST: sendBeacon cannot issue a PUT at all, and
  // `fetch(keepalive)` is only reliably queued by a closing tab for a method
  // the browser expects to queue. `/flush` is a POST alias for this PUT.
  const url = options.flush
    ? `/api/rankings/${id}/flush`
    : `/api/rankings/${id}`;
  const res = await fetch(url, {
    method: options.flush ? "POST" : "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    keepalive: options.flush,
  });

  if (res.status === 409) {
    const payload = await res.json();
    throw new RevConflictError(payload?.detail?.board as Board);
  }
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return (await res.json()) as Board;
}
