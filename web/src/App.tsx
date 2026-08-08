import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { BoardPanel } from "./components/BoardPanel";
import { CandidatePlot } from "./components/CandidatePlot";
import { DraftBoard } from "./components/DraftBoard";
import { PickTimer } from "./components/PickTimer";
import { ClockStrip } from "./components/ClockStrip";
import { RosterRail } from "./components/RosterRail";
import { SessionSetup } from "./components/SessionSetup";
import { WhyPanel } from "./components/WhyPanel";
import type { BoardPlayer, Candidate, League, Recommendation, SessionState } from "./types";

type Status = "idle" | "running" | "ready" | "error";

export default function App() {
  const [state, setState] = useState<SessionState | null>(null);
  const [league, setLeague] = useState<League | null>(null);
  const [board, setBoard] = useState<BoardPlayer[]>([]);
  const [rec, setRec] = useState<Recommendation | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [elapsed, setElapsed] = useState(0);
  const [choices, setChoices] = useState<Candidate[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [entry, setEntry] = useState("");
  const startedAt = useRef(0);

  const refresh = useCallback(async () => {
    // A 404 from /api/session is the NORMAL pre-draft state, not an error —
    // it is what puts the setup screen on screen. Only surface real failures.
    try {
      const l = await api.league();
      setLeague(l);
      if (!l.active) {
        setState(null);
        return;
      }
      const [s, b] = await Promise.all([api.session(), api.board(500)]);
      setState(s);
      setBoard(b.players);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const startSession = useCallback(
    async (
      seat: number,
      source: string,
      resume: boolean,
      archiveId: string | null = null,
    ) => {
      setError(null);
      try {
        await api.createSession(seat, source, resume, archiveId);
        await refresh();
      } catch (e) {
        const msg = String(e);
        setError(
          msg.includes("409")
            ? "A saved draft already exists — resume it, or archive it first."
            : msg,
        );
      }
    },
    [refresh],
  );

  // SSE is the push path; every frame is also reconstructible from the two
  // GETs above, so a dropped connection costs nothing but latency.
  useEffect(() => {
    const es = new EventSource("/api/stream");
    es.addEventListener("state", (e) => setState(JSON.parse((e as MessageEvent).data)));
    // Runs can start SERVER-side (`auto_recommend` fires when it becomes my
    // turn), so completion events alone leave the button looking idle while
    // the engine is busy.
    es.addEventListener("rec_started", () => {
      setStatus("running");
      setError(null);
    });
    es.addEventListener("rec_ready", (e) => {
      setRec(JSON.parse((e as MessageEvent).data));
      setStatus("ready");
    });
    es.addEventListener("narration_ready", (e) => {
      const { narration } = JSON.parse((e as MessageEvent).data);
      setRec((prev) => (prev ? { ...prev, narration } : prev));
    });
    es.addEventListener("rec_error", (e) => {
      setError(JSON.parse((e as MessageEvent).data).detail);
      setStatus("error");
    });
    es.addEventListener("source_status", (e) => {
      const s = JSON.parse((e as MessageEvent).data);
      setState((prev) => (prev ? { ...prev, source: s } : prev));
    });
    return () => es.close();
  }, []);

  useEffect(() => {
    if (status !== "running") return;
    startedAt.current = Date.now();
    const id = setInterval(
      () => setElapsed((Date.now() - startedAt.current) / 1000),
      200,
    );
    return () => clearInterval(id);
  }, [status]);

  const submit = useCallback(
    async (body: Record<string, unknown>) => {
      setError(null);
      setChoices(null);
      try {
        const res = await api.pick(body);
        if (res.status === "ambiguous") {
          setChoices(res.candidates as Candidate[]);
          return;
        }
        if (res.status === "unresolved" && !body.force_unresolved) {
          setError(`no match for "${body.raw_name}" — confirm to record it anyway`);
          setChoices([]);
          return;
        }
        // The recommendation was computed against a board that no longer
        // exists, so it stops being an answer the moment a pick lands. The
        // backend already discards its own superseded run on the generation
        // check; leaving the stale card on screen is the UI half of that.
        // `auto_recommend` repopulates it when it comes back around to me.
        setRec(null);
        setStatus("idle");
        setEntry("");
        await refresh();
      } catch (e) {
        setError(String(e));
      }
    },
    [refresh],
  );

  const clearSession = useCallback(async (purge: boolean) => {
    // Archives by default; `purge` deletes. The ledger is append-only and
    // untouched either way.
    await api.clearSession(purge);
    setState(null);
    setRec(null);
    setStatus("idle");
    setBoard([]);
    setEntry("");
    setError(null);
    await refresh();
  }, [refresh]);

  const runRecommendation = useCallback(async () => {
    setStatus("running");
    setElapsed(0);
    try {
      await api.recommend();
    } catch (e) {
      setError(String(e));
      setStatus("error");
    }
  }, []);

  if (!state) {
    if (!league) {
      return (
        <main className="grid min-h-screen place-items-center p-8">
          <p className="text-muted">{error ?? "connecting…"}</p>
        </main>
      );
    }
    return (
      <SessionSetup
        league={league}
        onStart={(seat, source, resume, archiveId) =>
          void startSession(seat, source, resume, archiveId)
        }
        onClear={(purge) => void clearSession(purge)}
        error={error}
      />
    );
  }

  const leaderCard = rec && rec.leader ? (
    <>
      <h1 className="text-3xl font-semibold">{rec.leader_name}</h1>
      <p className="tnum mt-1 text-lg">
        {rec.candidates.find((c) => c.player_id === rec.leader)?.e_dollars !== null && (
          <>
            <span className="text-accent">
              ${rec.candidates.find((c) => c.player_id === rec.leader)?.e_dollars?.toFixed(2)}
            </span>
            <span className="text-muted">
              {" ± "}
              {rec.candidates.find((c) => c.player_id === rec.leader)?.total_se?.toFixed(2)}
            </span>
          </>
        )}
        <span className="ml-3 rounded border border-line px-2 py-0.5 text-xs text-muted">
          tier {rec.tier}
        </span>
      </p>
    </>
  ) : (
    <h1 className="text-2xl text-muted">No recommendation yet</h1>
  );

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <ClockStrip state={state} onExit={() => void clearSession(false)} />

      <div className="flex min-h-[8rem] flex-1 flex-col overflow-hidden lg:flex-row">
        <RosterRail slots={state.roster_slots} />

        <main className="min-h-0 flex-1 overflow-y-auto p-6">
          {/* Timer sits with the recommendation, not in the Why rail: the pick
              and the clock are the two things read together. */}
          <div className="flex items-start justify-between gap-6">
            <div className="min-w-0">{leaderCard}</div>
            <PickTimer
              pickNumber={state.pick_number}
              seconds={state.pick_clock_seconds}
              size="text-4xl font-semibold shrink-0"
            />
          </div>

          <div className="mt-6 max-w-2xl">
            {status === "running" ? (
              <div className="space-y-3" role="status" aria-live="polite">
                <p className="tnum text-sm text-muted">
                  simulating… {elapsed.toFixed(1)}s
                </p>
                {[0, 1, 2].map((i) => (
                  <div key={i} className="h-5 animate-pulse rounded bg-surface" />
                ))}
              </div>
            ) : rec ? (
              <CandidatePlot candidates={rec.candidates} leader={rec.leader} />
            ) : null}
          </div>

          <form
            className="mt-8 flex max-w-2xl gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              if (entry.trim()) void submit({ raw_name: entry.trim() });
            }}
          >
            <label className="sr-only" htmlFor="pick">Record a pick</label>
            <input
              id="pick"
              value={entry}
              onChange={(e) => setEntry(e.target.value)}
              placeholder="type a pick…"
              className="focusable flex-1 rounded border border-line bg-surface px-3 py-2 text-sm"
            />
            <button
              type="submit"
              className="focusable cursor-pointer rounded bg-primary px-4 py-2 text-sm font-medium transition-colors duration-200 hover:bg-primary/80"
            >
              Record
            </button>
            <button
              type="button"
              onClick={() => void api.undo().then(refresh)}
              className="focusable cursor-pointer rounded border border-line px-4 py-2 text-sm transition-colors duration-200 hover:bg-surface"
            >
              Undo
            </button>
            <button
              type="button"
              onClick={() => void runRecommendation()}
              disabled={status === "running"}
              className="focusable cursor-pointer rounded border border-line px-4 py-2 text-sm transition-colors duration-200 hover:bg-surface disabled:opacity-50"
            >
              Recommend
            </button>
          </form>

          {error && <p className="mt-3 text-sm text-warn">{error}</p>}

          <BoardPanel
            players={board}
            query={entry}
            onPick={(player_id) => void submit({ player_id })}
          />


          {choices && choices.length > 0 && (
            <div className="mt-3 max-w-2xl">
              <p className="mb-2 text-sm text-muted">Which one?</p>
              <ul className="flex flex-wrap gap-2">
                {choices.map((c) => (
                  <li key={c.player_id}>
                    <button
                      type="button"
                      onClick={() => void submit({ player_id: c.player_id })}
                      className="focusable cursor-pointer rounded border border-line px-3 py-1.5 text-sm transition-colors duration-200 hover:bg-surface"
                    >
                      {c.name}{" "}
                      <span className="text-muted">{c.position}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {choices && choices.length === 0 && (
            <button
              type="button"
              onClick={() =>
                void submit({ raw_name: entry.trim(), force_unresolved: true })
              }
              className="focusable mt-3 cursor-pointer rounded border border-warn px-3 py-1.5 text-sm text-warn transition-colors duration-200 hover:bg-surface"
            >
              Record "{entry.trim()}" as unresolved (keeps the clock correct)
            </button>
          )}
        </main>

        <WhyPanel
          narration={rec?.narration ?? null}
          separatingAxis={rec?.separating_axis ?? ""}
        />
      </div>

      {/* Full width, below the three columns. Twelve teams do not fit inside
          the centre column — the last seats fell off the edge behind a
          horizontal scrollbar, and those are the ones whose picks you read
          between. */}
      <DraftBoard
        picks={state.picks}
        teamNames={state.team_names}
        rounds={state.rounds}
        mySeat={state.seat}
        onTheClock={state.on_the_clock}
        pickNumber={state.pick_number}
      />
    </div>
  );
}
