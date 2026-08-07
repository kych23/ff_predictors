import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { CandidatePlot } from "./components/CandidatePlot";
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
      const [s, b] = await Promise.all([api.session(), api.board(12)]);
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
    async (seat: number, source: string, resume: boolean) => {
      setError(null);
      try {
        await api.createSession(seat, source, resume);
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
        setEntry("");
        await refresh();
      } catch (e) {
        setError(String(e));
      }
    },
    [refresh],
  );

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
        onStart={(seat, source, resume) =>
          void startSession(seat, source, resume)
        }
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
    <div className="flex min-h-screen flex-col">
      <ClockStrip state={state} />

      <div className="flex flex-1 flex-col lg:flex-row">
        <RosterRail
          roster={state.my_roster}
          board={board}
          onPick={(player_id) => void submit({ player_id })}
        />

        <main className="flex-1 p-6">
          {leaderCard}

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
          staleFlags={rec?.stale_flags ?? []}
        />
      </div>
    </div>
  );
}
