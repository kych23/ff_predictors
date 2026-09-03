import { useEffect, useState } from "react";
import { Archive, Trash2 } from "lucide-react";
import { api } from "../api";
import type { League, ReplayOption } from "../types";

/** "8 Aug, 2:58 PM" — enough to tell two drafts apart, in local time. */
function whenLabel(iso: string | null): string {
  if (!iso) return "unknown time";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    day: "numeric",
    month: "short",
    hour: "numeric",
    minute: "2-digit",
  });
}

/**
 * Start a draft.
 *
 * This screen exists because without it the app is unusable: the cockpit
 * cannot begin a draft at all, and the placeholder it replaced told the
 * operator to POST to the API by hand.
 *
 * Seats are **0-indexed everywhere in the API** and 1-indexed nowhere except
 * the label a human reads — where they are also called "pick", because that is
 * what a drafter calls the slot they are in. The conversion happens here, once,
 * and this is the only place in the frontend allowed to do it.
 */
/** What each source actually DOES, rather than its internal name.
 *
 * "yahoo" is not "Yahoo instead of manual" — manual entry is never gated on
 * the source, so it is "Yahoo as well as manual". Presenting the raw enum
 * made them look mutually exclusive, which is the opposite of true and the
 * one thing an operator must not believe on a pick clock.
 */
const SOURCE_LABEL: Record<string, string> = {
  manual: "Manual only — I enter every pick",
  yahoo: "Yahoo live feed + manual (recommended)",
  replay: "Replay a saved draft",
};

export function SessionSetup({
  league,
  onStart,
  onClear,
  error,
}: {
  league: League;
  onStart: (
    seat: number,
    source: string,
    resume: boolean,
    archiveId: string | null,
  ) => void;
  onClear: (purge: boolean) => void;
  error: string | null;
}) {
  const [seat, setSeat] = useState(0);
  const [source, setSource] = useState(league.default_source);
  // Deleting is irreversible, so it takes a second, deliberate click. Archiving
  // is recoverable and does not.
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [replays, setReplays] = useState<ReplayOption[] | null>(null);
  const [archiveId, setArchiveId] = useState<string>("");
  const busy = false;

  // Fetched only when Replay is chosen. Loading a directory listing on every
  // visit to the start screen would be work nobody asked for.
  useEffect(() => {
    if (source !== "replay" || replays !== null) return;
    let live = true;
    void api
      .replays()
      .then((r) => live && setReplays(r.replays))
      .catch(() => live && setReplays([]));
    return () => {
      live = false;
    };
  }, [source, replays]);

  return (
    <main className="mx-auto grid min-h-screen max-w-lg place-items-center p-8">
      <div className="w-full space-y-6">
        <header>
          <h1 className="text-2xl font-semibold">Draft cockpit</h1>
        </header>

        {league.session_exists && (
          <div className="rounded border border-warn/40 bg-surface p-4">
            <p className="text-sm">
              A draft is already saved on this machine.
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => onStart(seat, source, true, null)}
                className="focusable cursor-pointer rounded bg-accent px-4 py-2 text-sm font-medium text-bg transition-colors duration-200 hover:bg-accent/80"
              >
                Resume it
              </button>

              <button
                type="button"
                onClick={() => {
                  setConfirmDelete(false);
                  onClear(false);
                }}
                title="Rename the log aside; you can recover it from disk"
                className="focusable flex cursor-pointer items-center gap-1.5 rounded border border-line px-3 py-2 text-sm transition-colors duration-200 hover:bg-bg"
              >
                <Archive size={14} aria-hidden="true" />
                Archive
              </button>

              {confirmDelete ? (
                <button
                  type="button"
                  onClick={() => {
                    setConfirmDelete(false);
                    onClear(true);
                  }}
                  className="focusable flex cursor-pointer items-center gap-1.5 rounded border border-red-500/60 bg-red-500/10 px-3 py-2 text-sm text-red-300 transition-colors duration-200 hover:bg-red-500/20"
                >
                  <Trash2 size={14} aria-hidden="true" />
                  Delete permanently?
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => setConfirmDelete(true)}
                  title="Erase the log. This cannot be undone."
                  className="focusable flex cursor-pointer items-center gap-1.5 rounded border border-line px-3 py-2 text-sm text-muted transition-colors duration-200 hover:border-red-500/50 hover:text-red-300"
                >
                  <Trash2 size={14} aria-hidden="true" />
                  Delete
                </button>
              )}
            </div>

            <p className="mt-2 text-xs text-muted">
              Starting a new draft refuses with a conflict until this one is
              archived or deleted. Archiving keeps the decision ledger; deleting
              removes this draft from it too.
            </p>
          </div>
        )}

        <div className="space-y-2">
          <label htmlFor="seat" className="block text-xs uppercase tracking-wider text-muted">
            Your pick
          </label>
          <select
            id="seat"
            value={seat}
            onChange={(e) => setSeat(Number(e.target.value))}
            className="focusable tnum w-full cursor-pointer rounded border border-line bg-surface px-3 py-2 text-sm"
          >
            {Array.from({ length: league.teams }, (_, i) => (
              <option key={i} value={i}>
                Pick {i + 1}
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-2">
          <label htmlFor="source" className="block text-xs uppercase tracking-wider text-muted">
            How picks arrive
          </label>
          <select
            id="source"
            value={source}
            onChange={(e) => setSource(e.target.value)}
            className="focusable w-full cursor-pointer rounded border border-line bg-surface px-3 py-2 text-sm"
          >
            {league.sources.map((s) => (
              <option key={s} value={s}>
                {SOURCE_LABEL[s] ?? s}
              </option>
            ))}
          </select>
          {source === "replay" && (
            <div className="space-y-2">
              <label
                htmlFor="replay"
                className="block text-xs uppercase tracking-wider text-muted"
              >
                Which draft
              </label>
              <select
                id="replay"
                value={archiveId}
                onChange={(e) => setArchiveId(e.target.value)}
                className="focusable w-full cursor-pointer rounded border border-line bg-surface px-3 py-2 text-sm"
              >
                <option value="">Sample draft (40 picks)</option>
                {(replays ?? []).map((r) => (
                  <option key={r.id} value={r.id} disabled={!r.readable}>
                    {whenLabel(r.started_at ?? r.archived_at)}
                    {" — "}
                    {r.picks} pick{r.picks === 1 ? "" : "s"}
                    {r.seat !== null ? `, pick ${r.seat + 1}` : ""}
                    {r.readable ? "" : " (unreadable)"}
                  </option>
                ))}
              </select>
              <p className="text-xs text-muted">
                {replays === null
                  ? "Loading archived drafts…"
                  : replays.length === 0
                    ? "No archived drafts yet — archive one and it appears here."
                    : `${replays.length} archived draft${replays.length === 1 ? "" : "s"}, newest first.`}
              </p>
            </div>
          )}

          {source === "yahoo" && (
            <p className="text-xs text-muted">
              The feed fills picks on its own, and you can always click a
              player off the board to fill one yourself — whichever gets there
              first wins and the other is ignored. If Yahoo stops responding
              the cockpit says so and you carry on clicking; nothing is lost.
            </p>
          )}
          {source === "manual" && (
            <p className="text-xs text-muted">
              Every pick typed or clicked by you. Choose Yahoo instead if you
              want the feed as a safety net — manual entry works there too.
            </p>
          )}
        </div>

        <button
          type="button"
          disabled={busy}
          onClick={() =>
            onStart(seat, source, false, source === "replay" ? archiveId || null : null)
          }
          className="focusable w-full cursor-pointer rounded bg-primary px-4 py-3 text-sm font-medium transition-colors duration-200 hover:bg-primary/80 disabled:opacity-50"
        >
          Start draft
        </button>

        {error && <p className="text-sm text-warn">{error}</p>}
      </div>
    </main>
  );
}
