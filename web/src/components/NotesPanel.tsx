import { useCallback, useEffect, useRef, useState } from "react";

/**
 * A scratch pad for the things the engine will never know.
 *
 * "Kevin is hoarding backs", "Zach said he needs a QB", "Raam took my guy" —
 * reads of the room that decide picks and have no column anywhere. The engine
 * is deliberately blind to them: nothing here reaches the recommendation, the
 * ledger, or any feature. It is a notepad that happens to sit next to a
 * simulator.
 *
 * **Persisted to localStorage, not the server.** Two reasons. It has to
 * survive a refresh — a browser reload mid-draft losing your reads would make
 * the pad useless, and draft night is exactly when a tab gets reloaded. But it
 * must NOT go through the API: `service.py` is a single-writer event loop on a
 * pick clock, and putting a keystroke-rate write on that path to store
 * something no code reads would be trading real risk for no gain.
 *
 * Keyed on `sessionId`, so a new draft opens a blank pad instead of inheriting
 * the last one's notes.
 */
const STORAGE_PREFIX = "ff.notes.";

/** Debounce writes. Every keystroke hitting localStorage is a synchronous
 *  main-thread write, and this sits beside a running simulation. */
const SAVE_DELAY_MS = 400;

function storageKey(sessionId: string): string {
  return `${STORAGE_PREFIX}${sessionId || "unknown"}`;
}

function readNotes(sessionId: string): string {
  try {
    return window.localStorage.getItem(storageKey(sessionId)) ?? "";
  } catch {
    // Private browsing and disabled storage both throw. A pad that does not
    // persist still beats a crashed draft screen.
    return "";
  }
}

export function NotesPanel({ sessionId }: { sessionId: string }) {
  const [text, setText] = useState(() => readNotes(sessionId));
  const [saved, setSaved] = useState(true);
  const timer = useRef<number | undefined>(undefined);

  // Reload when the draft changes — the pad belongs to a session, and
  // remounting is not guaranteed.
  useEffect(() => {
    setText(readNotes(sessionId));
    setSaved(true);
  }, [sessionId]);

  const flush = useCallback(
    (value: string) => {
      try {
        window.localStorage.setItem(storageKey(sessionId), value);
        setSaved(true);
      } catch {
        setSaved(false);
      }
    },
    [sessionId],
  );

  const onChange = (value: string) => {
    setText(value);
    setSaved(false);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => flush(value), SAVE_DELAY_MS);
  };

  // A pending debounce must not lose the last keystrokes when the component
  // goes away — which is what happens the moment the draft screen unmounts.
  useEffect(() => {
    const onHide = () => {
      window.clearTimeout(timer.current);
      flush(text);
    };
    window.addEventListener("beforeunload", onHide);
    return () => {
      window.removeEventListener("beforeunload", onHide);
      onHide();
    };
  }, [flush, text]);

  return (
    <section className="mt-6 border-t border-line pt-4">
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <h2 className="text-xs uppercase tracking-wider text-muted">Notes</h2>
        <span
          className="text-[11px] text-muted"
          aria-live="polite"
          // Only ever "saving…" or nothing: a permanent "saved" badge is noise
          // on a screen where every other element means something.
        >
          {saved ? "" : "saving…"}
        </span>
      </div>

      <label htmlFor="draft-notes" className="sr-only">
        Draft notes
      </label>
      <textarea
        id="draft-notes"
        value={text}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Reads on the room — who needs what, who is reaching…"
        spellCheck={false}
        className="focusable min-h-[8rem] w-full resize-y rounded border border-line bg-surface px-3 py-2 text-sm leading-relaxed placeholder:text-muted/70"
      />
    </section>
  );
}
