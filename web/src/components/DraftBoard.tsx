import { useEffect, useRef, useState } from "react";
import { POSITION_STYLE } from "../positions";
import type { DraftPick } from "../types";

/**
 * The room's board: teams across, rounds down.
 *
 * Answers the question a roster list cannot — *when* did that position dry up.
 * Seeing four RBs leave in round 2 is the reason to take one in round 3, and
 * no per-player number expresses it.
 *
 * Teams stay fully visible; ROUNDS scroll. That split is deliberate: a
 * horizontal scroll would hide the seats either side of yours, which are
 * exactly the ones whose picks you are reading between.
 */
/** Rounds visible before the grid scrolls, until the operator drags it. */
const VISIBLE_ROUNDS = 6;
const ROW_HEIGHT_PX = 52;
const DEFAULT_HEIGHT = VISIBLE_ROUNDS * ROW_HEIGHT_PX;
const MIN_HEIGHT = ROW_HEIGHT_PX;           // one round
const HEIGHT_KEY = "draftBoardHeight";

/**
 * Cap against the VIEWPORT, not a fixed pixel count.
 *
 * A fixed maximum taller than the window let the board crowd the three columns
 * out entirely — and because the drag handle sits on the board's top edge,
 * once it filled the screen there was nothing left to grab to shrink it again.
 * Eighty percent still leaves a strip of the columns and, more importantly,
 * keeps the handle on screen — at 100% there is nothing left to grab.
 */
function maxHeight(): number {
  const viewport = typeof window === "undefined" ? 900 : window.innerHeight;
  return Math.max(MIN_HEIGHT * 2, Math.round(viewport * 0.8));
}

function clamp(px: number) {
  return Math.min(maxHeight(), Math.max(MIN_HEIGHT, px));
}

export function DraftBoard({
  picks,
  teamNames,
  rounds,
  mySeat,
  onTheClock,
  pickNumber,
}: {
  picks: DraftPick[];
  teamNames: string[];
  rounds: number;
  mySeat: number;
  onTheClock: number;
  pickNumber: number;
}) {
  const bySeatRound = new Map<string, DraftPick>();
  for (const p of picks) bySeatRound.set(`${p.round}:${p.seat}`, p);

  const currentRound = Math.min(
    rounds,
    Math.floor((pickNumber - 1) / teamNames.length) + 1,
  );
  const shown = Array.from({ length: rounds }, (_, i) => i + 1);

  // Keep the live round in view. Fifteen rounds do not fit in six, and hunting
  // for the current pick is exactly the work this board exists to remove.
  const liveRow = useRef<HTMLDivElement>(null);
  useEffect(() => {
    liveRow.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [currentRound]);

  // How much board to show is a per-draft preference — early on you want the
  // recommendation, late on you want to read the room — so it is draggable and
  // it persists.
  const [height, setHeight] = useState(() => {
    const saved = Number(localStorage.getItem(HEIGHT_KEY));
    return clamp(
      Number.isFinite(saved) && saved > 0 ? saved : DEFAULT_HEIGHT,
    );
  });

  // Re-clamp when the window changes: a height saved on a big monitor would
  // otherwise fill a laptop screen on the next load, with no handle to grab.
  useEffect(() => {
    const onResize = () => setHeight((h) => clamp(h));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  const drag = useRef<{ y: number; from: number } | null>(null);

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    // Pointer capture, not mouse events: without it the drag is dropped the
    // moment the cursor outruns the 6px handle, which it always does.
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { y: e.clientY, from: height };
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!drag.current) return;
    // Dragging UP grows the board, so the delta is inverted.
    setHeight(clamp(drag.current.from + (drag.current.y - e.clientY)));
  };

  const endDrag = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!drag.current) return;
    e.currentTarget.releasePointerCapture(e.pointerId);
    drag.current = null;
    localStorage.setItem(HEIGHT_KEY, String(height));
  };

  const nudge = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? ROW_HEIGHT_PX : 12;
    if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
    e.preventDefault();
    const next = clamp(height + (e.key === "ArrowUp" ? step : -step));
    setHeight(next);
    localStorage.setItem(HEIGHT_KEY, String(next));
  };

  return (
    <section className="shrink-0 border-t border-line">
      {/* Drag handle. Also keyboard-operable — a resize that only works with a
          mouse is a resize half the operators cannot reach. */}
      <div
        role="separator"
        aria-orientation="horizontal"
        aria-label="Resize draft board"
        tabIndex={0}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onKeyDown={nudge}
        onDoubleClick={() => {
          setHeight(DEFAULT_HEIGHT);
          localStorage.setItem(HEIGHT_KEY, String(DEFAULT_HEIGHT));
        }}
        title="Drag to resize · double-click to reset · arrow keys when focused"
        className="focusable group flex h-3.5 w-full cursor-row-resize touch-none items-center justify-center bg-surface/40 transition-colors duration-200 hover:bg-surface"
      >
        <span className="h-1 w-12 rounded-full bg-line transition-colors duration-200 group-hover:bg-muted" />
      </div>

      <div className="px-4 pb-3">
      <h2 className="mb-2 text-xs uppercase tracking-wider text-muted">
        Draft board
      </h2>

      <div className="overflow-hidden rounded border border-line">
        {/* Team header stays put while the rounds scroll under it. */}
        <div
          className="grid border-b border-line bg-surface"
          style={{ gridTemplateColumns: `2.25rem repeat(${teamNames.length}, minmax(0, 1fr))` }}
        >
          <span className="px-1 py-1.5 text-[10px] uppercase text-muted">rd</span>
          {teamNames.map((name, seat) => (
            <span
              key={seat}
              className={`truncate px-2 py-1.5 text-center text-[11px] uppercase tracking-wide ${
                seat === mySeat ? "font-semibold text-accent" : "text-muted"
              }`}
              title={seat === mySeat ? `${name} (you)` : name}
            >
              {name}
            </span>
          ))}
        </div>

        <div className="overflow-y-auto" style={{ height: `${height}px` }}>
          {shown.map((round) => (
            <div
              key={round}
              ref={round === currentRound ? liveRow : undefined}
              className="grid border-b border-line/50 last:border-b-0"
              style={{
                gridTemplateColumns: `2.25rem repeat(${teamNames.length}, minmax(0, 1fr))`,
              }}
            >
              <span className="tnum flex items-center justify-center bg-surface/60 text-xs text-muted">
                {round}
              </span>
              {teamNames.map((_, seat) => {
                const pick = bySeatRound.get(`${round}:${seat}`);
                const isNext =
                  !pick && round === currentRound && seat === onTheClock;
                return (
                  <div
                    key={seat}
                    className={`min-h-[3.25rem] border-l border-line/40 px-1.5 py-1 ${
                      pick
                        ? `border-l-2 ${
                            POSITION_STYLE[pick.position ?? ""] ??
                            "bg-surface border-line"
                          }`
                        : isNext
                          ? "border-l-2 border-good bg-good/10"
                          : ""
                    }`}
                  >
                    {pick ? (
                      <>
                        <div className="flex items-baseline justify-between gap-1">
                          <span className="tnum text-[9px] uppercase text-ink/60">
                            {pick.name.split(" ")[0]}
                          </span>
                          <span className="tnum text-[9px] text-ink/60">
                            {pick.position ?? "—"}
                          </span>
                        </div>
                        <div className="truncate text-[12px] font-semibold uppercase leading-tight">
                          {pick.name.split(" ").slice(1).join(" ") || pick.name}
                        </div>
                      </>
                    ) : isNext ? (
                      <div className="flex h-full items-center justify-center text-[10px] uppercase tracking-wider text-good">
                        picking
                      </div>
                    ) : (
                      <span className="tnum text-[10px] text-muted/30">
                        {round}.{seat + 1}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
      </div>
    </section>
  );
}
