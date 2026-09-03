/**
 * A draggable vertical divider between the tier list and the player card.
 *
 * The width is in PIXELS, not a percentage: the detail panel's content is
 * fixed-width in character terms — stat labels, a six-column market grid — so
 * it should not reflow just because the window did. The tier list takes the
 * remainder.
 *
 * Persisted to localStorage per the precedent in
 * `components/DraftBoard.tsx`, which stores its own resized height the same
 * way. This is a viewing preference, not board content: it belongs to the
 * browser, not to the saved document, and it must not dirty the board or
 * trigger a save.
 */
import { useCallback, useEffect, useRef, useState } from "react";

const KEY = "rankingsPanelWidth";
const DEFAULT = 320;
/** Below this the market grid wraps into unreadable columns. */
const MIN = 260;
/** Always leave room for a player name plus its numbers. */
const MIN_LIST = 480;

const clamp = (value: number): number =>
  Math.max(MIN, Math.min(value, window.innerWidth - MIN_LIST));

export function useSplitPane() {
  const [width, setWidth] = useState<number>(() => {
    const saved = Number(window.localStorage.getItem(KEY));
    return Number.isFinite(saved) && saved >= MIN ? clamp(saved) : DEFAULT;
  });
  const dragging = useRef(false);
  const latest = useRef(width);
  latest.current = width;

  useEffect(() => {
    const onMove = (event: PointerEvent) => {
      if (!dragging.current) return;
      event.preventDefault();
      setWidth(clamp(window.innerWidth - event.clientX));
    };
    const onUp = () => {
      if (!dragging.current) return;
      dragging.current = false;
      // Restored on the BODY, not the divider: the cursor and the text-select
      // lock are applied globally while dragging so the pointer can leave the
      // 5px handle without the drag turning into a text selection.
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.localStorage.setItem(KEY, String(latest.current));
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, []);

  // A window narrower than the last saved width would otherwise push the list
  // off-screen with no way to drag it back.
  useEffect(() => {
    const onResize = () => setWidth((w) => clamp(w));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const onPointerDown = useCallback(() => {
    dragging.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }, []);

  /** Arrow keys move the divider too — a mouse-only splitter is unusable for
   * anyone who does not use a mouse, and it costs four lines. */
  const onKeyDown = useCallback((event: React.KeyboardEvent) => {
    const step = event.shiftKey ? 48 : 16;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      setWidth((w) => {
        const next = clamp(w + step);
        window.localStorage.setItem(KEY, String(next));
        return next;
      });
    }
    if (event.key === "ArrowRight") {
      event.preventDefault();
      setWidth((w) => {
        const next = clamp(w - step);
        window.localStorage.setItem(KEY, String(next));
        return next;
      });
    }
  }, []);

  return { width, onPointerDown, onKeyDown };
}
