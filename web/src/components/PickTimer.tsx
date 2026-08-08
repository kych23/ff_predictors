import { useEffect, useState } from "react";

/**
 * Countdown for the pick on the clock.
 *
 * Purely a display timer, and deliberately so: nothing server-side enforces a
 * pick clock, and the room's real clock is Yahoo's. Presenting this as
 * authoritative would be worse than not having it — it would be a number that
 * looks like a deadline and is not one. It resets whenever the pick changes,
 * from either source.
 */
export function PickTimer({ pickNumber, seconds, size = "text-sm" }: {
  pickNumber: number;
  seconds: number;
  size?: string;
}) {
  const [left, setLeft] = useState(seconds);

  useEffect(() => {
    setLeft(seconds);
    const started = Date.now();
    // Recomputed from a start timestamp rather than decremented, so a
    // backgrounded tab (where timers are throttled) does not drift.
    const id = setInterval(() => {
      const elapsed = Math.floor((Date.now() - started) / 1000);
      setLeft(Math.max(0, seconds - elapsed));
    }, 250);
    return () => clearInterval(id);
  }, [pickNumber, seconds]);

  const mm = Math.floor(left / 60);
  const ss = String(left % 60).padStart(2, "0");
  const tone =
    left === 0 ? "text-red-400"
      : left <= 10 ? "text-red-400"
        : left <= 20 ? "text-warn"
          : "text-muted";

  return (
    <span
      className={`tnum tabular-nums ${size} ${tone}`}
      role="timer"
      aria-live="off"
      title="Display only — the room's clock is authoritative"
    >
      {left === 0 ? "0:00" : `${mm}:${ss}`}
    </span>
  );
}
