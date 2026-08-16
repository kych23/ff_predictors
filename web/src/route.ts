/**
 * Hash routing, in about forty lines and no dependency.
 *
 * Three views do not justify react-router. The hash is used rather than the
 * History API because the app is served by FastAPI's `StaticFiles` mount,
 * which has no SPA rewrite — a real path would 404 on refresh.
 *
 * The **active scope tab is deliberately not in the route.** Clicking through
 * QB/RB/WR while building a board would otherwise stack seven history entries,
 * and Back would walk them one at a time instead of leaving the board.
 */
import { useEffect, useState } from "react";

export type View = "home" | "draft" | "board";

export interface Route {
  view: View;
  boardId: string | null;
}

export function parseHash(hash: string): Route {
  const path = hash.replace(/^#\/?/, "").split("?")[0];
  const [head, ...rest] = path.split("/").filter(Boolean);

  if (head === "draft") return { view: "draft", boardId: null };
  if (head === "board") {
    return { view: "board", boardId: rest[0] ? decodeURIComponent(rest[0]) : null };
  }
  // Anything unrecognised is home, not a 404 screen. A stale bookmark should
  // land somewhere useful.
  return { view: "home", boardId: null };
}

export function navigate(hash: string): void {
  window.location.hash = hash;
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() =>
    parseHash(window.location.hash),
  );

  useEffect(() => {
    const onChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  return route;
}
