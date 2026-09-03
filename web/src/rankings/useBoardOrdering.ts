/**
 * Order the cockpit's available-players list by one of your saved boards.
 *
 * **This is display only, and the distinction matters.** It changes what the
 * board panel shows and how it sorts; it does not reach the recommender. The
 * engine still prices every pick from its own objective and has never seen
 * your tiers — which is what keeps "the engine beats the market" a claim about
 * the model rather than about your opinion fed back to itself.
 *
 * Uses the board's `overall` scope even when a position filter is on: your
 * overall order filtered to running backs is still your ranking of running
 * backs, and the position scopes are independent lists you may not have
 * seeded at all.
 *
 * Defaults to ADP and falls back to ADP on any failure. Draft night is not
 * the moment to discover that a research feature can take the pick list away.
 */
import { useCallback, useEffect, useState } from "react";
import { api } from "../api";
import type { BoardSummary } from "../types";

const KEY = "cockpitOrderingBoard";

/** What a ranked player carries into the panel. */
export interface Ranked {
  rank: number;
  tierLabel: string;
  tierColor: string;
}

export interface Ordering {
  boards: BoardSummary[];
  boardId: string | null;
  boardName: string | null;
  /** null means "order by ADP" — the default. */
  ranking: Map<string, Ranked> | null;
  select: (id: string | null) => void;
  error: string | null;
}

export function useBoardOrdering(): Ordering {
  const [boards, setBoards] = useState<BoardSummary[]>([]);
  const [boardId, setBoardId] = useState<string | null>(
    () => window.localStorage.getItem(KEY),
  );
  const [ranking, setRanking] = useState<Map<string, Ranked> | null>(null);
  const [boardName, setBoardName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void api
      .boards()
      .then((r) => setBoards(r.boards))
      .catch(() => setBoards([]));
  }, []);

  useEffect(() => {
    if (!boardId) {
      setRanking(null);
      setBoardName(null);
      return;
    }
    let live = true;
    void api
      .rankingBoard(boardId)
      .then((board) => {
        if (!live) return;
        const map = new Map<string, Ranked>();
        let rank = 0;
        for (const tier of board.scopes.overall.tiers) {
          for (const id of tier.player_ids) {
            rank += 1;
            map.set(id, {
              rank,
              tierLabel: tier.label,
              tierColor: tier.color,
            });
          }
        }
        if (map.size === 0) {
          // An unseeded board would silently order everything as "unranked",
          // which looks identical to a broken feature. Say so and stay on ADP.
          setError(`${board.name} has nothing in Overall — showing ADP`);
          setRanking(null);
          setBoardName(null);
          return;
        }
        setError(null);
        setRanking(map);
        setBoardName(board.name);
      })
      .catch((e) => {
        if (!live) return;
        // A deleted board leaves a stale id in localStorage; drop it rather
        // than failing on every load from here on.
        setError(`could not load that board (${String(e)}) — showing ADP`);
        setRanking(null);
        setBoardName(null);
        window.localStorage.removeItem(KEY);
      });
    return () => {
      live = false;
    };
  }, [boardId]);

  const select = useCallback((id: string | null) => {
    setError(null);
    setBoardId(id);
    if (id) window.localStorage.setItem(KEY, id);
    else window.localStorage.removeItem(KEY);
  }, []);

  return { boards, boardId, boardName, ranking, select, error };
}
