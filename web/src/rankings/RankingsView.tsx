/**
 * My Board: the container. Owns board state, drag, and the save lifecycle.
 *
 * **Saving.** Debounced 1500 ms, one request in flight with trailing
 * coalescing, and a `beforeunload` flush. That is much slower than
 * `NotesPanel`'s 400 ms, on purpose: NotesPanel writes to localStorage, which
 * cannot fail and touches no server. This writes through FastAPI, and
 * `service.py` is a single-writer loop that may be running a pick clock.
 *
 * **The flush only fires when nothing is in flight.** If a PUT is outstanding,
 * the server has already advanced `rev` while this tab still holds the old
 * one, so a flush would 409 and lose the trailing edits with no page left to
 * retry on. When a request IS outstanding it is the last write anyway, so the
 * unsaved delta is bounded by one debounce tick.
 *
 * **Conflicts are shown, not swallowed.** Two tabs on one board is a real
 * thing to do; last-write-wins would silently eat an afternoon of research.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  DndContext,
  DragEndEvent,
  KeyboardSensor,
  PointerSensor,
  closestCenter,
  useSensor,
  useSensors,
} from "@dnd-kit/core";
import { sortableKeyboardCoordinates } from "@dnd-kit/sortable";
import { RevConflictError, api, saveBoard } from "../api";
import { navigate } from "../route";
import { RankTierList } from "./RankTierList";
import { PlayerDetail } from "./PlayerDetail";
import { ScopeTabs } from "./ScopeTabs";
import { SeedDialog } from "./SeedDialog";
import { TIER_COLORS } from "./palette";
import type { Board, ScopeName } from "./model";
import {
  addTier,
  deleteTier,
  flatten,
  movePlayer,
  recolorTier,
  renameTier,
} from "./model";
import type { BoardSummary, CataloguePlayer } from "../types";

const DEBOUNCE_MS = 1500;

type SaveState = "idle" | "saving" | "saved" | "conflict" | "error";

export function RankingsView({ boardId }: { boardId: string | null }) {
  const [board, setBoard] = useState<Board | null>(null);
  const [boards, setBoards] = useState<BoardSummary[]>([]);
  const [catalogue, setCatalogue] = useState<CataloguePlayer[]>([]);
  const [scope, setScope] = useState<ScopeName>("overall");
  const [selected, setSelected] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [seeding, setSeeding] = useState(false);
  const [newName, setNewName] = useState("");

  const pending = useRef<Board | null>(null);
  const inFlight = useRef(false);
  const timer = useRef<number | null>(null);

  const byId = useMemo(() => {
    const map = new Map<string, CataloguePlayer>();
    catalogue.forEach((p) => map.set(p.player_id, p));
    return map;
  }, [catalogue]);

  const staleIds = useMemo(
    () => new Set(board?.stale_player_ids ?? []),
    [board],
  );

  useEffect(() => {
    void api.boards().then((r) => setBoards(r.boards)).catch(() => undefined);
    void api
      .catalogue()
      .then((r) => setCatalogue(r.players))
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!boardId) {
      setBoard(null);
      return;
    }
    // A→B→A in quick succession can settle on the wrong board's contents
    // without this; `PlayerDetail` already uses the same guard.
    let live = true;
    void api
      .rankingBoard(boardId)
      .then((b) => live && setBoard(b))
      .catch((e) => live && setError(String(e)));
    return () => {
      live = false;
    };
  }, [boardId]);

  // ------------------------------------------------------------- saving
  const flushNow = useCallback(async () => {
    const next = pending.current;
    if (!next || inFlight.current) return;
    // `pending` is deliberately NOT cleared here. Identity comparison after
    // the await is what tells us whether a drag landed mid-flight, and it
    // keeps the queued edit reachable if the request fails.
    inFlight.current = true;
    setSaveState("saving");
    try {
      const saved = await saveBoard(next.board_id, {
        expected_rev: next.rev,
        scopes: next.scopes,
        name: next.name,
      });

      if (pending.current === next) {
        pending.current = null;
        setBoard(saved);
      } else {
        // REBASE, do not clobber. The edit made during the request carries the
        // PRE-save rev while the server is now at rev+1, so `setBoard(saved)`
        // would drop it visually AND the follow-up save would 409 against its
        // own predecessor — reporting "changed in another tab" in one tab.
        // Server metadata, local content.
        const queued = pending.current ?? next;
        const rebased: Board = {
          ...saved,
          scopes: queued.scopes,
          name: queued.name,
        };
        pending.current = rebased;
        setBoard(rebased);
      }
      setSaveState("saved");
    } catch (e) {
      if (e instanceof RevConflictError) {
        setBoard(e.current);
        setSaveState("conflict");
      } else {
        setError(String(e));
        setSaveState("error");
      }
    } finally {
      inFlight.current = false;
      if (pending.current) void flushNow();
    }
  }, []);

  const queueSave = useCallback(
    (next: Board) => {
      setBoard(next);
      pending.current = next;
      if (timer.current) window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => void flushNow(), DEBOUNCE_MS);
    },
    [flushNow],
  );

  useEffect(() => {
    const onUnload = () => {
      // Only when nothing is outstanding — see the module docstring.
      if (!pending.current || inFlight.current) return;
      const next = pending.current;
      void saveBoard(
        next.board_id,
        { expected_rev: next.rev, scopes: next.scopes, name: next.name },
        { flush: true },
      ).catch(() => undefined);
    };
    window.addEventListener("beforeunload", onUnload);
    return () => window.removeEventListener("beforeunload", onUnload);
  }, []);

  // --------------------------------------------------------------- drag
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 4 } }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  );

  const onDragEnd = useCallback(
    (event: DragEndEvent) => {
      const { active, over } = event;
      if (!board || !over || active.id === over.id) return;

      const activeId = String(active.id);
      const overId = String(over.id);
      const tiers = board.scopes[scope].tiers;

      // Dropped on an empty tier's droppable, not on another player.
      if (overId.startsWith("tier:")) {
        const tierId = overId.slice(5);
        const target = tiers.find((t) => t.id === tierId);
        queueSave(
          movePlayer(board, scope, activeId, tierId, target?.player_ids.length ?? 0),
        );
        return;
      }

      const target = tiers.find((t) => t.player_ids.includes(overId));
      if (!target) return;
      queueSave(
        movePlayer(board, scope, activeId, target.id,
          target.player_ids.indexOf(overId)),
      );
    },
    [board, scope, queueSave],
  );

  // ------------------------------------------------------------ actions
  const createBoard = useCallback(
    async (seedMethod: string | null) => {
      try {
        const created = await api.createBoard(newName.trim(), seedMethod);
        navigate(`#/board/${created.board_id}`);
      } catch (e) {
        setError(String(e));
      }
    },
    [newName],
  );

  const runSeed = useCallback(
    async (method: string, tierSize: number | null) => {
      if (!board) return;
      try {
        const seeded = await api.seedScope(board.board_id, {
          scope,
          method,
          expected_rev: board.rev,
          tier_size: tierSize,
        });
        setBoard(seeded);
        setSeeding(false);
      } catch (e) {
        setError(String(e));
      }
    },
    [board, scope],
  );

  // ------------------------------------------------------------ renders
  if (!boardId) {
    return (
      <div className="mx-auto max-w-2xl px-6 py-12">
        <h1 className="font-sans text-2xl text-ink">My Board</h1>
        {boards.length > 0 && (
          <ul className="mt-6 divide-y divide-line rounded border border-line">
            {boards.map((b) => (
              <li key={b.board_id}>
                <button
                  type="button"
                  onClick={() => navigate(`#/board/${b.board_id}`)}
                  className="flex w-full cursor-pointer items-center
                             justify-between px-4 py-3 text-left
                             transition-colors hover:bg-white/5"
                >
                  <span className="text-sm text-ink">{b.name}</span>
                  <span className="font-mono text-xs text-muted">
                    {b.counts.overall} ranked · rev {b.rev}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}

        <div className="mt-8 rounded border border-line p-4">
          <h2 className="font-sans text-sm text-ink">New board</h2>
          <input
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="My 2026 Board"
            className="mt-3 w-full rounded border border-line bg-bg px-3 py-2
                       text-sm text-ink outline-none focus:border-primary"
          />
          <div className="mt-3 flex flex-wrap gap-2">
            <button
              type="button"
              disabled={!newName.trim()}
              onClick={() => void createBoard("adp")}
              className="cursor-pointer rounded bg-primary px-3 py-1.5 text-sm
                         text-white transition-colors hover:bg-primary/80
                         disabled:cursor-not-allowed disabled:opacity-40"
            >
              Start from ADP
            </button>
            <button
              type="button"
              disabled={!newName.trim()}
              onClick={() => void createBoard("engine_vor")}
              className="cursor-pointer rounded border border-line px-3 py-1.5
                         text-sm text-ink transition-colors hover:bg-white/5
                         disabled:cursor-not-allowed disabled:opacity-40"
            >
              Start from the engine
            </button>
            <button
              type="button"
              disabled={!newName.trim()}
              onClick={() => void createBoard(null)}
              className="cursor-pointer px-3 py-1.5 text-sm text-muted
                         transition-colors hover:text-ink
                         disabled:cursor-not-allowed disabled:opacity-40"
            >
              Empty
            </button>
          </div>
          <p className="mt-3 text-xs text-muted">
            ADP orders by the market. The engine orders by projected points over
            replacement at each position — a different opinion, which is the
            point of having both.
          </p>
        </div>
        {error && <p className="mt-4 text-sm text-warn">{error}</p>}
      </div>
    );
  }

  if (!board) {
    return <div className="p-8 text-sm text-muted">Loading board…</div>;
  }

  const tiers = board.scopes[scope].tiers;
  const rankOffsets: number[] = [];
  tiers.reduce((acc, tier) => {
    rankOffsets.push(acc + 1);
    return acc + tier.player_ids.length;
  }, 0);

  const visible = filter.trim().toLowerCase();
  const matches = (id: string) =>
    !visible || (byId.get(id)?.name.toLowerCase().includes(visible) ?? false);

  return (
    <div className="flex h-[calc(100vh-41px)]">
      <div className="flex flex-1 flex-col overflow-hidden">
        <div className="flex flex-wrap items-center gap-3 border-b
                        border-line px-4 py-3">
          <h1 className="font-sans text-lg text-ink">{board.name}</h1>
          <span className="font-mono text-xs text-muted">
            {flatten(board, scope).length} in {scope}
          </span>
          <input
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter…"
            className="ml-auto w-40 rounded border border-line bg-bg px-2 py-1
                       text-xs text-ink outline-none focus:border-primary"
          />
          <button
            type="button"
            onClick={() => setSeeding(true)}
            className="cursor-pointer rounded border border-line px-2 py-1
                       text-xs text-ink transition-colors hover:bg-white/5"
          >
            Seed {scope}
          </button>
          <button
            type="button"
            onClick={() => queueSave(addTier(board, scope, tiers.length,
              TIER_COLORS[tiers.length % TIER_COLORS.length]))}
            className="cursor-pointer rounded border border-line px-2 py-1
                       text-xs text-ink transition-colors hover:bg-white/5"
          >
            + Tier
          </button>
          <span
            className={`font-mono text-xs ${
              saveState === "conflict" || saveState === "error"
                ? "text-warn"
                : "text-muted"
            }`}
          >
            {saveState === "saving" && "saving…"}
            {saveState === "saved" && "saved"}
            {saveState === "conflict" &&
              "changed in another tab — reloaded"}
            {saveState === "error" && "save failed"}
          </span>
        </div>

        <ScopeTabs scope={scope} board={board} onSelect={setScope} />

        {visible && (
          <p className="border-b border-line bg-warn/10 px-4 py-1 text-xs
                        text-warn">
            Dragging is off while a filter is active — an index in a filtered
            list is not the index the board stores.
          </p>
        )}

        <div className="flex-1 overflow-y-auto px-4 py-4">
          {tiers.length === 0 ? (
            <p className="py-12 text-center text-sm text-muted">
              Nothing in {scope} yet. Use “Seed {scope}”.
            </p>
          ) : (
            <DndContext
              sensors={sensors}
              collisionDetection={closestCenter}
              onDragEnd={onDragEnd}
            >
              {tiers.map((tier, i) => (
                <RankTierList
                  key={tier.id}
                  tier={{
                    ...tier,
                    player_ids: tier.player_ids.filter(matches),
                  }}
                  startRank={rankOffsets[i]}
                  catalogue={byId}
                  staleIds={staleIds}
                  selectedId={selected}
                  draggable={!visible}
                  canDelete={tiers.length > 1}
                  onSelect={setSelected}
                  onRename={(id, label) =>
                    queueSave(renameTier(board, scope, id, label))
                  }
                  onRecolor={(id, color) =>
                    queueSave(recolorTier(board, scope, id, color))
                  }
                  onDelete={(id) => {
                    try {
                      queueSave(deleteTier(board, scope, id));
                    } catch (e) {
                      setError(String(e));
                    }
                  }}
                />
              ))}
            </DndContext>
          )}
        </div>
      </div>

      <div className="w-80 shrink-0 border-l border-line">
        <PlayerDetail playerId={selected} />
      </div>

      {seeding && (
        <SeedDialog
          scope={scope}
          onCancel={() => setSeeding(false)}
          onSeed={(method, size) => void runSeed(method, size)}
          hasContent={tiers.length > 0}
        />
      )}
      {error && (
        <div className="absolute bottom-4 left-4 rounded border border-warn
                        bg-surface px-3 py-2 text-xs text-warn">
          {error}
          <button
            type="button"
            onClick={() => setError(null)}
            className="ml-2 cursor-pointer underline"
          >
            dismiss
          </button>
        </div>
      )}
    </div>
  );
}
