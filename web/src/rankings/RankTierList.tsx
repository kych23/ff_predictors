/**
 * One tier: a header you can rename, recolour and delete, and its players.
 *
 * Named `RankTier*` rather than `Tier*` because `web/src/tiers.ts` already
 * means something else entirely — the recommendation confidence ladder. Two
 * unrelated "tier" concepts in one frontend need distinct names or the next
 * person reads the wrong one.
 */
import { useState } from "react";
import { SortableContext, verticalListSortingStrategy } from "@dnd-kit/sortable";
import { useDroppable } from "@dnd-kit/core";
import { PlayerRow } from "./PlayerRow";
import { TIER_SWATCH, bodyClass, headerClass } from "./palette";
import { TIER_COLORS } from "./palette";
import type { Tier } from "./model";
import type { CataloguePlayer } from "../types";

export function RankTierList({
  tier,
  startRank,
  catalogue,
  staleIds,
  selectedId,
  draggable,
  canDelete,
  onSelect,
  onRename,
  onRecolor,
  onDelete,
}: {
  tier: Tier;
  startRank: number;
  catalogue: Map<string, CataloguePlayer>;
  staleIds: Set<string>;
  selectedId: string | null;
  draggable: boolean;
  canDelete: boolean;
  onSelect: (id: string) => void;
  onRename: (tierId: string, label: string) => void;
  onRecolor: (tierId: string, color: string) => void;
  onDelete: (tierId: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [picking, setPicking] = useState(false);
  const { setNodeRef } = useDroppable({ id: `tier:${tier.id}` });

  return (
    <section className={`mb-4 rounded-lg border ${bodyClass(tier.color)}`}>
      <header
        className={`flex items-center gap-2 rounded-t-lg border-b px-3 py-2
                    ${headerClass(tier.color)}`}
      >
        {editing ? (
          <input
            autoFocus
            defaultValue={tier.label}
            onBlur={(e) => {
              onRename(tier.id, e.target.value.trim() || tier.label);
              setEditing(false);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") (e.target as HTMLInputElement).blur();
              if (e.key === "Escape") setEditing(false);
            }}
            className="flex-1 rounded bg-black/30 px-2 py-0.5 text-sm
                       text-white outline-none"
          />
        ) : (
          <button
            type="button"
            onClick={() => setEditing(true)}
            className="flex-1 cursor-pointer text-left font-sans text-sm"
          >
            {tier.label}
          </button>
        )}

        <span className="font-mono text-xs opacity-70">
          {tier.player_ids.length}
        </span>

        {/* A bare swatch is invisible here — it sits on a header of its own
            colour. The white ring and the surface backing are what make it
            read as a control rather than a smudge. */}
        <button
          type="button"
          aria-label="Change tier colour"
          title="Change tier colour"
          onClick={() => setPicking((p) => !p)}
          className="flex h-5 w-5 cursor-pointer items-center justify-center
                     rounded border border-white/60 bg-black/25
                     transition-colors hover:bg-black/40"
        >
          <span
            className={`h-2.5 w-2.5 rounded-sm ring-1 ring-white/70
                        ${TIER_SWATCH[tier.color] ?? ""}`}
          />
        </button>
        <button
          type="button"
          aria-label="Delete tier"
          disabled={!canDelete}
          title={canDelete ? "Delete tier" : "A scope needs at least one tier"}
          onClick={() => onDelete(tier.id)}
          className="cursor-pointer px-1 text-xs opacity-70 transition-opacity
                     hover:opacity-100 disabled:cursor-not-allowed
                     disabled:opacity-25"
        >
          ✕
        </button>
      </header>

      {picking && (
        <div className="flex items-center gap-2 border-b border-line/40
                        bg-bg px-3 py-2">
          {TIER_COLORS.map((color, i) => (
            <button
              key={color}
              type="button"
              aria-label={`Tier colour ${i + 1}`}
              title={`Tier colour ${i + 1}`}
              onClick={() => {
                onRecolor(tier.id, color);
                setPicking(false);
              }}
              className={`h-6 w-6 cursor-pointer rounded transition-transform
                          hover:scale-110 ${TIER_SWATCH[color]} ${
                            tier.color === color
                              ? "ring-2 ring-white"
                              : "ring-1 ring-white/25"
                          }`}
            />
          ))}
          <span className="ml-1 text-xs text-muted">
            colour is yours to use — the engine never reads it
          </span>
        </div>
      )}

      <div ref={setNodeRef}>
        <SortableContext
          items={tier.player_ids}
          strategy={verticalListSortingStrategy}
        >
          {tier.player_ids.length === 0 && (
            <p className="px-3 py-4 text-center text-xs text-muted/60">
              Drop players here
            </p>
          )}
          {tier.player_ids.map((id, i) => (
            <PlayerRow
              key={id}
              playerId={id}
              rank={startRank + i}
              player={catalogue.get(id)}
              stale={staleIds.has(id)}
              selected={selectedId === id}
              draggable={draggable}
              onSelect={onSelect}
            />
          ))}
        </SortableContext>
      </div>
    </section>
  );
}
