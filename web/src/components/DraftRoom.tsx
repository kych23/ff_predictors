"use client";
import { useMemo } from "react";
import { useDraftState } from "@/hooks/useDraftState";
import { StartDraftForm } from "./StartDraftForm";
import { RecommendationPanel } from "./RecommendationPanel";
import { RosterPanel } from "./RosterPanel";
import { PickLog } from "./PickLog";

export function DraftRoom({
  sessionId,
  onSession,
}: {
  sessionId: string | null;
  onSession: (id: string) => void;
}) {
  const d = useDraftState(sessionId);
  const [min, max] = useMemo(() => {
    const ps = (d.recs ?? []).flatMap((r) => [r.p10, r.p90]);
    return ps.length ? [Math.min(...ps), Math.max(...ps)] : [0, 30];
  }, [d.recs]);

  if (!sessionId) {
    return (
      <div className="p-8">
        <h1 className="text-xl font-bold mb-4">Live draft</h1>
        <StartDraftForm onCreated={onSession} />
      </div>
    );
  }
  if (d.isLoading || !d.state) return <p className="p-8">Loading draft…</p>;
  const s = d.state;
  return (
    <div className="p-6 grid grid-cols-1 lg:grid-cols-[1fr_20rem] gap-6">
      <section>
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-xl font-bold">
            Pick {s.current_overall_pick} ·{" "}
            <span className={s.is_my_turn ? "text-emerald-700" : "text-slate-500"}>
              {s.is_my_turn ? "Your pick" : "Waiting on other teams"}
            </span>
          </h1>
          <div className="flex gap-2">
            <button
              onClick={() => d.skip.mutate()}
              className="border rounded px-3 py-1 text-sm"
            >
              Skip
            </button>
            <button
              onClick={() => d.undo.mutate()}
              className="border rounded px-3 py-1 text-sm"
            >
              Undo
            </button>
          </div>
        </div>
        <RecommendationPanel
          recs={d.recs ?? []}
          min={min}
          max={max}
          onDraft={(id) => d.pick.mutate({ player_id: id })}
        />
      </section>
      <aside className="space-y-6">
        <RosterPanel state={s} />
        <div>
          <h3 className="font-semibold mb-2">Pick log</h3>
          <PickLog state={s} />
        </div>
      </aside>
    </div>
  );
}
