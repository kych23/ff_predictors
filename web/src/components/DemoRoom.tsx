"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "@/lib/api";
import { useDraftState } from "@/hooks/useDraftState";
import { RecommendationPanel } from "./RecommendationPanel";
import { RosterPanel } from "./RosterPanel";

const DEMO_SEASON = 2024; // a completed season always has projections + ADP

export function DemoRoom() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const creating = useRef(false);
  useEffect(() => {
    if (sessionId || creating.current) return;
    creating.current = true;
    api.createSession(DEMO_SEASON, 1).then((s) => setSessionId(s.session_id));
  }, [sessionId]);

  const d = useDraftState(sessionId);
  const s = d.state;
  const advancing = d.botPick.isPending;

  useEffect(() => {
    if (!s || s.status !== "active" || s.is_my_turn || advancing) return;
    if (s.remaining_picks <= 0) return;
    d.botPick.mutate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s?.current_overall_pick, s?.is_my_turn, s?.status, advancing]);

  const [min, max] = useMemo(() => {
    const ps = (d.recs ?? []).flatMap((r) => [r.p10, r.p90]);
    return ps.length ? [Math.min(...ps), Math.max(...ps)] : [0, 30];
  }, [d.recs]);

  if (!s) return <p className="p-8">Setting up a mock draft…</p>;
  return (
    <div className="p-6 grid grid-cols-1 lg:grid-cols-[1fr_20rem] gap-6">
      <section>
        <h1 className="text-xl font-bold mb-1">Mock draft demo</h1>
        <p className="text-sm text-slate-500 mb-4">
          You draft from slot {s.draft_position}; the other {s.teams - 1} teams pick by ADP.
        </p>
        <div className="mb-3 text-sm">
          Pick {s.current_overall_pick} —{" "}
          {s.is_my_turn ? (
            <b className="text-emerald-700">your pick</b>
          ) : (
            "bots drafting…"
          )}
        </div>
        <RecommendationPanel
          recs={d.recs ?? []}
          min={min}
          max={max}
          onDraft={(id) => d.pick.mutate({ player_id: id })}
        />
      </section>
      <aside>
        <RosterPanel state={s} />
      </aside>
    </div>
  );
}
