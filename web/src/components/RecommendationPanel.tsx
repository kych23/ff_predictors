"use client";
import type { Recommendation } from "@/lib/types";
import { QuantileBar } from "./QuantileBar";

export function RecommendationPanel({
  recs,
  min,
  max,
  onDraft,
}: {
  recs: Recommendation[];
  min: number;
  max: number;
  onDraft?: (id: string) => void;
}) {
  if (!recs.length) return <p className="text-slate-500">No recommendations.</p>;
  return (
    <ul className="space-y-2">
      {recs.map((r) => (
        <li key={r.player_id} className="border rounded p-3 flex flex-col gap-1">
          <div className="flex items-center justify-between">
            <div>
              <span className="font-semibold">{r.name ?? r.player_id}</span>
              <span className="ml-2 text-xs text-slate-500">
                {r.position} · {r.team ?? "—"}
              </span>
              {r.forced_completion && (
                <span className="ml-2 text-xs bg-amber-200 rounded px-1">must fill slot</span>
              )}
            </div>
            {onDraft && (
              <button
                onClick={() => onDraft(r.player_id)}
                className="bg-emerald-600 text-white text-sm rounded px-3 py-1"
              >
                Draft
              </button>
            )}
          </div>
          <QuantileBar p10={r.p10} p50={r.p50} p90={r.p90} min={min} max={max} />
          <div className="flex justify-between text-xs text-slate-500">
            <span>VONA {r.vona_score.toFixed(1)}</span>
            <span>
              P10 {r.p10.toFixed(0)} · P50 {r.p50.toFixed(0)} · P90 {r.p90.toFixed(0)}
            </span>
            <span>ADP {r.adp?.toFixed(0) ?? "—"}</span>
          </div>
        </li>
      ))}
    </ul>
  );
}
