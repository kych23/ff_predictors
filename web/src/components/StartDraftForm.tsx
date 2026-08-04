"use client";
import { useState } from "react";
import { api } from "@/lib/api";

export function StartDraftForm({ onCreated }: { onCreated: (id: string) => void }) {
  const [season, setSeason] = useState(2026);
  const [pos, setPos] = useState(1);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      onCreated((await api.createSession(season, pos)).session_id);
    } catch (e2) {
      setErr(e2 instanceof Error ? e2.message : "failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="flex items-end gap-3">
      <label className="flex flex-col text-sm">
        Season
        <input
          type="number"
          value={season}
          onChange={(e) => setSeason(+e.target.value)}
          className="border rounded px-2 py-1"
        />
      </label>
      <label className="flex flex-col text-sm">
        Draft slot
        <input
          type="number"
          min={1}
          max={12}
          value={pos}
          onChange={(e) => setPos(+e.target.value)}
          className="border rounded px-2 py-1"
        />
      </label>
      <button disabled={busy} className="bg-emerald-600 text-white rounded px-4 py-1.5">
        {busy ? "Starting…" : "Start draft"}
      </button>
      {err && <span className="text-red-600 text-sm">{err}</span>}
    </form>
  );
}
