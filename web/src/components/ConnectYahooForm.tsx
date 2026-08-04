"use client";
import { useState } from "react";
import { api } from "@/lib/api";

export function ConnectYahooForm() {
  const [season, setSeason] = useState(2026);
  const [draftPosition, setDraftPosition] = useState<number | "">("");
  const [leagueKey, setLeagueKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (draftPosition === "" || !leagueKey.trim()) {
      setErr("Draft slot and league key are required");
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      const session = await api.createSession(season, draftPosition);
      window.location.href = api.yahooLoginUrl(session.session_id, leagueKey.trim());
    } catch (e2) {
      setErr(e2 instanceof Error ? e2.message : "failed to start");
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="flex flex-col gap-4 max-w-sm">
      <label className="flex flex-col text-sm gap-1">
        Season
        <input
          type="number"
          value={season}
          onChange={(e) => setSeason(+e.target.value)}
          className="border rounded px-2 py-1.5"
        />
      </label>
      <label className="flex flex-col text-sm gap-1">
        Your draft slot
        <input
          type="number"
          min={1}
          placeholder="e.g. 4 — check Yahoo's draft room once order is set"
          value={draftPosition}
          onChange={(e) => setDraftPosition(e.target.value === "" ? "" : +e.target.value)}
          className="border rounded px-2 py-1.5"
        />
      </label>
      <label className="flex flex-col text-sm gap-1">
        Yahoo league key
        <input
          type="text"
          placeholder="461.l.123456"
          value={leagueKey}
          onChange={(e) => setLeagueKey(e.target.value)}
          className="border rounded px-2 py-1.5"
        />
      </label>
      <button
        disabled={busy}
        className="bg-emerald-600 text-white rounded px-4 py-2 font-medium"
      >
        {busy ? "Connecting…" : "Connect Yahoo"}
      </button>
      {err && <span className="text-red-600 text-sm">{err}</span>}
    </form>
  );
}
