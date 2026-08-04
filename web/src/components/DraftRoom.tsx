"use client";
import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useDraftState } from "@/hooks/useDraftState";
import { RecommendationPanel } from "./RecommendationPanel";
import { RosterPanel } from "./RosterPanel";
import { PickLog } from "./PickLog";

const CONNECT_ERROR_MESSAGES: Record<string, string> = {
  team_not_found: "Couldn't find a team you own in that league — check the league key.",
  oauth_failed: "Yahoo authorization failed. Try connecting again.",
  invalid_request: "That connection link was invalid or expired.",
};

function secondsAgo(ts: number): number {
  return Math.max(0, Math.round((Date.now() - ts) / 1000));
}

export function DraftRoom({
  sessionId,
  connected,
  connectError,
}: {
  sessionId: string;
  connected: boolean;
  connectError: string | null;
}) {
  const [bannerDismissed, setBannerDismissed] = useState(false);

  if (connectError) {
    return (
      <div className="p-8 max-w-md">
        <h1 className="text-xl font-bold mb-2 text-red-700">Connection failed</h1>
        <p className="text-slate-600 mb-4">
          {CONNECT_ERROR_MESSAGES[connectError] ?? "Something went wrong connecting to Yahoo."}
        </p>
        <Link href="/draft" className="text-emerald-700 underline">
          Back to connect form
        </Link>
      </div>
    );
  }

  return (
    <DraftRoomLive
      sessionId={sessionId}
      showConnectedBanner={connected && !bannerDismissed}
      onDismissBanner={() => setBannerDismissed(true)}
    />
  );
}

function DraftRoomLive({
  sessionId,
  showConnectedBanner,
  onDismissBanner,
}: {
  sessionId: string;
  showConnectedBanner: boolean;
  onDismissBanner: () => void;
}) {
  const d = useDraftState(sessionId);
  const [min, max] = useMemo(() => {
    const ps = (d.recs ?? []).flatMap((r) => [r.p10, r.p90]);
    return ps.length ? [Math.min(...ps), Math.max(...ps)] : [0, 30];
  }, [d.recs]);
  const [nowTick, setNowTick] = useState(Date.now());

  useEffect(() => {
    const id = setInterval(() => setNowTick(Date.now()), 5000);
    return () => clearInterval(id);
  }, []);

  if (d.isLoading) return <p className="p-8">Loading draft…</p>;
  if (!d.state) {
    // First sync attempt failed before ever getting data — show the same
    // sync-error messaging DraftRoomLive uses once it has state, rather
    // than getting stuck on "Loading draft..." forever.
    return (
      <div className="p-8">
        {d.isUpstreamError ? (
          <p className="text-amber-800">
            Yahoo connection lost — reconnect from the connect form to keep syncing.
          </p>
        ) : (
          <p className="text-slate-600">Sync error, retrying…</p>
        )}
      </div>
    );
  }
  const s = d.state;
  const stale = d.lastSyncedAt !== null && nowTick - d.lastSyncedAt > 15000;

  return (
    <div className="p-6 grid grid-cols-1 lg:grid-cols-[1fr_20rem] gap-6">
      <section>
        {showConnectedBanner && (
          <div className="mb-4 bg-emerald-50 border border-emerald-200 text-emerald-800 rounded px-3 py-2 text-sm flex justify-between items-center">
            <span>Connected to Yahoo — syncing live.</span>
            <button onClick={onDismissBanner} className="underline">
              dismiss
            </button>
          </div>
        )}
        {d.isUpstreamError && (
          <div className="mb-4 bg-amber-50 border border-amber-200 text-amber-800 rounded px-3 py-2 text-sm">
            Yahoo connection lost — reconnect from the connect form to keep syncing.
          </div>
        )}
        {!d.isUpstreamError && d.error && (
          <div className="mb-4 bg-slate-50 border border-slate-200 text-slate-600 rounded px-3 py-2 text-sm">
            Sync error, retrying…
          </div>
        )}
        {!d.error && stale && d.lastSyncedAt && (
          <div className="mb-4 text-xs text-slate-500">
            Last synced {secondsAgo(d.lastSyncedAt)}s ago
          </div>
        )}
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-xl font-bold">
            Pick {s.current_overall_pick} ·{" "}
            <span className={s.is_my_turn ? "text-emerald-700" : "text-slate-500"}>
              {s.is_my_turn ? "Your pick" : "Waiting on other teams"}
            </span>
          </h1>
        </div>
        <RecommendationPanel recs={d.recs ?? []} min={min} max={max} />
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
