"use client";
import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ApiError, api } from "@/lib/api";

export function useDraftState(sessionId: string | null) {
  const enabled = !!sessionId;
  const key = ["draft-sync", sessionId];
  const [lastSyncedAt, setLastSyncedAt] = useState<number | null>(null);

  const stateQuery = useQuery({
    queryKey: key,
    enabled,
    queryFn: () => api.sync(sessionId as string),
    // Stop once the WHOLE draft is done (overall pick count exceeds
    // teams*rounds) — NOT once the connected user's own remaining_picks
    // hits 0, which happens up to teams-1 picks early for every seat
    // except the one that picks last. Keep retrying every 5s when there's
    // no data yet (including a failed first sync) — only a CONFIRMED
    // completed draft stops polling, never an error.
    refetchInterval: (q) => {
      const d = q.state.data;
      if (!d) return 5000;
      return d.current_overall_pick <= d.teams * d.rounds ? 5000 : false;
    },
  });

  useEffect(() => {
    if (stateQuery.isSuccess) setLastSyncedAt(Date.now());
  }, [stateQuery.isSuccess, stateQuery.dataUpdatedAt]);

  const recsQuery = useQuery({
    queryKey: ["recs", sessionId, stateQuery.data?.current_overall_pick],
    enabled: enabled && !!stateQuery.data,
    queryFn: () => api.recommendations(sessionId as string, 10),
  });

  const isUpstreamError =
    stateQuery.error instanceof ApiError && stateQuery.error.status === 502;

  return {
    state: stateQuery.data,
    recs: recsQuery.data,
    isLoading: stateQuery.isLoading,
    error: stateQuery.error,
    isUpstreamError,
    lastSyncedAt,
  };
}
