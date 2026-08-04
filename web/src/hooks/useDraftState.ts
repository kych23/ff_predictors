"use client";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { PickBody } from "@/lib/types";

export function useDraftState(sessionId: string | null) {
  const qc = useQueryClient();
  const enabled = !!sessionId;
  const key = ["draft", sessionId];

  const stateQuery = useQuery({
    queryKey: key,
    enabled,
    queryFn: () => api.getState(sessionId as string),
    refetchInterval: (q) => (q.state.data?.status === "active" ? 5000 : false),
  });

  const recsQuery = useQuery({
    queryKey: ["recs", sessionId, stateQuery.data?.current_overall_pick],
    enabled,
    queryFn: () => api.recommendations(sessionId as string, 10),
  });

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: key });
    qc.invalidateQueries({ queryKey: ["recs", sessionId] });
  };

  const pick = useMutation({
    mutationFn: (body: PickBody) => api.recordPick(sessionId as string, body),
    onSuccess: invalidate,
  });
  const skip = useMutation({
    mutationFn: () => api.recordPick(sessionId as string, { skip: true }),
    onSuccess: invalidate,
  });
  const undo = useMutation({
    mutationFn: () => api.undo(sessionId as string),
    onSuccess: invalidate,
  });
  const botPick = useMutation({
    mutationFn: () => api.botPick(sessionId as string),
    onSuccess: invalidate,
  });

  return {
    state: stateQuery.data,
    recs: recsQuery.data,
    isLoading: stateQuery.isLoading,
    error: stateQuery.error,
    pick,
    skip,
    undo,
    botPick,
    recsQuery,
  };
}
