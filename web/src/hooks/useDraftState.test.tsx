import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { vi, it, expect, beforeEach } from "vitest";
import React from "react";
import { useDraftState } from "./useDraftState";
import { api } from "@/lib/api";

function wrapper({ children }: { children: React.ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return React.createElement(QueryClientProvider, { client: qc }, children);
}

beforeEach(() => vi.restoreAllMocks());

it("loads state and recommendations for a session via sync", async () => {
  vi.spyOn(api, "sync").mockResolvedValue({
    session_id: "s",
    status: "active",
    is_my_turn: true,
    picks: [],
    my_roster: [],
    current_overall_pick: 1,
    teams: 12,
    rounds: 15,
  } as never);
  vi.spyOn(api, "recommendations").mockResolvedValue([
    { player_id: "P1", vona_score: 1 } as never,
  ]);
  const { result } = renderHook(() => useDraftState("s"), { wrapper });
  await waitFor(() => expect(result.current.state?.session_id).toBe("s"));
  await waitFor(() => expect(result.current.recs?.[0].player_id).toBe("P1"));
  await waitFor(() => expect(result.current.lastSyncedAt).not.toBeNull());
});

it("is disabled with a null session", () => {
  const sync = vi.spyOn(api, "sync");
  renderHook(() => useDraftState(null), { wrapper });
  expect(sync).not.toHaveBeenCalled();
});
