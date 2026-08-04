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

it("loads state and recommendations for a session", async () => {
  vi.spyOn(api, "getState").mockResolvedValue({
    session_id: "s",
    status: "active",
    is_my_turn: true,
    picks: [],
    my_roster: [],
    current_overall_pick: 1,
  } as never);
  vi.spyOn(api, "recommendations").mockResolvedValue([
    { player_id: "P1", vona_score: 1 } as never,
  ]);
  const { result } = renderHook(() => useDraftState("s"), { wrapper });
  await waitFor(() => expect(result.current.state?.session_id).toBe("s"));
  await waitFor(() => expect(result.current.recs?.[0].player_id).toBe("P1"));
});

it("is disabled with a null session", () => {
  const getState = vi.spyOn(api, "getState");
  renderHook(() => useDraftState(null), { wrapper });
  expect(getState).not.toHaveBeenCalled();
});
