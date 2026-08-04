import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { vi, it, expect, beforeEach } from "vitest";
import React from "react";
import { DemoRoom } from "./DemoRoom";
import { api } from "@/lib/api";

const wrap = (ui: React.ReactNode) => {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return React.createElement(QueryClientProvider, { client: qc }, ui);
};
beforeEach(() => vi.restoreAllMocks());

it("auto-creates a session and shows recs on the user's turn", async () => {
  vi.spyOn(api, "createSession").mockResolvedValue({ session_id: "demo" } as never);
  vi.spyOn(api, "getState").mockResolvedValue({
    session_id: "demo",
    status: "active",
    is_my_turn: true,
    current_overall_pick: 1,
    remaining_picks: 15,
    picks: [],
    my_roster: [],
    open_starters: {},
    teams: 12,
    rounds: 15,
    draft_position: 1,
  } as never);
  vi.spyOn(api, "recommendations").mockResolvedValue([
    {
      player_id: "P1",
      name: "Star",
      position: "WR",
      team: "SF",
      vona_score: 4,
      value: 18,
      p10: 12,
      p50: 18,
      p90: 24,
      adp: 5,
      draft_round: 1,
      target_quantile: 0.25,
      forced_completion: false,
    } as never,
  ]);
  render(wrap(<DemoRoom />));
  await waitFor(() => expect(screen.getByText("Star")).toBeInTheDocument());
});
