import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { vi, it, expect, beforeEach } from "vitest";
import React from "react";
import { DraftRoom } from "./DraftRoom";
import { api } from "@/lib/api";

const wrap = (ui: React.ReactNode) => {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return React.createElement(QueryClientProvider, { client: qc }, ui);
};
beforeEach(() => vi.restoreAllMocks());

it("shows the start form when no session", () => {
  render(wrap(<DraftRoom sessionId={null} onSession={() => {}} />));
  expect(screen.getByRole("button", { name: /start draft/i })).toBeInTheDocument();
});

it("renders recs and turn banner for an active session", async () => {
  vi.spyOn(api, "getState").mockResolvedValue({
    session_id: "s",
    status: "active",
    is_my_turn: true,
    current_overall_pick: 1,
    picks: [],
    my_roster: [],
    open_starters: {},
    teams: 12,
    rounds: 15,
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
  render(wrap(<DraftRoom sessionId="s" onSession={() => {}} />));
  await waitFor(() => expect(screen.getByText("Star")).toBeInTheDocument());
  expect(screen.getByText(/your pick/i)).toBeInTheDocument();
});
