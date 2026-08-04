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

it("renders recs, roster, and turn banner from a synced session", async () => {
  vi.spyOn(api, "sync").mockResolvedValue({
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
  render(wrap(<DraftRoom sessionId="s" connected={false} connectError={null} />));
  await waitFor(() => expect(screen.getByText("Star")).toBeInTheDocument());
  expect(screen.getByText(/your pick/i)).toBeInTheDocument();
});

it("renders no pick/skip/undo/bot-pick controls (read-only view)", async () => {
  vi.spyOn(api, "sync").mockResolvedValue({
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
    { player_id: "P1", name: "Star", position: "WR", team: "SF", vona_score: 4,
      value: 18, p10: 12, p50: 18, p90: 24, adp: 5, draft_round: 1,
      target_quantile: 0.25, forced_completion: false } as never,
  ]);
  render(wrap(<DraftRoom sessionId="s" connected={false} connectError={null} />));
  await waitFor(() => expect(screen.getByText("Star")).toBeInTheDocument());
  expect(screen.queryByRole("button", { name: /draft/i })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /skip/i })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /undo/i })).not.toBeInTheDocument();
});

it("shows a connect-error banner and never calls sync", () => {
  const syncSpy = vi.spyOn(api, "sync");
  render(
    wrap(<DraftRoom sessionId="s" connected={false} connectError="team_not_found" />)
  );
  expect(screen.getByText(/couldn't find a team/i)).toBeInTheDocument();
  expect(syncSpy).not.toHaveBeenCalled();
});

it("shows a dismissible connected banner when connected=true", async () => {
  vi.spyOn(api, "sync").mockResolvedValue({
    session_id: "s",
    status: "active",
    is_my_turn: false,
    current_overall_pick: 1,
    picks: [],
    my_roster: [],
    open_starters: {},
    teams: 12,
    rounds: 15,
  } as never);
  vi.spyOn(api, "recommendations").mockResolvedValue([]);
  render(wrap(<DraftRoom sessionId="s" connected={true} connectError={null} />));
  await waitFor(() =>
    expect(screen.getByText(/connected to yahoo/i)).toBeInTheDocument()
  );
});
