import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi, it, expect } from "vitest";
import { RecommendationPanel } from "./RecommendationPanel";
import type { Recommendation } from "@/lib/types";

const rec = (over: Partial<Recommendation>): Recommendation => ({
  player_id: "P1",
  name: "Star WR",
  position: "WR",
  team: "SF",
  vona_score: 4.2,
  value: 18,
  p10: 12,
  p50: 18,
  p90: 24,
  adp: 5,
  draft_round: 1,
  target_quantile: 0.25,
  forced_completion: false,
  ...over,
});

it("renders recs and fires onDraft with player_id", async () => {
  const onDraft = vi.fn();
  render(<RecommendationPanel recs={[rec({})]} min={0} max={30} onDraft={onDraft} />);
  expect(screen.getByText("Star WR")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: /draft/i }));
  expect(onDraft).toHaveBeenCalledWith("P1");
});

it("badges forced_completion picks", () => {
  render(
    <RecommendationPanel
      recs={[rec({ forced_completion: true })]}
      min={0}
      max={30}
      onDraft={() => {}}
    />,
  );
  expect(screen.getByText(/must fill/i)).toBeInTheDocument();
});
