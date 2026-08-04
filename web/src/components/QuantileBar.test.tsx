import { render, screen } from "@testing-library/react";
import { it, expect } from "vitest";
import { QuantileBar } from "./QuantileBar";

it("positions p50 marker proportionally within [min,max]", () => {
  render(<QuantileBar p10={5} p50={10} p90={15} min={0} max={20} />);
  const bar = screen.getByTestId("quantile-bar");
  expect(bar.getAttribute("data-p50-pct")).toBe("50"); // 10 of [0,20]
});

it("clamps to [0,100] when values exceed range", () => {
  render(<QuantileBar p10={-5} p50={25} p90={30} min={0} max={20} />);
  const bar = screen.getByTestId("quantile-bar");
  expect(Number(bar.getAttribute("data-p50-pct"))).toBeLessThanOrEqual(100);
});
