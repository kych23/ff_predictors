import { render, screen } from "@testing-library/react";
import { it, expect } from "vitest";
import Home from "./page";

it("links to the demo and the live draft", () => {
  render(<Home />);
  expect(screen.getByRole("link", { name: /try the demo/i })).toHaveAttribute("href", "/demo");
  expect(screen.getByRole("link", { name: /live draft/i })).toHaveAttribute("href", "/draft");
});
