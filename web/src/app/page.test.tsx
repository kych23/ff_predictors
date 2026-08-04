import { render, screen } from "@testing-library/react";
import { it, expect } from "vitest";
import Home from "./page";

it("links to the connect flow", () => {
  render(<Home />);
  expect(screen.getByRole("link", { name: /connect your league/i })).toHaveAttribute(
    "href",
    "/draft"
  );
});
