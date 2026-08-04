import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi, it, expect, beforeEach, afterEach } from "vitest";
import { ConnectYahooForm } from "./ConnectYahooForm";
import { api } from "@/lib/api";

const originalLocation = window.location;

beforeEach(() => {
  vi.restoreAllMocks();
  // jsdom's window.location isn't assignable by default; stub it so the
  // component's `window.location.href = ...` redirect is observable.
  Object.defineProperty(window, "location", {
    configurable: true,
    value: { ...originalLocation, href: "" },
  });
});

afterEach(() => {
  Object.defineProperty(window, "location", {
    configurable: true,
    value: originalLocation,
  });
});

it("blocks submit when draft slot or league key is empty", async () => {
  const createSession = vi.spyOn(api, "createSession");
  render(<ConnectYahooForm />);
  await userEvent.click(screen.getByRole("button", { name: /connect yahoo/i }));
  expect(createSession).not.toHaveBeenCalled();
  expect(screen.getByText(/required/i)).toBeInTheDocument();
});

it("creates a session then redirects to the Yahoo login URL", async () => {
  vi.spyOn(api, "createSession").mockResolvedValue({ session_id: "sess1" } as never);

  render(<ConnectYahooForm />);
  await userEvent.type(screen.getByLabelText(/draft slot/i), "4");
  await userEvent.type(screen.getByLabelText(/league key/i), "461.l.123456");
  await userEvent.click(screen.getByRole("button", { name: /connect yahoo/i }));

  expect(api.createSession).toHaveBeenCalledWith(2026, 4);
  expect(window.location.href).toBe(api.yahooLoginUrl("sess1", "461.l.123456"));
});

it("shows an error message if session creation fails", async () => {
  vi.spyOn(api, "createSession").mockRejectedValue(new Error("boom"));
  render(<ConnectYahooForm />);
  await userEvent.type(screen.getByLabelText(/draft slot/i), "4");
  await userEvent.type(screen.getByLabelText(/league key/i), "461.l.123456");
  await userEvent.click(screen.getByRole("button", { name: /connect yahoo/i }));
  expect(await screen.findByText("boom")).toBeInTheDocument();
});
