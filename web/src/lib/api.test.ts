import { describe, it, expect, vi, beforeEach } from "vitest";
import { api } from "./api";

const okJson = (body: unknown) =>
  Promise.resolve({
    ok: true,
    status: 200,
    json: () => Promise.resolve(body),
  } as Response);

beforeEach(() => vi.restoreAllMocks());

describe("api client", () => {
  it("createSession posts season + draft_position and returns state", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockReturnValue(okJson({ session_id: "abc", current_overall_pick: 1 }) as ReturnType<typeof fetch>);
    const st = await api.createSession(2026, 4);
    expect(st.session_id).toBe("abc");
    const [url, init] = spy.mock.calls[0];
    expect(String(url)).toContain("/draft/sessions");
    expect(JSON.parse(String((init as RequestInit).body))).toEqual({
      season: 2026,
      draft_position: 4,
    });
  });

  it("botPick posts to the bot-pick route", async () => {
    const spy = vi
      .spyOn(globalThis, "fetch")
      .mockReturnValue(okJson({ session_id: "abc" }) as ReturnType<typeof fetch>);
    await api.botPick("abc");
    expect(String(spy.mock.calls[0][0])).toContain("/draft/sessions/abc/bot-pick");
  });

  it("raises ApiError with status + detail on non-2xx", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: false,
      status: 400,
      json: () => Promise.resolve({ detail: "bad pick" }),
    } as Response);
    await expect(api.undo("x")).rejects.toMatchObject({
      name: "ApiError",
      status: 400,
      detail: "bad pick",
    });
  });
});
