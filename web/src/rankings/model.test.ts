import { describe, expect, it } from "vitest";
import {
  Board,
  SCOPES,
  ScopeName,
  Tier,
  addTier,
  deleteTier,
  flatten,
  movePlayer,
  moveTier,
  nextTierId,
  ranks,
  recolorTier,
  renameTier,
} from "./model";

function board(tiers: Array<[string, string[]]>): Board {
  const scopes = Object.fromEntries(
    SCOPES.map((s) => [s, { tiers: [] as Tier[] }]),
  ) as unknown as Board["scopes"];
  scopes.overall = {
    tiers: tiers.map(([id, players]) => ({
      id,
      label: id,
      color: "t1",
      player_ids: players,
    })),
  };
  return {
    schema_version: 1,
    rev: 1,
    board_id: "b",
    name: "B",
    created_at: "",
    updated_at: "",
    seeded_from: {},
    scopes,
  };
}

const O: ScopeName = "overall";

/** Every function must leave its input untouched — the view holds the previous
 * board in state and a mutation would corrupt it in place. */
function assertPure(before: Board, fn: () => void) {
  const snapshot = JSON.stringify(before);
  fn();
  expect(JSON.stringify(before)).toBe(snapshot);
}

/** The invariant the server refuses a save for. */
function assertNoDuplicates(b: Board, scope: ScopeName) {
  const all = flatten(b, scope);
  expect(new Set(all).size).toBe(all.length);
}

describe("movePlayer", () => {
  it("reorders within one tier", () => {
    const before = board([["t-1", ["a", "b", "c"]]]);
    const after = movePlayer(before, O, "c", "t-1", 0);
    expect(after.scopes.overall.tiers[0].player_ids).toEqual(["c", "a", "b"]);
  });

  it("moves across tiers", () => {
    const before = board([["t-1", ["a", "b"]], ["t-2", ["c"]]]);
    const after = movePlayer(before, O, "a", "t-2", 1);
    expect(after.scopes.overall.tiers[0].player_ids).toEqual(["b"]);
    expect(after.scopes.overall.tiers[1].player_ids).toEqual(["c", "a"]);
  });

  it("moves into an empty tier", () => {
    const before = board([["t-1", ["a"]], ["t-2", []]]);
    const after = movePlayer(before, O, "a", "t-2", 0);
    expect(after.scopes.overall.tiers[1].player_ids).toEqual(["a"]);
    expect(after.scopes.overall.tiers[0].player_ids).toEqual([]);
  });

  it("clamps an out-of-range index instead of dropping the player", () => {
    const before = board([["t-1", ["a", "b"]]]);
    expect(flatten(movePlayer(before, O, "a", "t-1", 99), O)).toEqual(["b", "a"]);
    expect(flatten(movePlayer(before, O, "b", "t-1", -5), O)).toEqual(["b", "a"]);
  });

  it("never leaves a duplicate", () => {
    const before = board([["t-1", ["a", "b"]], ["t-2", ["c"]]]);
    assertNoDuplicates(movePlayer(before, O, "a", "t-1", 1), O);
    assertNoDuplicates(movePlayer(before, O, "c", "t-1", 0), O);
  });

  it("is a no-op for an unknown target tier", () => {
    const before = board([["t-1", ["a"]]]);
    expect(movePlayer(before, O, "a", "nope", 0)).toBe(before);
  });

  it("does not mutate its input", () => {
    const before = board([["t-1", ["a", "b"]], ["t-2", []]]);
    assertPure(before, () => movePlayer(before, O, "a", "t-2", 0));
  });
});

describe("moveTier", () => {
  it("reorders tiers", () => {
    const before = board([["t-1", ["a"]], ["t-2", ["b"]], ["t-3", ["c"]]]);
    const after = moveTier(before, O, "t-3", 0);
    expect(after.scopes.overall.tiers.map((t) => t.id)).toEqual([
      "t-3", "t-1", "t-2",
    ]);
  });

  it("does not mutate its input", () => {
    const before = board([["t-1", ["a"]], ["t-2", ["b"]]]);
    assertPure(before, () => moveTier(before, O, "t-2", 0));
  });
});

describe("deleteTier", () => {
  it("relocates players into the tier below", () => {
    const before = board([["t-1", ["a"]], ["t-2", ["b"]], ["t-3", ["c"]]]);
    const after = deleteTier(before, O, "t-2");
    expect(after.scopes.overall.tiers.map((t) => t.id)).toEqual(["t-1", "t-3"]);
    expect(after.scopes.overall.tiers[1].player_ids).toEqual(["c", "b"]);
  });

  it("relocates upward when the deleted tier is last", () => {
    const before = board([["t-1", ["a"]], ["t-2", ["b"]]]);
    const after = deleteTier(before, O, "t-2");
    expect(after.scopes.overall.tiers[0].player_ids).toEqual(["a", "b"]);
  });

  it("never drops a player", () => {
    const before = board([["t-1", ["a"]], ["t-2", ["b", "c"]]]);
    expect(flatten(deleteTier(before, O, "t-1"), O).sort()).toEqual([
      "a", "b", "c",
    ]);
  });

  it("throws on the only tier rather than losing its players", () => {
    const before = board([["t-1", ["a", "b"]]]);
    expect(() => deleteTier(before, O, "t-1")).toThrow(/only tier/);
  });

  it("does not mutate its input", () => {
    const before = board([["t-1", ["a"]], ["t-2", ["b"]]]);
    assertPure(before, () => deleteTier(before, O, "t-1"));
  });
});

describe("renumbering", () => {
  it("closes the gap a delete opens", () => {
    const before = board([
      ["t-1", ["a"]], ["t-2", ["b"]], ["t-3", ["c"]], ["t-4", ["d"]],
    ]);
    // board() labels tiers by id; give them auto-shaped labels first.
    before.scopes.overall.tiers.forEach((t, i) => {
      t.label = `Tier ${i + 1}`;
    });
    const after = deleteTier(before, O, "t-2");
    expect(after.scopes.overall.tiers.map((t) => t.label)).toEqual([
      "Tier 1", "Tier 2", "Tier 3",
    ]);
  });

  it("renumbers after a tier is reordered", () => {
    const before = board([["t-1", ["a"]], ["t-2", ["b"]], ["t-3", ["c"]]]);
    before.scopes.overall.tiers.forEach((t, i) => {
      t.label = `Tier ${i + 1}`;
    });
    const after = moveTier(before, O, "t-3", 0);
    expect(after.scopes.overall.tiers.map((t) => t.label)).toEqual([
      "Tier 1", "Tier 2", "Tier 3",
    ]);
    // The MOVED tier keeps its identity, only its display number changed.
    expect(after.scopes.overall.tiers[0].id).toBe("t-3");
  });

  it("leaves a name you typed alone", () => {
    const before = board([["t-1", ["a"]], ["t-2", ["b"]], ["t-3", ["c"]]]);
    before.scopes.overall.tiers[0].label = "Tier 1";
    before.scopes.overall.tiers[1].label = "Elite";
    before.scopes.overall.tiers[2].label = "Tier 3";
    const after = deleteTier(before, O, "t-1");
    expect(after.scopes.overall.tiers.map((t) => t.label)).toEqual([
      "Elite", "Tier 2",
    ]);
  });

  it("does not rewrite a rename that happens to look auto-generated", () => {
    const before = board([["t-1", ["a"]], ["t-2", ["b"]], ["t-3", ["c"]]]);
    const after = renameTier(before, O, "t-2", "Tier 7");
    expect(after.scopes.overall.tiers[1].label).toBe("Tier 7");
  });

  it("gives an inserted tier the number its position earns", () => {
    const before = board([["t-1", ["a"]], ["t-2", ["b"]]]);
    before.scopes.overall.tiers.forEach((t, i) => {
      t.label = `Tier ${i + 1}`;
    });
    const after = addTier(before, O, 1, "t2");
    expect(after.scopes.overall.tiers.map((t) => t.label)).toEqual([
      "Tier 1", "Tier 2", "Tier 3",
    ]);
    expect(after.scopes.overall.tiers[1].id).toBe("t-3");
  });
});

describe("addTier", () => {
  it("mints max+1 within the scope", () => {
    const before = board([["t-1", []], ["t-7", []]]);
    expect(nextTierId(before, O)).toBe("t-8");
    expect(addTier(before, O, 1, "t2").scopes.overall.tiers[1].id).toBe("t-8");
  });

  it("is deterministic — no clock, no counter", () => {
    const before = board([["t-1", []]]);
    expect(addTier(before, O, 1, "t2").scopes.overall.tiers[1].id).toBe(
      addTier(before, O, 1, "t2").scopes.overall.tiers[1].id,
    );
  });

  it("inserts at the requested index", () => {
    const before = board([["t-1", []], ["t-2", []]]);
    expect(addTier(before, O, 0, "t3").scopes.overall.tiers[0].id).toBe("t-3");
  });

  it("does not mutate its input", () => {
    const before = board([["t-1", []]]);
    assertPure(before, () => addTier(before, O, 0, "t1"));
  });
});

describe("renameTier / recolorTier", () => {
  it("renames one tier and leaves the rest", () => {
    const before = board([["t-1", []], ["t-2", []]]);
    const after = renameTier(before, O, "t-1", "Elite");
    expect(after.scopes.overall.tiers[0].label).toBe("Elite");
    expect(after.scopes.overall.tiers[1].label).toBe("t-2");
  });

  it("recolors one tier", () => {
    const before = board([["t-1", []]]);
    expect(recolorTier(before, O, "t-1", "t5").scopes.overall.tiers[0].color)
      .toBe("t5");
  });

  it("does not mutate its input", () => {
    const before = board([["t-1", []]]);
    assertPure(before, () => renameTier(before, O, "t-1", "X"));
    assertPure(before, () => recolorTier(before, O, "t-1", "t8"));
  });
});

describe("scope independence", () => {
  it("editing overall leaves the position scopes alone", () => {
    const before = board([["t-1", ["a", "b"]]]);
    before.scopes.RB = {
      tiers: [{ id: "t-1", label: "T", color: "t1", player_ids: ["r1"] }],
    };
    const after = movePlayer(before, O, "b", "t-1", 0);
    expect(after.scopes.RB.tiers[0].player_ids).toEqual(["r1"]);
  });
});

describe("ranks", () => {
  it("numbers players across tiers, one-based", () => {
    const b = board([["t-1", ["a", "b"]], ["t-2", ["c"]]]);
    expect(Object.fromEntries(ranks(b, O))).toEqual({ a: 1, b: 2, c: 3 });
  });
});
