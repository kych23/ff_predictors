import { describe, expect, it } from "vitest";
import { makeScale } from "./scale";

describe("makeScale", () => {
  it("maps the domain into 0..1", () => {
    const s = makeScale([10, 20]);
    expect(s.at(10)).toBeGreaterThan(0);
    expect(s.at(20)).toBeLessThan(1);
    expect(s.at(10)).toBeLessThan(s.at(20));
  });

  it("survives a zero-width domain instead of dividing by zero", () => {
    const s = makeScale([44.64, 44.64, 44.64]);
    expect(Number.isFinite(s.at(44.64))).toBe(true);
    expect(s.at(44.64)).toBeCloseTo(0.5, 5);
  });

  it("survives an empty domain", () => {
    const s = makeScale([]);
    expect(s.at(1)).toBe(0.5);
  });

  it("ignores non-finite values rather than poisoning the domain", () => {
    const s = makeScale([10, NaN, 20, Infinity]);
    expect(Number.isFinite(s.lo)).toBe(true);
    expect(Number.isFinite(s.hi)).toBe(true);
  });

  it("clamps out-of-domain values", () => {
    const s = makeScale([10, 20]);
    expect(s.at(-100)).toBe(0);
    expect(s.at(1000)).toBe(1);
  });
});
