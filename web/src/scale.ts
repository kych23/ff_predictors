/**
 * Domain -> pixel mapping for the candidate dot-plot.
 *
 * Split out of the component because it is the one piece of chart code with a
 * real edge case: when every candidate has the same estimate the domain has
 * zero width, and the naive `(v - lo) / (hi - lo)` divides by zero and paints
 * NaN into the DOM. That happens in practice — a demoted tier reports no
 * dollars at all, so every value is null and the fallbacks collapse.
 */
export interface Scale {
  lo: number;
  hi: number;
  /** 0..1 position of a value within the padded domain. */
  at: (value: number) => number;
}

export function makeScale(values: number[], pad = 0.15): Scale {
  const finite = values.filter((v) => Number.isFinite(v));
  if (finite.length === 0) return { lo: 0, hi: 1, at: () => 0.5 };

  let lo = Math.min(...finite);
  let hi = Math.max(...finite);

  if (hi - lo < 1e-9) {
    // Zero-width domain: centre everything rather than divide by zero.
    const nudge = Math.max(Math.abs(hi) * 0.05, 0.5);
    lo -= nudge;
    hi += nudge;
  } else {
    const span = hi - lo;
    lo -= span * pad;
    hi += span * pad;
  }

  return {
    lo,
    hi,
    at: (value: number) => {
      if (!Number.isFinite(value)) return 0.5;
      return Math.min(1, Math.max(0, (value - lo) / (hi - lo)));
    },
  };
}
