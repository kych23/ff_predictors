"use client";

function pct(v: number, min: number, max: number): number {
  if (max <= min) return 0;
  return Math.max(0, Math.min(100, ((v - min) / (max - min)) * 100));
}

export function QuantileBar({
  p10,
  p50,
  p90,
  min,
  max,
}: {
  p10: number;
  p50: number;
  p90: number;
  min: number;
  max: number;
}) {
  const left = pct(p10, min, max);
  const right = pct(p90, min, max);
  const mid = pct(p50, min, max);
  return (
    <div
      data-testid="quantile-bar"
      data-p50-pct={String(Math.round(mid))}
      className="relative h-3 w-full rounded bg-slate-200"
    >
      <div
        className="absolute h-3 rounded bg-emerald-300"
        style={{ left: `${left}%`, width: `${Math.max(0, right - left)}%` }}
      />
      <div
        className="absolute top-[-2px] h-4 w-0.5 bg-emerald-700"
        style={{ left: `${mid}%` }}
        title={`P50 ${p50.toFixed(1)}`}
      />
    </div>
  );
}
