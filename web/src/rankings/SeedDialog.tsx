/**
 * Seeding a scope. The available methods depend on the scope, and the reasons
 * are worth saying out loud rather than hiding behind a disabled control.
 *
 * `overall` cannot use raw projected points: per-game value is not comparable
 * across positions when only one quarterback starts, and sorting the live
 * board by it puts 24 QBs in the top 40 with Daniel Jones (ADP 151) at 18th.
 * It also cannot use the engine's tiers, which are computed per position —
 * "tier 2 RB" and "tier 2 WR" are unrelated numbers.
 *
 * K and DST cannot use any projection sort at all: every kicker shares one
 * projected value and every defence shares another, so the ordering would be
 * player-id order wearing a projection's name.
 *
 * The server enforces all of this; this dialog just doesn't offer what would
 * be refused, and says why when asked.
 */
import { useState } from "react";
import { ScopeName } from "./model";

interface Method {
  id: string;
  label: string;
  blurb: string;
}

const ADP: Method = {
  id: "adp",
  label: "Market ADP",
  blurb: "Where the field is actually drafting them.",
};
const VOR: Method = {
  id: "engine_vor",
  label: "Engine, over replacement",
  blurb: "Projected points minus the replacement level at that position.",
};
const VALUE: Method = {
  id: "engine_value",
  label: "Engine, raw points",
  blurb: "Straight projected points per game. Comparable within a position.",
};
const TIERS: Method = {
  id: "engine_tiers",
  label: "Engine tiers",
  blurb: "The engine's own tier breaks for this position.",
};
const OVERALL: Method = {
  id: "from_overall",
  label: "From my overall list",
  blurb: "This position, in the order you already put them in Overall.",
};

function methodsFor(scope: ScopeName): Method[] {
  if (scope === "overall") return [ADP, VOR];
  if (scope === "K" || scope === "DST") return [ADP, TIERS, OVERALL];
  return [ADP, VOR, VALUE, TIERS, OVERALL];
}

const WHY: Partial<Record<ScopeName, string>> = {
  overall:
    "Raw projected points and engine tiers are both per-position measures, so neither can order a whole board.",
  K: "Every kicker shares one projected value, so a projection sort would be arbitrary.",
  DST: "Every defence shares one projected value, so a projection sort would be arbitrary.",
};

export function SeedDialog({
  scope,
  hasContent,
  onSeed,
  onCancel,
}: {
  scope: ScopeName;
  hasContent: boolean;
  onSeed: (method: string, tierSize: number | null) => void;
  onCancel: () => void;
}) {
  const [method, setMethod] = useState<string>("adp");
  const [size, setSize] = useState<string>("");

  const methods = methodsFor(scope);
  const why = WHY[scope];

  return (
    <div className="fixed inset-0 z-30 flex items-center justify-center
                    bg-black/60 p-4">
      <div className="w-full max-w-md rounded-lg border border-line
                      bg-surface p-5">
        <h2 className="font-sans text-lg text-ink">
          Seed {scope === "overall" ? "Overall" : scope}
        </h2>
        {hasContent && (
          <p className="mt-2 rounded border border-warn/40 bg-warn/10 px-3
                        py-2 text-xs text-warn">
            This replaces everything currently in {scope}. Other scopes are
            untouched.
          </p>
        )}

        <div className="mt-4 space-y-2">
          {methods.map((m) => (
            <label
              key={m.id}
              className="flex cursor-pointer gap-3 rounded border
                         border-line p-3 transition-colors hover:bg-white/5"
            >
              <input
                type="radio"
                name="seed-method"
                value={m.id}
                checked={method === m.id}
                onChange={() => setMethod(m.id)}
                className="mt-1 cursor-pointer"
              />
              <span>
                <span className="block text-sm text-ink">{m.label}</span>
                <span className="block text-xs text-muted">{m.blurb}</span>
              </span>
            </label>
          ))}
        </div>

        {why && <p className="mt-3 text-xs text-muted">{why}</p>}

        <label className="mt-4 flex items-center gap-2 text-xs text-muted">
          Players per tier
          <input
            type="number"
            min={2}
            max={50}
            value={size}
            onChange={(e) => setSize(e.target.value)}
            placeholder={scope === "overall" ? "12" : "6"}
            className="w-16 rounded border border-line bg-bg px-2 py-1
                       text-ink outline-none focus:border-primary"
          />
          {method === "engine_tiers" && (
            <span>ignored — the artifact defines the breaks</span>
          )}
        </label>

        <div className="mt-5 flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="cursor-pointer px-3 py-1.5 text-sm text-muted
                       transition-colors hover:text-ink"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => onSeed(method, size ? Number(size) : null)}
            className="cursor-pointer rounded bg-primary px-3 py-1.5 text-sm
                       text-white transition-colors hover:bg-primary/80"
          >
            Seed
          </button>
        </div>
      </div>
    </div>
  );
}
