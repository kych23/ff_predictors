/**
 * Overall plus one tab per position.
 *
 * The active tab is component state, deliberately NOT part of the route:
 * clicking through QB/RB/WR while building a board would otherwise stack seven
 * history entries and Back would walk them one at a time instead of leaving.
 *
 * Each scope is an independent list. A player who is RB3 in your RB tiers and
 * RB7 in overall is a legitimate state — that is a thing you are allowed to be
 * undecided about, and forcing the two into agreement would erase the
 * disagreement you were reasoning about.
 */
import { POSITION_TEXT } from "../positions";
import { Board, SCOPES, ScopeName } from "./model";

export function ScopeTabs({
  scope,
  board,
  onSelect,
}: {
  scope: ScopeName;
  board: Board;
  onSelect: (scope: ScopeName) => void;
}) {
  return (
    <nav className="flex gap-1 border-b border-line px-4">
      {SCOPES.map((name) => {
        const count = board.scopes[name].tiers.reduce(
          (acc, tier) => acc + tier.player_ids.length,
          0,
        );
        const active = name === scope;
        return (
          <button
            key={name}
            type="button"
            onClick={() => onSelect(name)}
            className={`cursor-pointer border-b-2 px-3 py-2 text-sm
                        transition-colors ${
                          active
                            ? "border-primary text-ink"
                            : "border-transparent text-muted hover:text-ink"
                        }`}
          >
            <span className={name === "overall" ? "" : POSITION_TEXT[name]}>
              {name === "overall" ? "Overall" : name}
            </span>
            <span className="ml-1.5 font-mono text-[10px] text-muted">
              {count || ""}
            </span>
          </button>
        );
      })}
    </nav>
  );
}
