/**
 * The entry point. Two things this app does, stated once.
 *
 * The cockpit and My Board are genuinely different modes — one is a live
 * decision surface on a pick clock, the other is research you do beforehand —
 * so putting them behind one screen is honest about that rather than burying
 * the second in a tab of the first.
 */
import { useEffect, useState } from "react";
import { api } from "../api";
import { navigate } from "../route";
import type { BoardSummary, League } from "../types";

function Card({
  title,
  blurb,
  detail,
  cta,
  onClick,
}: {
  title: string;
  blurb: string;
  detail: string;
  cta: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="group flex cursor-pointer flex-col rounded-lg border
                 border-line bg-surface p-6 text-left transition-colors
                 hover:border-primary/60 focus:outline-none focus-visible:ring-2
                 focus-visible:ring-primary"
    >
      <h2 className="font-sans text-xl text-ink">{title}</h2>
      <p className="mt-2 text-sm text-muted">{blurb}</p>
      <p className="mt-4 flex-1 font-mono text-xs text-muted/80">{detail}</p>
      <span className="mt-5 text-sm text-primary transition-colors
                       group-hover:text-primary/80">
        {cta} →
      </span>
    </button>
  );
}

export function HomeScreen() {
  const [league, setLeague] = useState<League | null>(null);
  const [boards, setBoards] = useState<BoardSummary[]>([]);

  useEffect(() => {
    void api.league().then(setLeague).catch(() => setLeague(null));
    void api
      .boards()
      .then((r) => setBoards(r.boards))
      .catch(() => setBoards([]));
  }, []);

  const active = league?.active === true;
  const boardCount = boards.length;

  return (
    <div className="mx-auto max-w-4xl px-6 py-16">
      <header className="mb-10">
        <h1 className="font-sans text-3xl text-ink">FantasyForecast</h1>
        <p className="mt-2 text-sm text-muted">
          {league
            ? `${league.teams}-team PPR · ${league.rounds} rounds · ${league.players} players on the board`
            : "Loading league…"}
        </p>
      </header>

      <div className="grid gap-5 sm:grid-cols-2">
        <Card
          title="Draft cockpit"
          blurb="Live recommendations priced in expected league winnings."
          detail={
            active
              ? "A draft is in progress — resume it."
              : "Start a session when you're on the clock."
          }
          cta={active ? "Resume draft" : "Start a draft"}
          onClick={() => navigate("#/draft")}
        />
        <Card
          title="My Board"
          blurb="Your own tiers, built by hand. The engine never reads them."
          detail={
            boardCount
              ? `${boardCount} saved board${boardCount === 1 ? "" : "s"}`
              : "No boards yet — start one from ADP or the engine."
          }
          cta={boardCount ? "Open a board" : "Build a board"}
          onClick={() => navigate("#/board")}
        />
      </div>

      {league && (
        <p className="mt-10 font-mono text-xs text-muted/70">
          snapshot {league.snapshot_id}
        </p>
      )}
    </div>
  );
}
