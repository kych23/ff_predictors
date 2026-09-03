/**
 * Yahoo Fantasy attribution, required by the API Access and Use Agreement.
 *
 * The obligation is specific: for web applications, attribution "must appear
 * in the footer of each page where Yahoo Fantasy Information is displayed and
 * must include a hyperlink to an official Yahoo Fantasy webpage."
 *
 * It renders on EVERY view rather than only the ones currently showing Yahoo
 * data. Two reasons. The board ordering is already Yahoo-derived — the Flock
 * export's `rank_yahoo` column is what `adp` is built from — so the cockpit
 * and My Board both display Yahoo-sourced information today, before the API is
 * wired at all. And once live draft results arrive over the API, every surface
 * can show them; a conditional footer would be one refactor away from being
 * quietly wrong. Over-attributing costs a line of grey text. Under-attributing
 * breaches the agreement.
 */
export function YahooAttribution() {
  return (
    <footer className="border-t border-line px-4 py-2 text-center text-[11px]
                       text-muted">
      Fantasy data provided by{" "}
      <a
        href="https://football.fantasysports.yahoo.com/"
        target="_blank"
        rel="noreferrer noopener"
        className="cursor-pointer underline transition-colors hover:text-ink"
      >
        Yahoo Fantasy
      </a>
    </footer>
  );
}
