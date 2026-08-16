/**
 * The routing shell, and nothing else.
 *
 * The cockpit used to live here. It was moved to `components/DraftCockpit.tsx`
 * unchanged for one specific reason: it opens an `EventSource("/api/stream")`
 * at component scope. If it stayed mounted across routes, the My Board view
 * would hold an open SSE connection and a live subscriber queue in
 * `service.py` for as long as the tab was open. Rendering it only under
 * `#/draft` means the connection opens and closes with the route.
 */
import { DraftCockpit } from "./components/DraftCockpit";
import { HomeScreen } from "./components/HomeScreen";
import { RankingsView } from "./rankings/RankingsView";
import { navigate, useRoute } from "./route";

function BackBar({ label }: { label: string }) {
  return (
    <div className="border-b border-line bg-surface/60 px-4 py-2">
      <button
        type="button"
        onClick={() => navigate("#/")}
        className="cursor-pointer text-sm text-muted transition-colors
                   hover:text-ink"
      >
        ← {label}
      </button>
    </div>
  );
}

export default function App() {
  const route = useRoute();

  if (route.view === "draft") {
    return (
      <>
        <BackBar label="Home" />
        <DraftCockpit />
      </>
    );
  }

  if (route.view === "board") {
    return (
      <>
        <BackBar label="Home" />
        <RankingsView boardId={route.boardId} />
      </>
    );
  }

  return <HomeScreen />;
}
