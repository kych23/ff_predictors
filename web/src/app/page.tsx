import Link from "next/link";

export default function Home() {
  return (
    <main className="max-w-3xl mx-auto p-10">
      <h1 className="text-4xl font-bold mb-3">FantasyForecast</h1>
      <p className="text-lg text-slate-600 mb-8">
        A draft assistant powered by quantile projection models (P10/P50/P90) and a
        VONA recommender — it tells you who to draft and how risky each pick is.
      </p>
      <div className="flex gap-4">
        <Link
          href="/draft"
          className="bg-emerald-600 text-white rounded px-5 py-2.5 font-medium"
        >
          Connect your league
        </Link>
      </div>
    </main>
  );
}
