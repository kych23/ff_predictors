'use client';

import { useState, useEffect } from 'react';
import { api, type Prediction } from '@/lib/api';
import Nav from '@/components/Nav';

const currentYear = new Date().getFullYear();

export default function Home() {
  const [season, setSeason] = useState(currentYear);
  const [week, setWeek] = useState(1);
  const [position, setPosition] = useState<string>('');
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [weeks, setWeeks] = useState<number[]>([]);
  const [apiOnline, setApiOnline] = useState(true);
  const [initialized, setInitialized] = useState(false);

  useEffect(() => {
    const checkApi = async () => {
      const isOnline = await api.healthCheck();
      setApiOnline(isOnline);
      
      // Fetch current week on initial load
      if (isOnline && !initialized) {
        try {
          const current = await api.getCurrentWeek();
          setSeason(current.season);
          setWeek(current.week);
          setInitialized(true);
        } catch (err) {
          console.error('Failed to fetch current week:', err);
        }
      }
    };
    checkApi();
  }, [initialized]);

  useEffect(() => {
    const fetchWeeks = async () => {
      try {
        const data = await api.getWeeks(season);
        setWeeks(data);
        if (data.length > 0 && !data.includes(week)) {
          setWeek(data[0]);
        }
      } catch (err) {
        console.error('Failed to fetch weeks:', err);
      }
    };
    fetchWeeks();
  }, [season, week]);

  useEffect(() => {
    const fetchPredictions = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await api.getPredictions(season, week, position || undefined);
        setPredictions(data);
      } catch (err) {
        setError('Failed to load predictions. Make sure the backend is running.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    
    if (apiOnline) {
      fetchPredictions();
    }
  }, [season, week, position, apiOnline]);

  if (!apiOnline) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 dark:from-slate-900 dark:via-slate-900 dark:to-slate-900">
        <Nav />
        <div className="mx-auto max-w-7xl px-4 py-20 sm:px-6 lg:px-8">
          <div className="mx-auto max-w-2xl rounded-2xl border border-red-200 bg-red-50 p-8 text-center shadow-xl dark:border-red-900 dark:bg-red-950">
            <div className="mb-4 text-6xl">⚡</div>
            <h2 className="text-2xl font-bold text-red-900 dark:text-red-100 mb-2">
              Backend Not Connected
            </h2>
            <p className="text-red-700 dark:text-red-300">
              Please start the FastAPI backend server on port 8000
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 dark:from-slate-900 dark:via-slate-900 dark:to-slate-900">
      <Nav />
      
      {/* Hero Section */}
      <div className="mx-auto max-w-7xl px-4 pt-12 pb-8 sm:px-6 lg:px-8">
        <div className="text-center mb-12 animate-fadeInUp">
          <h1 className="text-5xl font-extrabold gradient-text mb-4">
            Fantasy Predictions
          </h1>
          <div className="inline-block glass rounded-full px-6 py-2 mb-4 shadow-lg">
            <span className="text-lg font-bold text-slate-700 dark:text-slate-300">
              {season} Season • Week {week}
            </span>
          </div>
          <p className="text-xl text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
            AI-powered projections for every NFL player. Make smarter lineup decisions.
          </p>
        </div>

        {/* Filters */}
        <div className="glass rounded-2xl p-6 shadow-xl mb-8 animate-fadeInUp">
          <div className="flex flex-wrap items-end gap-4">
            <div className="flex-1 min-w-[200px]">
              <label htmlFor="season" className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
                Season
              </label>
              <input
                id="season"
                type="number"
                value={season}
                onChange={(e) => setSeason(parseInt(e.target.value))}
                className="w-full rounded-xl border-2 border-slate-200 bg-white px-4 py-3 shadow-sm transition-all focus:border-indigo-500 focus:ring-4 focus:ring-indigo-500/20 dark:border-slate-700 dark:bg-slate-800 dark:text-white"
                min="2012"
                max={currentYear + 1}
              />
            </div>
            
            <div className="flex-1 min-w-[200px]">
              <label htmlFor="week" className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
                Week
              </label>
              <select
                id="week"
                value={week}
                onChange={(e) => setWeek(parseInt(e.target.value))}
                className="w-full rounded-xl border-2 border-slate-200 bg-white px-4 py-3 shadow-sm transition-all focus:border-indigo-500 focus:ring-4 focus:ring-indigo-500/20 dark:border-slate-700 dark:bg-slate-800 dark:text-white"
              >
                {weeks.map((w) => (
                  <option key={w} value={w}>
                    Week {w}
                  </option>
                ))}
              </select>
            </div>
            
            <div className="flex-1 min-w-[200px]">
              <label htmlFor="position" className="block text-sm font-semibold text-slate-700 dark:text-slate-300 mb-2">
                Position
              </label>
              <select
                id="position"
                value={position}
                onChange={(e) => setPosition(e.target.value)}
                className="w-full rounded-xl border-2 border-slate-200 bg-white px-4 py-3 shadow-sm transition-all focus:border-indigo-500 focus:ring-4 focus:ring-indigo-500/20 dark:border-slate-700 dark:bg-slate-800 dark:text-white"
              >
                <option value="">All Positions</option>
                <option value="QB">Quarterback</option>
                <option value="RB">Running Back</option>
                <option value="WR">Wide Receiver</option>
                <option value="TE">Tight End</option>
              </select>
            </div>
          </div>
        </div>

        {error && (
          <div className="mb-8 rounded-2xl border-2 border-red-200 bg-red-50 p-6 shadow-xl dark:border-red-900 dark:bg-red-950 animate-fadeInUp">
            <p className="text-red-700 dark:text-red-300 font-medium">{error}</p>
          </div>
        )}

        {/* Results */}
        {loading ? (
          <div className="glass rounded-2xl p-20 text-center shadow-xl animate-fadeInUp">
            <div className="inline-block">
              <div className="mx-auto mb-4 h-16 w-16 animate-spin rounded-full border-4 border-indigo-200 border-t-indigo-600"></div>
              <p className="text-slate-600 dark:text-slate-400 font-medium">Loading predictions...</p>
            </div>
          </div>
        ) : predictions.length > 0 ? (
          <div className="glass rounded-2xl shadow-xl overflow-hidden animate-fadeInUp">
            {/* Header */}
            <div className="bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-4">
              <div className="flex items-center justify-between text-white">
                <div className="flex items-center space-x-2">
                  <span className="text-2xl">🏆</span>
                  <h2 className="text-xl font-bold">Top Projections</h2>
                </div>
                <div className="text-sm opacity-90">
                  {predictions.length} players
                </div>
              </div>
            </div>

            {/* Table */}
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-slate-50 dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700">
                  <tr>
                    <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider text-slate-600 dark:text-slate-400">
                      Rank
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider text-slate-600 dark:text-slate-400">
                      Player
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider text-slate-600 dark:text-slate-400">
                      Pos
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider text-slate-600 dark:text-slate-400">
                      Team
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider text-slate-600 dark:text-slate-400">
                      vs Opponent
                    </th>
                    <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider text-slate-600 dark:text-slate-400">
                      Points
                    </th>
                    {predictions[0].ci_lower && (
                      <th className="px-6 py-4 text-left text-xs font-bold uppercase tracking-wider text-slate-600 dark:text-slate-400">
                        Range
                      </th>
                    )}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 dark:divide-slate-700 bg-white dark:bg-slate-900">
                  {predictions.map((pred, idx) => {
                    const isTop = idx < 3;
                    return (
                      <tr 
                        key={`${pred.player_id}-${pred.season}-${pred.week}`} 
                        className={`transition-all hover:bg-slate-50 dark:hover:bg-slate-800 ${isTop ? 'bg-gradient-to-r from-indigo-50 to-transparent dark:from-indigo-950' : ''}`}
                        style={{ animationDelay: `${idx * 50}ms` }}
                      >
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            {isTop && <span className="mr-2 text-xl">{['🥇', '🥈', '🥉'][idx]}</span>}
                            <span className={`text-sm font-bold ${isTop ? 'text-indigo-600 dark:text-indigo-400' : 'text-slate-600 dark:text-slate-400'}`}>
                              #{idx + 1}
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <div>
                              <div className="text-sm font-semibold text-slate-900 dark:text-white">
                                {pred.name}
                              </div>
                              {pred.model_version && (
                                <div className="text-xs text-slate-500 dark:text-slate-400">
                                  {pred.model_version}
                                </div>
                              )}
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`position-badge position-${pred.position}`}>
                            {pred.position}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-900 dark:text-white font-medium">
                          {pred.team || '—'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-600 dark:text-slate-400">
                          vs {pred.opponent_team}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-baseline space-x-2">
                            <span className={`text-2xl font-black ${isTop ? 'gradient-text' : 'text-indigo-600 dark:text-indigo-400'}`}>
                              {pred.y_pred.toFixed(1)}
                            </span>
                            <span className="text-xs text-slate-400">pts</span>
                          </div>
                        </td>
                        {pred.ci_lower && (
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-600 dark:text-slate-400">
                            <div className="flex items-center space-x-1">
                              <span className="text-green-600 dark:text-green-400">•</span>
                              <span>{pred.ci_lower.toFixed(1)} - {pred.ci_upper?.toFixed(1)}</span>
                            </div>
                          </td>
                        )}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="glass rounded-2xl p-20 text-center shadow-xl animate-fadeInUp">
            <div className="text-6xl mb-4">📊</div>
            <p className="text-xl text-slate-600 dark:text-slate-400 font-medium">
              No predictions available for this week.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
