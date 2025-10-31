'use client';

import { useState, useEffect } from 'react';
import { api, type Player, type Prediction } from '@/lib/api';
import Nav from '@/components/Nav';

const currentYear = new Date().getFullYear();

export default function ComparePage() {
  const [season, setSeason] = useState(currentYear);
  const [week, setWeek] = useState(1);
  const [searchTerm1, setSearchTerm1] = useState('');
  const [searchTerm2, setSearchTerm2] = useState('');
  const [players1, setPlayers1] = useState<Player[]>([]);
  const [players2, setPlayers2] = useState<Player[]>([]);
  const [selected1, setSelected1] = useState<Player | null>(null);
  const [selected2, setSelected2] = useState<Player | null>(null);
  const [prediction1, setPrediction1] = useState<Prediction | null>(null);
  const [prediction2, setPrediction2] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [initialized, setInitialized] = useState(false);

  useEffect(() => {
    const fetchCurrentWeek = async () => {
      if (!initialized) {
        try {
          const current = await api.getCurrentWeek();
          setSeason(current.season);
          setWeek(current.week);
          setInitialized(true);
        } catch (err) {
          console.error('Failed to fetch current week:', err);
          setInitialized(true); // Still mark as initialized to avoid retry loop
        }
      }
    };
    fetchCurrentWeek();
  }, [initialized]);

  useEffect(() => {
    const fetchPlayers = async () => {
      if (searchTerm1.length >= 2) {
        const all = await api.getPlayers();
        const filtered = all.filter(p => 
          p.name.toLowerCase().includes(searchTerm1.toLowerCase())
        );
        setPlayers1(filtered.slice(0, 10));
      } else {
        setPlayers1([]);
      }
    };
    fetchPlayers();
  }, [searchTerm1]);

  useEffect(() => {
    const fetchPlayers = async () => {
      if (searchTerm2.length >= 2) {
        const all = await api.getPlayers();
        const filtered = all.filter(p => 
          p.name.toLowerCase().includes(searchTerm2.toLowerCase())
        );
        setPlayers2(filtered.slice(0, 10));
      } else {
        setPlayers2([]);
      }
    };
    fetchPlayers();
  }, [searchTerm2]);

  useEffect(() => {
    const runComparison = async () => {
      if (selected1 && selected2) {
        setLoading(true);
        try {
          const preds = await api.getPredictions(season, week);
          const pred1 = preds.find(p => p.player_id === selected1.player_id);
          const pred2 = preds.find(p => p.player_id === selected2.player_id);
          setPrediction1(pred1 || null);
          setPrediction2(pred2 || null);
        } catch (err) {
          console.error('Failed to fetch predictions:', err);
        } finally {
          setLoading(false);
        }
      }
    };
    runComparison();
  }, [selected1, selected2, season, week]);

  const winner = prediction1 && prediction2
    ? prediction1.y_pred > prediction2.y_pred ? 1 : 2
    : null;

  const diff = prediction1 && prediction2
    ? Math.abs(prediction1.y_pred - prediction2.y_pred)
    : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 dark:from-slate-900 dark:via-slate-900 dark:to-slate-900">
      <Nav />
      
      <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-12 animate-fadeInUp">
          <h1 className="text-5xl font-extrabold gradient-text mb-4">
            Player Comparison
          </h1>
          <div className="inline-block glass rounded-full px-6 py-2 mb-4 shadow-lg">
            <span className="text-lg font-bold text-slate-700 dark:text-slate-300">
              {season} Season • Week {week}
            </span>
          </div>
          <p className="text-xl text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
            Compare two players head-to-head to make the right start/sit decision
          </p>
        </div>

        {/* VS Comparison Cards */}
        <div className="grid gap-8 md:grid-cols-2 mb-8">
          {/* Player 1 */}
          <div className="animate-fadeInUp space-y-4">
            <h2 className="text-2xl font-bold text-slate-900 dark:text-white text-center">Player 1</h2>
            
            <div className="relative">
              <input
                type="text"
                value={searchTerm1}
                onChange={(e) => setSearchTerm1(e.target.value)}
                placeholder="Search for a player..."
                className="w-full rounded-xl border-2 border-slate-200 bg-white px-4 py-3 shadow-sm transition-all focus:border-indigo-500 focus:ring-4 focus:ring-indigo-500/20 dark:border-slate-700 dark:bg-slate-800 dark:text-white"
              />
              {players1.length > 0 && (
                <div className="absolute z-50 mt-2 w-full rounded-xl border-2 border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-slate-800 animate-fadeInUp">
                  {players1.map((player) => (
                    <button
                      key={player.player_id}
                      onClick={() => {
                        setSelected1(player);
                        setSearchTerm1(player.name);
                        setPlayers1([]);
                      }}
                      className="w-full px-4 py-3 text-left text-sm hover:bg-indigo-50 dark:hover:bg-indigo-950 transition-colors border-b border-slate-100 dark:border-slate-700 last:border-0"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-slate-900 dark:text-white">{player.name}</span>
                        <span className={`position-badge position-${player.position}`}>
                          {player.position}
                        </span>
                      </div>
                      <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                        {player.team_current || 'Free Agent'}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
            
            {selected1 && (
              <div className="glass rounded-2xl border-2 border-indigo-200 p-6 shadow-xl animate-fadeInUp">
                <div className="text-2xl font-bold text-slate-900 dark:text-white mb-2">{selected1.name}</div>
                <div className="flex items-center gap-2">
                  <span className={`position-badge position-${selected1.position}`}>
                    {selected1.position}
                  </span>
                  <span className="text-sm text-slate-600 dark:text-slate-400">
                    {selected1.team_current || 'Free Agent'}
                  </span>
                </div>
              </div>
            )}

            {prediction1 && (
              <div className="glass rounded-2xl border-2 border-indigo-300 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-950 dark:to-purple-950 p-8 shadow-2xl animate-fadeInUp">
                <div className="text-center">
                  <div className="text-sm font-semibold text-indigo-600 dark:text-indigo-400 mb-2 uppercase tracking-wide">
                    Projected Points
                  </div>
                  <div className="text-6xl font-black text-indigo-600 dark:text-indigo-400 mb-2">
                    {prediction1.y_pred.toFixed(1)}
                  </div>
                  {prediction1.ci_lower && (
                    <div className="text-sm text-indigo-700 dark:text-indigo-300 mb-4">
                      95% CI: {prediction1.ci_lower.toFixed(1)} - {prediction1.ci_upper?.toFixed(1)}
                    </div>
                  )}
                  <div className="inline-block rounded-full bg-indigo-100 dark:bg-indigo-900 px-4 py-2 text-sm font-semibold text-indigo-700 dark:text-indigo-300">
                    vs {prediction1.opponent_team}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* VS Divider */}
          <div className="hidden md:flex items-center justify-center">
            <div className="glass rounded-full p-6 shadow-2xl animate-pulse-glow">
              <div className="text-6xl font-black gradient-text">VS</div>
            </div>
          </div>

          {/* Player 2 */}
          <div className="animate-fadeInUp space-y-4" style={{ animationDelay: '100ms' }}>
            <h2 className="text-2xl font-bold text-slate-900 dark:text-white text-center">Player 2</h2>
            
            <div className="relative">
              <input
                type="text"
                value={searchTerm2}
                onChange={(e) => setSearchTerm2(e.target.value)}
                placeholder="Search for a player..."
                className="w-full rounded-xl border-2 border-slate-200 bg-white px-4 py-3 shadow-sm transition-all focus:border-indigo-500 focus:ring-4 focus:ring-indigo-500/20 dark:border-slate-700 dark:bg-slate-800 dark:text-white"
              />
              {players2.length > 0 && (
                <div className="absolute z-50 mt-2 w-full rounded-xl border-2 border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-slate-800 animate-fadeInUp">
                  {players2.map((player) => (
                    <button
                      key={player.player_id}
                      onClick={() => {
                        setSelected2(player);
                        setSearchTerm2(player.name);
                        setPlayers2([]);
                      }}
                      className="w-full px-4 py-3 text-left text-sm hover:bg-indigo-50 dark:hover:bg-indigo-950 transition-colors border-b border-slate-100 dark:border-slate-700 last:border-0"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-slate-900 dark:text-white">{player.name}</span>
                        <span className={`position-badge position-${player.position}`}>
                          {player.position}
                        </span>
                      </div>
                      <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                        {player.team_current || 'Free Agent'}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
            
            {selected2 && (
              <div className="glass rounded-2xl border-2 border-purple-200 p-6 shadow-xl animate-fadeInUp">
                <div className="text-2xl font-bold text-slate-900 dark:text-white mb-2">{selected2.name}</div>
                <div className="flex items-center gap-2">
                  <span className={`position-badge position-${selected2.position}`}>
                    {selected2.position}
                  </span>
                  <span className="text-sm text-slate-600 dark:text-slate-400">
                    {selected2.team_current || 'Free Agent'}
                  </span>
                </div>
              </div>
            )}

            {prediction2 && (
              <div className="glass rounded-2xl border-2 border-purple-300 bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-950 dark:to-pink-950 p-8 shadow-2xl animate-fadeInUp">
                <div className="text-center">
                  <div className="text-sm font-semibold text-purple-600 dark:text-purple-400 mb-2 uppercase tracking-wide">
                    Projected Points
                  </div>
                  <div className="text-6xl font-black text-purple-600 dark:text-purple-400 mb-2">
                    {prediction2.y_pred.toFixed(1)}
                  </div>
                  {prediction2.ci_lower && (
                    <div className="text-sm text-purple-700 dark:text-purple-300 mb-4">
                      95% CI: {prediction2.ci_lower.toFixed(1)} - {prediction2.ci_upper?.toFixed(1)}
                    </div>
                  )}
                  <div className="inline-block rounded-full bg-purple-100 dark:bg-purple-900 px-4 py-2 text-sm font-semibold text-purple-700 dark:text-purple-300">
                    vs {prediction2.opponent_team}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="text-center py-8">
            <div className="inline-block mx-auto mb-4 h-12 w-12 animate-spin rounded-full border-4 border-indigo-200 border-t-indigo-600"></div>
            <p className="text-slate-600 dark:text-slate-400 font-medium">Loading comparison...</p>
          </div>
        )}

        {/* Winner Card */}
        {winner && prediction1 && prediction2 && (
          <div className="glass rounded-2xl border-4 border-green-500 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-950 dark:to-emerald-950 p-10 text-center shadow-2xl animate-fadeInUp animate-pulse-glow">
            <div className="mb-4 text-7xl">
              {winner === 1 ? '🥇' : '🥈'}
            </div>
            <div className="mb-2 text-3xl font-black text-green-700 dark:text-green-300">
              Start Player {winner}!
            </div>
            <div className="text-lg font-semibold text-green-600 dark:text-green-400 mb-4">
              {winner === 1 ? selected1?.name : selected2?.name}
            </div>
            <div className="text-4xl font-black text-green-700 dark:text-green-300 mb-2">
              +{diff.toFixed(1)}
            </div>
            <div className="text-sm font-semibold text-green-600 dark:text-green-400 uppercase tracking-wide">
              Projected Points Advantage
            </div>
          </div>
        )}

        {/* Helper Text */}
        {!selected1 || !selected2 ? (
          <div className="text-center py-8">
            <div className="glass rounded-2xl p-8 max-w-md mx-auto shadow-xl animate-fadeInUp">
              <div className="text-5xl mb-4">🎯</div>
              <p className="text-slate-600 dark:text-slate-400 font-medium">
                {!selected1 && !selected2 
                  ? 'Select two players to compare their projections' 
                  : 'Select another player to start the comparison'}
              </p>
            </div>
          </div>
        ) : !prediction1 || !prediction2 ? (
          <div className="text-center py-8">
            <div className="glass rounded-2xl border-2 border-orange-200 bg-orange-50 dark:bg-orange-950 p-6 max-w-md mx-auto shadow-xl animate-fadeInUp">
              <p className="text-orange-700 dark:text-orange-300 font-medium">
                One or both players don&apos;t have predictions for this week.
              </p>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
