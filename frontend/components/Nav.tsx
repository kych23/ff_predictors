import Link from 'next/link';

export default function Nav() {
  return (
    <nav className="sticky top-0 z-50 border-b border-slate-200 bg-white/80 backdrop-blur-md dark:border-slate-800 dark:bg-slate-900/80">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-600 to-purple-600 shadow-lg">
              <span className="text-2xl">🏈</span>
            </div>
            <div>
              <Link href="/" className="text-xl font-bold gradient-text hover:opacity-90 transition-opacity">
                Fantasy Predictors
              </Link>
              <p className="text-xs text-slate-500 dark:text-slate-400">ML-Powered Insights</p>
            </div>
          </div>
          <div className="flex items-center space-x-1">
            <Link
              href="/"
              className="group relative px-4 py-2 text-sm font-medium text-slate-700 dark:text-slate-300 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
            >
              Predictions
              <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-gradient-to-r from-indigo-600 to-purple-600 group-hover:w-full transition-all duration-300" />
            </Link>
            <Link
              href="/compare"
              className="group relative px-4 py-2 text-sm font-medium text-slate-700 dark:text-slate-300 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
            >
              Compare
              <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-gradient-to-r from-indigo-600 to-purple-600 group-hover:w-full transition-all duration-300" />
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}
