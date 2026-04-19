import { useEffect, useState } from 'react';
import Navigation, { type RouteId, ROUTES } from './components/Navigation';
import Intro from './components/Intro';
import SessionExplorer from './components/SessionExplorer';
import AugmentationView from './components/AugmentationView';
import AlignmentDemo from './components/AlignmentDemo';
import CoefficientsView from './components/CoefficientsView';
import DecayCurves from './components/DecayCurves';
import PerformancePanel from './components/PerformancePanel';

function getRouteFromHash(): RouteId {
  const h = window.location.hash.replace(/^#\/?/, '');
  return (ROUTES.find((r) => r.id === h)?.id ?? 'intro') as RouteId;
}

export default function App() {
  const [route, setRoute] = useState<RouteId>(getRouteFromHash);

  useEffect(() => {
    const onHash = () => setRoute(getRouteFromHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-slate-200 bg-white">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-slate-900">
              HRT Datathon 2026 — Solution Walkthrough
            </h1>
            <p className="text-xs text-slate-500 font-mono">
              Ridge regression + alignment-weighted FinBERT sentiment · LB Sharpe 3.027
            </p>
          </div>
          <Navigation current={route} />
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-8">
        {route === 'intro' && <Intro />}
        {route === 'session' && <SessionExplorer />}
        {route === 'augmentation' && <AugmentationView />}
        {route === 'alignment' && <AlignmentDemo />}
        {route === 'coefficients' && <CoefficientsView />}
        {route === 'decay' && <DecayCurves />}
        {route === 'performance' && <PerformancePanel />}
      </main>

      <footer className="border-t border-slate-200 bg-white text-xs text-slate-500 py-3 text-center">
        Static visualization · built with Vite + React + Plotly · model frozen at submission time
      </footer>
    </div>
  );
}
