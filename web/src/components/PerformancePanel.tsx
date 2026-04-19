import { useEffect, useMemo, useState } from 'react';
import { loadModel, loadSessions } from '../lib/dataLoaders';
import type { ModelData, SessionData } from '../lib/types';
import { fmtSharpe, fmtSigned } from '../lib/format';
import Plot from '../lib/Plot';

function pearson(xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n === 0) return NaN;
  let mx = 0,
    my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let num = 0,
    dx2 = 0,
    dy2 = 0;
  for (let i = 0; i < n; i++) {
    const a = xs[i] - mx;
    const b = ys[i] - my;
    num += a * b;
    dx2 += a * a;
    dy2 += b * b;
  }
  return num / Math.sqrt(dx2 * dy2);
}

export default function PerformancePanel() {
  const [m, setM] = useState<ModelData | null>(null);
  const [sessions, setSessions] = useState<SessionData[] | null>(null);

  useEffect(() => {
    loadModel().then(setM);
    loadSessions().then(setSessions);
  }, []);

  const stats = useMemo(() => {
    if (!sessions) return null;
    const yhat = sessions.map((s) => s.prediction_vol_adj);
    const y = sessions.map((s) => s.actual_vol_adj);
    const pnl = sessions.map((s) => s.pnl_session);
    const r = pearson(yhat, y);
    const r2 = r * r;
    let sumPnl = 0,
      sumPnl2 = 0;
    for (const p of pnl) {
      sumPnl += p;
      sumPnl2 += p * p;
    }
    const meanPnl = sumPnl / pnl.length;
    const stdPnl = Math.sqrt(sumPnl2 / pnl.length - meanPnl * meanPnl);
    const inSampleSharpe = stdPnl > 0 ? (meanPnl / stdPnl) * 16 : NaN;
    return { yhat, y, pnl, r, r2, meanPnl, stdPnl, inSampleSharpe };
  }, [sessions]);

  if (!m) return <div className="panel">Loading model…</div>;

  const std =
    Math.sqrt(
      m.cv_fold_sharpes.reduce(
        (a, s) => a + (s - m.cv_mean_sharpe) ** 2,
        0,
      ) / m.cv_fold_sharpes.length,
    );

  return (
    <div className="space-y-4">
      {/* Summary banner */}
      <div className="panel">
        <h2 className="text-2xl font-semibold text-slate-900 mb-3">
          Performance summary
        </h2>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div>
            <div className="stat-label">CV Sharpe (5-fold)</div>
            <div className="stat">
              {fmtSharpe(m.cv_mean_sharpe)}
              <span className="text-sm text-slate-500 ml-2">
                ± {fmtSharpe(std)}
              </span>
            </div>
          </div>
          <div>
            <div className="stat-label">α*</div>
            <div className="stat">{m.alpha_star}</div>
          </div>
          <div>
            <div className="stat-label">LB Sharpe (public)</div>
            <div className="stat">{fmtSharpe(m.lb_score_public)}</div>
          </div>
          <div>
            <div className="stat-label">In-sample Sharpe</div>
            <div className="stat">
              {stats ? fmtSharpe(stats.inSampleSharpe) : '…'}
            </div>
          </div>
        </div>
      </div>

      {/* Two-up: fold bars + alpha sweep */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="panel">
          <h3 className="text-sm font-semibold text-slate-700 mb-2">
            Per-fold Sharpe at α* = {m.alpha_star}
          </h3>
          <Plot
            data={[
              {
                type: 'bar',
                x: m.cv_fold_sharpes.map((_, i) => `fold ${i + 1}`),
                y: m.cv_fold_sharpes,
                marker: { color: '#4f46e5' },
                text: m.cv_fold_sharpes.map((s) => s.toFixed(2)),
                textposition: 'outside',
                hovertemplate: '%{x}: %{y:.4f}<extra></extra>',
              },
              {
                type: 'scatter',
                mode: 'lines',
                x: ['fold 1', `fold ${m.cv_fold_sharpes.length}`],
                y: [m.cv_mean_sharpe, m.cv_mean_sharpe],
                line: { color: '#94a3b8', dash: 'dash', width: 1.5 },
                name: `mean ${fmtSharpe(m.cv_mean_sharpe)}`,
                hovertemplate: 'mean = %{y:.4f}<extra></extra>',
              },
            ]}
            layout={{
              height: 280,
              margin: { l: 50, r: 20, t: 10, b: 40 },
              yaxis: { gridcolor: '#e2e8f0', title: 'Sharpe' },
              xaxis: { gridcolor: '#e2e8f0' },
              paper_bgcolor: 'white',
              plot_bgcolor: 'white',
              showlegend: false,
              font: { family: 'system-ui', size: 12, color: '#0f172a' },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
            useResizeHandler
          />
        </div>

        <div className="panel">
          <h3 className="text-sm font-semibold text-slate-700 mb-2">
            α grid sweep (5-fold mean Sharpe)
          </h3>
          <Plot
            data={[
              {
                type: 'scatter',
                mode: 'lines+markers',
                x: m.alpha_grid.map((r) => r.alpha),
                y: m.alpha_grid.map((r) => r.cv_sharpe),
                line: { color: '#4f46e5', width: 2 },
                marker: { color: '#4f46e5', size: 6 },
                hovertemplate: 'α = %{x}<br>Sharpe = %{y:.4f}<extra></extra>',
              },
              {
                type: 'scatter',
                mode: 'markers',
                x: [m.alpha_star],
                y: [
                  m.alpha_grid.find((r) => r.alpha === m.alpha_star)
                    ?.cv_sharpe ?? m.cv_mean_sharpe,
                ],
                marker: { color: '#dc2626', size: 14, symbol: 'star' },
                name: `α* = ${m.alpha_star}`,
                hovertemplate: 'α* = %{x}<br>%{y:.4f}<extra></extra>',
              },
            ]}
            layout={{
              height: 280,
              margin: { l: 50, r: 20, t: 10, b: 40 },
              xaxis: {
                type: 'log',
                title: 'α (log scale)',
                gridcolor: '#e2e8f0',
              },
              yaxis: { gridcolor: '#e2e8f0', title: 'CV Sharpe' },
              paper_bgcolor: 'white',
              plot_bgcolor: 'white',
              showlegend: false,
              font: { family: 'system-ui', size: 12, color: '#0f172a' },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
            useResizeHandler
          />
        </div>
      </div>

      {/* Predicted vs actual scatter */}
      <div className="panel">
        <h3 className="text-sm font-semibold text-slate-700 mb-2">
          Predicted vs actual (vol-adjusted, all 1000 in-sample sessions)
        </h3>
        {stats ? (
          <Plot
            data={[
              {
                type: 'scattergl',
                mode: 'markers',
                x: stats.yhat,
                y: stats.y,
                marker: (() => {
                  const maxAbs = Math.max(...stats.pnl.map(Math.abs));
                  return {
                    color: stats.pnl,
                    colorscale: 'RdBu',
                    cmin: -maxAbs,
                    cmax: maxAbs,
                    size: 5,
                    opacity: 0.7,
                    colorbar: { title: { text: 'PnL' }, thickness: 12 },
                  };
                })(),
                hovertemplate:
                  'ŷ = %{x:.3f}<br>y = %{y:.3f}<br>PnL = %{marker.color:.3f}<extra></extra>',
              },
              {
                type: 'scatter',
                mode: 'lines',
                x: [
                  Math.min(...stats.yhat, ...stats.y),
                  Math.max(...stats.yhat, ...stats.y),
                ],
                y: [
                  Math.min(...stats.yhat, ...stats.y),
                  Math.max(...stats.yhat, ...stats.y),
                ],
                line: { color: '#94a3b8', dash: 'dot', width: 1 },
                showlegend: false,
                hoverinfo: 'skip',
              },
            ]}
            layout={{
              height: 460,
              margin: { l: 60, r: 20, t: 10, b: 50 },
              xaxis: {
                title: 'predicted ŷ (vol-adjusted)',
                gridcolor: '#e2e8f0',
                zerolinecolor: '#94a3b8',
              },
              yaxis: {
                title: 'actual y (vol-adjusted)',
                gridcolor: '#e2e8f0',
                zerolinecolor: '#94a3b8',
              },
              paper_bgcolor: 'white',
              plot_bgcolor: 'white',
              showlegend: false,
              annotations: [
                {
                  x: 0.02,
                  y: 0.98,
                  xref: 'paper',
                  yref: 'paper',
                  xanchor: 'left',
                  yanchor: 'top',
                  showarrow: false,
                  text:
                    `R² = ${stats.r2.toFixed(4)} · ` +
                    `Pearson r = ${fmtSigned(stats.r, 4)}`,
                  font: { family: 'monospace', size: 12, color: '#475569' },
                  bgcolor: 'rgba(255,255,255,0.85)',
                  bordercolor: '#cbd5e1',
                  borderwidth: 1,
                  borderpad: 4,
                },
              ],
              font: { family: 'system-ui', size: 12, color: '#0f172a' },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
            useResizeHandler
          />
        ) : (
          <div className="text-sm text-slate-500">
            Loading per-session predictions…
          </div>
        )}
      </div>

      {/* Full alpha grid table */}
      <div className="panel">
        <h3 className="text-sm font-semibold text-slate-700 mb-2">
          Full α grid
        </h3>
        <table className="w-full text-sm font-mono">
          <thead className="text-xs uppercase text-slate-500 border-b">
            <tr>
              <th className="text-left py-1">α</th>
              <th className="text-right py-1">CV Sharpe</th>
            </tr>
          </thead>
          <tbody>
            {m.alpha_grid.map((r) => (
              <tr
                key={r.alpha}
                className={`border-b border-slate-100 ${
                  r.alpha === m.alpha_star
                    ? 'text-indigo-700 font-semibold bg-indigo-50'
                    : ''
                }`}
              >
                <td className="py-1">{r.alpha}</td>
                <td className="text-right py-1">{fmtSharpe(r.cv_sharpe)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
