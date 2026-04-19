import { useEffect, useMemo, useState } from 'react';
import { loadModel, loadSessions } from '../lib/dataLoaders';
import type { ModelData, SessionData } from '../lib/types';
import { fmtSigned, fmtSciOrFixed } from '../lib/format';
import Plot from '../lib/Plot';

const LLM_SYMBOL: Record<string, string> = {
  pos: 'triangle-up',
  neutral: 'circle',
  neg: 'triangle-down',
};

export default function SessionExplorer() {
  const [m, setM] = useState<ModelData | null>(null);
  const [sessions, setSessions] = useState<SessionData[] | null>(null);
  const [sid, setSid] = useState(0);
  const [hoveredHeadline, setHoveredHeadline] = useState<number | null>(null);

  useEffect(() => {
    loadModel().then(setM);
    loadSessions().then(setSessions);
  }, []);

  const session = sessions?.[sid] ?? null;

  const contributions = useMemo(() => {
    if (!m || !session) return null;
    return m.feature_cols
      .map((name) => {
        const raw = session.features[name];
        const mean = m.feature_means[name];
        const std = m.feature_stds[name];
        const standardized = std > 0 ? (raw - mean) / std : 0;
        const coef = m.coefs[name];
        const contribution = standardized * coef;
        return { name, raw, standardized, coef, contribution };
      })
      .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  }, [m, session]);

  if (!m || !sessions || !session)
    return <div className="panel">Loading sessions ({sessions?.length ?? 0})…</div>;

  const N = sessions.length;
  const seen = session.seen_bars;
  const unseen = session.unseen_bars;
  const closeMap = new Map<number, number>();
  for (const b of seen) closeMap.set(b.b, b.c);
  for (const b of unseen) closeMap.set(b.b, b.c);

  const headlineX = session.headlines.map((h) => h.b);
  const headlineY = session.headlines.map((h) => closeMap.get(h.b) ?? 1);
  const headlineColor = session.headlines.map((h) => h.fb);
  const headlineSymbol = session.headlines.map(
    (h) => LLM_SYMBOL[h.llm] ?? 'circle',
  );

  const allClose = [
    ...seen.map((b) => b.c),
    ...unseen.map((b) => b.c),
  ];
  const yMin = Math.min(...seen.map((b) => b.l ?? b.c));
  const yMax = Math.max(...seen.map((b) => b.h ?? b.c), ...allClose);

  return (
    <div className="space-y-4">
      {/* Picker bar */}
      <div className="panel flex flex-wrap items-center gap-3">
        <label className="text-sm text-slate-700">Session</label>
        <button
          type="button"
          onClick={() => setSid((s) => Math.max(0, s - 1))}
          className="px-2 py-1 text-sm rounded border border-slate-300 hover:bg-slate-100"
        >
          ←
        </button>
        <input
          type="number"
          min={0}
          max={N - 1}
          value={sid}
          onChange={(e) => {
            const v = parseInt(e.target.value, 10);
            if (!Number.isNaN(v) && v >= 0 && v < N) setSid(v);
          }}
          className="w-24 px-2 py-1 text-sm border border-slate-300 rounded font-mono"
        />
        <span className="text-sm text-slate-500 font-mono">/ {N - 1}</span>
        <button
          type="button"
          onClick={() => setSid((s) => Math.min(N - 1, s + 1))}
          className="px-2 py-1 text-sm rounded border border-slate-300 hover:bg-slate-100"
        >
          →
        </button>
        <button
          type="button"
          onClick={() => setSid(Math.floor(Math.random() * N))}
          className="ml-2 px-3 py-1 text-sm rounded bg-indigo-600 text-white hover:bg-indigo-700"
        >
          random
        </button>
        <span className="text-xs text-slate-500 font-mono ml-auto">
          {session.headlines.length} headlines · vol ={' '}
          {session.vol.toFixed(5)}
        </span>
      </div>

      {/* Bottom strip — moved to top for visibility */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="panel">
          <div className="stat-label">ŷ (vol-adj)</div>
          <div
            className={`stat ${
              session.prediction_vol_adj >= 0
                ? 'text-teal-700'
                : 'text-orange-700'
            }`}
          >
            {fmtSigned(session.prediction_vol_adj, 3)}
          </div>
        </div>
        <div className="panel">
          <div className="stat-label">y (vol-adj)</div>
          <div
            className={`stat ${
              session.actual_vol_adj >= 0 ? 'text-teal-700' : 'text-orange-700'
            }`}
          >
            {fmtSigned(session.actual_vol_adj, 3)}
          </div>
        </div>
        <div className="panel">
          <div className="stat-label">position</div>
          <div className="stat">
            {fmtSigned(session.prediction_raw_position, 2)}
          </div>
        </div>
        <div className="panel">
          <div className="stat-label">PnL = pos · raw return</div>
          <div
            className={`stat ${
              session.pnl_session >= 0 ? 'text-teal-700' : 'text-orange-700'
            }`}
          >
            {fmtSigned(session.pnl_session, 3)}
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="panel">
        <Plot
          data={[
            {
              type: 'candlestick',
              x: seen.map((b) => b.b),
              open: seen.map((b) => b.o ?? b.c),
              high: seen.map((b) => b.h ?? b.c),
              low: seen.map((b) => b.l ?? b.c),
              close: seen.map((b) => b.c),
              name: 'seen (0-49)',
              increasing: { line: { color: '#0d9488' } },
              decreasing: { line: { color: '#ea580c' } },
              showlegend: false,
            },
            {
              type: 'scatter',
              mode: 'lines',
              x: unseen.map((b) => b.b),
              y: unseen.map((b) => b.c),
              name: 'unseen (50-99)',
              line: { color: '#94a3b8', dash: 'dash', width: 2 },
              showlegend: false,
              hovertemplate: 'bar %{x}<br>close %{y:.4f}<extra>unseen</extra>',
            },
            {
              type: 'scatter',
              mode: 'markers',
              x: headlineX,
              y: headlineY,
              name: 'headlines',
              text: session.headlines.map(
                (h, i) => {
                  const tail =
                    h.align3 !== undefined && h.final3 !== undefined
                      ? `align3 ${fmtSigned(h.align3, 3)} · final3 ${fmtSigned(
                          h.final3,
                          3,
                        )}`
                      : `(in unseen window — no alignment)`;
                  return (
                    `<b>bar ${h.b}</b><br>${h.t.replace(/</g, '&lt;')}<br>` +
                    `FinBERT ${fmtSigned(h.fb, 3)} · LLM ${h.llm}<br>` +
                    `${tail}<extra>#${i}</extra>`
                  );
                },
              ),
              hovertemplate: '%{text}',
              marker: {
                color: headlineColor,
                colorscale: 'RdBu',
                cmin: -1,
                cmax: 1,
                size: 13,
                symbol: headlineSymbol,
                line: { color: '#1e293b', width: 1 },
                colorbar: {
                  title: { text: 'FinBERT' },
                  thickness: 10,
                  len: 0.6,
                  x: 1.02,
                },
              },
              showlegend: false,
            },
          ]}
          layout={{
            height: 460,
            margin: { l: 50, r: 80, t: 30, b: 40 },
            xaxis: {
              title: 'bar',
              range: [-1, 100],
              gridcolor: '#e2e8f0',
              rangeslider: { visible: false },
            },
            yaxis: {
              title: 'price',
              gridcolor: '#e2e8f0',
              range: [yMin - (yMax - yMin) * 0.05, yMax + (yMax - yMin) * 0.05],
            },
            shapes: [
              {
                type: 'line',
                x0: 49.5,
                x1: 49.5,
                yref: 'paper',
                y0: 0,
                y1: 1,
                line: { color: '#475569', width: 1.5, dash: 'dot' },
              },
            ],
            annotations: [
              {
                x: 49.5,
                yref: 'paper',
                y: 1.0,
                yanchor: 'bottom',
                text: 'prediction point',
                showarrow: false,
                font: { size: 10, color: '#475569' },
              },
              {
                x: 24,
                yref: 'paper',
                y: 0.02,
                yanchor: 'bottom',
                text: 'seen',
                showarrow: false,
                font: { size: 10, color: '#94a3b8' },
              },
              {
                x: 74,
                yref: 'paper',
                y: 0.02,
                yanchor: 'bottom',
                text: 'unseen (true future)',
                showarrow: false,
                font: { size: 10, color: '#94a3b8' },
              },
            ],
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            font: { family: 'system-ui', size: 12, color: '#0f172a' },
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%' }}
          useResizeHandler
        />
        <p className="text-xs text-slate-500 mt-2">
          Marker symbol: ▲ pos · ● neutral · ▼ neg (LLM label). Marker color:
          FinBERT score (red = negative, blue = positive). Hover for the
          headline text.
        </p>
      </div>

      {/* Headlines + Features */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="panel">
          <h3 className="text-sm font-semibold text-slate-700 mb-2">
            Headlines ({session.headlines.length})
          </h3>
          {session.headlines.length === 0 ? (
            <p className="text-sm text-slate-500">No headlines this session.</p>
          ) : (
            <ul className="space-y-2">
              {session.headlines.map((h, i) => (
                <li
                  key={i}
                  onMouseEnter={() => setHoveredHeadline(i)}
                  onMouseLeave={() => setHoveredHeadline(null)}
                  className={`text-sm p-2 rounded border ${
                    hoveredHeadline === i
                      ? 'border-indigo-300 bg-indigo-50'
                      : 'border-slate-200'
                  }`}
                >
                  <div className="text-xs text-slate-500 font-mono">
                    bar {h.b} · fb {fmtSigned(h.fb, 3)} · llm {h.llm}
                    {h.final3 !== undefined && (
                      <> · final3 {fmtSigned(h.final3, 3)}</>
                    )}
                  </div>
                  <div className="text-slate-800">{h.t}</div>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="panel">
          <h3 className="text-sm font-semibold text-slate-700 mb-2">
            Feature contributions
          </h3>
          <p className="text-xs text-slate-500 mb-2">
            contribution = (raw − μ)/σ × coef. Sorted by |contribution|. Top 3
            in <span className="font-semibold">bold</span>.
          </p>
          <table className="w-full text-xs font-mono">
            <thead className="text-slate-500 border-b">
              <tr>
                <th className="text-left py-1">Feature</th>
                <th className="text-right py-1">Raw</th>
                <th className="text-right py-1">Std</th>
                <th className="text-right py-1">Coef</th>
                <th className="text-right py-1">Contrib</th>
              </tr>
            </thead>
            <tbody>
              {contributions?.map((r, i) => (
                <tr
                  key={r.name}
                  className={`border-b border-slate-100 ${
                    i < 3 ? 'font-semibold' : ''
                  }`}
                >
                  <td className="py-1">{r.name}</td>
                  <td className="text-right py-1">
                    {fmtSciOrFixed(r.raw)}
                  </td>
                  <td className="text-right py-1">
                    {fmtSigned(r.standardized, 3)}
                  </td>
                  <td className="text-right py-1">{fmtSigned(r.coef, 3)}</td>
                  <td
                    className={`text-right py-1 ${
                      r.contribution >= 0
                        ? 'text-teal-700'
                        : 'text-orange-700'
                    }`}
                  >
                    {fmtSigned(r.contribution, 3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
