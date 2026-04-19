import { useEffect, useMemo, useState } from 'react';
import { loadAlignmentExamples, loadSessions } from '../lib/dataLoaders';
import type { AlignmentExample, SessionData } from '../lib/types';
import { fmtSigned, fmtNumber } from '../lib/format';
import Plot from '../lib/Plot';
import MathBlock from './MathBlock';

const QUADRANT_LABEL: Record<string, { color: string; tag: string }> = {
  pos_aligned: { color: 'bg-teal-100 text-teal-800', tag: 'pos · aligned' },
  pos_antialigned: {
    color: 'bg-amber-100 text-amber-800',
    tag: 'pos · anti-aligned (muted)',
  },
  neg_aligned: {
    color: 'bg-orange-100 text-orange-800',
    tag: 'neg · aligned',
  },
  neg_antialigned: {
    color: 'bg-amber-100 text-amber-800',
    tag: 'neg · anti-aligned (muted)',
  },
};

function VectorSVG({ ex }: { ex: AlignmentExample }) {
  const W = 380;
  const H = 380;
  const PAD = 40;
  // Auto-scale so both vectors fit comfortably with some headroom.
  const scale =
    1.4 *
    Math.max(
      Math.abs(ex.u3),
      Math.abs(ex.v3),
      Math.abs(ex.u5),
      Math.abs(ex.v5),
      0.05,
    );
  const cx = W / 2;
  const cy = H / 2;
  const xToPx = (u: number) =>
    cx + ((u + scale) / (2 * scale)) * (W - 2 * PAD) - (W - 2 * PAD) / 2;
  const yToPx = (v: number) =>
    cy - ((v + scale) / (2 * scale)) * (H - 2 * PAD) + (H - 2 * PAD) / 2;

  function arrow(u: number, v: number, color: string, dashed = false) {
    const x2 = xToPx(u);
    const y2 = yToPx(v);
    return (
      <>
        <line
          x1={cx}
          y1={cy}
          x2={x2}
          y2={y2}
          stroke={color}
          strokeWidth={2.5}
          strokeDasharray={dashed ? '5 4' : undefined}
          markerEnd={`url(#arrowhead-${color.replace('#', '')})`}
        />
        <defs>
          <marker
            id={`arrowhead-${color.replace('#', '')}`}
            viewBox="0 0 10 10"
            refX="8"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill={color} />
          </marker>
        </defs>
      </>
    );
  }

  const sIsPos = ex.fb_score > 0;
  const align3Color = ex.align3 > 0 ? '#0d9488' : '#ea580c';
  const align5Color = ex.align5 > 0 ? '#0891b2' : '#dc2626';
  const final3Clipped = ex.fb_score * Math.max(0, ex.align3) === 0 && ex.align3 !== 0;
  const final5Clipped = ex.fb_score * Math.max(0, ex.align5) === 0 && ex.align5 !== 0;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-[400px] h-auto">
      {/* Quadrant background — the "consistent" half-plane */}
      <rect
        x={PAD}
        y={sIsPos ? PAD : cy}
        width={W - 2 * PAD}
        height={(H - 2 * PAD) / 2}
        fill={sIsPos ? '#ecfdf5' : '#fff7ed'}
        opacity={0.6}
      />
      {/* Axes */}
      <line
        x1={PAD}
        y1={cy}
        x2={W - PAD}
        y2={cy}
        stroke="#cbd5e1"
        strokeWidth={1}
      />
      <line
        x1={cx}
        y1={PAD}
        x2={cx}
        y2={H - PAD}
        stroke="#cbd5e1"
        strokeWidth={1}
      />
      {/* Axis labels */}
      <text x={W - PAD + 4} y={cy + 4} fontSize="11" fill="#64748b">
        u →
      </text>
      <text x={cx + 4} y={PAD - 4} fontSize="11" fill="#64748b">
        v ↑
      </text>
      <text
        x={W - PAD}
        y={sIsPos ? PAD - 6 : H - PAD + 14}
        fontSize="10"
        fill="#0d9488"
        textAnchor="end"
      >
        consistent half-plane (s × v ≥ 0)
      </text>
      {/* Origin dot */}
      <circle cx={cx} cy={cy} r={4} fill="#0f172a" />
      <text x={cx + 6} y={cy - 6} fontSize="11" fill="#0f172a">
        anchor (b={ex.bar_ix})
      </text>

      {/* Vectors */}
      {arrow(ex.u3, ex.v3, align3Color, final3Clipped)}
      {arrow(ex.u5, ex.v5, align5Color, final5Clipped)}

      {/* Vector labels */}
      <text
        x={xToPx(ex.u3) + 6}
        y={yToPx(ex.v3) - 4}
        fontSize="11"
        fill={align3Color}
        fontFamily="monospace"
      >
        k=3 · align={fmtSigned(ex.align3, 3)}
      </text>
      <text
        x={xToPx(ex.u5) + 6}
        y={yToPx(ex.v5) - 4}
        fontSize="11"
        fill={align5Color}
        fontFamily="monospace"
      >
        k=5 · align={fmtSigned(ex.align5, 3)}
      </text>
    </svg>
  );
}

export default function AlignmentDemo() {
  const [examples, setExamples] = useState<AlignmentExample[] | null>(null);
  const [sessions, setSessions] = useState<SessionData[] | null>(null);
  const [idx, setIdx] = useState(0);

  useEffect(() => {
    loadAlignmentExamples().then(setExamples);
    loadSessions().then(setSessions);
  }, []);

  const ex = examples?.[idx] ?? null;

  const priceSnippet = useMemo(() => {
    if (!ex || !sessions) return null;
    const sess = sessions.find((s) => s.session === ex.session);
    if (!sess) return null;
    const lo = Math.max(0, ex.bar_ix - 2);
    const hi = Math.min(49, ex.bar_ix + 6);
    const barsSeen = sess.seen_bars.filter((b) => b.b >= lo && b.b <= hi);
    const barsUnseen = sess.unseen_bars
      .filter((b) => b.b >= lo && b.b <= hi)
      .map((b) => ({ b: b.b, c: b.c }));
    return { barsSeen, barsUnseen, lo, hi };
  }, [ex, sessions]);

  if (!examples || !ex) return <div className="panel">Loading…</div>;

  const q = QUADRANT_LABEL[ex.quadrant];

  return (
    <div className="space-y-4">
      {/* Math intro */}
      <div className="panel space-y-2">
        <h2 className="text-2xl font-semibold text-slate-900">
          Alignment geometry
        </h2>
        <p className="text-slate-700 leading-relaxed">
          For a headline at bar <MathBlock formula="b_h" /> with FinBERT score{' '}
          <MathBlock formula="s" />, look ahead{' '}
          <MathBlock formula="k \in \{3, 5\}" /> bars and form the normalized
          motion vector{' '}
          <MathBlock formula="(u_k, v_k)" />:
        </p>
        <MathBlock
          display
          formula="u_k = k / x_{\text{range}}, \quad v_k = (p_k - p_0) / y_{\text{range}}"
        />
        <p className="text-slate-700 leading-relaxed">
          The alignment scalar is the signed sine of the angle the vector makes
          with the time axis, scaled by sentiment:
        </p>
        <MathBlock
          display
          formula="\text{align}_k = s \cdot \frac{v_k}{\sqrt{u_k^2 + v_k^2 + 10^{-8}}}, \qquad \mathrm{final}_k = s \cdot \max(0, \text{align}_k)"
        />
        <p className="text-sm text-slate-500">
          Headlines whose price motion <em>contradicts</em> their sentiment
          land in the muted half-plane and contribute zero to the feature.
        </p>
      </div>

      {/* Picker */}
      <div className="panel flex flex-wrap items-center gap-3">
        <label className="text-sm text-slate-700">Curated example</label>
        <select
          value={idx}
          onChange={(e) => setIdx(parseInt(e.target.value, 10))}
          className="text-sm border border-slate-300 rounded px-2 py-1 max-w-md"
        >
          {examples.map((e, i) => (
            <option key={i} value={i}>
              {i + 1}. [{QUADRANT_LABEL[e.quadrant].tag}] sess {e.session}, bar{' '}
              {e.bar_ix} — {e.headline.slice(0, 50)}
              {e.headline.length > 50 ? '…' : ''}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={() => setIdx((i) => (i + 1) % examples.length)}
          className="ml-auto px-3 py-1 text-sm rounded bg-indigo-600 text-white hover:bg-indigo-700"
        >
          next →
        </button>
      </div>

      {/* Headline card */}
      <div className="panel space-y-1">
        <div className="flex items-center gap-2 text-xs">
          <span className={`px-2 py-0.5 rounded font-mono ${q.color}`}>
            {q.tag}
          </span>
          <span className="text-slate-500 font-mono">
            session {ex.session} · bar {ex.bar_ix}
          </span>
        </div>
        <p className="text-slate-900 leading-relaxed">{ex.headline}</p>
        <div className="text-xs font-mono text-slate-500">
          FinBERT = {fmtSigned(ex.fb_score, 3)} · LLM = {ex.llm_sent}
        </div>
      </div>

      {/* Price snippet + SVG vectors */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="panel">
          <h3 className="text-sm font-semibold text-slate-700 mb-2">
            Price around bar {ex.bar_ix}
          </h3>
          {priceSnippet ? (
            <Plot
              data={[
                {
                  type: 'scatter',
                  mode: 'lines+markers',
                  x: priceSnippet.barsSeen.map((b) => b.b),
                  y: priceSnippet.barsSeen.map((b) => b.c),
                  line: { color: '#1e293b', width: 2 },
                  marker: { color: '#1e293b', size: 6 },
                  name: 'seen',
                  hovertemplate: 'bar %{x}<br>close %{y:.4f}<extra></extra>',
                },
                priceSnippet.barsUnseen.length > 0
                  ? {
                      type: 'scatter',
                      mode: 'lines+markers',
                      x: priceSnippet.barsUnseen.map((b) => b.b),
                      y: priceSnippet.barsUnseen.map((b) => b.c),
                      line: { color: '#94a3b8', dash: 'dash', width: 2 },
                      marker: { color: '#94a3b8', size: 6 },
                      name: 'unseen',
                      hovertemplate:
                        'bar %{x}<br>close %{y:.4f}<extra>unseen</extra>',
                    }
                  : { x: [], y: [] },
                {
                  type: 'scatter',
                  mode: 'text+markers',
                  x: [ex.bar_ix, ex.bar_ix + 3, ex.bar_ix + 5],
                  y: [ex.p0, ex.p3, ex.p5],
                  marker: {
                    color: ['#dc2626', '#0d9488', '#0891b2'],
                    size: 12,
                    symbol: 'circle-open',
                    line: { width: 2 },
                  },
                  text: ['p₀', 'p₃', 'p₅'],
                  textposition: 'top center',
                  showlegend: false,
                  hovertemplate: '%{text} = %{y:.4f}<extra></extra>',
                },
              ]}
              layout={{
                height: 320,
                margin: { l: 50, r: 20, t: 10, b: 40 },
                xaxis: { title: 'bar', gridcolor: '#e2e8f0' },
                yaxis: { title: 'close', gridcolor: '#e2e8f0' },
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                showlegend: false,
                shapes: [
                  {
                    type: 'line',
                    x0: ex.bar_ix,
                    x1: ex.bar_ix,
                    yref: 'paper',
                    y0: 0,
                    y1: 1,
                    line: { color: '#dc2626', width: 1, dash: 'dot' },
                  },
                ],
                annotations: [
                  {
                    x: ex.bar_ix,
                    yref: 'paper',
                    y: 1,
                    yanchor: 'bottom',
                    text: 'headline',
                    showarrow: false,
                    font: { size: 10, color: '#dc2626' },
                  },
                ],
                font: { family: 'system-ui', size: 12, color: '#0f172a' },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: '100%' }}
              useResizeHandler
            />
          ) : (
            <p className="text-sm text-slate-500">Loading price context…</p>
          )}
        </div>

        <div className="panel">
          <h3 className="text-sm font-semibold text-slate-700 mb-2">
            (u, v) motion vectors (normalized)
          </h3>
          <div className="flex justify-center">
            <VectorSVG ex={ex} />
          </div>
          <p className="text-xs text-slate-500 mt-2">
            Shaded background = the consistent half-plane (sign(s) · v ≥ 0).
            Vectors landing outside it have align &lt; 0 → final clipped to 0.
            Dashed arrows are clipped this turn.
          </p>
        </div>
      </div>

      {/* Numeric readout */}
      <div className="panel">
        <h3 className="text-sm font-semibold text-slate-700 mb-2">
          Geometry breakdown
        </h3>
        <table className="w-full text-sm font-mono">
          <thead className="text-xs uppercase text-slate-500 border-b">
            <tr>
              <th className="text-left py-1">Quantity</th>
              <th className="text-right py-1">k = 3</th>
              <th className="text-right py-1">k = 5</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-slate-100">
              <td className="py-1">p₀ → pₖ</td>
              <td className="text-right py-1">
                {fmtNumber(ex.p0, 4)} → {fmtNumber(ex.p3, 4)}
              </td>
              <td className="text-right py-1">
                {fmtNumber(ex.p0, 4)} → {fmtNumber(ex.p5, 4)}
              </td>
            </tr>
            <tr className="border-b border-slate-100">
              <td className="py-1">x_range</td>
              <td className="text-right py-1" colSpan={2}>
                {fmtNumber(ex.x_range, 1)}
              </td>
            </tr>
            <tr className="border-b border-slate-100">
              <td className="py-1">y_range</td>
              <td className="text-right py-1" colSpan={2}>
                {fmtNumber(ex.y_range, 4)}
              </td>
            </tr>
            <tr className="border-b border-slate-100">
              <td className="py-1">u = k / x_range</td>
              <td className="text-right py-1">{fmtNumber(ex.u3, 4)}</td>
              <td className="text-right py-1">{fmtNumber(ex.u5, 4)}</td>
            </tr>
            <tr className="border-b border-slate-100">
              <td className="py-1">v = (pₖ - p₀) / y_range</td>
              <td className="text-right py-1">{fmtSigned(ex.v3, 4)}</td>
              <td className="text-right py-1">{fmtSigned(ex.v5, 4)}</td>
            </tr>
            <tr className="border-b border-slate-100">
              <td className="py-1">norm = √(u² + v² + ε)</td>
              <td className="text-right py-1">{fmtNumber(ex.norm3, 4)}</td>
              <td className="text-right py-1">{fmtNumber(ex.norm5, 4)}</td>
            </tr>
            <tr className="border-b border-slate-100">
              <td className="py-1">align = s · v / norm</td>
              <td
                className={`text-right py-1 ${
                  ex.align3 >= 0 ? 'text-teal-700' : 'text-orange-700'
                }`}
              >
                {fmtSigned(ex.align3, 4)}
              </td>
              <td
                className={`text-right py-1 ${
                  ex.align5 >= 0 ? 'text-teal-700' : 'text-orange-700'
                }`}
              >
                {fmtSigned(ex.align5, 4)}
              </td>
            </tr>
            <tr className="font-semibold">
              <td className="py-1">final = s · max(0, align)</td>
              <td className="text-right py-1">{fmtSigned(ex.final3, 4)}</td>
              <td className="text-right py-1">{fmtSigned(ex.final5, 4)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
