import { useEffect, useMemo, useState } from 'react';
import { loadModel, loadSessions } from '../lib/dataLoaders';
import type { ModelData, SessionData } from '../lib/types';
import { fmtSharpe, fmtSigned } from '../lib/format';
import Plot from '../lib/Plot';

// ────────────────────────────────────────────────────────────────────────────
// Pitch — single-page dashboard for live presentation.
// Pick a session, walk through its candlestick + the per-bar sentiment
// evolution (the model's view of "what does the news add up to so far"),
// and the resulting Ridge contribution breakdown.
// ────────────────────────────────────────────────────────────────────────────

const LLM_SYMBOL: Record<string, string> = {
  pos: 'triangle-up',
  neutral: 'circle',
  neg: 'triangle-down',
};

// Compute the per-bar (t = 0..49) accumulation of every sentiment feature
// the model uses. This is "what would the model see at bar t if asked to
// predict now". Mirrors solution.extract_features but evaluated at every t.
function computeSentimentEvolution(
  session: SessionData,
  decayPos: number,
  decayNeg: number,
) {
  const T = 50;
  const ts = Array.from({ length: T }, (_, t) => t);
  const headlines = session.headlines.filter(
    (h) => h.b <= 49 && h.fb !== 0,
  );

  const fp = new Array(T).fill(0);
  const fn = new Array(T).fill(0);
  const fpA3 = new Array(T).fill(0);
  const fnA3 = new Array(T).fill(0);
  const fpA5 = new Array(T).fill(0);
  const fnA5 = new Array(T).fill(0);
  const llmNeg = new Array(T).fill(0);

  for (let t = 0; t < T; t++) {
    for (const h of headlines) {
      if (h.b > t) continue;
      const age = t - h.b;
      const wp = Math.exp(-decayPos * age);
      const wn = Math.exp(-decayNeg * age);
      if (h.fb > 0) {
        fp[t] += h.fb * wp;
        if (h.final3 !== undefined) fpA3[t] += h.final3 * wp;
        if (h.final5 !== undefined) fpA5[t] += h.final5 * wp;
      } else {
        fn[t] += h.fb * wn;
        if (h.final3 !== undefined) fnA3[t] += h.final3 * wn;
        if (h.final5 !== undefined) fnA5[t] += h.final5 * wn;
      }
      if (h.llm === 'neg') {
        llmNeg[t] -= Math.exp(-decayNeg * age);
      }
    }
  }
  const conf = ts.map((_, t) => {
    const net = fp[t] + fn[t];
    const gross = fp[t] + Math.abs(fn[t]);
    return (net * Math.abs(net)) / (gross + 1e-8);
  });
  return { ts, fp, fn, conf, fpA3, fnA3, fpA5, fnA5, llmNeg };
}

// Compute per-bar vol-normalized returns (ret_all/last5/last20) using a
// running Parkinson-vol estimate up to bar t.
function computeReturnsTimeSeries(session: SessionData) {
  const bars = session.seen_bars;
  const T = bars.length;
  const o0 = bars[0].o ?? bars[0].c;
  const closes = bars.map((b) => b.c);

  // Running Parkinson vol up to each t (inclusive of bars 0..t).
  const ln2 = Math.log(2);
  const parkinsonAccum = new Array(T).fill(0);
  let sumSq = 0;
  for (let t = 0; t < T; t++) {
    const h = bars[t].h ?? bars[t].c;
    const l = bars[t].l ?? bars[t].c;
    const r = Math.log(h / l);
    sumSq += r * r;
    const variance = sumSq / (4 * ln2 * (t + 1));
    parkinsonAccum[t] = Math.max(Math.sqrt(variance), 1e-6);
  }
  const retAll = closes.map((c, t) => (c / o0 - 1) / parkinsonAccum[t]);
  const ret5 = closes.map((c, t) => {
    const k = Math.max(0, t - 5);
    return (c / closes[k] - 1) / parkinsonAccum[t];
  });
  const ret20 = closes.map((c, t) => {
    const k = Math.max(0, t - 20);
    return (c / closes[k] - 1) / parkinsonAccum[t];
  });

  // Running cand_up_ratio.
  const upRatio = new Array(T).fill(0);
  let upCount = 0;
  for (let t = 1; t < T; t++) {
    if (Math.log(closes[t]) > Math.log(closes[t - 1])) upCount += 1;
    upRatio[t] = upCount / t;
  }

  return {
    ts: bars.map((b) => b.b),
    retAll,
    ret5,
    ret20,
    upRatio,
    parkinson: parkinsonAccum,
  };
}

// ────────────────────────────────────────────────────────────────────────────
// Sub-components
// ────────────────────────────────────────────────────────────────────────────

function StatCard({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: string;
}) {
  return (
    <div className="bg-white rounded-lg border border-slate-200 px-4 py-3">
      <div className="text-xs uppercase tracking-wide text-slate-500">
        {label}
      </div>
      <div
        className="text-2xl font-mono font-semibold mt-0.5"
        style={{ color: accent ?? '#0f172a' }}
      >
        {value}
      </div>
    </div>
  );
}

function CandleChart({ session }: { session: SessionData }) {
  const seen = session.seen_bars;
  const unseen = session.unseen_bars;
  const closeMap = new Map<number, number>();
  for (const b of seen) closeMap.set(b.b, b.c);
  for (const b of unseen) closeMap.set(b.b, b.c);
  const all = [...seen, ...unseen];
  const yMin = Math.min(...all.map((b) => b.l ?? b.c));
  const yMax = Math.max(...all.map((b) => b.h ?? b.c));
  const yPad = (yMax - yMin) * 0.05;

  return (
    <Plot
      data={[
        {
          type: 'candlestick',
          x: seen.map((b) => b.b),
          open: seen.map((b) => b.o ?? b.c),
          high: seen.map((b) => b.h ?? b.c),
          low: seen.map((b) => b.l ?? b.c),
          close: seen.map((b) => b.c),
          increasing: { line: { color: '#0d9488' } },
          decreasing: { line: { color: '#ea580c' } },
          showlegend: false,
        },
        {
          type: 'scatter',
          mode: 'lines',
          x: unseen.map((b) => b.b),
          y: unseen.map((b) => b.c),
          line: { color: '#94a3b8', dash: 'dash', width: 2 },
          showlegend: false,
          hovertemplate: 'bar %{x}<br>close %{y:.4f}<extra>unseen</extra>',
        },
        {
          type: 'scatter',
          mode: 'markers',
          x: session.headlines.map((h) => h.b),
          y: session.headlines.map((h) => closeMap.get(h.b) ?? 1),
          marker: {
            color: session.headlines.map((h) => h.fb),
            colorscale: 'RdBu',
            cmin: -1,
            cmax: 1,
            size: 11,
            symbol: session.headlines.map((h) => LLM_SYMBOL[h.llm] ?? 'circle'),
            line: { color: '#1e293b', width: 1 },
          },
          text: session.headlines.map(
            (h) =>
              `<b>bar ${h.b}</b><br>${h.t.replace(/</g, '&lt;')}<br>` +
              `FinBERT ${fmtSigned(h.fb, 3)} · LLM ${h.llm}<extra></extra>`,
          ),
          hovertemplate: '%{text}',
          showlegend: false,
        },
      ]}
      layout={{
        height: 360,
        margin: { l: 50, r: 20, t: 10, b: 40 },
        xaxis: { title: 'bar', range: [-1, 100], gridcolor: '#e2e8f0', rangeslider: { visible: false } },
        yaxis: { title: 'price', gridcolor: '#e2e8f0', range: [yMin - yPad, yMax + yPad] },
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
            y: 1,
            yanchor: 'bottom',
            text: 'prediction point',
            showarrow: false,
            font: { size: 10, color: '#475569' },
          },
        ],
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        font: { family: 'system-ui', size: 11, color: '#0f172a' },
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
      useResizeHandler
    />
  );
}

function SentimentEvolutionChart({
  session,
  m,
}: {
  session: SessionData;
  m: ModelData;
}) {
  const data = useMemo(
    () => computeSentimentEvolution(session, m.decay_pos, m.decay_neg),
    [session, m.decay_pos, m.decay_neg],
  );
  return (
    <Plot
      data={[
        { x: data.ts, y: data.fp,    type: 'scatter', mode: 'lines', name: 'finbert_pos',         line: { color: '#15803d', width: 2 } },
        { x: data.ts, y: data.fn,    type: 'scatter', mode: 'lines', name: 'finbert_neg',         line: { color: '#dc2626', width: 2 } },
        { x: data.ts, y: data.conf,  type: 'scatter', mode: 'lines', name: 'finbert_conf_belief', line: { color: '#0f172a', width: 2 } },
        { x: data.ts, y: data.fpA3,  type: 'scatter', mode: 'lines', name: 'pos_align3',          line: { color: '#22c55e', width: 1.3, dash: 'dash' } },
        { x: data.ts, y: data.fnA3,  type: 'scatter', mode: 'lines', name: 'neg_align3',          line: { color: '#fb7185', width: 1.3, dash: 'dash' } },
        { x: data.ts, y: data.fpA5,  type: 'scatter', mode: 'lines', name: 'pos_align5',          line: { color: '#16a34a', width: 1.3, dash: 'dot' } },
        { x: data.ts, y: data.fnA5,  type: 'scatter', mode: 'lines', name: 'neg_align5',          line: { color: '#b91c1c', width: 1.3, dash: 'dot' } },
        { x: data.ts, y: data.llmNeg, type: 'scatter', mode: 'lines', name: 'llm_neg_decay',      line: { color: '#7c2d12', width: 1.5, dash: 'dashdot' } },
      ]}
      layout={{
        height: 320,
        margin: { l: 50, r: 20, t: 10, b: 40 },
        xaxis: { title: 'bar (prediction-time-equivalent)', range: [0, 49], gridcolor: '#e2e8f0' },
        yaxis: { title: 'sentiment score', gridcolor: '#e2e8f0', zerolinecolor: '#94a3b8' },
        legend: {
          orientation: 'v',
          x: 1.02,
          y: 1,
          font: { family: 'monospace', size: 10 },
        },
        shapes: [
          {
            type: 'line',
            x0: 49,
            x1: 49,
            yref: 'paper',
            y0: 0,
            y1: 1,
            line: { color: '#475569', width: 1, dash: 'dot' },
          },
        ],
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        font: { family: 'system-ui', size: 11, color: '#0f172a' },
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
      useResizeHandler
    />
  );
}

function FeatureContributionChart({
  session,
  m,
}: {
  session: SessionData;
  m: ModelData;
}) {
  const rows = useMemo(() => {
    return m.feature_cols
      .map((name) => {
        const raw = session.features[name];
        const mean = m.feature_means[name];
        const std = m.feature_stds[name];
        const standardized = std > 0 ? (raw - mean) / std : 0;
        const coef = m.coefs[name];
        const contribution = standardized * coef;
        return { name, contribution };
      })
      .sort((a, b) => Math.abs(a.contribution) - Math.abs(b.contribution));
  }, [session, m]);

  return (
    <Plot
      data={[
        {
          type: 'bar',
          orientation: 'h',
          y: rows.map((r) => r.name),
          x: rows.map((r) => r.contribution),
          marker: {
            color: rows.map((r) => (r.contribution >= 0 ? '#0d9488' : '#ea580c')),
          },
          text: rows.map((r) => fmtSigned(r.contribution, 3)),
          textposition: 'outside',
          hovertemplate: '<b>%{y}</b><br>contribution = %{x:+.4f}<extra></extra>',
        },
      ]}
      layout={{
        height: 360,
        margin: { l: 160, r: 60, t: 10, b: 40 },
        xaxis: {
          title: 'standardized × coef',
          zeroline: true,
          zerolinecolor: '#94a3b8',
          gridcolor: '#e2e8f0',
        },
        yaxis: { automargin: true, tickfont: { family: 'monospace', size: 11 } },
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        showlegend: false,
        font: { family: 'system-ui', size: 11, color: '#0f172a' },
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
      useResizeHandler
    />
  );
}

function MiniLineChart({
  ts,
  series,
  height = 180,
  yZero = false,
  yLabel,
}: {
  ts: number[];
  series: { name: string; y: number[]; color: string }[];
  height?: number;
  yZero?: boolean;
  yLabel?: string;
}) {
  return (
    <Plot
      data={series.map((s) => ({
        x: ts,
        y: s.y,
        type: 'scatter',
        mode: 'lines',
        name: s.name,
        line: { color: s.color, width: 1.5 },
      }))}
      layout={{
        height,
        margin: { l: 40, r: 10, t: 10, b: 50 },
        xaxis: { range: [0, 49], gridcolor: '#e2e8f0' },
        yaxis: {
          gridcolor: '#e2e8f0',
          zerolinecolor: '#94a3b8',
          ...(yLabel ? { title: yLabel } : {}),
          ...(yZero ? { zeroline: true } : {}),
        },
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        legend: {
          orientation: 'h',
          x: 0.5,
          xanchor: 'center',
          y: -0.15,
          yanchor: 'top',
          font: { family: 'monospace', size: 9 },
        },
        font: { family: 'system-ui', size: 10, color: '#0f172a' },
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
      useResizeHandler
    />
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Main
// ────────────────────────────────────────────────────────────────────────────

export default function Presentation() {
  const [m, setM] = useState<ModelData | null>(null);
  const [sessions, setSessions] = useState<SessionData[] | null>(null);
  const [sid, setSid] = useState(0);

  useEffect(() => {
    loadModel().then(setM);
    loadSessions().then(setSessions);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      if (!sessions) return;
      if (e.key === 'ArrowRight') setSid((s) => Math.min(sessions.length - 1, s + 1));
      else if (e.key === 'ArrowLeft') setSid((s) => Math.max(0, s - 1));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [sessions]);

  if (!m || !sessions)
    return <div className="panel">Loading…</div>;

  const session = sessions[sid];
  const N = sessions.length;
  const returns = computeReturnsTimeSeries(session);

  return (
    <div className="space-y-3">
      {/* Header banner */}
      <div className="panel space-y-3">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <div>
            <div className="text-xs uppercase tracking-widest text-indigo-600 font-semibold">
              HRT Datathon 2026 · Pitch dashboard
            </div>
            <h2 className="text-2xl font-semibold text-slate-900">
              Ridge + alignment-weighted FinBERT, augmented training
            </h2>
          </div>
          <div className="flex flex-wrap gap-2">
            <StatCard label="Public LB Sharpe" value={fmtSharpe(m.lb_score_public)} accent="#4338ca" />
            <StatCard label="CV Sharpe (5-fold GroupKFold)" value={fmtSharpe(m.cv_mean_sharpe)} accent="#0d9488" />
            <StatCard label="α*" value={String(m.alpha_star)} />
            <StatCard
              label="Train (orig + aug)"
              value={`${m.n_train_original ?? '—'} + ${m.n_train_augmented ?? '—'}`}
            />
          </div>
        </div>

        {/* Session picker */}
        <div className="flex flex-wrap items-center gap-3 pt-1 border-t border-slate-100">
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
            {session.headlines.length} headlines · vol = {session.vol.toFixed(5)} ·
            ŷ = <span className={session.prediction_vol_adj >= 0 ? 'text-teal-700' : 'text-orange-700'}>
              {fmtSigned(session.prediction_vol_adj, 3)}
            </span> ·
            y = <span className={session.actual_vol_adj >= 0 ? 'text-teal-700' : 'text-orange-700'}>
              {fmtSigned(session.actual_vol_adj, 3)}
            </span> ·
            position = {fmtSigned(session.prediction_raw_position, 2)} ·
            PnL = <span className={session.pnl_session >= 0 ? 'text-teal-700' : 'text-orange-700'}>
              {fmtSigned(session.pnl_session, 3)}
            </span>
          </span>
        </div>
      </div>

      {/* Section 1 — Candlestick + headlines */}
      <section className="panel">
        <h3 className="text-sm font-semibold text-slate-700 mb-2">
          Price + headlines · candles 0-49 (seen) · dashed line 50-99 (unseen)
        </h3>
        <CandleChart session={session} />
      </section>

      {/* Section 2 — Sentiment evolution (the headline visual) */}
      <section className="panel">
        <h3 className="text-sm font-semibold text-slate-700 mb-1">
          Sentiment evolution · running totals over bars 0-49
        </h3>
        <p className="text-xs text-slate-500 mb-2">
          Each line is what the model would see at bar t if asked to predict
          there. Solid = raw FinBERT pos / neg / confidence. Dashed/dotted =
          alignment-weighted variants ({'k = 3, 5'}). Dash-dot brown ={' '}
          time-decayed count of LLM-tagged negative headlines.
        </p>
        <SentimentEvolutionChart session={session} m={m} />
      </section>

      {/* Section 3 — Feature contributions */}
      <section className="panel">
        <h3 className="text-sm font-semibold text-slate-700 mb-1">
          Prediction breakdown · standardised feature × coef for this session
        </h3>
        <p className="text-xs text-slate-500 mb-2">
          Sum of bars + intercept ={' '}
          <span className="font-mono text-slate-900">
            {fmtSigned(session.prediction_vol_adj, 3)}
          </span>{' '}
          (vol-adjusted prediction). Multiply by 1/vol and rescale to get the
          raw position {fmtSigned(session.prediction_raw_position, 2)}.
        </p>
        <FeatureContributionChart session={session} m={m} />
      </section>

      {/* Section 4 — supporting time-series */}
      <section className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <div className="panel">
          <h3 className="text-sm font-semibold text-slate-700 mb-1">
            Vol-normalised returns
          </h3>
          <p className="text-xs text-slate-500 mb-2">
            Sharpe-like quantities computed at each bar with a running Parkinson vol.
          </p>
          <MiniLineChart
            ts={returns.ts}
            yZero
            series={[
              { name: 'ret_all_vol', y: returns.retAll, color: '#7c3aed' },
              { name: 'ret_last5_vol', y: returns.ret5, color: '#f59e0b' },
              { name: 'ret_last20_vol', y: returns.ret20, color: '#1d4ed8' },
            ]}
          />
        </div>
        <div className="panel">
          <h3 className="text-sm font-semibold text-slate-700 mb-1">
            cand_up_ratio
          </h3>
          <p className="text-xs text-slate-500 mb-2">
            Fraction of up-bars to date — 0.5 line = unbiased random walk.
          </p>
          <MiniLineChart
            ts={returns.ts}
            series={[{ name: 'cand_up_ratio', y: returns.upRatio, color: '#06b6d4' }]}
          />
        </div>
        <div className="panel">
          <h3 className="text-sm font-semibold text-slate-700 mb-1">
            Parkinson volatility (running)
          </h3>
          <p className="text-xs text-slate-500 mb-2">
            Range-based vol estimate using bars 0..t.
          </p>
          <MiniLineChart
            ts={returns.ts}
            series={[{ name: 'parkinson', y: returns.parkinson, color: '#0d9488' }]}
          />
        </div>
      </section>

      <p className="text-xs text-slate-400 text-center">
        Use ← / → to flip through sessions live.
      </p>

      <ReferenceDashboardToggle />
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Hidden reference: collapsible toggle for the original matplotlib dashboard
// (kept as a backup view to flash up during the talk if needed).
// Source image lives at web/public/matplotlib-dashboard.jpeg.
// ────────────────────────────────────────────────────────────────────────────
function ReferenceDashboardToggle() {
  const [open, setOpen] = useState(false);
  const [error, setError] = useState(false);
  const src = `${import.meta.env.BASE_URL}matplotlib-dashboard.jpeg`;

  return (
    <section className="panel">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-slate-700">
            Original matplotlib dashboard (reference)
          </h3>
          <p className="text-xs text-slate-500">
            The static all-in-one figure we built first. Toggle to flash it up
            during the talk.
          </p>
        </div>
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="px-3 py-1 text-sm rounded bg-slate-700 text-white hover:bg-slate-800"
        >
          {open ? 'hide' : 'show reference dashboard'}
        </button>
      </div>
      {open && (
        <div className="mt-4">
          {error ? (
            <p className="text-sm text-orange-700 font-mono">
              Could not load <code>{src}</code>. Save the image at{' '}
              <code>web/public/matplotlib-dashboard.jpeg</code> and rebuild.
            </p>
          ) : (
            <img
              src={src}
              alt="Original matplotlib model-walkthrough dashboard"
              className="w-full h-auto rounded border border-slate-200"
              onError={() => setError(true)}
            />
          )}
        </div>
      )}
    </section>
  );
}
