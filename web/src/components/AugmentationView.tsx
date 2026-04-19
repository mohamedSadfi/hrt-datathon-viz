import { useEffect, useMemo, useState } from 'react';
import { loadModel, loadSessions } from '../lib/dataLoaders';
import type { ModelData, SessionData, HeadlineEntry, Bar } from '../lib/types';
import Plot from '../lib/Plot';
import MathBlock from './MathBlock';
import { fmtSigned } from '../lib/format';

const LLM_SYMBOL: Record<string, string> = {
  pos: 'triangle-up',
  neutral: 'circle',
  neg: 'triangle-down',
};

// ────────────────────────────────────────────────────────────────────────────
// Static SVG: timeline of the two splits along the bar 0-99 axis.
// Exported so Intro can render a small version too.
// ────────────────────────────────────────────────────────────────────────────
export function TimelineDiagram({ splitBar }: { splitBar: number }) {
  const W = 720;
  const H = 200;
  const padL = 50;
  const padR = 30;
  const innerW = W - padL - padR;
  const xOf = (b: number) => padL + (b / 99) * innerW;

  const augStart = splitBar - 49;
  const rowYOrig = 60;
  const rowYAug = 130;
  const rowH = 28;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
      {/* Axis */}
      <line
        x1={padL}
        y1={H - 30}
        x2={W - padR}
        y2={H - 30}
        stroke="#94a3b8"
        strokeWidth={1}
      />
      {[0, 25, 49, 50, splitBar, splitBar + 1, 99].map((b) => (
        <g key={b}>
          <line
            x1={xOf(b)}
            x2={xOf(b)}
            y1={H - 30}
            y2={H - 26}
            stroke="#94a3b8"
            strokeWidth={1}
          />
          <text
            x={xOf(b)}
            y={H - 14}
            textAnchor="middle"
            fontSize="11"
            fill="#475569"
            fontFamily="monospace"
          >
            {b}
          </text>
        </g>
      ))}
      <text
        x={W / 2}
        y={H - 2}
        textAnchor="middle"
        fontSize="10"
        fill="#94a3b8"
      >
        bar index
      </text>

      {/* Overlap shading (visible behind both rows) */}
      <rect
        x={xOf(augStart)}
        y={rowYOrig - 5}
        width={xOf(49) - xOf(augStart)}
        height={rowYAug + rowH - rowYOrig + 10}
        fill="#fef3c7"
        opacity={0.5}
      />

      {/* Original row */}
      <text
        x={padL - 8}
        y={rowYOrig + rowH / 2 + 4}
        textAnchor="end"
        fontSize="12"
        fill="#0f172a"
        fontWeight="600"
      >
        Original
      </text>
      <rect
        x={xOf(0)}
        y={rowYOrig}
        width={xOf(49) - xOf(0)}
        height={rowH}
        fill="#0d9488"
        rx={3}
      />
      <text
        x={(xOf(0) + xOf(49)) / 2}
        y={rowYOrig + rowH / 2 + 4}
        textAnchor="middle"
        fontSize="11"
        fill="white"
        fontWeight="600"
      >
        seen 0-49
      </text>
      <rect
        x={xOf(50)}
        y={rowYOrig}
        width={xOf(99) - xOf(50)}
        height={rowH}
        fill="none"
        stroke="#94a3b8"
        strokeDasharray="4 3"
        strokeWidth={1.5}
        rx={3}
      />
      <text
        x={(xOf(50) + xOf(99)) / 2}
        y={rowYOrig + rowH / 2 + 4}
        textAnchor="middle"
        fontSize="11"
        fill="#64748b"
      >
        unseen 50-99
      </text>

      {/* Augmented row */}
      <text
        x={padL - 8}
        y={rowYAug + rowH / 2 + 4}
        textAnchor="end"
        fontSize="12"
        fill="#0f172a"
        fontWeight="600"
      >
        Augmented
      </text>
      <rect
        x={xOf(augStart)}
        y={rowYAug}
        width={xOf(splitBar) - xOf(augStart)}
        height={rowH}
        fill="#4f46e5"
        rx={3}
      />
      <text
        x={(xOf(augStart) + xOf(splitBar)) / 2}
        y={rowYAug + rowH / 2 + 4}
        textAnchor="middle"
        fontSize="11"
        fill="white"
        fontWeight="600"
      >
        seen {augStart}-{splitBar}
      </text>
      <rect
        x={xOf(splitBar + 1)}
        y={rowYAug}
        width={xOf(99) - xOf(splitBar + 1)}
        height={rowH}
        fill="none"
        stroke="#94a3b8"
        strokeDasharray="4 3"
        strokeWidth={1.5}
        rx={3}
      />
      <text
        x={(xOf(splitBar + 1) + xOf(99)) / 2}
        y={rowYAug + rowH / 2 + 4}
        textAnchor="middle"
        fontSize="11"
        fill="#64748b"
      >
        unseen {splitBar + 1}-99
      </text>

      {/* Header */}
      <text x={padL} y={30} fontSize="13" fill="#0f172a" fontWeight="600">
        Two views of the same 100-bar session
      </text>
      <text x={padL} y={46} fontSize="10" fill="#64748b">
        Yellow band = overlap region: bars {augStart}-49 are in BOTH seen
        windows.
      </text>
    </svg>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Static SVG: GroupKFold vs vanilla KFold leakage diagram.
// ────────────────────────────────────────────────────────────────────────────
function GroupKFoldDiagram() {
  const W = 700;
  const H = 240;
  const padL = 130;
  const padR = 30;
  const innerW = W - padL - padR;
  const N_FOLDS = 5;
  const foldW = innerW / N_FOLDS;

  // For visual clarity: place orig and aug in different folds (bad), then
  // same fold (good).
  const origFoldBad = 0;
  const augFoldBad = 3;
  const sameFoldGood = 2;

  const dotR = 7;
  const FOLD_COLORS = ['#fee2e2', '#fef3c7', '#dbeafe', '#dcfce7', '#ede9fe'];

  function row({
    y,
    label,
    sub,
    origFold,
    augFold,
    sameLabel,
  }: {
    y: number;
    label: string;
    sub: string;
    origFold: number;
    augFold: number;
    sameLabel?: boolean;
  }) {
    const dotY = y + 38;
    return (
      <g key={label}>
        <text
          x={padL - 10}
          y={y + 28}
          textAnchor="end"
          fontSize="13"
          fontWeight="600"
          fill="#0f172a"
        >
          {label}
        </text>
        <text
          x={padL - 10}
          y={y + 44}
          textAnchor="end"
          fontSize="10"
          fill="#64748b"
        >
          {sub}
        </text>
        {Array.from({ length: N_FOLDS }, (_, i) => (
          <g key={i}>
            <rect
              x={padL + i * foldW + 4}
              y={y + 12}
              width={foldW - 8}
              height={50}
              fill={FOLD_COLORS[i]}
              stroke="#cbd5e1"
              strokeWidth={1}
              rx={4}
            />
            <text
              x={padL + i * foldW + foldW / 2}
              y={y + 8}
              textAnchor="middle"
              fontSize="9"
              fill="#94a3b8"
            >
              fold {i + 1}
            </text>
          </g>
        ))}
        {/* Original session dot */}
        <circle
          cx={padL + origFold * foldW + foldW * 0.35}
          cy={dotY}
          r={dotR}
          fill="#0d9488"
          stroke="#0f172a"
          strokeWidth={1.5}
        />
        <text
          x={padL + origFold * foldW + foldW * 0.35}
          y={dotY + 4}
          textAnchor="middle"
          fontSize="9"
          fill="white"
          fontWeight="700"
        >
          O
        </text>
        {/* Augmented twin dot */}
        <circle
          cx={padL + augFold * foldW + foldW * (sameLabel ? 0.65 : 0.35)}
          cy={dotY}
          r={dotR}
          fill="#0d9488"
          stroke="#0f172a"
          strokeWidth={1.5}
          strokeDasharray="2 2"
        />
        <text
          x={padL + augFold * foldW + foldW * (sameLabel ? 0.65 : 0.35)}
          y={dotY + 4}
          textAnchor="middle"
          fontSize="9"
          fill="white"
          fontWeight="700"
        >
          A
        </text>
      </g>
    );
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
      <text x={20} y={20} fontSize="13" fontWeight="600" fill="#0f172a">
        Why GroupKFold? (5 folds, one session's twins shown)
      </text>

      {row({
        y: 30,
        label: 'Plain KFold',
        sub: '(leakage)',
        origFold: origFoldBad,
        augFold: augFoldBad,
        sameLabel: false,
      })}
      {row({
        y: 130,
        label: 'GroupKFold',
        sub: '(safe)',
        origFold: sameFoldGood,
        augFold: sameFoldGood,
        sameLabel: true,
      })}

      <text x={20} y={222} fontSize="10" fill="#64748b">
        O = original session · A = augmented twin (same color = same group).
        With plain KFold the twins can split across train/test → the model
        sees the &quot;future&quot; of a held-out session through its
        original. GroupKFold pins both to the same fold.
      </text>
    </svg>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Plotly chart: one split's view of a session.
// ────────────────────────────────────────────────────────────────────────────
function SplitChart({
  title,
  subtitle,
  seenStart,
  seenEnd,
  allBars,
  headlines,
  accent,
}: {
  title: string;
  subtitle: string;
  seenStart: number;
  seenEnd: number;
  allBars: Bar[];
  headlines: HeadlineEntry[];
  accent: string;
}) {
  const seen = allBars.filter((b) => b.b >= seenStart && b.b <= seenEnd);
  const unseen = allBars.filter((b) => b.b > seenEnd);
  const closeMap = new Map<number, number>();
  for (const b of allBars) closeMap.set(b.b, b.c);

  const yMin = Math.min(...allBars.map((b) => b.l ?? b.c));
  const yMax = Math.max(...allBars.map((b) => b.h ?? b.c));
  const yPad = (yMax - yMin) * 0.05;

  const headlineX = headlines.map((h) => h.b);
  const headlineY = headlines.map((h) => closeMap.get(h.b) ?? 1);
  const headlineColor = headlines.map((h) => h.fb);
  const headlineSymbol = headlines.map((h) => LLM_SYMBOL[h.llm] ?? 'circle');
  const headlineText = headlines.map(
    (h) =>
      `<b>bar ${h.b}</b><br>${h.t.replace(/</g, '&lt;')}<br>` +
      `FinBERT ${fmtSigned(h.fb, 3)} · LLM ${h.llm}<extra></extra>`,
  );

  return (
    <div className="panel">
      <div className="flex items-baseline justify-between mb-2">
        <h3 className="text-sm font-semibold" style={{ color: accent }}>
          {title}
        </h3>
        <span className="text-xs font-mono text-slate-500">{subtitle}</span>
      </div>
      <Plot
        data={[
          {
            type: 'candlestick',
            x: seen.map((b) => b.b),
            open: seen.map((b) => b.o ?? b.c),
            high: seen.map((b) => b.h ?? b.c),
            low: seen.map((b) => b.l ?? b.c),
            close: seen.map((b) => b.c),
            name: 'seen',
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
            x: headlineX,
            y: headlineY,
            text: headlineText,
            hovertemplate: '%{text}',
            marker: {
              color: headlineColor,
              colorscale: 'RdBu',
              cmin: -1,
              cmax: 1,
              size: 11,
              symbol: headlineSymbol,
              line: { color: '#1e293b', width: 1 },
              showscale: false,
            },
            showlegend: false,
          },
        ]}
        layout={{
          height: 320,
          margin: { l: 50, r: 20, t: 10, b: 40 },
          xaxis: {
            title: 'bar',
            range: [-1, 100],
            gridcolor: '#e2e8f0',
            rangeslider: { visible: false },
          },
          yaxis: {
            title: 'price',
            gridcolor: '#e2e8f0',
            range: [yMin - yPad, yMax + yPad],
          },
          shapes: [
            // Highlight the seen window
            {
              type: 'rect',
              x0: seenStart - 0.5,
              x1: seenEnd + 0.5,
              yref: 'paper',
              y0: 0,
              y1: 1,
              fillcolor: accent,
              opacity: 0.06,
              line: { width: 0 },
            },
            // Prediction-point line
            {
              type: 'line',
              x0: seenEnd + 0.5,
              x1: seenEnd + 0.5,
              yref: 'paper',
              y0: 0,
              y1: 1,
              line: { color: '#475569', width: 1.5, dash: 'dot' },
            },
          ],
          annotations: [
            {
              x: seenEnd + 0.5,
              yref: 'paper',
              y: 1.0,
              yanchor: 'bottom',
              text: 'prediction point',
              showarrow: false,
              font: { size: 10, color: '#475569' },
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
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────
// Main component.
// ────────────────────────────────────────────────────────────────────────────
export default function AugmentationView() {
  const [m, setM] = useState<ModelData | null>(null);
  const [sessions, setSessions] = useState<SessionData[] | null>(null);
  const [sid, setSid] = useState(0);

  useEffect(() => {
    loadModel().then(setM);
    loadSessions().then(setSessions);
  }, []);

  const splitBar = m?.augmentation?.split_bar ?? 74;
  const augStart = splitBar - 49;

  const session = sessions?.[sid] ?? null;
  const allBars = useMemo(() => {
    if (!session) return [];
    return [...session.seen_bars, ...session.unseen_bars].sort(
      (a, b) => a.b - b.b,
    );
  }, [session]);

  const headlinesInWindow = useMemo(() => {
    if (!session) return { orig: 0, aug: 0, only_in_aug: 0 };
    const orig = session.headlines.filter((h) => h.b <= 49).length;
    const aug = session.headlines.filter(
      (h) => h.b >= augStart && h.b <= splitBar,
    ).length;
    const only_in_aug = session.headlines.filter(
      (h) => h.b > 49 && h.b <= splitBar,
    ).length;
    return { orig, aug, only_in_aug };
  }, [session, augStart, splitBar]);

  if (!m || !sessions || !session)
    return <div className="panel">Loading…</div>;
  if (!m.augmentation)
    return (
      <div className="panel">
        Augmentation is disabled in this build (no `augmentation` field in
        model.json).
      </div>
    );

  const N = sessions.length;

  return (
    <div className="space-y-4">
      {/* Intro / explanation */}
      <section className="panel space-y-3">
        <h2 className="text-2xl font-semibold text-slate-900">
          Shifted-split data augmentation
        </h2>
        <p className="text-slate-700 leading-relaxed">
          Each labelled session contains 100 bars of OHLC, but at prediction
          time the model only ever sees 50 (bars 0-49). Test sessions are
          unlabelled, so we can&apos;t use them for training — but the
          remaining 50 bars of every <em>training</em> session
          (bars 50-99) are sitting there with full ground truth. We re-use
          them by sliding the &quot;seen&quot; window forward to bar{' '}
          {splitBar} and treating the result as a synthetic second sample of
          the same session.
        </p>
        <p className="text-slate-700 leading-relaxed">
          Concretely, with{' '}
          <MathBlock formula={`\\text{split\\_bar} = ${splitBar}`} />:
        </p>
        <div className="font-mono text-sm bg-slate-50 rounded p-3 text-slate-800">
          original split: seen = bars 0-49, unseen = bars 50-99
          <br />
          augmented split: seen = bars {augStart}-{splitBar}, unseen = bars{' '}
          {splitBar + 1}-99 &nbsp;
          <span className="text-slate-500">
            (re-indexed to 0-49 internally so feature engineering is
            identical)
          </span>
        </div>
        <p className="text-slate-700 leading-relaxed">
          This doubles the labelled training set from{' '}
          <span className="font-mono">{m.n_train_original}</span> to{' '}
          <span className="font-mono">{m.n_train}</span>. Augmented session
          IDs are offset by{' '}
          <span className="font-mono">{m.augmentation.session_offset}</span>{' '}
          to avoid collision with the originals.
        </p>
      </section>

      {/* Timeline diagram */}
      <section className="panel">
        <TimelineDiagram splitBar={splitBar} />
      </section>

      {/* Picker */}
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
          {session.headlines.length} headlines · {headlinesInWindow.orig} in
          original window · {headlinesInWindow.aug} in augmented window ·{' '}
          {headlinesInWindow.only_in_aug} new for augmented
        </span>
      </div>

      {/* Side-by-side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <SplitChart
          title="Original split"
          subtitle={`seen 0-49, unseen 50-99`}
          seenStart={0}
          seenEnd={49}
          allBars={allBars}
          headlines={session.headlines}
          accent="#0d9488"
        />
        <SplitChart
          title="Augmented split"
          subtitle={`seen ${augStart}-${splitBar}, unseen ${splitBar + 1}-99`}
          seenStart={augStart}
          seenEnd={splitBar}
          allBars={allBars}
          headlines={session.headlines}
          accent="#4f46e5"
        />
      </div>
      <p className="text-xs text-slate-500">
        Same session, two windows. Headlines that fall in bars 50-{splitBar}{' '}
        are <em>future news</em> in the original split (off the right of the
        candles, on the dashed line) but become <em>training context</em> in
        the augmented split. The model sees them through different feature
        values in each sample.
      </p>

      {/* GroupKFold */}
      <section className="panel space-y-3">
        <h2 className="text-2xl font-semibold text-slate-900">
          Why GroupKFold?
        </h2>
        <p className="text-slate-700 leading-relaxed">
          Augmenting doubles the data but introduces a subtle leakage risk:
          the original sample at bars 0-49 and its augmented twin at bars{' '}
          {augStart}-{splitBar} share most of the seen window (bars{' '}
          {augStart}-49 overlap, the yellow band in the timeline). If a vanilla{' '}
          <span className="font-mono">KFold</span> happens to put one in the
          training fold and the other in the test fold, the model effectively
          sees the future of a held-out session.
        </p>
        <GroupKFoldDiagram />
        <p className="text-slate-700 leading-relaxed">
          We use{' '}
          <span className="font-mono">
            sklearn.model_selection.GroupKFold
          </span>{' '}
          with <em>group = original session id</em>; the augmented twin
          inherits the same group as its original, so both always land in the
          same fold. The reported {m.cv_fold_sharpes.length}-fold CV Sharpe
          of <span className="font-mono">{m.cv_mean_sharpe.toFixed(3)}</span>{' '}
          is therefore a faithful generalisation estimate, not an
          augmentation-inflated one.
        </p>
      </section>
    </div>
  );
}
