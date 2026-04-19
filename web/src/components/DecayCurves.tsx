import { useEffect, useState } from 'react';
import { loadModel } from '../lib/dataLoaders';
import type { ModelData } from '../lib/types';
import Plot from '../lib/Plot';
import MathBlock from './MathBlock';

const AGE_MAX = 49;
const ages = Array.from({ length: AGE_MAX + 1 }, (_, i) => i);

function decayCurve(d: number) {
  return ages.map((a) => Math.exp(-d * a));
}

export default function DecayCurves() {
  const [m, setM] = useState<ModelData | null>(null);
  const [dPos, setDPos] = useState<number | null>(null);
  const [dNeg, setDNeg] = useState<number | null>(null);

  useEffect(() => {
    loadModel().then((data) => {
      setM(data);
      setDPos(data.decay_pos);
      setDNeg(data.decay_neg);
    });
  }, []);

  if (!m || dPos == null || dNeg == null)
    return <div className="panel">Loading…</div>;

  const reset = () => {
    setDPos(m.decay_pos);
    setDNeg(m.decay_neg);
  };

  const halfLife = (d: number) => Math.log(2) / d;

  return (
    <div className="space-y-4">
      <div className="panel space-y-2">
        <h2 className="text-2xl font-semibold text-slate-900">Decay curves</h2>
        <p className="text-sm text-slate-600">
          A headline at age <MathBlock formula="a = 49 - b_h" /> contributes its
          score with weight{' '}
          <MathBlock formula="w(a) = e^{-d \cdot a}" />. The decays are{' '}
          <em>learned</em> by differentiating Sharpe through the Ridge
          closed-form (see Intro). Asymmetric d₊ vs d₋ lets positive and
          negative news fade at different rates.
        </p>
        <p className="text-xs text-slate-500">
          Move the sliders to explore — these adjust the curves visually only,
          they do not retrain the model.
        </p>
      </div>

      <div className="panel">
        <Plot
          data={[
            {
              x: ages,
              y: decayCurve(dPos),
              type: 'scatter',
              mode: 'lines',
              name: `d₊ = ${dPos.toFixed(5)}`,
              line: { color: '#0d9488', width: 2.5 },
              hovertemplate: 'age %{x} → w = %{y:.3f}<extra>d₊</extra>',
            },
            {
              x: ages,
              y: decayCurve(dNeg),
              type: 'scatter',
              mode: 'lines',
              name: `d₋ = ${dNeg.toFixed(5)}`,
              line: { color: '#ea580c', width: 2.5 },
              hovertemplate: 'age %{x} → w = %{y:.3f}<extra>d₋</extra>',
            },
            {
              x: [0, AGE_MAX],
              y: [0.5, 0.5],
              type: 'scatter',
              mode: 'lines',
              line: { color: '#94a3b8', dash: 'dot', width: 1 },
              showlegend: false,
              hoverinfo: 'skip',
            },
          ]}
          layout={{
            height: 360,
            margin: { l: 50, r: 20, t: 20, b: 50 },
            xaxis: {
              title: 'age (bars before prediction point)',
              range: [0, AGE_MAX],
              gridcolor: '#e2e8f0',
            },
            yaxis: {
              title: 'weight',
              range: [0, 1.02],
              gridcolor: '#e2e8f0',
            },
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            legend: {
              orientation: 'h',
              x: 0.5,
              xanchor: 'center',
              y: 1.08,
              font: { family: 'monospace' },
            },
            annotations: [
              {
                x: AGE_MAX,
                y: 0.5,
                text: 'half-weight',
                xanchor: 'right',
                yanchor: 'bottom',
                showarrow: false,
                font: { size: 10, color: '#64748b' },
              },
            ],
            font: { family: 'system-ui', size: 12, color: '#0f172a' },
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%' }}
          useResizeHandler
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="panel space-y-3">
          <div className="flex items-baseline justify-between">
            <span className="font-mono text-teal-700 font-semibold">
              d₊ (positive headlines)
            </span>
            <span className="font-mono text-sm">{dPos.toFixed(5)}</span>
          </div>
          <input
            type="range"
            min={0.005}
            max={0.3}
            step={0.001}
            value={dPos}
            onChange={(e) => setDPos(parseFloat(e.target.value))}
            className="w-full accent-teal-600"
          />
          <div className="text-xs text-slate-500 font-mono">
            half-life ≈ {halfLife(dPos).toFixed(1)} bars · learned ={' '}
            {m.decay_pos.toFixed(5)}
          </div>
        </div>

        <div className="panel space-y-3">
          <div className="flex items-baseline justify-between">
            <span className="font-mono text-orange-700 font-semibold">
              d₋ (negative headlines)
            </span>
            <span className="font-mono text-sm">{dNeg.toFixed(5)}</span>
          </div>
          <input
            type="range"
            min={0.005}
            max={0.3}
            step={0.001}
            value={dNeg}
            onChange={(e) => setDNeg(parseFloat(e.target.value))}
            className="w-full accent-orange-600"
          />
          <div className="text-xs text-slate-500 font-mono">
            half-life ≈ {halfLife(dNeg).toFixed(1)} bars · learned ={' '}
            {m.decay_neg.toFixed(5)}
          </div>
        </div>
      </div>

      <div className="text-right">
        <button
          type="button"
          onClick={reset}
          className="text-sm text-indigo-700 hover:text-indigo-900 underline"
        >
          reset to learned values
        </button>
      </div>
    </div>
  );
}
