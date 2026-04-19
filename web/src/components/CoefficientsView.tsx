import { useEffect, useMemo, useState } from 'react';
import { loadModel } from '../lib/dataLoaders';
import type { ModelData } from '../lib/types';
import { fmtSigned, fmtSharpe } from '../lib/format';
import Plot from '../lib/Plot';
import MathBlock from './MathBlock';

const FEATURE_INTUITION: Record<string, { formula: string; reads: string }> = {
  ret_all_vol: {
    formula: '(C_{49}/O_0 - 1) / \\text{vol}',
    reads: 'Full first-half return, vol-scaled.',
  },
  ret_last5_vol: {
    formula: '(C_{49}/C_{44} - 1) / \\text{vol}',
    reads: 'Short-term momentum (5 bars).',
  },
  ret_last20_vol: {
    formula: '(C_{49}/C_{29} - 1) / \\text{vol}',
    reads: 'Medium-term mean-reversion (20 bars).',
  },
  finbert_pos_align3: {
    formula: '\\sum_{s>0} \\mathrm{final}_3 \\cdot e^{-d_+ \\cdot \\text{age}}',
    reads: 'Positive headlines whose 3-bar price motion confirmed them.',
  },
  finbert_neg_align3: {
    formula: '\\sum_{s<0} \\mathrm{final}_3 \\cdot e^{-d_- \\cdot \\text{age}}',
    reads: 'Negative headlines whose 3-bar price motion confirmed them.',
  },
  finbert_pos_align5: {
    formula: '\\sum_{s>0} \\mathrm{final}_5 \\cdot e^{-d_+ \\cdot \\text{age}}',
    reads: 'Same as above with a 5-bar look-ahead.',
  },
  finbert_neg_align5: {
    formula: '\\sum_{s<0} \\mathrm{final}_5 \\cdot e^{-d_- \\cdot \\text{age}}',
    reads: 'Same as above with a 5-bar look-ahead.',
  },
  finbert_conf_belief: {
    formula: '\\frac{\\text{net} \\cdot |\\text{net}|}{\\text{gross} + \\epsilon}',
    reads: 'Quadratic in net sentiment; suppressed when pos and neg disagree.',
  },
  llm_neg_decay: {
    formula: '-\\!\\sum_{\\text{LLM}=\\text{neg}} e^{-d_- \\cdot \\text{age}}',
    reads:
      'Time-decayed count of LLM-labelled negative headlines. Signed negative by construction.',
  },
  cand_up_ratio: {
    formula: '\\tfrac{1}{49}\\#\\{t : \\log C_{t+1} > \\log C_t\\}',
    reads: 'Fraction of up-bars in the seen half — captures path shape.',
  },
};

export default function CoefficientsView() {
  const [m, setM] = useState<ModelData | null>(null);
  useEffect(() => {
    loadModel().then(setM);
  }, []);

  const sorted = useMemo(() => {
    if (!m) return [];
    return m.feature_cols
      .map((name) => ({ name, coef: m.coefs[name] ?? 0 }))
      .sort((a, b) => Math.abs(a.coef) - Math.abs(b.coef));
  }, [m]);

  if (!m) return <div className="panel">Loading…</div>;

  const colors = sorted.map((r) => (r.coef >= 0 ? '#0d9488' : '#ea580c'));

  return (
    <div className="space-y-4">
      <div className="panel">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <h2 className="text-2xl font-semibold text-slate-900">
            Standardized Ridge coefficients
          </h2>
          <div className="text-sm text-slate-500 font-mono">
            α* = {m.alpha_star} · CV Sharpe ={' '}
            <span className="text-slate-900 font-semibold">
              {fmtSharpe(m.cv_mean_sharpe)}
            </span>{' '}
            · intercept = {fmtSigned(m.intercept, 4)}
          </div>
        </div>
        <p className="text-sm text-slate-600 mt-2">
          Each feature is z-standardised (zero mean, unit variance on the
          training set) before fitting, so the magnitudes are directly
          comparable. Positive (teal) = feature ↑ ⇒ prediction ↑; negative
          (orange) = feature ↑ ⇒ prediction ↓.
        </p>
      </div>

      <div className="panel">
        <Plot
          data={[
            {
              type: 'bar',
              orientation: 'h',
              y: sorted.map((r) => r.name),
              x: sorted.map((r) => r.coef),
              marker: { color: colors },
              text: sorted.map((r) => fmtSigned(r.coef, 4)),
              textposition: 'outside',
              hovertemplate: '<b>%{y}</b><br>coef = %{x:+.5f}<extra></extra>',
            },
          ]}
          layout={{
            height: 420,
            margin: { l: 160, r: 60, t: 20, b: 40 },
            xaxis: {
              title: 'standardized coefficient',
              zeroline: true,
              zerolinecolor: '#94a3b8',
              gridcolor: '#e2e8f0',
            },
            yaxis: { automargin: true, tickfont: { family: 'monospace' } },
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

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {[...sorted].reverse().map(({ name, coef }) => {
          const info = FEATURE_INTUITION[name];
          return (
            <div key={name} className="panel">
              <div className="flex items-baseline justify-between gap-2">
                <span className="font-mono text-sm text-slate-900">{name}</span>
                <span
                  className={`font-mono text-sm font-semibold ${
                    coef >= 0 ? 'text-teal-700' : 'text-orange-700'
                  }`}
                >
                  {fmtSigned(coef, 5)}
                </span>
              </div>
              {info && (
                <>
                  <div className="my-1 text-slate-700">
                    <MathBlock formula={info.formula} />
                  </div>
                  <p className="text-xs text-slate-500 leading-snug">
                    {info.reads}
                  </p>
                </>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
