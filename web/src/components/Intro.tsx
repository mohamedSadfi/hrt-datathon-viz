import { useEffect, useState } from 'react';
import { loadModel } from '../lib/dataLoaders';
import type { ModelData } from '../lib/types';
import { fmtSharpe } from '../lib/format';
import MathBlock from './MathBlock';
import { TimelineDiagram } from './AugmentationView';

export default function Intro() {
  const [m, setM] = useState<ModelData | null>(null);
  useEffect(() => {
    loadModel().then(setM);
  }, []);

  return (
    <div className="space-y-6">
      {/* Headline metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="panel">
          <div className="stat-label">CV Sharpe (5-fold)</div>
          <div className="stat">{m ? fmtSharpe(m.cv_mean_sharpe) : '…'}</div>
        </div>
        <div className="panel">
          <div className="stat-label">LB Sharpe (public)</div>
          <div className="stat">{m ? fmtSharpe(m.lb_score_public) : '…'}</div>
        </div>
      </div>

      {/* Problem */}
      <section className="panel space-y-3">
        <h2 className="text-2xl font-semibold text-slate-900">
          Problem setup
        </h2>
        <p className="text-slate-700 leading-relaxed">
          Each session has 100 bars of OHLC, normalized to start at 1.0. Bars
          0-49 are <em>seen</em> at prediction time; bars 50-99 are{' '}
          <em>unseen</em>. Each session also has ~10 news headlines about
          various companies at bars{' '}
          <MathBlock formula="b_h \in [0, 49]" />. The objective is a Sharpe on
          the per-session PnL{' '}
          <MathBlock formula="\pi_i = \text{position}_i \cdot y_i" />, where the
          actual return is{' '}
          <MathBlock formula="y_i = C_{99}/C_{49} - 1" />:
        </p>
        <MathBlock
          display
          formula="\text{Sharpe} \;=\; \frac{\overline{\pi}}{\sigma(\pi)} \cdot 16"
        />
        <p className="text-sm text-slate-500">
          Sharpe is scale-invariant under positive rescaling — only the{' '}
          <em>direction</em> and <em>relative magnitude</em> of positions
          matter.
        </p>
      </section>

      {/* Features */}
      <section className="panel space-y-3">
        <h2 className="text-2xl font-semibold text-slate-900">
          The 10 features
        </h2>
        <p className="text-slate-700">
          All features are dimensionless. Price features are vol-normalized
          (Sharpe-like), so two sessions with the same{' '}
          <em>signal-to-noise</em> get the same value regardless of absolute
          volatility.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead className="text-xs uppercase text-slate-500 border-b">
              <tr>
                <th className="text-left py-2 pr-4">Feature</th>
                <th className="text-left py-2 pr-4">Formula</th>
                <th className="text-left py-2">Reads</th>
              </tr>
            </thead>
            <tbody className="text-slate-800">
              <tr className="border-b border-slate-100">
                <td className="py-2 pr-4 font-mono text-xs">ret_all_vol</td>
                <td className="py-2 pr-4">
                  <MathBlock formula="(C_{49}/O_0 - 1) / \text{vol}" />
                </td>
                <td className="py-2 text-slate-600">
                  Full first-half return, vol-scaled
                </td>
              </tr>
              <tr className="border-b border-slate-100">
                <td className="py-2 pr-4 font-mono text-xs">ret_last5_vol</td>
                <td className="py-2 pr-4">
                  <MathBlock formula="(C_{49}/C_{44} - 1) / \text{vol}" />
                </td>
                <td className="py-2 text-slate-600">5-bar momentum</td>
              </tr>
              <tr className="border-b border-slate-100">
                <td className="py-2 pr-4 font-mono text-xs">ret_last20_vol</td>
                <td className="py-2 pr-4">
                  <MathBlock formula="(C_{49}/C_{29} - 1) / \text{vol}" />
                </td>
                <td className="py-2 text-slate-600">
                  20-bar mean-reversion (largest neg coef)
                </td>
              </tr>
              <tr className="border-b border-slate-100">
                <td className="py-2 pr-4 font-mono text-xs">cand_up_ratio</td>
                <td className="py-2 pr-4">
                  <MathBlock formula="\tfrac{1}{49}\#\{t : \log C_{t+1} > \log C_t\}" />
                </td>
                <td className="py-2 text-slate-600">
                  Up-bar ratio — captures path shape
                </td>
              </tr>
              <tr className="border-b border-slate-100">
                <td className="py-2 pr-4 font-mono text-xs">
                  finbert_pos_align3
                </td>
                <td className="py-2 pr-4">
                  <MathBlock formula="\sum_{s>0} \mathrm{final}_3 \cdot e^{-d_+ \cdot \text{age}}" />
                </td>
                <td className="py-2 text-slate-600">
                  3-bar alignment-weighted positive signal
                </td>
              </tr>
              <tr className="border-b border-slate-100">
                <td className="py-2 pr-4 font-mono text-xs">
                  finbert_neg_align3
                </td>
                <td className="py-2 pr-4">
                  <MathBlock formula="\sum_{s<0} \mathrm{final}_3 \cdot e^{-d_- \cdot \text{age}}" />
                </td>
                <td className="py-2 text-slate-600">
                  3-bar alignment-weighted negative signal
                </td>
              </tr>
              <tr className="border-b border-slate-100">
                <td className="py-2 pr-4 font-mono text-xs">
                  finbert_pos_align5
                </td>
                <td className="py-2 pr-4">
                  <MathBlock formula="\sum_{s>0} \mathrm{final}_5 \cdot e^{-d_+ \cdot \text{age}}" />
                </td>
                <td className="py-2 text-slate-600">
                  5-bar horizon variant
                </td>
              </tr>
              <tr className="border-b border-slate-100">
                <td className="py-2 pr-4 font-mono text-xs">
                  finbert_neg_align5
                </td>
                <td className="py-2 pr-4">
                  <MathBlock formula="\sum_{s<0} \mathrm{final}_5 \cdot e^{-d_- \cdot \text{age}}" />
                </td>
                <td className="py-2 text-slate-600">
                  5-bar horizon variant
                </td>
              </tr>
              <tr className="border-b border-slate-100">
                <td className="py-2 pr-4 font-mono text-xs">
                  finbert_conf_belief
                </td>
                <td className="py-2 pr-4">
                  <MathBlock formula="\frac{\text{net} \cdot |\text{net}|}{\text{gross} + \epsilon}" />
                </td>
                <td className="py-2 text-slate-600">
                  Quadratic belief — penalises pos/neg disagreement
                </td>
              </tr>
              <tr>
                <td className="py-2 pr-4 font-mono text-xs">llm_neg_decay</td>
                <td className="py-2 pr-4">
                  <MathBlock formula="-\!\!\!\sum_{\text{LLM}=\text{neg}} e^{-d_- \cdot \text{age}}" />
                </td>
                <td className="py-2 text-slate-600">
                  Time-decayed count of LLM-labelled negative headlines
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* Alignment */}
      <section className="panel space-y-3">
        <h2 className="text-2xl font-semibold text-slate-900">
          The alignment mechanism
        </h2>
        <p className="text-slate-700 leading-relaxed">
          For each headline at bar{' '}
          <MathBlock formula="b_h" /> with FinBERT score{' '}
          <MathBlock formula="s \in [-1,1]" />, we look ahead{' '}
          <MathBlock formula="k \in \{3, 5\}" /> bars and form a normalized
          (bar-position, price) motion vector. The horizontal range is the
          session span padded by 10; the vertical range is the high-low
          envelope, floored at <MathBlock formula="10^{-8}" />:
        </p>
        <MathBlock
          display
          formula="u_k = \frac{k}{x_{\text{range}}}, \quad v_k = \frac{p_k - p_0}{y_{\text{range}}}, \quad \text{align}_k = s \cdot \frac{v_k}{\sqrt{u_k^2 + v_k^2 + 10^{-8}}}"
        />
        <p className="text-slate-700 leading-relaxed">
          Then the per-headline weight clips to the consistent half-line:
        </p>
        <MathBlock
          display
          formula="\mathrm{final}_k = s \cdot \max\!\bigl(0,\ \text{align}_k\bigr)"
        />
        <p className="text-slate-700 leading-relaxed">
          A bullish headline followed by a price rise →{' '}
          <MathBlock formula="\text{align} > 0" /> →{' '}
          <MathBlock formula="\mathrm{final} > 0" />. A bullish headline
          followed by a drop →{' '}
          <MathBlock formula="\text{align} < 0" /> →{' '}
          <MathBlock formula="\mathrm{final} = 0" /> (clipped). The result:
          headlines that the market subsequently <em>contradicts</em>{' '}
          self-mute. See the <a className="text-indigo-700 hover:underline" href="#/alignment">Alignment</a> tab for the
          geometry on real curated examples.
        </p>
      </section>

      {/* Decay optimization */}
      <section className="panel space-y-3">
        <h2 className="text-2xl font-semibold text-slate-900">
          Decay optimization
        </h2>
        <p className="text-slate-700 leading-relaxed">
          The decay rates{' '}
          <MathBlock formula="(d_+, d_-)" /> are <em>learned</em>, not
          grid-searched. Ridge regression has a closed form that's fully
          differentiable in PyTorch via <code className="font-mono">torch.linalg.solve</code>:
        </p>
        <MathBlock
          display
          formula="\hat{w}(d_+, d_-) = (X(d_+, d_-)^\top X + \alpha I)^{-1} X^\top y"
        />
        <p className="text-slate-700 leading-relaxed">
          The training objective is the Sharpe of{' '}
          <MathBlock formula="X \hat{w} \cdot y" />; gradients flow through the
          inverse back to{' '}
          <MathBlock formula="\log d_+, \log d_-" /> (positivity via exp).
          600 Adam steps at lr=0.02 converge deterministically; a single seeded
          run suffices.
        </p>
        {m && (
          <div className="grid grid-cols-2 gap-4 pt-2">
            <div className="bg-slate-50 rounded p-3 text-center">
              <div className="stat-label">d₊ (positive headlines)</div>
              <div className="font-mono text-lg">{m.decay_pos.toFixed(5)}</div>
              <div className="text-xs text-slate-500">
                half-life ≈ {(Math.log(2) / m.decay_pos).toFixed(1)} bars
              </div>
            </div>
            <div className="bg-slate-50 rounded p-3 text-center">
              <div className="stat-label">d₋ (negative headlines)</div>
              <div className="font-mono text-lg">{m.decay_neg.toFixed(5)}</div>
              <div className="text-xs text-slate-500">
                half-life ≈ {(Math.log(2) / m.decay_neg).toFixed(1)} bars
              </div>
            </div>
          </div>
        )}
      </section>

      {/* Parkinson */}
      <section className="panel space-y-3">
        <h2 className="text-2xl font-semibold text-slate-900">
          Volatility — Parkinson estimator
        </h2>
        <p className="text-slate-700">
          We use the Parkinson estimator (range-based) instead of std-of-log-returns: it extracts more
          information from each bar's high/low and is robust to non-Gaussian returns.
        </p>
        <MathBlock
          display
          formula="\sigma^2_{\text{park}} = \frac{1}{4 \ln 2} \cdot \overline{\left(\ln \frac{H_t}{L_t}\right)^2} \quad\Rightarrow\quad \text{vol} = \max\!\bigl(\sqrt{\sigma^2_{\text{park}}},\ 10^{-6}\bigr)"
        />
      </section>

      {/* Data augmentation */}
      {m?.augmentation && (
        <section className="panel space-y-3">
          <h2 className="text-2xl font-semibold text-slate-900">
            Data augmentation — shifted train/test split
          </h2>
          <p className="text-slate-700 leading-relaxed">
            Each labeled session contains 100 bars but the model only ever sees
            50 (bars 0-49) at prediction time. We exploit the remaining 50 bars
            of <em>training</em> sessions by re-using them as a second sample:
            the augmented session uses bars {m.augmentation.split_bar - 49}-
            {m.augmentation.split_bar} as &quot;seen&quot; and bars{' '}
            {m.augmentation.split_bar + 1}-99 as &quot;unseen&quot;,
            re-indexed to 0-49 so feature engineering is identical. Augmented
            session IDs are offset by{' '}
            <span className="font-mono">{m.augmentation.session_offset}</span>{' '}
            to avoid collision.
          </p>
          <TimelineDiagram splitBar={m.augmentation.split_bar} />
          <p className="text-slate-700 leading-relaxed">
            This doubles the labelled set from{' '}
            <span className="font-mono">{m.n_train_original}</span> to{' '}
            <span className="font-mono">{m.n_train}</span> samples. Critically,
            CV uses{' '}
            <span className="font-mono">GroupKFold</span> with{' '}
            <em>group = original session id</em>, so the augmented twin and its
            original never end up split across train and test — without that
            guard the CV Sharpe would be artificially inflated by the model
            essentially seeing the future of a held-out session. See the{' '}
            <a
              className="text-indigo-700 hover:underline"
              href="#/augmentation"
            >
              Augmentation
            </a>{' '}
            tab for the side-by-side session viewer and the GroupKFold
            leakage diagram.
          </p>
        </section>
      )}

      {/* Training + Inference */}
      <section className="panel space-y-3">
        <h2 className="text-2xl font-semibold text-slate-900">
          Training & inference
        </h2>
        <p className="text-slate-700">
          Ridge target is the vol-scaled second-half return{' '}
          <MathBlock formula="y_{\text{train}} = (C_{99}/C_{49} - 1) / \text{vol}" />
          . α is selected by 5-fold GroupKFold CV over a 17-point grid spanning
          α ∈ [25, 1000]; a broad flat ridge near the optimum indicates the
          model is robust to that knob. Inference position-sizes each session
          by <MathBlock formula="1/\text{vol}" /> (inverse-vol Kelly):
        </p>
        <MathBlock
          display
          formula="\text{position}_i = \frac{\hat y_i}{\text{vol}_i} \cdot \frac{100}{\sigma(\hat y / \text{vol})}"
        />
        <p className="text-sm text-slate-500">
          The 100/σ rescaling is cosmetic — Sharpe is invariant to global
          multiplicative scaling.
        </p>
      </section>
    </div>
  );
}
