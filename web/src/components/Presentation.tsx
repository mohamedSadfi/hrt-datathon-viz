import { useEffect, useState, type ReactNode } from 'react';
import { loadModel } from '../lib/dataLoaders';
import type { ModelData } from '../lib/types';
import { fmtSharpe } from '../lib/format';
import MathBlock from './MathBlock';
import { TimelineDiagram } from './AugmentationView';

// ────────────────────────────────────────────────────────────────────────────
// 6-slide pitch deck (~30 s per slide, target 2-3 min).
// Built around the live numbers in model.json so it stays in sync with the
// model (LB Sharpe, CV mean, decays, train counts, alpha*).
// ────────────────────────────────────────────────────────────────────────────

interface SlideProps {
  m: ModelData;
}

function Slide({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="panel min-h-[540px] flex flex-col">
      <div className="text-xs uppercase tracking-widest text-indigo-600 font-semibold mb-2">
        {eyebrow}
      </div>
      <h2 className="text-3xl sm:text-4xl font-semibold text-slate-900 mb-6 leading-tight">
        {title}
      </h2>
      <div className="flex-1 text-slate-700 text-lg leading-relaxed space-y-4">
        {children}
      </div>
    </div>
  );
}

function SlideTitle({ m }: SlideProps) {
  return (
    <Slide eyebrow="HRT Datathon 2026" title="Predicting market direction from price + news">
      <p>
        Each session: 100 bars of OHLC, ~10 news headlines, one anonymous
        company is the trading subject. We see the first 50 bars and have to
        decide a long/short position size, scored by Sharpe across all
        sessions.
      </p>
      <div className="grid grid-cols-2 gap-6 pt-4">
        <div className="bg-slate-50 rounded-lg p-6 text-center">
          <div className="text-xs uppercase tracking-wide text-slate-500">
            Public LB Sharpe
          </div>
          <div className="text-5xl font-semibold text-indigo-700 mt-2 font-mono">
            {fmtSharpe(m.lb_score_public)}
          </div>
        </div>
        <div className="bg-slate-50 rounded-lg p-6 text-center">
          <div className="text-xs uppercase tracking-wide text-slate-500">
            5-fold GroupKFold CV Sharpe
          </div>
          <div className="text-5xl font-semibold text-teal-700 mt-2 font-mono">
            {fmtSharpe(m.cv_mean_sharpe)}
          </div>
        </div>
      </div>
      <p className="text-sm text-slate-500 pt-4">
        Ridge regression on 10 hand-engineered features. Linear, deterministic,
        ~6 s end-to-end retraining. The novel piece is how we turn news into
        self-validating signals.
      </p>
    </Slide>
  );
}

function SlideProblem(_: SlideProps) {
  return (
    <Slide eyebrow="The challenge" title="A noisy signal, half the data, one Sharpe number">
      <ul className="space-y-3 list-disc list-inside">
        <li>
          <strong>Inputs per session:</strong> 50 bars of OHLC (prices
          normalised to start at 1.0) + ~10 news headlines about{' '}
          <em>various</em> companies, of which exactly one is the subject —
          identity not labelled.
        </li>
        <li>
          <strong>Output:</strong> a scalar{' '}
          <code className="font-mono">target_position</code> at bar 49.
        </li>
        <li>
          <strong>Scoring:</strong>{' '}
          <MathBlock formula="\text{Sharpe} = \frac{\overline{\pi}}{\sigma(\pi)} \cdot 16, \quad \pi_i = \text{position}_i \cdot (C_{99}/C_{49} - 1)" />
        </li>
      </ul>
      <p className="pt-4">
        Sharpe is scale-invariant — only direction and relative magnitude of
        positions matter. So this is fundamentally a <em>signed return
        prediction</em> problem with a generous noise budget.
      </p>
    </Slide>
  );
}

function SlideFeatures(_: SlideProps) {
  return (
    <Slide eyebrow="The model" title="Ridge on 10 vol-normalised features">
      <p>
        All features are dimensionless (Sharpe-like quantities). Three
        families:
      </p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
        <div className="bg-teal-50 rounded-lg p-4">
          <div className="text-xs uppercase tracking-wide text-teal-700 font-semibold">
            Price (3)
          </div>
          <ul className="mt-2 space-y-1 text-sm font-mono text-slate-800">
            <li>ret_all_vol</li>
            <li>ret_last5_vol</li>
            <li>ret_last20_vol</li>
          </ul>
          <p className="text-xs text-slate-600 mt-2">
            Mean-reversion on the 20-bar window dominates.
          </p>
        </div>
        <div className="bg-amber-50 rounded-lg p-4">
          <div className="text-xs uppercase tracking-wide text-amber-700 font-semibold">
            Path shape (1)
          </div>
          <ul className="mt-2 space-y-1 text-sm font-mono text-slate-800">
            <li>cand_up_ratio</li>
          </ul>
          <p className="text-xs text-slate-600 mt-2">
            Fraction of up-bars — distinguishes steady climb vs few big jumps.
          </p>
        </div>
        <div className="bg-indigo-50 rounded-lg p-4">
          <div className="text-xs uppercase tracking-wide text-indigo-700 font-semibold">
            News sentiment (6)
          </div>
          <ul className="mt-2 space-y-1 text-sm font-mono text-slate-800">
            <li>finbert_pos_align{'{3,5}'}</li>
            <li>finbert_neg_align{'{3,5}'}</li>
            <li>finbert_conf_belief</li>
            <li>llm_neg_decay</li>
          </ul>
          <p className="text-xs text-slate-600 mt-2">
            FinBERT scores filtered through an alignment scalar (next slide) +
            an LLM-tagged negative count.
          </p>
        </div>
      </div>
      <p className="pt-4 text-sm text-slate-500">
        Ridge α chosen by 5-fold GroupKFold over a 17-point grid; the optimum
        sits on a broad flat ridge ≈ α ∈ [300, 750], so the model is
        insensitive to that knob.
      </p>
    </Slide>
  );
}

function SlideAlignment(_: SlideProps) {
  return (
    <Slide
      eyebrow="The novel bit"
      title={
        <>
          Headlines that the market <em>contradicts</em> self-mute
        </>
      }
    >
      <p>
        For a headline at bar <MathBlock formula="b_h" /> with FinBERT score{' '}
        <MathBlock formula="s \in [-1, 1]" />, look ahead k ∈ {'{3, 5}'} bars
        and form the normalised motion vector{' '}
        <MathBlock formula="(u_k, v_k)" />. The alignment scalar is the signed
        sine of its angle:
      </p>
      <MathBlock
        display
        formula="\text{align}_k = s \cdot \frac{v_k}{\sqrt{u_k^2 + v_k^2 + 10^{-8}}}, \qquad \mathrm{final}_k = s \cdot \max(0, \text{align}_k)"
      />
      <div className="bg-slate-50 rounded-lg p-4 grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
        <div>
          <div className="text-sm font-semibold text-teal-700">
            Bullish news + price rises
          </div>
          <div className="text-xs text-slate-600 mt-1">
            sign(s) = sign(v) → align &gt; 0 → final {'>'} 0. Counts.
          </div>
        </div>
        <div>
          <div className="text-sm font-semibold text-orange-700">
            Bullish news + price falls
          </div>
          <div className="text-xs text-slate-600 mt-1">
            sign(s) ≠ sign(v) → align &lt; 0 → final = 0. Muted.
          </div>
        </div>
      </div>
      <p className="pt-2">
        The result: irrelevant or wrong-company headlines that don&apos;t move
        the price stop contributing to the feature. Decay rates{' '}
        <MathBlock formula="(d_+, d_-)" /> for sentiment age are{' '}
        <em>learned</em> by differentiating Sharpe through the Ridge
        closed-form (PyTorch + <code className="font-mono">torch.linalg.solve</code>).
      </p>
    </Slide>
  );
}

function SlideAugmentation({ m }: SlideProps) {
  const split = m.augmentation?.split_bar ?? 74;
  const augStart = split - 49;
  return (
    <Slide
      eyebrow="More signal from the same labels"
      title="Shifted-split augmentation + GroupKFold"
    >
      <p>
        We re-use the back half of every <em>training</em> session by sliding
        the seen window forward to bar {split}, generating a synthetic second
        sample of the same stock&apos;s trajectory.
      </p>
      <div className="bg-slate-50 rounded-lg p-3">
        <TimelineDiagram splitBar={split} />
      </div>
      <div className="grid grid-cols-3 gap-4 text-center pt-2">
        <div>
          <div className="text-xs uppercase tracking-wide text-slate-500">
            Original
          </div>
          <div className="text-2xl font-semibold font-mono">
            {m.n_train_original?.toLocaleString() ?? '—'}
          </div>
        </div>
        <div className="text-3xl text-slate-400 self-center">→</div>
        <div>
          <div className="text-xs uppercase tracking-wide text-slate-500">
            With augmentation
          </div>
          <div className="text-2xl font-semibold font-mono text-indigo-700">
            {m.n_train.toLocaleString()}
          </div>
        </div>
      </div>
      <p>
        Honest CV requires <code className="font-mono">GroupKFold</code> with{' '}
        <em>group = original session id</em> — the augmented twin shares its
        original&apos;s seen window from bar {augStart} to 49, so a vanilla
        KFold split would leak the &quot;future&quot; into the test fold.
      </p>
    </Slide>
  );
}

function SlideResults({ m }: SlideProps) {
  const std =
    Math.sqrt(
      m.cv_fold_sharpes.reduce(
        (a, s) => a + (s - m.cv_mean_sharpe) ** 2,
        0,
      ) / m.cv_fold_sharpes.length,
    );
  return (
    <Slide eyebrow="Results & next steps" title="Honest 3.0+ Sharpe, deterministic, fast">
      <div className="grid grid-cols-2 gap-6">
        <div className="bg-slate-50 rounded-lg p-6">
          <div className="text-xs uppercase tracking-wide text-slate-500">
            CV Sharpe (5-fold GroupKFold)
          </div>
          <div className="text-4xl font-semibold text-teal-700 mt-2 font-mono">
            {fmtSharpe(m.cv_mean_sharpe)}
            <span className="text-base text-slate-500 ml-2">
              ± {fmtSharpe(std)}
            </span>
          </div>
        </div>
        <div className="bg-slate-50 rounded-lg p-6">
          <div className="text-xs uppercase tracking-wide text-slate-500">
            Public LB Sharpe
          </div>
          <div className="text-4xl font-semibold text-indigo-700 mt-2 font-mono">
            {fmtSharpe(m.lb_score_public)}
          </div>
        </div>
        <div className="bg-slate-50 rounded-lg p-6">
          <div className="text-xs uppercase tracking-wide text-slate-500">
            α*
          </div>
          <div className="text-3xl font-semibold mt-2 font-mono">
            {m.alpha_star}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            on a broad flat ridge of optima
          </div>
        </div>
        <div className="bg-slate-50 rounded-lg p-6">
          <div className="text-xs uppercase tracking-wide text-slate-500">
            Learned decays
          </div>
          <div className="text-xl font-mono mt-2">
            d₊ = {m.decay_pos.toFixed(3)} · d₋ = {m.decay_neg.toFixed(3)}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            half-life ≈ {(Math.log(2) / m.decay_pos).toFixed(0)} /{' '}
            {(Math.log(2) / m.decay_neg).toFixed(0)} bars
          </div>
        </div>
      </div>
      <div className="pt-4 text-base text-slate-600">
        <p className="font-semibold text-slate-900">Explore the rest of the site:</p>
        <ul className="mt-2 grid grid-cols-2 sm:grid-cols-3 gap-x-6 gap-y-1 text-sm">
          <li>
            <a className="text-indigo-700 hover:underline" href="#/session">
              Sessions
            </a>{' '}
            — per-session predictions
          </li>
          <li>
            <a
              className="text-indigo-700 hover:underline"
              href="#/augmentation"
            >
              Augmentation
            </a>{' '}
            — side-by-side splits
          </li>
          <li>
            <a className="text-indigo-700 hover:underline" href="#/alignment">
              Alignment
            </a>{' '}
            — geometry of headline weights
          </li>
          <li>
            <a
              className="text-indigo-700 hover:underline"
              href="#/coefficients"
            >
              Coefficients
            </a>{' '}
            — standardised Ridge weights
          </li>
          <li>
            <a className="text-indigo-700 hover:underline" href="#/decay">
              Decay
            </a>{' '}
            — learned decay curves
          </li>
          <li>
            <a
              className="text-indigo-700 hover:underline"
              href="#/performance"
            >
              Performance
            </a>{' '}
            — folds, α sweep, scatter
          </li>
        </ul>
      </div>
    </Slide>
  );
}

const SLIDES = [SlideTitle, SlideProblem, SlideFeatures, SlideAlignment, SlideAugmentation, SlideResults];

export default function Presentation() {
  const [m, setM] = useState<ModelData | null>(null);
  const [idx, setIdx] = useState(0);

  useEffect(() => {
    loadModel().then(setM);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      if (e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === ' ') {
        setIdx((i) => Math.min(SLIDES.length - 1, i + 1));
      } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
        setIdx((i) => Math.max(0, i - 1));
      } else if (e.key === 'Home') {
        setIdx(0);
      } else if (e.key === 'End') {
        setIdx(SLIDES.length - 1);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  if (!m) return <div className="panel">Loading…</div>;

  const SlideComponent = SLIDES[idx];

  return (
    <div className="space-y-4">
      <SlideComponent m={m} />

      {/* Footer controls */}
      <div className="flex items-center justify-between gap-4">
        <button
          type="button"
          onClick={() => setIdx((i) => Math.max(0, i - 1))}
          disabled={idx === 0}
          className="px-4 py-2 text-sm rounded border border-slate-300 hover:bg-slate-100 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          ← prev
        </button>

        <div className="flex items-center gap-2">
          {SLIDES.map((_, i) => (
            <button
              type="button"
              key={i}
              onClick={() => setIdx(i)}
              aria-label={`Go to slide ${i + 1}`}
              className={`w-2.5 h-2.5 rounded-full transition-colors ${
                i === idx
                  ? 'bg-indigo-600'
                  : 'bg-slate-300 hover:bg-slate-400'
              }`}
            />
          ))}
          <span className="text-xs text-slate-500 font-mono ml-2">
            {idx + 1} / {SLIDES.length}
          </span>
        </div>

        <button
          type="button"
          onClick={() =>
            setIdx((i) => Math.min(SLIDES.length - 1, i + 1))
          }
          disabled={idx === SLIDES.length - 1}
          className="px-4 py-2 text-sm rounded bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          next →
        </button>
      </div>

      <p className="text-xs text-slate-400 text-center">
        Use ← / → (or PageUp/PageDown) to navigate. Space advances. Home/End
        jump to ends.
      </p>
    </div>
  );
}
