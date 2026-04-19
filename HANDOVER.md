# HANDOVER — Visualization Platform for Lenny's Solution (HRT Datathon 2026)

> Copy this file into the new repository (e.g. as `HANDOVER.md` or `docs/PLAN.md`). It is self-contained: it explains the problem, the solution, what to bring over from the old repo, the full site architecture, and verification steps.

---

## 1. Purpose

Build a **GitHub-Pages-hosted, static visualization site** that presents our winning model for the HRT Datathon 2026 to a technical/academic jury. The model itself is frozen (`Lenny/solution.py`, leaderboard Sharpe ~2.88). The site is a *presentation* artifact, not a model-development environment.

Stack (already decided):
- **Vite + React + TypeScript + Plotly.js** frontend
- **Python (+ uv) + torch + scikit-learn** for one-shot data prep
- **GitHub Actions** deploying to **GitHub Pages**

Audience: datathon judges. Expect math, coefficients, explicit methodology. Not an ELI5 site.

---

## 2. What to bring over from the old repo

Copy the minimal set; leave the rest behind. The old repo has dozens of failed experiments that we don't want cluttering the new one.

### Mandatory (the model + its inputs)

| Source (old repo) | Destination (new repo) | Why |
|---|---|---|
| `Lenny/solution.py` | `model/solution.py` | The frozen model. Data-prep script imports from it. |
| `data/bars_seen_train.parquet` | `data/bars_seen_train.parquet` | 1000 train sessions × 50 seen bars (OHLC). ~590 KB. |
| `data/bars_unseen_train.parquet` | `data/bars_unseen_train.parquet` | 1000 train sessions × 50 unseen bars (for ground-truth targets). ~600 KB. |
| `data/headlines_seen_train.parquet` | `data/headlines_seen_train.parquet` | 9,740 train headlines with (session, bar_ix, headline). ~220 KB. |
| `Pyrros/finbert_scores.parquet` | `data/finbert_scores.parquet` | 34,182 unique headlines → FinBERT score ∈ [-1, +1]. ~860 KB. |
| `Momo/data/processed/headline_annotations.parquet` | `data/headline_annotations.parquet` | 34,182 LLM annotations (sentiment, numbers, number_contexts). ~420 KB. |

Test data (~11 MB × 2) is **not needed** — we only visualize labeled training sessions.

### Optional references (documentation / context)

| Source (old repo) | Destination | Why |
|---|---|---|
| `Lenny/notes.md` | `docs/methodology_notes.md` | Lenny's original narrative about mean reversion + sentiment findings. |
| `Lenny/plots/*.png` | `docs/legacy_plots/` | Static plots that inspired the interactive versions. |
| `Momo/reports/v36_build_trail.md` | `docs/findings_log.md` | Ablation history — shows we discovered `llm_neg_decay` (which Lenny later adopted) and `cand_up_ratio`. Nice for the judges to see the process. |

### Do NOT bring over

- All `Momo/scripts/train_v*.py` (50+ files of failed experiments).
- Old `Momo/submissions/` CSVs.
- Any of the `Momo/data/models/*.joblib` files (except as needed — data-prep will regenerate the final model).
- Old `__pycache__`, `.venv`, notebook checkpoints, etc.

### Paths inside `Lenny/solution.py` that will need adjusting

The current `solution.py` uses relative paths to locate data:

```python
PYRROS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Pyrros")
FINBERT_CACHE = os.path.join(PYRROS_DIR, "finbert_scores.parquet")
MOMO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Momo")
LLM_ANNOTATIONS_PATH = os.path.join(MOMO_DIR, "data", "processed", "headline_annotations.parquet")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
```

In the new repo, move these resolution lines to the top or into a small `config.py`:

```python
# model/config.py
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FINBERT_CACHE = DATA_DIR / "finbert_scores.parquet"
LLM_ANNOTATIONS_PATH = DATA_DIR / "headline_annotations.parquet"
OUTPUT_DIR = ROOT / "web" / "public" / "data"   # where prep writes JSON
```

Then patch the three `os.path.join(...)` lines in `solution.py` to import from `config`. Keep everything else in `solution.py` untouched.

---

## 3. The solution — detailed technical writeup

This section is the one the judges actually care about. Keep the same explanation on the site's "Intro" page.

### 3.1 Problem setup

- 1,000 labeled training sessions + 10,000 public test + 10,000 private test.
- Each session = 100 bars of OHLC data. Prices normalized to start at 1.0.
- Bars 0–49 are "seen"; bars 50–99 are "unseen" at prediction time.
- Each session has ~10 news headlines at various `bar_ix ∈ [0, 49]`.
- Headlines are about **multiple companies** per session. Exactly one company is the "subject" whose price we're trading, but the identity isn't labeled.
- **Target**: emit a scalar `target_position` per test session. Score = Sharpe across all test sessions:

  $$\text{Sharpe} = \frac{\bar{\pi}}{\sigma(\pi)} \cdot 16, \qquad \pi_i = \text{target\_position}_i \cdot y_i, \qquad y_i = \frac{\text{close}[99]_i}{\text{close}[49]_i} - 1$$

  The factor 16 is an annualization convention (4 bars/day × √252 / √…). Scale of positions is irrelevant — Sharpe is scale-invariant within positive rescaling.

### 3.2 Volatility estimator — Parkinson

Instead of `std(log_returns)`, the solution uses the **Parkinson** estimator which extracts more info from OHLC by using the high/low range:

$$\sigma^2_{\text{park}} = \frac{1}{4 \ln 2} \cdot \overline{\left(\ln \frac{H_t}{L_t}\right)^2}$$

$$\text{vol} = \max\!\big(\sqrt{\sigma^2_{\text{park}}},\ 10^{-6}\big)$$

Averaged over the 50 seen bars. More robust than std(log returns) because it doesn't assume returns are Gaussian and uses intra-bar information.

### 3.3 The 10 features

**Price features (vol-normalized, so they're dimensionless "Sharpe-like" quantities):**

| Feature | Formula | Interpretation |
|---|---|---|
| `ret_all_vol` | $\frac{C_{49}/O_0 - 1}{\text{vol}}$ | Full-first-half return, vol-scaled |
| `ret_last5_vol` | $\frac{C_{49}/C_{44} - 1}{\text{vol}}$ | 5-bar momentum |
| `ret_last20_vol` | $\frac{C_{49}/C_{29} - 1}{\text{vol}}$ | 20-bar mean-reversion signal |

The 20-bar feature ends up with the largest-magnitude (negative) coefficient — the signal is mean-reversion, not momentum, on that window.

**Price-shape feature:**

| Feature | Formula | Interpretation |
|---|---|---|
| `cand_up_ratio` | $\frac{1}{49}\#\{t : \log C_{t+1} > \log C_t\}$ | Fraction of positive 1-bar log returns. Captures *how* the return was accumulated: steady climb (high ratio) vs. few big jumps (low ratio). |

The coefficient ends up negative — sessions with many up-bars tend to mean-revert. Distinct from `ret_last20_vol` because two sessions can have the same 20-bar return but different up-bar ratios.

**Sentiment features — the alignment-weighted FinBERT signals (the novel bit):**

For a headline $h$ at bar $b_h$ with FinBERT score $s \in [-1, 1]$:

1. Identify the closest price at or before $b_h$: $p_0 = C_{b_h}$.
2. Pick horizons $k \in \{3, 5\}$, look ahead: $p_k = C_{\min(b_h + k, \ 49)}$.
3. Form the (bar-position, price) motion vector, normalized by the session's ranges:
   - $x_{\text{range}} = \text{bars}[-1] - \text{bars}[0] + 10$
   - $y_{\text{range}} = \max(H) - \min(L)$, floored at $10^{-8}$
   - $u_k = \frac{k}{x_{\text{range}}},\qquad v_k = \frac{p_k - p_0}{y_{\text{range}}}$
4. Compute the alignment scalar:
   $$\text{align}_k = s \cdot \frac{v_k}{\sqrt{u_k^2 + v_k^2 + 10^{-8}}}$$

   Note this is **signed** — same sign as $v_k$ (price direction) times the sign of $s$.
5. Final per-headline weight:
   $$\text{final}_k = s \cdot \max(0, \text{align}_k)$$

**Semantic**: a headline's FinBERT sentiment only "counts" when the price subsequently moves in the direction the sentiment predicted. A bullish headline followed by a price rise → $\text{align} > 0$ → $\text{final} > 0$. A bullish headline followed by a drop → $\text{align} < 0$ → clipped to 0. This **geometrically couples text sentiment to realized price motion**, making noisy or irrelevant headlines self-mute.

The per-session feature is then the time-decayed sum over all matching headlines:

$$\text{finbert\_pos\_align}_k = \sum_{h : s_h > 0} \text{final}_k^{(h)} \cdot e^{-d_+ \cdot (49 - b_h)}$$

$$\text{finbert\_neg\_align}_k = \sum_{h : s_h < 0} \text{final}_k^{(h)} \cdot e^{-d_- \cdot (49 - b_h)}$$

Two horizons × two signs = 4 features: `finbert_{pos,neg}_align{3,5}`.

**Confidence feature:**

Let $f_+ = \sum_{s_h>0} s_h \cdot e^{-d_+ \cdot \text{age}}$ and $f_- = \sum_{s_h<0} s_h \cdot e^{-d_- \cdot \text{age}}$ (raw, no alignment).

$$\text{net} = f_+ + f_- \qquad \text{gross} = f_+ + |f_-|$$

$$\text{finbert\_conf\_belief} = \frac{\text{net} \cdot |\text{net}|}{\text{gross} + 10^{-8}} = \text{sign}(\text{net}) \cdot \frac{\text{net}^2}{\text{gross}+\epsilon}$$

Quadratic in net. Dampens sessions where pos and neg headlines disagree (net near 0 despite heavy gross).

**LLM sentiment feature:**

$$\text{llm\_neg\_decay} = -\sum_{h : \text{LLM}(h)=\text{neg}} e^{-d_- \cdot (49 - b_h)}$$

Time-decayed count of headlines labeled "neg" by an LLM annotator (separate from FinBERT). Signed negative by construction so Ridge interprets it as a signed feature. Historically this was our key discovery: LLM categorical labels are cleaner than FinBERT's continuous scores for negative headlines.

### 3.4 Decay optimization (differentiable Ridge closed-form)

The decay rates $(d_+, d_-)$ are *learned* rather than grid-searched. The trick: Ridge regression has a closed-form solution

$$\hat{w} = (X^\top X + \alpha I)^{-1} X^\top y$$

which is fully differentiable in PyTorch via `torch.linalg.solve`. The optimization loop (600 Adam steps, lr=0.02):

1. Parametrize $d_+ = e^{\ell_+}$, $d_- = e^{\ell_-}$ (positivity via exp).
2. Build the (internal) design matrix $X$ from price features + sentiment features computed with current $(d_+, d_-)$.
3. Solve Ridge with $\alpha = 50$ (fixed during decay fit): $\hat w$, $\hat y = X_s \hat w$.
4. Compute Sharpe loss: $\mathcal{L} = -\frac{\overline{\hat y \cdot y}}{\sigma(\hat y \cdot y) + \epsilon} \cdot 16$.
5. `loss.backward()` → gradients flow through the Ridge inverse back to $(\ell_+, \ell_-)$.
6. `opt.step()`.

Converges deterministically (within ~150 epochs — the remaining ~450 are a safety margin). A single run suffices.

**Note**: the decay fit uses an *8-feature internal model* (price + `finbert_pos`, `finbert_neg`, `finbert_pos_align3`, `finbert_neg_align3`, `finbert_pos_align5`, `finbert_neg_align5`, `finbert_conf_belief`). The **final** Ridge uses a slightly different 10-feature set (no `finbert_pos`/`finbert_neg` raw; adds `llm_neg_decay`, `cand_up_ratio`). This decoupling is intentional — the decay search needs a smooth differentiable objective; the final model is what predicts.

### 3.5 Training

- **Target**: $y_{\text{train}} = (C_{99}/C_{49} - 1) / \text{vol}$ — raw second-half return vol-scaled.
- **CV**: `KFold(n_splits=5, shuffle=True, random_state=42)`. Deterministic folds.
- **Alpha search**: grid `[25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 600, 750, 1000]`. For each α, compute mean CV Sharpe over 5 folds, pick argmax.
- **Final fit**: single Ridge on all 1000 train sessions at best α.

### 3.6 Inference position sizing

$$\text{target\_position} = \frac{\hat y}{\text{vol}} \cdot \frac{100}{\sigma(\hat y / \text{vol})}$$

The division by vol gives "proportional risk" positions: high-vol sessions get smaller, low-vol sessions bigger (inverse-vol Kelly style). The std=100 rescaling is cosmetic — Sharpe is invariant to global multiplicative scaling.

### 3.7 Why this particular architecture works

- **Linear model with few features** (10) on a small labeled set (1000): Ridge's L2 keeps every coefficient bounded, preventing the kind of overfitting we repeatedly saw with tree-based models (GBM best CV was 2.74 vs Ridge's 3.1+).
- **Features are directional** (signed): each has a clear sign interpretation (mean-reversion = −, up-ratio = −, conf_belief = +, llm_neg_decay = +).
- **Alignment mechanism** makes sentiment self-mute when the market disagrees — this is what sets the solution apart from a naive FinBERT sum.
- **Vol-normalization of both features and target** gives scale-invariance across sessions.

---

## 4. New-repo layout

```
hrt-datathon-viz/
├── README.md                              # public-facing, links to deployed site
├── HANDOVER.md                            # this file
├── pyproject.toml                         # Python deps (uv)
├── model/
│   ├── __init__.py
│   ├── config.py                          # path resolution (see §2)
│   └── solution.py                        # copied verbatim from Lenny/solution.py
├── data/
│   ├── bars_seen_train.parquet
│   ├── bars_unseen_train.parquet
│   ├── headlines_seen_train.parquet
│   ├── finbert_scores.parquet
│   └── headline_annotations.parquet
├── scripts/
│   └── build_site_data.py                 # runs solution → dumps JSON to web/public/data/
├── web/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── tailwind.config.js                 # optional
│   ├── postcss.config.js                  # optional
│   ├── public/
│   │   ├── favicon.ico
│   │   └── data/                          # JSON artifacts (git-ignored if generated by CI; committed if generated locally)
│   └── src/
│       ├── main.tsx
│       ├── App.tsx                        # hash router, top nav
│       ├── styles.css
│       ├── lib/
│       │   ├── types.ts
│       │   ├── dataLoaders.ts
│       │   └── format.ts                  # number/percent/Sharpe formatters
│       └── components/
│           ├── Navigation.tsx
│           ├── Intro.tsx                  # problem + architecture (copy §3 content)
│           ├── SessionExplorer.tsx
│           ├── AlignmentDemo.tsx
│           ├── CoefficientsView.tsx
│           ├── DecayCurves.tsx
│           ├── PerformancePanel.tsx
│           └── MathBlock.tsx              # KaTeX wrapper
├── docs/
│   ├── methodology_notes.md               # from Lenny/notes.md
│   ├── findings_log.md                    # from Momo/reports/v36_build_trail.md
│   └── legacy_plots/                      # PNGs from Lenny/plots/
└── .github/
    └── workflows/
        └── pages.yml                      # build + deploy
```

---

## 5. Data-prep script (`scripts/build_site_data.py`)

This script runs **once** per code change (or on every CI run if we automate). It recreates the full model and dumps everything the frontend needs as JSON.

### Inputs
- Reads: `data/*.parquet`
- Imports: `model.solution`

### Outputs (written to `web/public/data/`)

All JSON; gzipped by Vite at build time.

**`model.json`** (~10 KB)
```json
{
  "decay_pos": 0.02812,
  "decay_neg": 0.02804,
  "alpha_star": 150,
  "alpha_grid": [
    {"alpha": 25, "cv_sharpe": 3.041},
    {"alpha": 50, "cv_sharpe": 3.082},
    ...
  ],
  "cv_fold_sharpes": [3.15, 3.08, 3.12, 3.10, 3.11],
  "cv_mean_sharpe": 3.112,
  "coefs": {
    "ret_all_vol": 0.xxx,
    "ret_last5_vol": 0.xxx,
    ...
  },
  "intercept": 0.xxx,
  "feature_cols": ["ret_all_vol", ...],
  "feature_means": {"ret_all_vol": 0.xxx, ...},
  "feature_stds":  {"ret_all_vol": 0.xxx, ...},
  "lb_score_public": 2.88,
  "lb_score_private": null,
  "n_train": 1000
}
```

**`sessions.json`** (~3–6 MB — the bulk of the payload)

An array of 1000 session objects:
```json
{
  "session": 0,
  "seen_bars":   [{"b": 0, "o": 1.0, "h": 1.02, "l": 0.98, "c": 1.01}, ... 50 entries],
  "unseen_bars": [{"b": 50, "c": 1.03}, ... 50 entries],  // close-only, to save bytes
  "headlines": [
    {"b": 6, "t": "Relvos Biosciences opens new office...", "fb": 0.45, "llm": "pos", "align3": 0.12, "align5": 0.08, "final3": 0.054, "final5": 0.036},
    ...
  ],
  "features": {"ret_all_vol": 0.xxx, ...},
  "prediction_vol_adj": 0.xxx,
  "actual_vol_adj": 0.xxx,
  "prediction_raw_position": 0.xxx,   // final target_position from pipeline
  "pnl_session": 0.xxx                // position * raw_return (for backtest plot)
}
```

**`alignment_examples.json`** (~20 KB)

~12 curated headlines spanning:
- 3 well-aligned bullish (price rose after positive news)
- 3 anti-aligned bullish (positive FinBERT, price dropped → clipped to 0)
- 3 well-aligned bearish
- 3 anti-aligned bearish (negative FinBERT, price rose → clipped to 0)

Each entry has the full geometric quantities for the Alignment Demo to render.

```json
{
  "session": 42,
  "bar_ix": 35,
  "headline": "...",
  "fb_score": 0.7,
  "llm_sent": "pos",
  "p0": 1.034,
  "p3": 1.051,
  "p5": 1.048,
  "x_range": 59,
  "y_range": 0.08,
  "u3": 0.0508, "v3": 0.2125, "norm3": 0.2184,
  "u5": 0.0847, "v5": 0.175, "norm5": 0.194,
  "align3": 0.681, "align5": 0.631,
  "final3": 0.477, "final5": 0.442
}
```

**`features_corr.json`** (~2 KB): 11×11 Pearson correlation matrix (10 features + target).

### Implementation sketch

```python
# scripts/build_site_data.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import solution, config

DATA_DIR = config.DATA_DIR
OUT_DIR = config.OUTPUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load
bars_seen = pd.read_parquet(DATA_DIR / "bars_seen_train.parquet")
bars_unseen = pd.read_parquet(DATA_DIR / "bars_unseen_train.parquet")
hl = pd.read_parquet(DATA_DIR / "headlines_seen_train.parquet")

# 2. Decay fit
dp, dn = solution._optimize_decay_params(bars_seen, bars_unseen, hl)

# 3. Features
feats = solution.extract_features(bars_seen, hl, decay_pos=dp, decay_neg=dn)

# 4. Target
close_end = bars_unseen.groupby("session")["close"].last()
feats = feats.set_index("session")
y_raw = (close_end.reindex(feats.index) / feats["halfway_close"] - 1)
y = (y_raw / feats["vol"]).values

# 5. Alpha sweep (full curve, not just winner)
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
alpha_grid_results = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)
X = feats[solution.FEATURE_COLS].values
for a in solution.ALPHA_GRID:
    sharpes = []
    for tr, vl in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=a).fit(sc.transform(X[tr]), y[tr])
        pnl = m.predict(sc.transform(X[vl])) * y[vl]
        if pnl.std() > 0:
            sharpes.append(pnl.mean() / pnl.std() * 16)
    alpha_grid_results.append({"alpha": float(a), "cv_sharpe": float(np.mean(sharpes))})
alpha_star = max(alpha_grid_results, key=lambda r: r["cv_sharpe"])["alpha"]

# 6. Per-fold Sharpe at alpha_star
fold_sharpes = solution.cross_validate(X, y, alpha=alpha_star)

# 7. Final model
model, scaler = solution.train_final_model(X, y, alpha=alpha_star)

# 8. Predictions on train sessions
preds = model.predict(scaler.transform(X))
positions = preds / feats["vol"].values
positions = positions * (100.0 / np.std(positions))

# 9. Dump model.json
(OUT_DIR / "model.json").write_text(json.dumps({
    "decay_pos": float(dp),
    "decay_neg": float(dn),
    "alpha_star": float(alpha_star),
    "alpha_grid": alpha_grid_results,
    "cv_fold_sharpes": [float(s) for s in fold_sharpes],
    "cv_mean_sharpe": float(np.mean(fold_sharpes)),
    "coefs": dict(zip(solution.FEATURE_COLS, map(float, model.coef_))),
    "intercept": float(model.intercept_),
    "feature_cols": solution.FEATURE_COLS,
    "feature_means": {c: float(scaler.mean_[i]) for i, c in enumerate(solution.FEATURE_COLS)},
    "feature_stds":  {c: float(scaler.scale_[i]) for i, c in enumerate(solution.FEATURE_COLS)},
    "lb_score_public": 2.88,
    "lb_score_private": None,
    "n_train": len(X),
}, indent=2))

# 10. Dump sessions.json (detailed per-session)
# ... iterate 1000 sessions, build the nested object per §5 spec ...

# 11. Dump alignment_examples.json
# ... pick 12 curated headlines, compute alignment geometry ...

# 12. Dump features_corr.json
import numpy as np
corr = np.corrcoef(np.column_stack([X, y]), rowvar=False)
# write as nested list with row/col labels

print(f"Wrote artifacts to {OUT_DIR}")
```

**Expected runtime**: ~30–60 seconds on a modern laptop (decay fit is 600 Adam steps, but small tensors; the rest is fast sklearn).

---

## 6. Frontend — component specifications

### 6.1 App shell

- `App.tsx`: hash-based router (e.g. `#/`, `#/session`, `#/alignment`, `#/features`, `#/performance`).
- Top-nav `<Navigation>` with 5 tabs.
- Each section is a self-contained component that fetches its needed JSON on mount.

### 6.2 Dependencies (in `web/package.json`)

```json
{
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "plotly.js-dist-min": "^2.35.3",
    "react-plotly.js": "^2.6.0",
    "katex": "^0.16.11"
  },
  "devDependencies": {
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "@types/react-plotly.js": "^2.6.3",
    "@vitejs/plugin-react": "^4.3.1",
    "typescript": "^5.5.3",
    "vite": "^5.4.0",
    "tailwindcss": "^3.4.10",
    "postcss": "^8.4.41",
    "autoprefixer": "^10.4.20"
  }
}
```

Plotly is the biggest bundled dep (~3 MB). Acceptable for a data-viz site; consider lazy-loading it behind the initial intro page if budget-sensitive.

### 6.3 `vite.config.ts`

```ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/<repo-name>/',   // replace with actual GH Pages sub-path
  build: { outDir: 'dist', assetsDir: 'assets' },
});
```

### 6.4 `src/lib/types.ts`

```ts
export interface Bar { b: number; o?: number; h?: number; l?: number; c: number; }
export interface HeadlineEntry {
  b: number; t: string; fb: number; llm: 'pos' | 'neutral' | 'neg';
  align3: number; align5: number; final3: number; final5: number;
}
export interface SessionData {
  session: number;
  seen_bars: Bar[];
  unseen_bars: Bar[];
  headlines: HeadlineEntry[];
  features: Record<string, number>;
  prediction_vol_adj: number;
  actual_vol_adj: number;
  prediction_raw_position: number;
  pnl_session: number;
}
export interface ModelData {
  decay_pos: number;
  decay_neg: number;
  alpha_star: number;
  alpha_grid: { alpha: number; cv_sharpe: number }[];
  cv_fold_sharpes: number[];
  cv_mean_sharpe: number;
  coefs: Record<string, number>;
  intercept: number;
  feature_cols: string[];
  feature_means: Record<string, number>;
  feature_stds: Record<string, number>;
  lb_score_public: number;
  lb_score_private: number | null;
  n_train: number;
}
export interface AlignmentExample {
  session: number; bar_ix: number; headline: string;
  fb_score: number; llm_sent: 'pos' | 'neutral' | 'neg';
  p0: number; p3: number; p5: number;
  x_range: number; y_range: number;
  u3: number; v3: number; norm3: number;
  u5: number; v5: number; norm5: number;
  align3: number; align5: number; final3: number; final5: number;
}
```

### 6.5 `src/components/SessionExplorer.tsx`

- State: `selectedSessionId`, `hoveredHeadline`.
- On mount: fetch `sessions.json`, `model.json`.
- Layout (3 columns on desktop, stacked on mobile):
  - **Left (20%)**: searchable numeric input for session id (0–999), prev/next buttons, session count, link to "random".
  - **Center (55%)**: `<Plot>` with:
    - Candlestick trace for bars 0–49 (colored normally).
    - Line trace for bars 50–99 (dashed gray — "the unseen future").
    - Vertical shape at x=49 labeled "prediction point".
    - Overlay scatter of headlines: `x=b, y=close[b]`, `color=fb_score` (RdBu colorscale), `symbol` by `llm` (triangle-up/circle/triangle-down), on hover show text.
  - **Right (25%)**: features panel. Table of 10 rows:
    | Feature | Raw value | Standardized | Coef | Contribution |
    |---|---|---|---|---|
    where `contribution = std_value × coef`. Sort by `|contribution|` desc; bold the top 3.
  - **Bottom strip**: three big numbers with icons:
    - `ŷ = prediction_vol_adj` (with up/down chip)
    - `y = actual_vol_adj`
    - `PnL = ŷ · y`

### 6.6 `src/components/AlignmentDemo.tsx`

- On mount: fetch `alignment_examples.json`, `sessions.json` (for zoomed price context).
- Static intro (KaTeX-rendered math from §3.3).
- Example picker: dropdown of 12, labeled "pos + aligned", "pos + anti-aligned", etc.
- For the selected example, render an **SVG canvas** (not Plotly — need precise vector arrows):
  - Left panel: mini candlestick of the full 100 bars of the session for context, with `bar_ix` marked.
  - Right panel: zoomed view of bars `[bar_ix, bar_ix+5]`:
    - Small price line from `p0` to `p5`.
    - Arrow `(u3, v3)` drawn as SVG line with arrowhead, starting from `(b_ix, p0)`.
    - Arrow `(u5, v5)` in a different color.
    - Normalization box showing `x_range`, `y_range`.
    - Annotations for each scalar: `align3 = 0.68`, `final3 = 0.477`, etc.
- Right column: numeric readout (table of every intermediate quantity).
- "Next example" button.

### 6.7 `src/components/CoefficientsView.tsx`

- Fetch `model.json`.
- Horizontal bar chart (Plotly `Bar` with `orientation='h'`) of the 10 standardized coefficients.
- Color split: positive = teal, negative = orange.
- Sort bars by magnitude.
- Below the chart: 10-row card list with each feature's 1-line formula + intuitive sentence (taken from §3.3).
- Show α* prominently.

### 6.8 `src/components/DecayCurves.tsx`

- Fetch `model.json`.
- Plotly line chart. X: age ∈ [0, 49]. Y: weight.
- Two traces: `exp(-d_pos · age)` and `exp(-d_neg · age)`.
- Slider for each decay (starts at learned value, client-side only — doesn't retrain).
- Annotation: "Half-life = ln 2 / d ≈ X bars".
- Reset button.

### 6.9 `src/components/PerformancePanel.tsx`

Four sub-plots using Plotly subplots or a grid of Plot components:

1. **CV fold Sharpes**: Bar chart, 5 bars + dashed line for mean. Tooltip: fold index and Sharpe.
2. **Alpha sweep**: Line plot, log-x. Marker at α\*. Hover: `α = 175, Sharpe = 3.11`.
3. **Prediction vs actual**: Scatter of 1000 points. X = ŷ, Y = y, both vol-adjusted. Diagonal y=x line. Color = fold assignment. R² annotation.
4. **Summary card**: "CV Sharpe 3.11 ± 0.02 over 5 folds. α\* = 150. Decays: d+ = 0.028, d− = 0.028. LB: 2.88."

### 6.10 Styling

Use Tailwind for speed. A muted palette: slate/indigo/amber. Plotly charts use the Tailwind slate scheme for axes/grid. Avoid gratuitous animation — judges prefer data density.

---

## 7. GitHub Actions deployment

`.github/workflows/pages.yml`:

```yaml
name: Build & Deploy Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Install Python deps
        run: uv sync --frozen

      - name: Build site data (one-shot train + dump JSON)
        run: uv run python scripts/build_site_data.py

      - name: Set up Node
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: web/package-lock.json

      - name: Install frontend deps
        working-directory: web
        run: npm ci

      - name: Build frontend
        working-directory: web
        run: npm run build

      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: web/dist

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy
        id: deployment
        uses: actions/deploy-pages@v4
```

**Repo settings**: Settings → Pages → Source: **GitHub Actions**.

### Alternative: local-build + committed `docs/`

If the user doesn't want CI to run torch + sklearn (adds ~1–2 min to every build), instead:

1. Build locally: `python scripts/build_site_data.py && cd web && npm run build`.
2. Commit `web/dist/` to a `docs/` folder on `main`.
3. Settings → Pages → Source: **Deploy from a branch**, path: `/docs`.

Simpler, but requires re-running manually when the model or data changes.

---

## 8. Verification (end-to-end)

```bash
# 0. From a fresh clone of the new repo
git clone git@github.com:<user>/hrt-datathon-viz.git
cd hrt-datathon-viz

# 1. Python deps
uv sync

# 2. Data prep (30-60 seconds)
uv run python scripts/build_site_data.py
ls -la web/public/data/
# expected: model.json (~10KB), sessions.json (~3-6MB),
#           alignment_examples.json (~20KB), features_corr.json (~2KB)

# 3. Sanity check: does solution.py still converge to the same numbers?
uv run python -c "
import json
from pathlib import Path
m = json.loads(Path('web/public/data/model.json').read_text())
print('CV mean:', m['cv_mean_sharpe'])   # should be ~3.1
print('alpha*:', m['alpha_star'])        # should be in [100, 200]
print('coefs:', m['coefs'])
"

# 4. Frontend dev
cd web
npm install
npm run dev   # http://localhost:5173
# - Verify Intro page renders, math formulas appear via KaTeX
# - Session Explorer: pick session 0, 42, 500, 999; confirm candlestick + headlines + features
# - Alignment Demo: cycle through all 12 examples; SVG arrows render correctly
# - Coefficients: 10 horizontal bars; signs match solution.py's console output
# - Decay Curves: 2 curves visible, slider moves them; half-life annotation updates
# - Performance: 5 fold bars + alpha curve + scatter all render with real data

# 5. Prod build
npm run build
ls dist/   # index.html + assets/ + data/

# 6. Preview static build (identical to deployed)
npm run preview

# 7. Push → CI builds & deploys
cd ..
git add .
git commit -m "Initial viz platform"
git push origin main
# Watch Actions tab; once green, visit https://<user>.github.io/hrt-datathon-viz/
```

---

## 9. Risks / edge cases

- **sessions.json size**: at 1000 sessions with full 100-bar OHLC + headlines, expect ~5 MB uncompressed, ~1.5 MB gzipped. If it balloons past 10 MB, split into 10 chunks of 100 sessions each, lazy-load on selection.
- **torch install in CI**: PyTorch's CPU wheel is ~150 MB. Uv caches it; first CI run is ~2 min, cached runs ~30 s.
- **Plotly bundle size**: ~3 MB. Use `plotly.js-dist-min` (not `plotly.js-dist`). Consider code-splitting by route if initial page load gets sluggish.
- **Vite `base` path**: must match the GH Pages URL. If the repo is named `hrt-datathon-viz`, the URL is `https://<user>.github.io/hrt-datathon-viz/`, so `base: '/hrt-datathon-viz/'`.
- **SVG arrow geometry**: alignment demo's vectors are in normalized `(u, v)` space. The SVG render must apply an affine transform to map `(b_ix, close)` → screen pixels. Double-check the Y-axis flip (SVG y grows downward).
- **KaTeX rendering**: use `katex` package + `<span dangerouslySetInnerHTML={{__html: katex.renderToString(formula)}} />` or a small `<MathBlock>` component. Import KaTeX CSS in `main.tsx`.
- **Data checksum**: if CI rebuilds data every push, the `model.json` numbers should be identical across runs (decay fit is deterministic, KFold is seeded). Add a CI assertion that `CV mean ∈ [3.05, 3.20]` to catch regressions.

---

## 10. Out of scope for v1

- Test-set (20k) exploration — only 1000 labeled train sessions are worth visualizing (they have ground truth).
- Re-training in the browser (Pyodide) — overkill.
- Feature-flag A/B over past model versions — the old experiments stay in the old repo.
- User accounts, analytics, PR review app.
- Mobile-first design — judge audience primarily uses desktops; basic responsive layout is enough.

---

## 11. Quick-start for the implementer

If you're picking this up cold:

1. Create the repo. `git init && mkdir -p model data scripts web/src/{components,lib} web/public/data docs/legacy_plots .github/workflows`.
2. Copy over the 6 mandatory files from §2.
3. Adapt `model/solution.py` paths per §2 (three `os.path.join` lines → use `config.py`).
4. Write `scripts/build_site_data.py` following §5.
5. Run it locally to generate `web/public/data/*.json`.
6. `cd web && npm create vite@latest . -- --template react-ts`, then install the deps from §6.2.
7. Implement components top-down: `App.tsx` → `Navigation` → `Intro` → `SessionExplorer` → `CoefficientsView` → `DecayCurves` → `PerformancePanel` → `AlignmentDemo` (hardest — last).
8. Verify locally per §8.
9. Commit, push, enable GH Pages, watch CI, ship.

Expected total build time (solo, focused): **~2 days** for a polished v1. The SessionExplorer and AlignmentDemo take the most time; the rest are 1–2 hours each.
