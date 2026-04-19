# HRT Datathon 2026 — Solution Walkthrough

Interactive visualization of our Ridge regression solution for the HRT Datathon 2026. A frozen model trained on 2 000 labelled samples (1 000 original + 1 000 generated via shifted-split augmentation) is wrapped in a static React + Plotly site that judges can click through to inspect features, coefficients, learned decay rates, the alignment-weighted FinBERT mechanism, and per-session predictions.

> **Live demo:** <https://mohamedsadfi.github.io/hrt-datathon-viz/>

## Approach in one paragraph

Each labelled session has 100 OHLC bars + ~10 news headlines about various companies, of which exactly one is the trading subject. The model predicts a vol-normalized second-half return from 10 hand-engineered features: 3 vol-scaled price-return signals (full first-half, 5-bar momentum, 20-bar mean-reversion), a path-shape signal (`cand_up_ratio`), 4 alignment-weighted FinBERT sentiment aggregates (3-bar and 5-bar look-ahead × pos/neg sign), a quadratic confidence-belief term, and a time-decayed count of LLM-labelled negative headlines. The two exponential decay rates `(d₊, d₋)` for sentiment age are *learned* by differentiating the in-sample Sharpe through the Ridge closed-form (PyTorch + `torch.linalg.solve`); 600 Adam steps converge deterministically. Cross-validation uses 5-fold GroupKFold so that an original session and its shifted-split augmented twin always end up in the same fold.

## Stack

- **Model** — Python 3.12, PyTorch (CPU), scikit-learn ([model/solution.py](model/solution.py))
- **Data prep** — one-shot script ([scripts/build_site_data.py](scripts/build_site_data.py)) emits 4 JSON artifacts
- **Frontend** — Vite + React 18 + TypeScript + Plotly.js + KaTeX + Tailwind
- **CI/CD** — GitHub Actions → GitHub Pages ([.github/workflows/pages.yml](.github/workflows/pages.yml))

## Local quickstart

```bash
# 1. Python: install + generate the site's JSON artifacts
uv sync
uv run python scripts/build_site_data.py
# → writes web/public/data/{model,sessions,alignment_examples,features_corr}.json (~7 MB)

# 2. Frontend: install + dev server
cd web
npm install
npm run dev
# → http://localhost:5173/
```

Production build: `npm run build && npm run preview`.

## Deployment

Push to `main` → GitHub Actions runs the model end-to-end (decay fit → α grid sweep → CV → final fit → JSON dump → React build) and publishes to Pages. The workflow auto-derives `VITE_BASE` from the repo name so asset paths resolve under any project page. **One-time setup:** in repo Settings → Pages → Source, pick **GitHub Actions**.

For a user/org page or custom domain, edit [.github/workflows/pages.yml](.github/workflows/pages.yml) to set `VITE_BASE: /` instead.

## Repo layout

```text
model/
  solution.py             frozen model — Ridge + alignment-weighted FinBERT + LLM neg + augmentation
  config.py               path resolution
data/                     6 input parquets (~2.9 MB) — bars (seen/unseen), headlines (seen/unseen),
                          FinBERT scores, LLM annotations
scripts/
  build_site_data.py      one-shot pipeline: trains model + dumps JSON artifacts
web/
  src/components/         8 React components (Intro, Sessions, Alignment, Coefficients, Decay,
                          Performance, Navigation, MathBlock)
  src/lib/                types, data loaders, format helpers, Plotly factory wrapper
  public/data/            generated JSON (gitignored — run build_site_data.py)
.github/workflows/
  pages.yml               CI: build + deploy to GitHub Pages
```

## What's in the site

| Tab | Shows |
| --- | --- |
| **Intro** | Problem setup, Parkinson volatility, the 10 features (with formulas), the alignment mechanism, differentiable-Ridge decay optimization, shifted-split data augmentation, training & inference. |
| **Sessions** | Per-session candlestick + headline overlay (color = FinBERT, shape = LLM label) + feature contribution table sorted by \|coef × std\|, with ŷ / y / position / PnL strip up top. |
| **Alignment** | 12 curated headlines spanning the 4 (sentiment × alignment) quadrants, with the (u, v) motion vectors drawn in normalized space and the consistent half-plane shaded. |
| **Coefficients** | Standardized Ridge coefficients sorted by magnitude, color-coded by sign, each with its formula and 1-line interpretation. |
| **Decay** | The two learned exponential decay curves with sliders (visualize-only, do not retrain) and their half-life annotations. |
| **Performance** | 5-fold GroupKFold per-fold Sharpe bars, full α grid sweep on log-x, predicted-vs-actual scatter (R² + Pearson r), full α table. |
