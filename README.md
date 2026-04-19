# HRT Datathon 2026 — Solution Walkthrough

Interactive visualization of the winning Ridge regression model for the HRT Datathon 2026. Frozen model (CV Sharpe ~3.2, public LB 2.88) is wrapped in a static React + Plotly site that judges can click through to inspect features, coefficients, learned decay rates, the alignment-weighted FinBERT mechanism, and per-session predictions.

> Live demo: _set after first deploy — `https://<user>.github.io/<repo-name>/`_

## Stack

- **Model**: Python 3.12, PyTorch (CPU), scikit-learn, frozen at submission time in [model/solution.py](model/solution.py)
- **Data prep**: one-shot script [scripts/build_site_data.py](scripts/build_site_data.py) emits 4 JSON artifacts
- **Frontend**: Vite + React 18 + TypeScript + Plotly.js + KaTeX + Tailwind
- **CI/CD**: GitHub Actions → GitHub Pages ([.github/workflows/pages.yml](.github/workflows/pages.yml))

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

Push to `main` → GitHub Actions runs the model end-to-end, builds the React app with `VITE_BASE=/<repo-name>/`, and publishes to Pages. **One-time setup:** in repo Settings → Pages → Source, pick **GitHub Actions**.

If you're deploying to a user/org page or a custom domain, override the base path: edit [.github/workflows/pages.yml](.github/workflows/pages.yml) to set `VITE_BASE: /` instead.

## Repo layout

```
model/                    frozen model (do not modify)
  solution.py             Ridge + alignment-weighted FinBERT, ~3.2 CV Sharpe
  config.py               path resolution
data/                     5 input parquets (~2.7 MB) — bars, headlines, FinBERT, LLM annotations
scripts/
  build_site_data.py      one-shot: trains model + dumps JSON for the site
web/
  src/components/         7 React components (Intro, Sessions, Alignment, Coefficients, Decay, Performance, MathBlock)
  src/lib/                types, data loaders, format helpers, Plotly factory wrapper
  public/data/            generated JSON (gitignored — run build_site_data.py)
docs/                     methodology notes, findings log, legacy plots
HANDOVER.md               full technical writeup of the model + site spec
```

## What's in the site

| Tab | Shows |
|---|---|
| **Intro** | Problem setup, Parkinson volatility, the 10 features (with formulas), the alignment mechanism, differentiable-Ridge decay optimization, training & inference. |
| **Sessions** | Per-session candlestick + headline overlay (color = FinBERT, shape = LLM label) + feature contribution table sorted by \|coef × std\|. |
| **Alignment** | 12 curated headlines spanning the 4 (sentiment × alignment) quadrants, with the (u, v) motion vectors drawn in normalized space and the muted vs aligned half-plane shaded. |
| **Coefficients** | Standardized Ridge coefficients sorted by magnitude, color-coded by sign, each with its formula and 1-line interpretation. |
| **Decay** | The two learned exponential decay curves with sliders to explore alternatives (visualization only — sliders do not retrain). |
| **Performance** | 5-fold CV Sharpe bars, full α grid sweep on log-x, predicted-vs-actual scatter (R² + Pearson r), full α table. |

## Methodology

See [HANDOVER.md](HANDOVER.md) for the full technical writeup. In short: a Ridge on 10 vol-normalized features, including 4 FinBERT signals weighted by an alignment scalar that measures how much the price moved in the direction the headline predicted within a 3- or 5-bar lookahead. Decay rates for sentiment age are learned by differentiating Sharpe through the Ridge closed-form (PyTorch + `torch.linalg.solve`).
