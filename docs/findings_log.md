# v36 build trail

## Starting point: submission_17 (our v34 replica)

Lenny's best LB submission. Architecture:

- **6 features**: 3 vol-normalized price returns (`ret_all_vol`, `ret_last5_vol`, `ret_last20_vol`) where `vol = std(diff(log(close)))` over the 50 seen bars, plus 3 FinBERT sentiment aggregates (`finbert_pos`, `finbert_neg`, `finbert_conf_belief = net·|net|/gross`).
- **Target**: `y = (close[99]/close[49] - 1) / vol` — predict a vol-adjusted return, not raw.
- **Decays**: `(dp, dn)` learned by gradient descent *through* the Ridge closed-form solution on Sharpe loss (converged to `dp=0.0978, dn=0.0798`).
- **Ridge α**: grid search with `TimeSeriesSplit(5)`, α=200 wins.
- **Position at inference**: `pos = preds / vol`, then rescale to `std=100` (inverse-vol Kelly sizing).

## What was investigated for v36

### EDA 1 — oracle ceiling

For each training session, tested what Sharpe we'd get using only one company's headlines (first 2 words = company, company with max signed sentiment picked per session with hindsight).

- FinBERT oracle: **+10.71**
- LLM oracle: **+11.34**
- All-summed baseline: +1.70

Confirms there's a huge subject-company signal IF we can identify it without the target.

### EDA 2 — winning company profile

Averaged properties of the oracle-winning company vs losing ones across 1000 sessions:

| Property | Winner mean | Non-win mean | Ratio |
|---|---|---|---|
| `last_bar` | 37.8 | 32.2 | **1.17×** (more recent headlines) |
| `n_num` | 1.0 | 0.8 | **1.20×** (more quantitative headlines) |
| `n_pos_llm` | 1.41 | 1.16 | **1.21×** (more pos-tagged news) |
| `n_headlines` | 2.86 | 2.59 | 1.10× |

Small but consistent directional signals.

### EDA 3 — number-context lift

For each LLM-extracted event type, what fraction of occurrences are in the winning company? Baseline rate: 27.3%.

| Event type | N | Win% | Lift |
|---|---|---|---|
| `customer_acquisition_growth_pct` | 183 | 37.7% | **1.38×** |
| `revenue_growth_yoy_pct` | 40 | 37.5% | **1.37×** |
| `operating_income_decline_pct` | 225 | 36.9% | **1.35×** |
| `margin_improvement_pct` | 212 | 34.0% | **1.24×** |
| `contract_value_usd` | 1602 | 31.3% | 1.15× |
| `share_buyback_usd` | 239 | 25.9% | 0.95× |
| `revenue_miss_pct` | 237 | 27.4% | 1.00× |

Fundamental-quality news lifts; routine news doesn't.

## What was built

Started with 6 additive candidate features on top of the 6 baselines:

1. `evt_margin_improve` — time-decayed count of `margin_improvement_pct` events
2. `evt_op_decline` — time-decayed count of `operating_income_decline_pct` events
3. `evt_cust_acq` — time-decayed count of `customer_acquisition_growth_pct`
4. `evt_contract_log` — `log1p(Σ contract_value × decay)`
5. `llm_pos_decay` — time-decayed count of LLM-"pos"-tagged headlines
6. `llm_neg_decay` — time-decayed count of LLM-"neg"-tagged headlines (signed negative)

## Ablation surprise

Added each feature alone to the 6-feature baseline, measured TimeSeriesSplit CV Sharpe Δ:

| Feature added | α | CV Sharpe | Δ |
|---|---|---|---|
| baseline (6 feat) | 200 | +3.2522 | — |
| +evt_margin_improve | 400 | +3.1664 | **−0.086** |
| +evt_op_decline | 250 | +3.2348 | −0.017 |
| +evt_cust_acq | 250 | +3.2348 | −0.017 |
| +evt_contract_log | 350 | +3.2522 | 0 |
| +llm_pos_decay | 200 | +3.2433 | −0.009 |
| **+llm_neg_decay** | 150 | **+3.3017** | **+0.049** |

All 6 together: α=500, Sharpe +3.1766 (worse than baseline). Classic overfit — each event feature steals regularization budget from the useful ones.

## v36 final

**7 features = 6 baseline + `llm_neg_decay`**. At α=150:

- `llm_neg_decay` gets **+0.455** coef (big)
- `finbert_neg` gets flipped to **−0.313** (was +0.29 in v34)

Ridge is effectively **replacing FinBERT's continuous neg signal with LLM's cleaner categorical neg labels**. FinBERT gives `score = P(pos) − P(neg)` per headline (noisy continuous). LLM gives discrete neg/pos/neutral (less noisy, less granular). For negatives, discreteness wins.

## Difference from submission_17

Same everything except one extra feature that's a pure "number of negative-labeled headlines (time-decayed)" signal. Cheap to compute, one dimension added, Ridge can't easily overfit a single well-motivated feature. Corr 0.984 with submission_17.

## Why this is safer than v35

v35 added 3 *per-session-argmax-selected* sentiment features (highest, last, largest_abs). Those are non-linear selectors — they pick *different companies* per session, which Ridge can't regularize the selection rule. Got +0.098 CV but LB collapsed to 2.68.

v36 adds one *globally-computed* feature (sum over session). No per-sample selection. If it overfits, α should shrink it. The +0.049 CV lift is smaller but more robust by construction.
