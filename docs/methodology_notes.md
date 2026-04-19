# Lenny's Solution Notes

## Problem Understanding

We're given simulated stock trading data:
- **1000 training sessions**, each with **100 bars** (time steps) of OHLC data
- Bars 0-49 are "seen" (available), bars 50-99 are "unseen" (future)
- Each session also has news headlines (~10 per session)
- Prices are normalized to start at 1.0

**Task**: At the halfway point (bar 49), decide a `target_position` (shares to buy/sell).
We buy at close of bar 49 and sell at close of bar 99.

**Metric**: Sharpe ratio across sessions
```
pnl_i = target_position_i * (close_end / close_halfway - 1)
sharpe = mean(pnl) / std(pnl) * 16
```

Since Sharpe is scale-invariant, only the **direction and relative sizing** of positions matter.

## Key Findings

### Price Signal: Mean Reversion on ~20-bar window
- Strong mean reversion: stocks that went up in the first half tend to come back
- `ret_last20` (return over last 20 bars) has correlation **-0.08** with future return
- Mean reversion Sharpe (scaled by ret_last20): **1.12**

### Price Signal: Short-term Momentum (last 5 bars)
- Counterintuitively, very recent momentum (last 5 bars) **continues** into the future
- `ret_last5` has a positive but weaker correlation (+0.036)
- These two signals are complementary (reversion on longer window, momentum on shorter)

### Headline Sentiment
- Simple keyword-based sentiment score correlates with future return (+0.05)
- Standalone sentiment strategy Sharpe: **2.03** — very strong!
- Interestingly, sentiment of "other companies" in the session has higher correlation (0.065) than the main company (0.004)
  - This makes sense: the challenge says "not all headlines are relevant"

### Volatility
- Higher volatility is mildly predictive of positive returns (+0.07 correlation)
- But adding `vol` to the model actually hurts CV Sharpe (overfitting risk)

## Model Selection

### Ridge Regression wins
- **Best features**: `ret_full`, `ret_last5`, `ret_last20`, `sentiment`
- **Best alpha**: 1.0 (or 50.0 with slightly higher Sharpe in some seeds)
- **5-fold CV Sharpe**: ~3.06 (robust across 10 random seeds: range 2.88-3.12)

### Why not tree-based models?
- GBM best CV Sharpe: ~2.74 (much worse than Ridge)
- Random Forest: <2.5
- The signal is fundamentally linear — price returns combine linearly with sentiment

### Position scaling
- Using predictions directly (power=1.0) is near-optimal
- Slight improvement with power=0.7 (Sharpe 3.07) — dampens extreme positions

## Solution Architecture

1. **Feature Engineering**: Extract 4 features per session from seen bars + headlines
2. **Model**: Ridge regression (alpha=1.0) trained on all training data
3. **Position**: Model prediction used as target_position (scale doesn't matter)
4. **Generate**: Predictions for both public and private test sets
