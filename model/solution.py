"""
Datathon 2026 HRT Challenge — Lenny's Solution
================================================

Predicts the second-half return direction and magnitude using:
  1. Vol-normalized full first-half return
  2. Vol-normalized short-term momentum (~5-bar return)
  3. Vol-normalized medium-term mean-reversion (~20-bar return)
  4. FinBERT sentiment with optimized asymmetric exponential decay:
       - finbert_neg:        negative-headline signal (decay_neg optimized)
       - finbert_conf_belief: confidence-weighted combined signal
                              = net * |net| / (gross + ε)
                              Penalizes sessions where headlines disagree.
     Decay rates are found by differentiating through the Ridge closed-form
     to directly maximise Sharpe on training data. Optimizer converges
     deterministically — a single run is sufficient.
     Note: finbert_pos is intentionally omitted — empirically it receives a
     near-zero coefficient and wastes Ridge regularisation budget.
  5. Ridge alpha is selected via CV grid search over [25, 50, ..., 1000].
  6. LLM negative sentiment: llm_neg_decay = time-decayed count of LLM-"neg"
     headlines.
  7. Price-path shape: cand_up_ratio = fraction of first-half bars that
     close above the prior bar (up-bar ratio).

Model: Ridge regression (alpha found by KFold CV)
CV Sharpe: ~3.1+ (5-fold, robust across seeds)
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

# ─── FinBERT Sentiment Cache ─────────────────────────────────────────────────

from .config import FINBERT_CACHE, LLM_ANNOTATIONS_PATH, DATA_DIR as _DATA_DIR


def _load_finbert_scores() -> dict:
    fb_df = pd.read_parquet(FINBERT_CACHE)
    return dict(zip(fb_df["headline"], fb_df["score"]))


def _load_llm_sentiments() -> dict:
    """Load LLM headline annotations → {headline: 'pos'|'neutral'|'neg'}."""
    if not LLM_ANNOTATIONS_PATH.exists():
        print("  WARNING: LLM annotations not found; llm_neg_decay will be zero.")
        return {}
    ann = pd.read_parquet(LLM_ANNOTATIONS_PATH)
    return dict(zip(ann["headline"], ann["sentiment"]))


FINBERT_SCORES = _load_finbert_scores()
LLM_SENTIMENTS = _load_llm_sentiments()

# ─── Configuration ───────────────────────────────────────────────────────────

DATA_DIR = str(_DATA_DIR)
OUTPUT_DIR = os.path.dirname(__file__)

RIDGE_ALPHA = 50.0  # default; overridden by alpha grid search at runtime
# Extended grid: small datasets often need high regularisation (alpha > 200)
ALPHA_GRID = [25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0,
              250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 600.0, 750.0, 1000.0]

FEATURE_COLS = ["ret_all_vol", "ret_last5_vol", "ret_last20_vol",
                "finbert_pos_align3", "finbert_neg_align3",
                "finbert_pos_align5", "finbert_neg_align5",
                "finbert_conf_belief",
                "llm_neg_decay", "cand_up_ratio"]

def compute_alignments(bars_df, headlines_df):
    """
    Returns scores multiplied by max(0, alignment) for 3-bar and 5-bar horizons.
    """
    scores3 = pd.Series(0.0, index=headlines_df.index)
    scores5 = pd.Series(0.0, index=headlines_df.index)
    
    for session_id, h_idx in headlines_df.groupby("session").groups.items():
        b_group = bars_df[bars_df["session"] == session_id]
        if len(b_group) == 0: continue
        b_group = b_group.sort_values("bar_ix")
        
        bars = b_group["bar_ix"].values
        closes = b_group["close"].values
        ymax = b_group["high"].max()
        ymin = b_group["low"].min()
        y_range = ymax - ymin
        if y_range == 0: y_range = 1e-8
        x_range = bars[-1] - bars[0] + 10
        
        for idx_in_df in h_idx:
            headline = headlines_df.at[idx_in_df, "headline"]
            score = FINBERT_SCORES.get(headline, 0.0)
            if score == 0.0: continue
                
            b_ix = headlines_df.at[idx_in_df, "bar_ix"]
            valid_idx = np.where(bars <= b_ix)[0]
            if len(valid_idx) == 0:
                continue
            idx = valid_idx[-1]
            p0 = closes[idx]
            
            idx3 = min(idx + 3, len(closes) - 1)
            p3 = closes[idx3]
            if idx3 > idx:
                u3 = (bars[idx3] - bars[idx]) / x_range
                v3 = (p3 - p0) / y_range
                norm3 = np.sqrt(u3**2 + v3**2 + 1e-8)
                align3 = score * (v3 / norm3)
            else:
                align3 = abs(score)
            
            idx5 = min(idx + 5, len(closes) - 1)
            p5 = closes[idx5]
            if idx5 > idx:
                u5 = (bars[idx5] - bars[idx]) / x_range
                v5 = (p5 - p0) / y_range
                norm5 = np.sqrt(u5**2 + v5**2 + 1e-8)
                align5 = score * (v5 / norm5)
            else:
                align5 = abs(score)
            
            scores3.at[idx_in_df] = score * max(0.0, align3)
            scores5.at[idx_in_df] = score * max(0.0, align5)
            
    return scores3.values, scores5.values


# ─── Data Augmentation ──────────────────────────────────────────────────────

def create_augmented_data(
    bars_seen: pd.DataFrame,
    bars_unseen: pd.DataFrame,
    headlines_seen: pd.DataFrame,
    headlines_unseen: pd.DataFrame,
    split_bar: int = 74,
) -> tuple:
    """
    Create augmented training samples by shifting the seen/unseen split point.

    For split_bar=74 (default):
      Original : seen = bars 0–49,  unseen = bars 50–99
      Augmented: seen = bars 25–74, unseen = bars 75–99

    The augmented "seen" bars are re-indexed to bar_ix 0–49 so that
    feature engineering (which pivots on bar_ix) works identically.
    Augmented session IDs are offset to avoid collision with originals.

    Returns (aug_bars_seen, aug_bars_unseen, aug_headlines_seen, session_offset).
    """
    all_bars = pd.concat([bars_seen, bars_unseen], ignore_index=True)
    all_bars = all_bars.sort_values(["session", "bar_ix"])
    all_headlines = pd.concat([headlines_seen, headlines_unseen], ignore_index=True)
    all_headlines = all_headlines.sort_values(["session", "bar_ix"])

    n_seen = 50  # always 50 bars in the seen window
    aug_start = split_bar - (n_seen - 1)  # e.g. 74 − 49 = 25

    # ── Augmented "seen" bars: [aug_start, split_bar] ─────────────────────
    mask_seen = (all_bars["bar_ix"] >= aug_start) & (all_bars["bar_ix"] <= split_bar)
    aug_bs = all_bars[mask_seen].copy()
    aug_bs["bar_ix"] = aug_bs["bar_ix"] - aug_start  # re-index → 0..49

    # ── Augmented "unseen" bars: (split_bar, 99] ─────────────────────────
    aug_bu = all_bars[all_bars["bar_ix"] > split_bar].copy()

    # Keep only sessions that have both seen AND unseen bars
    valid = set(aug_bs["session"].unique()) & set(aug_bu["session"].unique())
    # … and exactly n_seen bars in the seen window
    counts = aug_bs[aug_bs["session"].isin(valid)].groupby("session").size()
    valid = set(counts[counts == n_seen].index)

    aug_bs = aug_bs[aug_bs["session"].isin(valid)].copy()
    aug_bu = aug_bu[aug_bu["session"].isin(valid)].copy()

    # ── Augmented headlines: bar_ix ∈ [aug_start, split_bar] ──────────────
    mask_hdl = (
        (all_headlines["bar_ix"] >= aug_start)
        & (all_headlines["bar_ix"] <= split_bar)
        & all_headlines["session"].isin(valid)
    )
    aug_hdl = all_headlines[mask_hdl].copy()
    aug_hdl["bar_ix"] = aug_hdl["bar_ix"] - aug_start

    # ── Offset session IDs ───────────────────────────────────────────────
    session_offset = int(bars_seen["session"].max()) + 1
    aug_bs["session"]  = aug_bs["session"]  + session_offset
    aug_bu["session"]  = aug_bu["session"]  + session_offset
    aug_hdl["session"] = aug_hdl["session"] + session_offset

    return aug_bs, aug_bu, aug_hdl, session_offset


# Decay rate defaults (used as optimizer starting point; final values learned at runtime)
_DEFAULT_DECAY_POS = 0.028
_DEFAULT_DECAY_NEG = 0.028

# ─── Sentiment Decay Optimization ───────────────────────────────────────────

def _optimize_decay_params(
    bars_seen_train: pd.DataFrame,
    bars_unseen_train: pd.DataFrame,
    headlines_seen_train: pd.DataFrame,
    n_epochs: int = 600,
    lr: float = 0.02,
    seed: int = 42,
) -> tuple[float, float]:
    """
    One run of the decay optimizer. Differentiates through the Ridge closed-
    form to maximise in-sample Sharpe w.r.t. (decay_pos, decay_neg).

    Uses an internal 6-feature model [price, finbert_pos, finbert_neg, conf]
    just to learn (dp, dn); the final Ridge uses a different feature set.
    """
    torch.manual_seed(seed)
    sessions = np.sort(bars_seen_train["session"].unique())
    n = len(sessions)
    s2i = {s: i for i, s in enumerate(sessions)}

    # Vectorised price features
    bars = bars_seen_train.sort_values(["session", "bar_ix"])
    C = bars.pivot(index="session", columns="bar_ix", values="close").reindex(sessions).to_numpy()
    O = bars.pivot(index="session", columns="bar_ix", values="open").reindex(sessions).to_numpy()
    H = bars.pivot(index="session", columns="bar_ix", values="high").reindex(sessions).to_numpy()
    L = bars.pivot(index="session", columns="bar_ix", values="low").reindex(sessions).to_numpy()
    last = C[:, -1]
    parkinson_var = (1.0 / (4.0 * np.log(2.0))) * np.mean((np.log(H / L)) ** 2, axis=1)
    vol = np.maximum(np.sqrt(parkinson_var), 1e-6)
    price_rows = np.column_stack([
        (last / O[:, 0]  - 1) / vol,
        (last / C[:, -5] - 1) / vol,
        (last / C[:, -20] - 1) / vol,
    ])
    price_X = torch.tensor(price_rows, dtype=torch.float32)

    # Target: raw second-half return scaled by vol
    close_half = bars_seen_train.groupby("session")["close"].last().reindex(sessions).values
    close_end  = bars_unseen_train.groupby("session")["close"].last().reindex(sessions).values
    y_vol_adj  = torch.tensor((close_end / close_half - 1) / vol, dtype=torch.float32)

    # Sentiment tensors
    last_bar_ix = int(bars_seen_train["bar_ix"].max())
    df = headlines_seen_train.copy()
    scores3, scores5 = compute_alignments(bars_seen_train, df)
    df["score"] = df["headline"].map(FINBERT_SCORES).fillna(0.0)
    df["score3"] = scores3
    df["score5"] = scores5
    df["age"]   = last_bar_ix - df["bar_ix"].values
    df["sidx"]  = df["session"].map(s2i)
    df = df[df["score"] != 0.0].reset_index(drop=True)

    pos_df = df[df["score"] > 0].reset_index(drop=True)
    neg_df = df[df["score"] < 0].reset_index(drop=True)

    pos_scores  = torch.tensor(pos_df["score"].values, dtype=torch.float32)
    pos_scores3 = torch.tensor(pos_df["score3"].values, dtype=torch.float32)
    pos_scores5 = torch.tensor(pos_df["score5"].values, dtype=torch.float32)
    pos_ages    = torch.tensor(pos_df["age"].values,   dtype=torch.float32)
    pos_sidx    = torch.tensor(pos_df["sidx"].values,  dtype=torch.long)
    
    neg_scores  = torch.tensor(neg_df["score"].values, dtype=torch.float32)
    neg_scores3 = torch.tensor(neg_df["score3"].values, dtype=torch.float32)
    neg_scores5 = torch.tensor(neg_df["score5"].values, dtype=torch.float32)
    neg_ages    = torch.tensor(neg_df["age"].values,   dtype=torch.float32)
    neg_sidx    = torch.tensor(neg_df["sidx"].values,  dtype=torch.long)

    alpha_r = torch.tensor(float(RIDGE_ALPHA))
    log_dp  = torch.tensor(float(np.log(_DEFAULT_DECAY_POS)), requires_grad=True)
    log_dn  = torch.tensor(float(np.log(_DEFAULT_DECAY_NEG)), requires_grad=True)
    opt     = torch.optim.Adam([log_dp, log_dn], lr=lr)

    for epoch in range(n_epochs):
        opt.zero_grad()

        dp = torch.exp(log_dp)
        dn = torch.exp(log_dn)

        fp = torch.zeros(n).scatter_add(0, pos_sidx, pos_scores * torch.exp(-dp * pos_ages))
        fn = torch.zeros(n).scatter_add(0, neg_sidx, neg_scores * torch.exp(-dn * neg_ages))
        fp3 = torch.zeros(n).scatter_add(0, pos_sidx, pos_scores3 * torch.exp(-dp * pos_ages))
        fn3 = torch.zeros(n).scatter_add(0, neg_sidx, neg_scores3 * torch.exp(-dn * neg_ages))
        fp5 = torch.zeros(n).scatter_add(0, pos_sidx, pos_scores5 * torch.exp(-dp * pos_ages))
        fn5 = torch.zeros(n).scatter_add(0, neg_sidx, neg_scores5 * torch.exp(-dn * neg_ages))

        net   = fp + fn
        gross = fp + fn.abs()
        conf  = net * net.abs() / (gross + 1e-8)

        X = torch.cat([price_X, fp3.unsqueeze(1), fn3.unsqueeze(1), fp5.unsqueeze(1), fn5.unsqueeze(1), conf.unsqueeze(1)], dim=1)
        mu  = X.mean(0)
        sig = X.std(0) + 1e-8
        Xs  = (X - mu) / sig

        A = Xs.T @ Xs + alpha_r * torch.eye(8)
        w = torch.linalg.solve(A, Xs.T @ y_vol_adj)

        preds = Xs @ w
        pnl   = preds * y_vol_adj  # Evaluated on y_vol_adj (equivalent to proportional risk scaling)
        loss  = -(pnl.mean() / (pnl.std() + 1e-8) * 16)
        loss.backward()
        opt.step()

    return float(torch.exp(log_dp).detach()), float(torch.exp(log_dn).detach())


# ─── Feature Engineering ─────────────────────────────────────────────────────

def extract_features(
    bars_seen: pd.DataFrame,
    headlines_seen: pd.DataFrame,
    decay_pos: float = _DEFAULT_DECAY_POS,
    decay_neg: float = _DEFAULT_DECAY_NEG,
) -> pd.DataFrame:
    """
    Build session-level features (vectorised numpy).

    Returns a DataFrame with columns:
      ret_all_vol, ret_last5_vol, ret_last20_vol  — vol-normalised price returns
      finbert_pos                                 — FinBERT pos sentiment (decayed)
      finbert_neg                                 — FinBERT neg sentiment (decayed)
      finbert_conf_belief                         — confidence-weighted belief
      llm_neg_decay                               — LLM neg headline count (decayed)
      cand_up_ratio                               — fraction of up-bars
      halfway_close                               — last seen-half close price
      vol                                         — realised volatility of log returns
    """
    bars = bars_seen.sort_values(["session", "bar_ix"])
    sessions = np.sort(bars["session"].unique())

    C = bars.pivot(index="session", columns="bar_ix", values="close").reindex(sessions).to_numpy()
    O = bars.pivot(index="session", columns="bar_ix", values="open").reindex(sessions).to_numpy()
    H = bars.pivot(index="session", columns="bar_ix", values="high").reindex(sessions).to_numpy()
    L = bars.pivot(index="session", columns="bar_ix", values="low").reindex(sessions).to_numpy()
    last = C[:, -1]
    log_rets = np.diff(np.log(C), axis=1)
    parkinson_var = (1.0 / (4.0 * np.log(2.0))) * np.mean((np.log(H / L)) ** 2, axis=1)
    vol = np.maximum(np.sqrt(parkinson_var), 1e-6)

    df = pd.DataFrame({
        "ret_all_vol":    (last / O[:, 0]   - 1) / vol,
        "ret_last5_vol":  (last / C[:, -5]  - 1) / vol,
        "ret_last20_vol": (last / C[:, -20] - 1) / vol,
        "cand_up_ratio":  (log_rets > 0).mean(axis=1).astype(np.float32),
        "halfway_close":  last,
        "vol":            vol.astype(np.float32),
    }, index=pd.Index(sessions, name="session"))

    # ── FinBERT sentiment (decayed) ──────────────────────────────────────────
    scores3, scores5 = compute_alignments(bars, headlines_seen)

    last_bar_ix = int(bars_seen["bar_ix"].max())
    h_fb = headlines_seen.copy()
    h_fb["score"] = h_fb["headline"].map(FINBERT_SCORES).fillna(0.0).astype(np.float32)
    h_fb["score3"] = scores3.astype(np.float32)
    h_fb["score5"] = scores5.astype(np.float32)
    h_fb = h_fb[h_fb["score"] != 0.0].copy()
    h_fb["age"] = last_bar_ix - h_fb["bar_ix"].values

    pos_h = h_fb[h_fb["score"] > 0].copy()
    neg_h = h_fb[h_fb["score"] < 0].copy()
    pos_h["w"] = pos_h["score"].to_numpy() * np.exp(-decay_pos * pos_h["age"].to_numpy())
    neg_h["w"] = neg_h["score"].to_numpy() * np.exp(-decay_neg * neg_h["age"].to_numpy())
    pos_h["w3"] = pos_h["score3"].to_numpy() * np.exp(-decay_pos * pos_h["age"].to_numpy())
    pos_h["w5"] = pos_h["score5"].to_numpy() * np.exp(-decay_pos * pos_h["age"].to_numpy())
    neg_h["w3"] = neg_h["score3"].to_numpy() * np.exp(-decay_neg * neg_h["age"].to_numpy())
    neg_h["w5"] = neg_h["score5"].to_numpy() * np.exp(-decay_neg * neg_h["age"].to_numpy())

    fp = pos_h.groupby("session")["w"].sum().reindex(sessions, fill_value=0.0).to_numpy()
    fn = neg_h.groupby("session")["w"].sum().reindex(sessions, fill_value=0.0).to_numpy()
    fp3 = pos_h.groupby("session")["w3"].sum().reindex(sessions, fill_value=0.0).to_numpy()
    fn3 = neg_h.groupby("session")["w3"].sum().reindex(sessions, fill_value=0.0).to_numpy()
    fp5 = pos_h.groupby("session")["w5"].sum().reindex(sessions, fill_value=0.0).to_numpy()
    fn5 = neg_h.groupby("session")["w5"].sum().reindex(sessions, fill_value=0.0).to_numpy()

    net   = fp + fn
    gross = fp + np.abs(fn)

    df["finbert_pos_align3"]  = fp3
    df["finbert_neg_align3"]  = fn3
    df["finbert_pos_align5"]  = fp5
    df["finbert_neg_align5"]  = fn5
    df["finbert_conf_belief"] = net * np.abs(net) / (gross + 1e-8)

    # ── LLM negative sentiment (decayed) ─────────────────────────────────────
    if LLM_SENTIMENTS:
        h_llm = headlines_seen.copy()
        h_llm["llm_sent"] = h_llm["headline"].map(LLM_SENTIMENTS).fillna("neutral")
        h_llm["age"] = last_bar_ix - h_llm["bar_ix"].values
        h_neg_llm = h_llm[h_llm["llm_sent"] == "neg"].copy()
        llm_neg = -h_neg_llm.groupby("session")["age"].apply(
            lambda a: np.exp(-decay_neg * a.to_numpy()).sum()
        ).reindex(sessions, fill_value=0.0).astype(np.float32)
        df["llm_neg_decay"] = llm_neg.to_numpy()
    else:
        df["llm_neg_decay"] = 0.0

    return df.reset_index()


# ─── Alpha Grid Search ────────────────────────────────────────────────────────

def _find_best_alpha(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
) -> float:
    """CV grid search over ALPHA_GRID; return the alpha with the highest mean Sharpe.
    Fits and evaluates on the raw (unclipped) target.
    Uses GroupKFold when groups are provided (to prevent augmented-sample leakage).
    """
    kf_results: dict[float, list[float]] = {a: [] for a in ALPHA_GRID}
    if groups is not None:
        kf = GroupKFold(n_splits=n_splits)
        splits = list(kf.split(X, groups=groups))
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kf.split(X))

    for train_idx, test_idx in splits:
        sc = StandardScaler().fit(X[train_idx])
        Xtr = sc.transform(X[train_idx])
        Xvl = sc.transform(X[test_idx])
        for alpha in ALPHA_GRID:
            m = Ridge(alpha=alpha).fit(Xtr, y[train_idx])
            pnl = m.predict(Xvl) * y[test_idx]
            if pnl.std() > 0:
                kf_results[alpha].append(pnl.mean() / pnl.std() * 16)

    mean_sharpes = {a: float(np.mean(v)) if v else 0.0 for a, v in kf_results.items()}
    best_alpha = max(mean_sharpes, key=mean_sharpes.__getitem__)
    print("  Alpha grid search results:")
    for a in ALPHA_GRID:
        marker = "  <--" if a == best_alpha else ""
        print(f"    alpha={a:6.0f}  mean CV Sharpe={mean_sharpes[a]:+.4f}{marker}")
    return best_alpha


# ─── Training & Evaluation ────────────────────────────────────────────────────

def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = RIDGE_ALPHA,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
) -> list[float]:
    """Run cross-validation and return per-fold Sharpe ratios.
    Fits and evaluates on the raw (unclipped) target.
    Uses GroupKFold when groups are provided (to prevent augmented-sample leakage).
    """
    if groups is not None:
        kf = GroupKFold(n_splits=n_splits)
        splits = list(kf.split(X, groups=groups))
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kf.split(X))
    fold_sharpes = []

    for train_idx, test_idx in splits:
        scaler = StandardScaler().fit(X[train_idx])
        X_train = scaler.transform(X[train_idx])
        X_test  = scaler.transform(X[test_idx])

        model = Ridge(alpha=alpha).fit(X_train, y[train_idx])
        preds = model.predict(X_test)

        pnl = preds * y[test_idx]
        if pnl.std() > 0:
            fold_sharpes.append(pnl.mean() / pnl.std() * 16)

    return fold_sharpes


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = RIDGE_ALPHA,
) -> tuple:
    """Fit on all training data with the raw (unclipped) target. Returns (model, scaler)."""
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    model = Ridge(alpha=alpha).fit(X_scaled, y)
    return model, scaler


# ─── Prediction ───────────────────────────────────────────────────────────────

def predict_positions(
    model,
    scaler,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate target positions for test sessions."""
    X = features_df[FEATURE_COLS].values
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    # Scale predictions by 1/vol to simulate taking proportional risk
    positions = predictions / features_df["vol"].values

    # We scale to a reasonable magnitude for interpretability
    scale_factor = 100.0 / np.std(positions)
    positions = positions * scale_factor

    return pd.DataFrame({
        "session": features_df["session"].values,
        "target_position": positions,
    })


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Datathon 2026 HRT — Lenny's Solution")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[1/8] Loading data...")
    bars_seen_train   = pd.read_parquet(os.path.join(DATA_DIR, "bars_seen_train.parquet"))
    bars_unseen_train = pd.read_parquet(os.path.join(DATA_DIR, "bars_unseen_train.parquet"))
    headlines_seen_train = pd.read_parquet(os.path.join(DATA_DIR, "headlines_seen_train.parquet"))
    headlines_unseen_train = pd.read_parquet(os.path.join(DATA_DIR, "headlines_unseen_train.parquet"))

    bars_seen_pub = pd.read_parquet(os.path.join(DATA_DIR, "bars_seen_public_test.parquet"))
    headlines_seen_pub = pd.read_parquet(os.path.join(DATA_DIR, "headlines_seen_public_test.parquet"))

    bars_seen_priv = pd.read_parquet(os.path.join(DATA_DIR, "bars_seen_private_test.parquet"))
    headlines_seen_priv = pd.read_parquet(os.path.join(DATA_DIR, "headlines_seen_private_test.parquet"))

    print(f"  Training sessions  : {bars_seen_train.session.nunique()}")
    print(f"  Public test        : {bars_seen_pub.session.nunique()}")
    print(f"  Private test       : {bars_seen_priv.session.nunique()}")
    print(f"  LLM annotations    : {len(LLM_SENTIMENTS):,} headlines")

    # ── Data augmentation: shifted split ──────────────────────────────────────
    print("\n[2/8] Augmenting training data (shifted split at bar 74)...")
    aug_bs, aug_bu, aug_hdl, session_offset = create_augmented_data(
        bars_seen_train, bars_unseen_train,
        headlines_seen_train, headlines_unseen_train, split_bar=74,
    )
    combined_bars_seen = pd.concat([bars_seen_train, aug_bs], ignore_index=True)
    combined_bars_unseen = pd.concat([bars_unseen_train, aug_bu], ignore_index=True)
    combined_headlines = pd.concat([headlines_seen_train, aug_hdl], ignore_index=True)
    print(f"  Augmented sessions : {aug_bs.session.nunique()}")
    print(f"  Combined sessions  : {combined_bars_seen.session.nunique()}")
    print(f"  Combined headlines : {len(combined_headlines):,}")

    # ── Optimize sentiment decay rates (single-seed) ────────────────────
    print("\n[3/8] Optimizing sentiment decay parameters...")
    decay_pos, decay_neg = _optimize_decay_params(
        combined_bars_seen, combined_bars_unseen, combined_headlines
    )
    print(f"  Final decay_pos={decay_pos:.5f}  decay_neg={decay_neg:.5f}")

    # ── Feature engineering ────────────────────────────────────────────────────
    print("\n[4/8] Engineering features...")
    train_feat = extract_features(combined_bars_seen, combined_headlines,
                                  decay_pos=decay_pos, decay_neg=decay_neg)
    print(f"  Features: {FEATURE_COLS}")
    print(f"  Training samples: {len(train_feat)}")

    # Compute training target: second-half return scaled by vol
    close_end_map = combined_bars_unseen.groupby("session")["close"].last()
    train_feat = train_feat.set_index("session")
    raw_return = close_end_map.reindex(train_feat.index) / train_feat["halfway_close"] - 1
    y_train = (raw_return / train_feat["vol"]).values

    # Group mapping for CV: augmented sessions share group with their original
    groups_all = np.array([int(s) if s < session_offset else int(s) - session_offset
                           for s in train_feat.index])
    train_feat = train_feat.reset_index()

    idx_keep = ~np.isnan(y_train)
    X_train = train_feat[FEATURE_COLS].values[idx_keep]
    y_train = y_train[idx_keep]
    groups_train = groups_all[idx_keep]

    print(f"  Training samples after alignment: {len(X_train)}")

    # ── Alpha grid search ──────────────────────────────────────────────────────
    print("\n[5/8] Finding best Ridge alpha...")
    best_alpha = _find_best_alpha(X_train, y_train, groups=groups_train)
    print(f"  Best alpha: {best_alpha}")

    # ── Cross-validation report ────────────────────────────────────────────────
    print("\n[6/8] Cross-validating with best alpha...")
    sharpes = cross_validate(X_train, y_train, alpha=best_alpha, groups=groups_train)
    mean_sharpe = np.mean(sharpes)
    print(f"  GroupKFold sharpes = {[f'{s:.2f}' for s in sharpes]}, mean = {mean_sharpe:.4f}")

    # ── Train final model ──────────────────────────────────────────────────────
    print("\n[7/8] Training final model on all data...")
    model, scaler = train_final_model(X_train, y_train, alpha=best_alpha)

    print("  Coefficients (standardized):")
    for feat, coef in zip(FEATURE_COLS, model.coef_):
        print(f"    {feat:28s}: {coef:+.6f}")
    print(f"    {'intercept':28s}: {model.intercept_:+.6f}")

    # ── Generate predictions ───────────────────────────────────────────────────
    print("\n[8/8] Generating predictions...")
    pub_feat  = extract_features(bars_seen_pub,  headlines_seen_pub,
                                 decay_pos=decay_pos, decay_neg=decay_neg)
    priv_feat = extract_features(bars_seen_priv, headlines_seen_priv,
                                 decay_pos=decay_pos, decay_neg=decay_neg)

    pub_pos  = predict_positions(model, scaler, pub_feat)
    priv_pos = predict_positions(model, scaler, priv_feat)

    pub_path  = os.path.join(OUTPUT_DIR, "submission_public.csv")
    priv_path = os.path.join(OUTPUT_DIR, "submission_private.csv")
    pub_pos.to_csv(pub_path,   index=False)
    priv_pos.to_csv(priv_path, index=False)

    print(f"\n  Public  submission: {pub_path}")
    print(f"  Private submission: {priv_path}")
    print(f"  Public  — {len(pub_pos):,} sessions, "
          f"pos range [{pub_pos.target_position.min():.2f}, {pub_pos.target_position.max():.2f}], "
          f"long={( pub_pos.target_position > 0).sum():,}, "
          f"short={(pub_pos.target_position  < 0).sum():,}")
    print(f"  Private — {len(priv_pos):,} sessions, "
          f"pos range [{priv_pos.target_position.min():.2f}, {priv_pos.target_position.max():.2f}], "
          f"long={( priv_pos.target_position > 0).sum():,}, "
          f"short={(priv_pos.target_position  < 0).sum():,}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()