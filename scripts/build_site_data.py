"""
Build site data: one-shot pipeline that runs the frozen winning model
end-to-end on the labeled training set and dumps four JSON artifacts
into web/public/data/ for the future React frontend to consume.

Outputs:
  - model.json              ~10 KB   — decays, alpha sweep, CV folds, coefs
  - sessions.json           ~3-6 MB  — per-session bars + headlines + predictions
  - alignment_examples.json ~20 KB   — 12 curated headlines with full geometry
  - features_corr.json      ~2 KB    — 11x11 Pearson correlation (features + target)

Run: uv run python scripts/build_site_data.py  (~30-60 s)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import config, solution

OUT_DIR = config.OUTPUT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Alignment helper (re-derives the unclipped geometry that solution.compute_alignments hides) ──

def per_headline_alignment(
    bars_seen: pd.DataFrame,
    headlines: pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per headline with full alignment geometry.

    Mirrors solution.compute_alignments line-for-line, but additionally
    surfaces p0/p3/p5/u/v/norm/unclipped-align so the frontend can
    visualize anti-aligned (clipped-to-zero) cases.

    Returned columns (one row per headline, indexed identically to `headlines`):
      fb_score, p0, p3, p5, x_range, y_range,
      u3, v3, norm3, align3 (unclipped), final3 (clipped),
      u5, v5, norm5, align5 (unclipped), final5 (clipped)
    """
    out = pd.DataFrame(
        {
            "fb_score": 0.0,
            "p0": np.nan, "p3": np.nan, "p5": np.nan,
            "x_range": np.nan, "y_range": np.nan,
            "u3": np.nan, "v3": np.nan, "norm3": np.nan,
            "align3": 0.0, "final3": 0.0,
            "u5": np.nan, "v5": np.nan, "norm5": np.nan,
            "align5": 0.0, "final5": 0.0,
        },
        index=headlines.index,
    )

    for sid, h_idx in headlines.groupby("session").groups.items():
        b_group = bars_seen[bars_seen["session"] == sid].sort_values("bar_ix")
        if len(b_group) == 0:
            continue
        bars = b_group["bar_ix"].to_numpy()
        closes = b_group["close"].to_numpy()
        y_range = max(b_group["high"].max() - b_group["low"].min(), 1e-8)
        x_range = bars[-1] - bars[0] + 10

        for ix in h_idx:
            headline_text = headlines.at[ix, "headline"]
            score = solution.FINBERT_SCORES.get(headline_text, 0.0)
            if score == 0.0:
                continue
            b_ix = headlines.at[ix, "bar_ix"]
            valid = np.where(bars <= b_ix)[0]
            if len(valid) == 0:
                continue
            idx = valid[-1]
            p0 = float(closes[idx])

            idx3 = min(idx + 3, len(closes) - 1)
            idx5 = min(idx + 5, len(closes) - 1)
            p3 = float(closes[idx3])
            p5 = float(closes[idx5])

            if idx3 > idx:
                u3 = (bars[idx3] - bars[idx]) / x_range
                v3 = (p3 - p0) / y_range
                norm3 = float(np.sqrt(u3**2 + v3**2 + 1e-8))
                align3 = float(score * (v3 / norm3))
            else:
                u3 = v3 = norm3 = 0.0
                align3 = float(abs(score))

            if idx5 > idx:
                u5 = (bars[idx5] - bars[idx]) / x_range
                v5 = (p5 - p0) / y_range
                norm5 = float(np.sqrt(u5**2 + v5**2 + 1e-8))
                align5 = float(score * (v5 / norm5))
            else:
                u5 = v5 = norm5 = 0.0
                align5 = float(abs(score))

            out.loc[ix, "fb_score"] = float(score)
            out.loc[ix, "p0"] = p0
            out.loc[ix, "p3"] = p3
            out.loc[ix, "p5"] = p5
            out.loc[ix, "x_range"] = float(x_range)
            out.loc[ix, "y_range"] = float(y_range)
            out.loc[ix, "u3"] = float(u3)
            out.loc[ix, "v3"] = float(v3)
            out.loc[ix, "norm3"] = norm3
            out.loc[ix, "align3"] = align3
            out.loc[ix, "final3"] = float(score * max(0.0, align3))
            out.loc[ix, "u5"] = float(u5)
            out.loc[ix, "v5"] = float(v5)
            out.loc[ix, "norm5"] = norm5
            out.loc[ix, "align5"] = align5
            out.loc[ix, "final5"] = float(score * max(0.0, align5))

    return out


def round_floats(obj, sig: int = 6):
    """Recursively round floats for compact JSON. Skips NaN/Inf -> None."""
    if isinstance(obj, dict):
        return {k: round_floats(v, sig) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, sig) for v in obj]
    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        if obj == 0.0:
            return 0.0
        from math import floor, log10
        digits = sig - int(floor(log10(abs(obj)))) - 1
        return round(obj, max(0, digits))
    if isinstance(obj, (np.floating,)):
        return round_floats(float(obj), sig)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def write_json(path: Path, payload, indent: int | None = None):
    path.write_text(json.dumps(payload, indent=indent))
    return path.stat().st_size


def main():
    t0 = time.time()
    print(f"OUT_DIR = {OUT_DIR}")

    # ── 1. Load ─────────────────────────────────────────────────────────────
    print("\n[1/8] Loading parquet inputs...")
    bars_seen = pd.read_parquet(config.DATA_DIR / "bars_seen_train.parquet")
    bars_unseen = pd.read_parquet(config.DATA_DIR / "bars_unseen_train.parquet")
    headlines = pd.read_parquet(config.DATA_DIR / "headlines_seen_train.parquet").reset_index(drop=True)
    print(f"  bars_seen   : {len(bars_seen):,} rows, {bars_seen['session'].nunique()} sessions")
    print(f"  bars_unseen : {len(bars_unseen):,} rows, {bars_unseen['session'].nunique()} sessions")
    print(f"  headlines   : {len(headlines):,}")

    # ── 2. Decay optimization ──────────────────────────────────────────────
    print("\n[2/8] Optimizing sentiment decay parameters (~30s)...")
    decay_pos, decay_neg = solution._optimize_decay_params(
        bars_seen, bars_unseen, headlines
    )
    print(f"  decay_pos={decay_pos:.5f}  decay_neg={decay_neg:.5f}")

    # ── 3. Features + target ────────────────────────────────────────────────
    print("\n[3/8] Extracting features...")
    feats = solution.extract_features(
        bars_seen, headlines, decay_pos=decay_pos, decay_neg=decay_neg
    ).set_index("session")
    close_end = bars_unseen.groupby("session")["close"].last()
    close_end = close_end.reindex(feats.index)
    raw_return = (close_end / feats["halfway_close"] - 1).to_numpy()
    y = (raw_return / feats["vol"].to_numpy())
    keep = ~np.isnan(y)
    feats = feats.iloc[keep]
    y = y[keep]
    raw_return = raw_return[keep]
    sessions_arr = feats.index.to_numpy()
    X = feats[solution.FEATURE_COLS].to_numpy()
    print(f"  {len(feats)} sessions after dropping NaN target")

    # ── 4. Alpha sweep (full curve) ────────────────────────────────────────
    print("\n[4/8] Alpha grid search...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    alpha_grid_results = []
    fold_indices = list(kf.split(X))
    for a in solution.ALPHA_GRID:
        sharpes = []
        for tr, vl in fold_indices:
            sc = StandardScaler().fit(X[tr])
            m = Ridge(alpha=a).fit(sc.transform(X[tr]), y[tr])
            pnl = m.predict(sc.transform(X[vl])) * y[vl]
            if pnl.std() > 0:
                sharpes.append(pnl.mean() / pnl.std() * 16)
        alpha_grid_results.append({"alpha": float(a), "cv_sharpe": float(np.mean(sharpes)) if sharpes else 0.0})
    alpha_star = max(alpha_grid_results, key=lambda r: r["cv_sharpe"])["alpha"]
    print(f"  alpha_star = {alpha_star}")

    # ── 5. Per-fold Sharpe at alpha_star ───────────────────────────────────
    print("\n[5/8] Cross-validating at alpha_star...")
    fold_sharpes = solution.cross_validate(X, y, alpha=alpha_star)
    cv_mean = float(np.mean(fold_sharpes))
    print(f"  fold sharpes: {[round(float(s), 3) for s in fold_sharpes]}, mean = {cv_mean:.4f}")

    # ── 6. Final model ──────────────────────────────────────────────────────
    print("\n[6/8] Training final model...")
    model, scaler = solution.train_final_model(X, y, alpha=alpha_star)
    preds_vol_adj = model.predict(scaler.transform(X))      # ŷ (vol-adj)
    positions_raw = preds_vol_adj / feats["vol"].to_numpy()
    pos_std = float(np.std(positions_raw))
    positions = positions_raw * (100.0 / pos_std) if pos_std > 0 else positions_raw
    pnl_session = positions * raw_return                    # raw return, not vol-adj

    # ── 7. Per-headline alignment geometry ─────────────────────────────────
    print("\n[7/8] Computing per-headline alignment geometry...")
    align_df = per_headline_alignment(bars_seen, headlines)

    # ────────────────────────────────────────────────────────────────────────
    # Write model.json
    # ────────────────────────────────────────────────────────────────────────
    model_payload = round_floats({
        "decay_pos": float(decay_pos),
        "decay_neg": float(decay_neg),
        "alpha_star": float(alpha_star),
        "alpha_grid": alpha_grid_results,
        "cv_fold_sharpes": [float(s) for s in fold_sharpes],
        "cv_mean_sharpe": cv_mean,
        "coefs": {c: float(model.coef_[i]) for i, c in enumerate(solution.FEATURE_COLS)},
        "intercept": float(model.intercept_),
        "feature_cols": list(solution.FEATURE_COLS),
        "feature_means": {c: float(scaler.mean_[i]) for i, c in enumerate(solution.FEATURE_COLS)},
        "feature_stds": {c: float(scaler.scale_[i]) for i, c in enumerate(solution.FEATURE_COLS)},
        "lb_score_public": 2.88,
        "lb_score_private": None,
        "n_train": int(len(X)),
    })
    sz_model = write_json(OUT_DIR / "model.json", model_payload, indent=2)

    # ────────────────────────────────────────────────────────────────────────
    # Write sessions.json
    # ────────────────────────────────────────────────────────────────────────
    print("\n[8/8] Building sessions.json (1000 sessions)...")
    seen_by_sess = {sid: g.sort_values("bar_ix") for sid, g in bars_seen.groupby("session")}
    unseen_by_sess = {sid: g.sort_values("bar_ix") for sid, g in bars_unseen.groupby("session")}
    headlines_by_sess = {sid: g for sid, g in headlines.groupby("session")}

    llm_map = solution.LLM_SENTIMENTS
    sess_to_pos = {int(s): i for i, s in enumerate(sessions_arr)}

    sessions_payload = []
    for sid in sessions_arr:
        sid_int = int(sid)
        i = sess_to_pos[sid_int]

        sb = seen_by_sess[sid_int]
        seen_bars_list = [
            {"b": int(r.bar_ix), "o": float(r.open), "h": float(r.high),
             "l": float(r.low), "c": float(r.close)}
            for r in sb.itertuples(index=False)
        ]
        ub = unseen_by_sess.get(sid_int)
        unseen_bars_list = (
            [{"b": int(r.bar_ix), "c": float(r.close)} for r in ub.itertuples(index=False)]
            if ub is not None else []
        )

        hl_list = []
        if sid_int in headlines_by_sess:
            hg = headlines_by_sess[sid_int].sort_values("bar_ix")
            for ix, row in hg.iterrows():
                a = align_df.loc[ix]
                fb = float(a["fb_score"])
                hl_list.append({
                    "b": int(row["bar_ix"]),
                    "t": str(row["headline"]),
                    "fb": fb,
                    "llm": str(llm_map.get(row["headline"], "neutral")),
                    "align3": float(a["align3"]),
                    "align5": float(a["align5"]),
                    "final3": float(a["final3"]),
                    "final5": float(a["final5"]),
                })

        sessions_payload.append({
            "session": sid_int,
            "seen_bars": seen_bars_list,
            "unseen_bars": unseen_bars_list,
            "headlines": hl_list,
            "features": {c: float(feats.iloc[i][c]) for c in solution.FEATURE_COLS},
            "vol": float(feats.iloc[i]["vol"]),
            "halfway_close": float(feats.iloc[i]["halfway_close"]),
            "prediction_vol_adj": float(preds_vol_adj[i]),
            "actual_vol_adj": float(y[i]),
            "prediction_raw_position": float(positions[i]),
            "raw_return": float(raw_return[i]),
            "pnl_session": float(pnl_session[i]),
        })

    sz_sessions = write_json(OUT_DIR / "sessions.json", round_floats(sessions_payload))

    # ────────────────────────────────────────────────────────────────────────
    # Write alignment_examples.json: 12 curated, 3-from-each-quadrant
    # ────────────────────────────────────────────────────────────────────────
    print("\nCurating alignment_examples.json...")
    valid = align_df.dropna(subset=["p0"]).copy()
    valid["abs_fb"] = valid["fb_score"].abs()
    valid = valid.join(headlines[["session", "bar_ix", "headline"]])
    quadrants = {
        "pos_aligned":     (valid["fb_score"] > 0) & (valid["align3"] > 0),
        "pos_antialigned": (valid["fb_score"] > 0) & (valid["align3"] < 0),
        "neg_aligned":     (valid["fb_score"] < 0) & (valid["align3"] < 0),
        "neg_antialigned": (valid["fb_score"] < 0) & (valid["align3"] > 0),
    }
    examples = []
    for label, mask in quadrants.items():
        bucket = valid[mask].sort_values("abs_fb", ascending=False).head(3)
        for ix, r in bucket.iterrows():
            examples.append({
                "quadrant": label,
                "session": int(r["session"]),
                "bar_ix": int(r["bar_ix"]),
                "headline": str(r["headline"]),
                "fb_score": float(r["fb_score"]),
                "llm_sent": str(llm_map.get(r["headline"], "neutral")),
                "p0": float(r["p0"]), "p3": float(r["p3"]), "p5": float(r["p5"]),
                "x_range": float(r["x_range"]), "y_range": float(r["y_range"]),
                "u3": float(r["u3"]), "v3": float(r["v3"]), "norm3": float(r["norm3"]),
                "u5": float(r["u5"]), "v5": float(r["v5"]), "norm5": float(r["norm5"]),
                "align3": float(r["align3"]), "align5": float(r["align5"]),
                "final3": float(r["final3"]), "final5": float(r["final5"]),
            })
    sz_align = write_json(OUT_DIR / "alignment_examples.json", round_floats(examples), indent=2)

    # ────────────────────────────────────────────────────────────────────────
    # Write features_corr.json
    # ────────────────────────────────────────────────────────────────────────
    print("Building features_corr.json...")
    labels = list(solution.FEATURE_COLS) + ["target"]
    corr = np.corrcoef(np.column_stack([X, y]), rowvar=False)
    corr_payload = {
        "labels": labels,
        "matrix": [[float(v) for v in row] for row in corr],
    }
    sz_corr = write_json(OUT_DIR / "features_corr.json", round_floats(corr_payload), indent=2)

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 64)
    print(f"Done in {elapsed:.1f}s")
    print(f"  model.json              {sz_model:>10,} bytes")
    print(f"  sessions.json           {sz_sessions:>10,} bytes")
    print(f"  alignment_examples.json {sz_align:>10,} bytes")
    print(f"  features_corr.json      {sz_corr:>10,} bytes")
    print(f"  cv_mean_sharpe          {cv_mean:.4f}")
    print(f"  alpha_star              {alpha_star}")
    print(f"  decays                  pos={decay_pos:.5f}  neg={decay_neg:.5f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
