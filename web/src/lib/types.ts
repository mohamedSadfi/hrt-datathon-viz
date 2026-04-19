export interface Bar {
  b: number;
  o?: number;
  h?: number;
  l?: number;
  c: number;
}

export type LlmSentiment = 'pos' | 'neutral' | 'neg';

export interface HeadlineEntry {
  b: number;
  t: string;
  fb: number;
  llm: LlmSentiment;
  align3: number;
  align5: number;
  final3: number;
  final5: number;
}

export interface SessionData {
  session: number;
  seen_bars: Bar[];
  unseen_bars: Bar[];
  headlines: HeadlineEntry[];
  features: Record<string, number>;
  vol: number;
  halfway_close: number;
  prediction_vol_adj: number;
  actual_vol_adj: number;
  prediction_raw_position: number;
  raw_return: number;
  pnl_session: number;
}

export interface AlphaGridEntry {
  alpha: number;
  cv_sharpe: number;
}

export interface ModelData {
  decay_pos: number;
  decay_neg: number;
  alpha_star: number;
  alpha_grid: AlphaGridEntry[];
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

export type AlignmentQuadrant =
  | 'pos_aligned'
  | 'pos_antialigned'
  | 'neg_aligned'
  | 'neg_antialigned';

export interface AlignmentExample {
  quadrant: AlignmentQuadrant;
  session: number;
  bar_ix: number;
  headline: string;
  fb_score: number;
  llm_sent: LlmSentiment;
  p0: number;
  p3: number;
  p5: number;
  x_range: number;
  y_range: number;
  u3: number;
  v3: number;
  norm3: number;
  u5: number;
  v5: number;
  norm5: number;
  align3: number;
  align5: number;
  final3: number;
  final5: number;
}

export interface FeaturesCorr {
  labels: string[];
  matrix: number[][];
}
