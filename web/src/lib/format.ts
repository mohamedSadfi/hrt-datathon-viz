export function fmtNumber(x: number, digits = 4): string {
  if (!Number.isFinite(x)) return '—';
  return x.toFixed(digits);
}

export function fmtSigned(x: number, digits = 4): string {
  if (!Number.isFinite(x)) return '—';
  return (x >= 0 ? '+' : '') + x.toFixed(digits);
}

export function fmtPercent(x: number, digits = 2): string {
  if (!Number.isFinite(x)) return '—';
  return (x * 100).toFixed(digits) + '%';
}

export function fmtSharpe(x: number): string {
  if (!Number.isFinite(x)) return '—';
  return x.toFixed(3);
}

export function fmtSciOrFixed(x: number, threshold = 1e-3): string {
  if (!Number.isFinite(x)) return '—';
  if (x !== 0 && Math.abs(x) < threshold) return x.toExponential(2);
  return x.toFixed(4);
}

export const FEATURE_NAMES: Record<string, string> = {
  ret_all_vol: 'ret_all_vol',
  ret_last5_vol: 'ret_last5_vol',
  ret_last20_vol: 'ret_last20_vol',
  finbert_pos_align3: 'finbert_pos_align3',
  finbert_neg_align3: 'finbert_neg_align3',
  finbert_pos_align5: 'finbert_pos_align5',
  finbert_neg_align5: 'finbert_neg_align5',
  finbert_conf_belief: 'finbert_conf_belief',
  llm_neg_decay: 'llm_neg_decay',
  cand_up_ratio: 'cand_up_ratio',
};
