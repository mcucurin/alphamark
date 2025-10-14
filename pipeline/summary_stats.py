# =============================
# summary_stats.py (includes dcor helpers)
# =============================
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr, linregress

# dcor (distance correlation) helper lives here now
try:
    import dcor as _dcor
    def _distance_correlation(x, y):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        n = min(x.size, y.size)
        if n < 3:
            return np.nan
        if np.all(x == x[0]) or np.all(y == y[0]):
            return np.nan
        return float(_dcor.distance_correlation(x, y))
except Exception:
    def _distance_correlation(x, y):
        return np.nan

# Reuse structure helper from daily module
from .daily_stats import create_5d_stats

# ===================== SUMMARY over multiple days (nested) ====================
def compute_summary_stats_over_days(
    df,
    date_col,
    signal_cols,
    target_cols,
    quantiles=[1.0, 0.75, 0.5, 0.25],
    bet_size_cols=['betsize_equal'],
    type_quantile='cumulative',   # 'cumulative' (>=thr) or 'quantEach' (exact bucket)
    add_spearman=True,
    add_dcor=False
):
    """
    Returns ONE summary value per (signal, qrank, target, bet).

    Invariants:
      - Notional uses sum|b|; PnL uses |b| as magnitude.
      - Daily PPD for Sharpe is (pnl_day / notional_day) with notional_day >= 0.

    Output access pattern:
      out[stat_type][signal][qrank][target][bet] = value
    """
    import pandas as pd
    out = create_5d_stats()

    need = list(set([date_col] + signal_cols + target_cols + bet_size_cols))
    df = df[need].copy().replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[date_col])
    if df.empty:
        return out

    # Accumulators
    daily_ppd = defaultdict(list)   # key=(signal, qlabel, target, bet) -> list of daily PPD
    pooled_pairs = defaultdict(lambda: {'s': [], 'y': []})  # for r2/t, spearman, dcor
    hit_num = defaultdict(int)
    hit_den = defaultdict(int)
    long_num = defaultdict(int)     # key=(signal, qlabel, bet)
    long_den = defaultdict(int)

    # Per-day loop
    for day, d in df.sort_values(date_col).groupby(date_col, sort=True):
        d = d.dropna(how='any')
        if d.empty:
            continue

        for signal in signal_cols:
            if signal not in d.columns:
                continue

            s_all = d[signal].to_numpy(float)
            sgn_all = np.sign(s_all)
            sabs_all = np.abs(s_all)

            # thresholds/buckets for this day and signal
            day_thresholds = {}
            for q in quantiles:
                day_thresholds[f'qr_{int(q*100)}'] = np.nanquantile(sabs_all, 1.0 - q)

            if type_quantile == 'quantEach':
                K = len(quantiles)
                probs = [i / K for i in range(K + 1)]
                edges = np.nanquantile(sabs_all, probs)

            for q in quantiles:
                qlabel = f'qr_{int(q*100)}'

                if type_quantile == 'cumulative':
                    mask = (sabs_all >= day_thresholds[qlabel])
                else:  # 'quantEach'
                    j = quantiles.index(q) + 1  # 1..K
                    lo = edges[j-1]
                    hi = edges[j]
                    mask = (sabs_all >= lo) & (sabs_all <= hi)

                if not np.any(mask):
                    continue

                s = s_all[mask]
                sgn = sgn_all[mask]
                n_names = int(s.size)

                for bet in bet_size_cols:
                    if bet not in d.columns:
                        continue
                    b = d.loc[mask, bet].to_numpy(float)
                    b_abs = np.abs(b)

                    # long ratio counts per (signal, qlabel, bet)
                    long_num[(signal, qlabel, bet)] += int(np.sum(sgn > 0))
                    long_den[(signal, qlabel, bet)] += n_names

                    for target in target_cols:
                        if target not in d.columns:
                            continue
                        y = d.loc[mask, target].to_numpy(float)

                        # daily PPD for Sharpe (one per day)
                        pnl_day = float(np.nansum(np.sign(s) * y * b_abs))
                        notional_day = float(np.nansum(b_abs))
                        key_full = (signal, qlabel, target, bet)
                        if notional_day > 0:
                            daily_ppd[key_full].append(pnl_day / notional_day)

                        # pooled pairs for r2/t, spearman, dcor
                        m = np.isfinite(s) & np.isfinite(y)
                        if m.any():
                            pooled_pairs[key_full]['s'].append(s[m])
                            pooled_pairs[key_full]['y'].append(y[m])

                        # hit ratio counts (exclude y==0)
                        y_sign = np.sign(y)
                        denom_mask = (y_sign != 0.0)
                        if denom_mask.any():
                            hit_num[key_full] += int(np.sum(np.sign(s)[denom_mask] == y_sign[denom_mask]))
                            hit_den[key_full] += int(np.sum(denom_mask))

    # Finalize one value per combo
    sqrt_252 = np.sqrt(252.0)

    # Sharpe on daily PPD
    for key_full, series in daily_ppd.items():
        arr = np.array(series, float)
        arr = arr[np.isfinite(arr)]
        if arr.size >= 2:
            mu = float(arr.mean())
            sd = float(arr.std(ddof=0))
            sharpe = mu / sd * sqrt_252 if sd > 0 else np.nan
        else:
            sharpe = np.nan
        signal, qlabel, target, bet = key_full
        out['sharpe'][signal][qlabel][target][bet] = float(sharpe) if np.isfinite(sharpe) else np.nan

    # r2, t_stat, spearman, dcor
    for key_full, bits in pooled_pairs.items():
        x = np.concatenate(bits['s']) if bits['s'] else np.array([])
        y = np.concatenate(bits['y']) if bits['y'] else np.array([])
        if x.size >= 3 and x.size == y.size:
            try:
                slope, intercept, r_val, p_val, stderr = linregress(x, y)
                r2 = float(r_val**2) if np.isfinite(r_val) else np.nan
                t_stat = float(slope / stderr) if (stderr is not None and stderr > 0) else np.nan
            except Exception:
                r2 = np.nan
                t_stat = np.nan
            if add_spearman:
                try:
                    sp = float(spearmanr(x, y, nan_policy='omit').correlation)
                except Exception:
                    sp = np.nan
                signal, qlabel, target, bet = key_full
                out['spearman'][signal][qlabel][target][bet] = sp if np.isfinite(sp) else np.nan
            if add_dcor:
                try:
                    dc = float(_distance_correlation(x, y))
                except Exception:
                    dc = np.nan
                signal, qlabel, target, bet = key_full
                out['dcor'][signal][qlabel][target][bet] = dc if np.isfinite(dc) else np.nan
        else:
            r2 = np.nan
            t_stat = np.nan

        signal, qlabel, target, bet = key_full
        out['r2'][signal][qlabel][target][bet] = r2 if np.isfinite(r2) else np.nan
        out['t_stat'][signal][qlabel][target][bet] = t_stat if np.isfinite(t_stat) else np.nan

    # hit ratio
    for key_full, hn in hit_num.items():
        hd = hit_den[key_full]
        signal, qlabel, target, bet = key_full
        out['hit_ratio'][signal][qlabel][target][bet] = (hn / hd) if hd > 0 else np.nan

    # finalize long_ratio (accumulated per (signal, qlabel, bet), broadcast to targets)
    for (signal, qlabel, bet), ln in long_num.items():
        ld = long_den[(signal, qlabel, bet)]
        val = (ln / ld) if ld > 0 else np.nan
        for target in target_cols:
            out['long_ratio'][signal][qlabel][target][bet] = val

    return out


__all__ = [
    'compute_summary_stats_over_days',
    '_distance_correlation',
]
