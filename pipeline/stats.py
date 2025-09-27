import numpy as np
from scipy.stats import spearmanr, linregress
from collections import defaultdict
from itertools import product

# ---- optional dcor ----
try:
    import dcor as _dcor
    def _distance_correlation(x, y):
        return _dcor.distance_correlation(x, y)
except Exception:
    def _distance_correlation(x, y):
        return np.nan


def create_5d_stats():
    """
    Access pattern:
      stats[stat_type][signal][qrank][target][bet_size] = value
    """
    return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))


# ========================= DAILY (nested loops, lean) =========================
def compute_daily_stats(
    df,
    signal_cols,
    target_cols,
    quantiles=[1.0, 0.75, 0.5, 0.25],
    bet_size_cols=['betsize_equal']
):
    """
    Per-day cross-section stats ONLY (no correlations / Sharpe here):
      - pnl (sum over names in bucket)
      - ppd  (pnl / notional)
      - sizeNotional (sum |sig|*bet)
      - nrInstr (names in bucket)
      - bet_size (mean bet for names in bucket)

    Quantile semantics (cumulative): for q in {1.0,0.75,0.5,0.25}
      mask = |sig| >= quantile_{day}(|sig|, 1 - q)
      -> qr_100 is the whole universe; qr_25 is the top 25% by |sig|.
    """
    stats = create_5d_stats()

    needed = list(set(signal_cols + target_cols + bet_size_cols))
    df = df[needed].replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if df.empty:
        return stats

    for signal in signal_cols:
        if signal not in df.columns:
            continue

        # thresholds per q (per-day, but here we do it once on the full df passed in)
        sig_abs = df[signal].abs().to_numpy(float)
        thresholds = {}
        for q in quantiles:
            thr = np.nanquantile(sig_abs, 1.0 - q)
            thresholds[f'qr_{int(q*100)}'] = thr

        for target, bet in product(target_cols, bet_size_cols):
            if target not in df.columns or bet not in df.columns:
                continue

            for qlabel, thr in thresholds.items():
                mask = (df[signal].abs().to_numpy(float) >= thr)
                if not np.any(mask):
                    continue

                sub = df.loc[mask, [signal, target, bet]].dropna(how="any")
                if sub.empty:
                    continue

                s = sub[signal].to_numpy(float)
                y = sub[target].to_numpy(float)
                b = sub[bet].to_numpy(float)

                pnl_vec = np.sign(s) * y * b
                pnl = float(pnl_vec.sum())
                notional = float(np.abs(s * b).sum())

                stats['pnl'][signal][qlabel][target][bet] = pnl
                stats['ppd'][signal][qlabel][target][bet] = (pnl / notional) if notional > 0 else np.nan
                stats['sizeNotional'][signal][qlabel][target][bet] = notional
                stats['nrInstr'][signal][qlabel][target][bet] = int(s.size)
                stats['bet_size'][signal][qlabel][target][bet] = float(b.mean())

    return stats


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
    Returns ONE summary value per (signal, qrank, target, bet):
      - sharpe     : time-series Sharpe of DAILY PPD (not rolling)
      - hit_ratio  : pooled fraction sign(sig)==sign(target), EXCLUDING target==0
      - long_ratio : pooled fraction sign(sig)>0 among traded names (zeros excluded)
      - r2, t_stat : from pooled linear regression y ~ s (scipy linregress over all pooled pairs)
      - spearman   : optional pooled Spearman(s, y)
      - dcor       : optional pooled distance correlation(s, y)

    Implementation: plain nested loops (proof-of-concept, easy to vectorize later).
    """
    out = create_5d_stats()

    need = list(set([date_col] + signal_cols + target_cols + bet_size_cols))
    df = df[need].copy().replace([np.inf, -np.inf], np.nan)
    df[date_col] = np.asarray(df[date_col], dtype='datetime64[ns]')
    df = df.dropna(subset=[date_col])
    if df.empty:
        return out

    # Accumulators
    daily_ppd = defaultdict(list)   # key=(signal, qlabel, target, bet) -> list of daily PPD
    pooled_pairs = defaultdict(lambda: {'s': [], 'y': []})  # for r2/t, spearman, dcor
    hit_num = defaultdict(int)
    hit_den = defaultdict(int)
    long_num = defaultdict(int)     # key=(signal, qlabel, bet) (independent of target)
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

            # Build thresholds/buckets for this day and signal
            # cumulative: thr_q = quantile(|s|, 1-q)
            day_thresholds = {}
            for q in quantiles:
                day_thresholds[f'qr_{int(q*100)}'] = np.nanquantile(sabs_all, 1.0 - q)

            # For quantEach, precompute bin edges (equal-frequency) once per day/signal
            if type_quantile == 'quantEach':
                # K = len(quantiles); edges from 0..1 equally spaced on |s|
                K = len(quantiles)
                probs = [i / K for i in range(K + 1)]
                edges = np.nanquantile(sabs_all, probs)
                # bucket j (1..K): [edges[j-1], edges[j]]; we'll match labels in same order as quantiles list
                # map j -> qlabel like qr_{int(q*100)} in the given list order
                qlabels_ordered = [f'qr_{int(q*100)}' for q in quantiles]  # assumes order like [1.0,0.75,0.5,0.25]
                # but for quantEach we typically want equal-width buckets; we’ll still name them using that order.

            for q in quantiles:
                qlabel = f'qr_{int(q*100)}'

                if type_quantile == 'cumulative':
                    mask = (sabs_all >= day_thresholds[qlabel])
                else:  # 'quantEach'
                    # find which j corresponds to this q in the provided order
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

                    for target in target_cols:
                        if target not in d.columns:
                            continue
                        y = d.loc[mask, target].to_numpy(float)

                        # --- daily PPD for Sharpe (one per day) ---
                        pnl_day = float(np.nansum(np.sign(s) * y * b))
                        notional_day = float(np.nansum(np.abs(s * b)))
                        key_full = (signal, qlabel, target, bet)
                        if notional_day > 0:
                            daily_ppd[key_full].append(pnl_day / notional_day)

                        # --- pooled pairs for r2/t, spearman, dcor ---
                        # pairwise finite filter
                        m = np.isfinite(s) & np.isfinite(y)
                        if m.any():
                            pooled_pairs[key_full]['s'].append(s[m])
                            pooled_pairs[key_full]['y'].append(y[m])

                        # --- hit ratio counts (exclude y==0) ---
                        y_sign = np.sign(y)
                        denom_mask = (y_sign != 0.0)
                        if denom_mask.any():
                            hit_num[key_full] += int(np.sum(np.sign(s)[denom_mask] == y_sign[denom_mask]))
                            hit_den[key_full] += int(np.sum(denom_mask))

                    # --- long ratio counts (zeros excluded) ---
                    long_num[(signal, qlabel, bet)] += int(np.sum(sgn > 0))
                    long_den[(signal, qlabel, bet)] += n_names

    # Finalize one value per combo
    sqrt_252 = np.sqrt(252.0)

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

    for key_full, bits in pooled_pairs.items():
        x = np.concatenate(bits['s']) if bits['s'] else np.array([])
        y = np.concatenate(bits['y']) if bits['y'] else np.array([])
        if x.size >= 3 and x.size == y.size:
            # r2 / t_stat via linregress
            try:
                slope, intercept, r_val, p_val, stderr = linregress(x, y)
                r2 = float(r_val**2) if np.isfinite(r_val) else np.nan
                t_stat = float(slope / stderr) if (stderr is not None and stderr > 0) else np.nan
            except Exception:
                r2 = np.nan
                t_stat = np.nan
            # spearman / dcor (optional)
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

    for key_full, hn in hit_num.items():
        hd = hit_den[key_full]
        signal, qlabel, target, bet = key_full
        out['hit_ratio'][signal][qlabel][target][bet] = (hn / hd) if hd > 0 else np.nan

    for (signal, qlabel, bet), ln in long_num.items():
        ld = long_den[(signal, qlabel, bet)]
        for target in target_cols:
            out['long_ratio'][signal][qlabel][target][bet] = (ln / ld) if ld > 0 else np.nan

    return out
