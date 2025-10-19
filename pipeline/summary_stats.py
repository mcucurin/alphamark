# =============================
# summary_stats.py (fast + parallel + robust columns)
# =============================
from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Sequence, Dict, Tuple, List
from scipy.stats import spearmanr, linregress

# Optional distance correlation
try:
    import dcor as _dcor
    def _distance_correlation(x, y):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        n = min(x.size, y.size)
        if n < 3 or np.all(x == x[0]) or np.all(y == y[0]):
            return np.nan
        return float(_dcor.distance_correlation(x, y))
except Exception:
    def _distance_correlation(x, y):
        return np.nan

# Joblib for parallelism across days
try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

# Reuse nested store
from .daily_stats import create_5d_stats


# ----------------------------
# Helpers
# ----------------------------
def _qlabel(q: float) -> str:
    return f"qr_{int(round(q*100))}"


def _float_clean(arr: np.ndarray) -> np.ndarray:
    """Return float64 with non-finites -> NaN."""
    out = np.asarray(arr, dtype="float64")
    out[~np.isfinite(out)] = np.nan
    return out


def _sanitize_list(cols: Sequence) -> List[str]:
    """Drop Ellipsis/None/non-strings; keep order & dedupe."""
    seen = set()
    out: List[str] = []
    for c in cols or []:
        if c is None or c is Ellipsis:
            continue
        if isinstance(c, str) and c not in seen:
            out.append(c); seen.add(c)
    return out


def _merge_daily_accumulators(
    dest_daily_ppd, dest_pairs, dest_hit_num, dest_hit_den, dest_long_num, dest_long_den, src
):
    (daily_ppd, pooled_pairs, hit_num, hit_den, long_num, long_den) = src
    for k, v in daily_ppd.items():
        dest_daily_ppd[k].extend(v)
    for k, d in pooled_pairs.items():
        if d['s']:
            dest_pairs[k]['s'].extend(d['s'])
            dest_pairs[k]['y'].extend(d['y'])
    for k, c in hit_num.items():
        dest_hit_num[k] += c
    for k, c in hit_den.items():
        dest_hit_den[k] += c
    for k, c in long_num.items():
        dest_long_num[k] += c
    for k, c in long_den.items():
        dest_long_den[k] += c


def _summarize_one_day(
    day_df,
    signal_cols: Sequence[str],
    target_cols: Sequence[str],
    quantiles: Sequence[float],
    bet_size_cols: Sequence[str],
    type_quantile: str,
    add_spearman: bool,
    add_dcor: bool,
):
    """
    Compute per-day accumulators (not the final summary), using NumPy ops.
    Returns:
      (daily_ppd, pooled_pairs, hit_num, hit_den, long_num, long_den)
    """
    # Clean numeric columns to float64 once
    d = day_df.copy()
    for c in signal_cols:
        if c in d.columns: d[c] = _float_clean(d[c].to_numpy())
    for c in target_cols:
        if c in d.columns: d[c] = _float_clean(d[c].to_numpy())
    for c in bet_size_cols:
        if c in d.columns: d[c] = _float_clean(d[c].to_numpy())

    # Dict accumulators
    daily_ppd = defaultdict(list)   # key=(signal, qlabel, target, bet) -> [ppd_day,...]
    pooled_pairs = defaultdict(lambda: {'s': [], 'y': []})
    hit_num = defaultdict(int)
    hit_den = defaultdict(int)
    long_num = defaultdict(int)     # key=(signal, qlabel, bet)
    long_den = defaultdict(int)

    n = len(d)
    if n == 0:
        return (daily_ppd, pooled_pairs, hit_num, hit_den, long_num, long_den)

    # Pre-build absolute bet arrays
    bet_abs = {}
    for b in bet_size_cols:
        if b in d.columns:
            bet_abs[b] = np.abs(d[b].to_numpy(dtype='float64', copy=False))
        else:
            bet_abs[b] = np.full(n, np.nan, dtype='float64')

    # Main vectorized loops
    for signal in signal_cols:
        if signal not in d.columns:
            continue

        s_all = d[signal].to_numpy(dtype='float64', copy=False)
        m_fin = np.isfinite(s_all)
        if not m_fin.any():
            continue

        sgn_all  = np.sign(s_all)
        sabs_all = np.abs(s_all[m_fin])
        if sabs_all.size == 0:
            continue

        if type_quantile == 'cumulative':
            thr_map = { _qlabel(q): float(np.nanquantile(sabs_all, 1.0 - q)) for q in quantiles }
            edges = None
        else:
            K = len(quantiles)
            probs = np.linspace(0.0, 1.0, K + 1)
            edges = np.nanquantile(sabs_all, probs)
            thr_map = None

        for q in quantiles:
            qlbl = _qlabel(q)
            if type_quantile == 'cumulative':
                thr = thr_map[qlbl]
                mask_q = m_fin & (np.abs(s_all) >= thr)
            else:
                j = quantiles.index(q) + 1
                lo, hi = edges[j-1], edges[j]
                mask_q = m_fin & (np.abs(s_all) >= lo) & (np.abs(s_all) <= hi)

            if not mask_q.any():
                continue

            s_q   = s_all[mask_q]
            sgn_q = sgn_all[mask_q]
            n_names = int(s_q.size)

            for bet in bet_size_cols:
                b = bet_abs[bet]
                b_q = b[mask_q]
                b_q_fin = np.isfinite(b_q)
                if not b_q_fin.any():
                    continue

                # long ratio counts per (signal, qlabel, bet)
                long_num[(signal, qlbl, bet)] += int(np.sum(sgn_q > 0))
                long_den[(signal, qlbl, bet)] += n_names

                for target in target_cols:
                    if target not in d.columns:
                        continue
                    y = d[target].to_numpy(dtype='float64', copy=False)
                    y_q = y[mask_q]
                    m = np.isfinite(y_q) & b_q_fin
                    if not m.any():
                        continue

                    # Daily PPD for Sharpe (one number per day/key)
                    pnl_day = float(np.nansum(sgn_q[m] * y_q[m] * b_q[m]))
                    notional_day = float(np.nansum(b_q[m]))
                    if notional_day > 0.0:
                        daily_ppd[(signal, qlbl, target, bet)].append(pnl_day / notional_day)

                    # pooled pairs for r2/t, spearman, dcor
                    s_use = s_q[m]
                    y_use = y_q[m]
                    if s_use.size > 0:
                        pooled_pairs[(signal, qlbl, target, bet)]['s'].append(s_use)
                        pooled_pairs[(signal, qlbl, target, bet)]['y'].append(y_use)

                    # Hit ratio (exclude y==0)
                    y_sign = np.sign(y_q[m])
                    denom_mask = (y_sign != 0.0)
                    if denom_mask.any():
                        hit_num[(signal, qlbl, target, bet)] += int(np.sum(np.sign(s_use)[denom_mask] == y_sign[denom_mask]))
                        hit_den[(signal, qlbl, target, bet)] += int(np.sum(denom_mask))

    return (daily_ppd, pooled_pairs, hit_num, hit_den, long_num, long_den)


# ===================== SUMMARY over multiple days (parallel) ====================
def compute_summary_stats_over_days(
    df,
    date_col: str,
    signal_cols: Sequence[str],
    target_cols: Sequence[str],
    quantiles: Sequence[float] = (1.0, 0.75, 0.5, 0.25),
    bet_size_cols: Sequence[str] = ('betsize_equal',),
    type_quantile: str = 'cumulative',   # 'cumulative' (>=thr) or 'quantEach' (exact bucket)
    add_spearman: bool = True,
    add_dcor: bool = False,
    n_jobs: int | None = None,           # None/0 -> serial; -1 -> all cores
    backend: str = "loky",               # 'loky' (proc), 'threading' if I/O bound
):
    """
    Returns ONE summary value per (signal, qrank, target, bet).
    Output access pattern:
      out[stat_type][signal][qrank][target][bet] = value
    """
    import pandas as pd

    out = create_5d_stats()

    # Sanitize lists
    signal_cols  = _sanitize_list(signal_cols)
    target_cols  = _sanitize_list(target_cols)
    bet_size_cols = _sanitize_list(bet_size_cols)

    # Validate date_col & resolve present columns
    if date_col not in df.columns:
        raise KeyError(f"[summary_stats] date_col '{date_col}' not found in DataFrame.")

    want = [date_col] + signal_cols + target_cols + bet_size_cols
    present = [c for c in want if c in df.columns]
    missing = [c for c in want if c not in df.columns]
    if missing:
        print(f"[WARN][summary_stats] Ignoring missing columns: {missing}")

    df = df[present].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if df.empty:
        return out

    # Pre-clean numeric columns to float64 once for all days
    numeric_cols = [c for c in present if c != date_col]
    for c in numeric_cols:
        df[c] = _float_clean(df[c].to_numpy())

    grouped = list(df.sort_values(date_col).groupby(date_col, sort=True))

    # Accumulators to be merged from per-day pieces
    daily_ppd   = defaultdict(list)   # (signal, qlabel, target, bet) -> [ppd_day,...]
    pooled_pairs = defaultdict(lambda: {'s': [], 'y': []})
    hit_num     = defaultdict(int)
    hit_den     = defaultdict(int)
    long_num    = defaultdict(int)     # (signal, qlabel, bet)
    long_den    = defaultdict(int)

    # Run per-day workers (parallel if requested and joblib available)
    if _HAS_JOBLIB and (n_jobs is not None) and (int(n_jobs) != 0):
        results = Parallel(n_jobs=n_jobs, backend=backend, prefer="processes")(
            delayed(_summarize_one_day)(
                day_df,
                signal_cols,
                target_cols,
                quantiles,
                bet_size_cols,
                type_quantile,
                add_spearman,
                add_dcor,
            )
            for _, day_df in grouped
        )
        for part in results:
            _merge_daily_accumulators(daily_ppd, pooled_pairs, hit_num, hit_den, long_num, long_den, part)
    else:
        for _, day_df in grouped:
            part = _summarize_one_day(
                day_df,
                signal_cols,
                target_cols,
                quantiles,
                bet_size_cols,
                type_quantile,
                add_spearman,
                add_dcor,
            )
            _merge_daily_accumulators(daily_ppd, pooled_pairs, hit_num, hit_den, long_num, long_den, part)

    # ---------- Finalize one value per combo ----------
    sqrt_252 = np.sqrt(252.0)

    # Sharpe from daily PPD
    for key_full, series in daily_ppd.items():
        arr = np.asarray(series, dtype='float64')
        arr = arr[np.isfinite(arr)]
        if arr.size >= 2:
            mu = float(arr.mean())
            sd = float(arr.std(ddof=0))
            sharpe = (mu / sd * sqrt_252) if sd > 0 else np.nan
        else:
            sharpe = np.nan
        s, ql, t, b = key_full
        out['sharpe'][s][ql][t][b] = float(sharpe) if np.isfinite(sharpe) else np.nan

    # r2, t_stat, spearman, dcor (pooled over all days)
    for key_full, bits in pooled_pairs.items():
        x = np.concatenate(bits['s']) if bits['s'] else np.array([], dtype='float64')
        y = np.concatenate(bits['y']) if bits['y'] else np.array([], dtype='float64')
        if x.size >= 3 and x.size == y.size:
            try:
                slope, intercept, r_val, p_val, stderr = linregress(x, y)
                r2 = float(r_val**2) if np.isfinite(r_val) else np.nan
                t_stat = float(slope / stderr) if (stderr is not None and stderr > 0) else np.nan
            except Exception:
                r2 = np.nan
                t_stat = np.nan
            # Spearman
            if add_spearman:
                try:
                    sp = float(spearmanr(x, y, nan_policy='omit').correlation)
                except Exception:
                    sp = np.nan
                s, ql, t, b = key_full
                out['spearman'][s][ql][t][b] = sp if np.isfinite(sp) else np.nan
            # dCor
            if add_dcor:
                try:
                    dc = float(_distance_correlation(x, y))
                except Exception:
                    dc = np.nan
                s, ql, t, b = key_full
                out['dcor'][s][ql][t][b] = dc if np.isfinite(dc) else np.nan
        else:
            r2 = np.nan
            t_stat = np.nan

        s, ql, t, b = key_full
        out['r2'][s][ql][t][b] = r2 if np.isfinite(r2) else np.nan
        out['t_stat'][s][ql][t][b] = t_stat if np.isfinite(t_stat) else np.nan

    # hit ratio
    for key_full, hn in hit_num.items():
        hd = hit_den[key_full]
        s, ql, t, b = key_full
        out['hit_ratio'][s][ql][t][b] = (hn / hd) if hd > 0 else np.nan

    # long ratio (stored per (signal, q, bet); broadcast to all targets)
    for (s, ql, b), ln in long_num.items():
        ld = long_den[(s, ql, b)]
        val = (ln / ld) if ld > 0 else np.nan
        for t in target_cols:
            out['long_ratio'][s][ql][t][b] = val

    return out


__all__ = [
    'compute_summary_stats_over_days',
    '_distance_correlation',
]
