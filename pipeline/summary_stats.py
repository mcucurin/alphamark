# =============================
# summary_stats.py (vectorized, fast + parallel)
# =============================
from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Sequence, List
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


def _topk_mask_desc(abs_vals: np.ndarray, finite_mask: np.ndarray, q: float) -> np.ndarray:
    """
    Rank-based cumulative selection: pick exactly ceil(q * N) largest |s| among finite entries.
    Deterministic tie handling via stable argsort (mergesort) on the finite slice.
    """
    idx_fin = np.where(finite_mask)[0]
    if idx_fin.size == 0 or q <= 0.0:
        return np.zeros_like(finite_mask, dtype=bool)
    if q >= 1.0:
        out = np.zeros_like(finite_mask, dtype=bool)
        out[idx_fin] = True
        return out
    k = int(np.ceil(q * idx_fin.size))
    order = np.argsort(-abs_vals[idx_fin], kind="mergesort")
    choose = idx_fin[order[:k]]
    out = np.zeros_like(finite_mask, dtype=bool)
    out[choose] = True
    return out


def _merge_daily_accumulators(
    dest_daily_ppd, dest_pairs, dest_hit_num, dest_hit_den, dest_long_num, dest_long_den, src
):
    (daily_ppd, pooled_pairs, hit_num, hit_den, long_num, long_den) = src
    for k, v in daily_ppd.items():
        if v:
            dest_daily_ppd[k].extend(v)
    for k, d in pooled_pairs.items():
        if d['s']:
            dest_pairs[k]['s'].extend(d['s'])
            dest_pairs[k]['y'].extend(d['y'])
    for k, c in hit_num.items():
        if c:
            dest_hit_num[k] += c
    for k, c in hit_den.items():
        if c:
            dest_hit_den[k] += c
    for k, c in long_num.items():
        if c:
            dest_long_num[k] += c
    for k, c in long_den.items():
        if c:
            dest_long_den[k] += c


# -------- Vectorized single-day accumulator --------
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
    Compute per-day accumulators (not the final summary).
    Returns:
      (daily_ppd, pooled_pairs, hit_num, hit_den, long_num, long_den)
    """
    import pandas as pd

    d = day_df.copy()

    # Keep only present columns and build 2D blocks (n x ns/nt/nb)
    sig_names = [c for c in signal_cols if c in d.columns]
    tgt_names = [c for c in target_cols if c in d.columns]
    bet_names = [c for c in bet_size_cols if c in d.columns]

    if len(sig_names) == 0 or len(tgt_names) == 0 or len(bet_names) == 0 or len(d) == 0:
        return (
            defaultdict(list),
            defaultdict(lambda: {'s': [], 'y': []}),
            defaultdict(int),
            defaultdict(int),
            defaultdict(int),
            defaultdict(int),
        )

    S = np.column_stack([_float_clean(d[c].to_numpy()) for c in sig_names])              # (n, ns)
    Y = np.column_stack([_float_clean(d[c].to_numpy()) for c in tgt_names])              # (n, nt)
    B = np.column_stack([np.abs(_float_clean(d[c].to_numpy())) for c in bet_names])      # (n, nb)

    n, ns = S.shape
    _, nt = Y.shape
    _, nb = B.shape

    daily_ppd   = defaultdict(list)                 # key=(signal, qlabel, target, bet) -> [ppd_day,...]
    pooled_pairs = defaultdict(lambda: {'s': [], 'y': []})
    hit_num     = defaultdict(int)
    hit_den     = defaultdict(int)
    long_num    = defaultdict(int)                  # key=(signal, qlabel, bet)
    long_den    = defaultdict(int)

    for si, s_name in enumerate(sig_names):
        s_all = S[:, si]
        m_fin = np.isfinite(s_all)
        if not m_fin.any():
            continue

        sgn = np.sign(s_all)
        sabs_fin = np.abs(s_all[m_fin])

        # Precompute stable edges only if bucket mode; cumulative uses rank-based top-K
        if type_quantile != 'cumulative' and sabs_fin.size:
            K = len(quantiles)
            probs = np.linspace(0.0, 1.0, K + 1)
            edges = np.nanquantile(sabs_fin, probs)
        else:
            edges = None

        for q in quantiles:
            qlbl = _qlabel(q)

            if type_quantile == 'cumulative':
                mask_q = _topk_mask_desc(np.abs(s_all), m_fin, q)
            else:
                j = quantiles.index(q) + 1
                lo, hi = edges[j-1], edges[j]
                mask_q = m_fin & (np.abs(s_all) >= lo) & (np.abs(s_all) <= hi)

            if not mask_q.any():
                continue

            # Slice once for this (signal, quantile)
            s_q   = s_all[mask_q]                          # (m,)
            sgn_q = sgn[mask_q]                            # (m,)
            Y_q   = Y[mask_q, :]                           # (m, nt)
            B_q   = B[mask_q, :]                           # (m, nb)

            # Finite masks and zero-filled views for safe math
            Y_fin = np.isfinite(Y_q)                       # (m, nt)
            B_fin = np.isfinite(B_q)                       # (m, nb)
            Yz    = np.where(Y_fin, Y_q, 0.0)              # (m, nt)
            Bz    = np.where(B_fin, B_q, 0.0)              # (m, nb)

            # --------- Vectorized PPD per (target, bet) ---------
            # pnl[t, b] = sum_r (sgn_q[r] * Y_q[r,t] * B_q[r,b]) over rows with finite Y and B
            # notional[t, b] = sum_r (1_{Y finite} * B_q[r,b])
            pnl_mat = ( (Yz * sgn_q[:, None]).T @ Bz )     # (nt, nb)
            not_mat = ( Y_fin.astype(float).T @ Bz )       # (nt, nb)
            ppd_mat = np.divide(pnl_mat, not_mat, out=np.full_like(pnl_mat, np.nan), where=(not_mat > 0))

            # Save daily PPDs (tiny loops over nt * nb only)
            for ti, t_name in enumerate(tgt_names):
                vals = ppd_mat[ti, :]
                for bi, b_name in enumerate(bet_names):
                    v = vals[bi]
                    if np.isfinite(v):
                        daily_ppd[(s_name, qlbl, t_name, b_name)].append(float(v))

            # --------- Vectorized hit-ratio per (target, bet) ---------
            y_sign = np.sign(Y_q)                          # (m, nt)
            nonzero = (y_sign != 0.0) & Y_fin              # valid denom rows (finite and non-zero)
            # Counts across rows using B_fin as the “pair mask”:
            denom_mat = (nonzero.astype(float).T @ B_fin.astype(float))   # (nt, nb)
            eq_sign = ((np.sign(s_q)[:, None] == y_sign) & nonzero)       # (m, nt)
            numer_mat = (eq_sign.astype(float).T @ B_fin.astype(float))   # (nt, nb)

            for ti, t_name in enumerate(tgt_names):
                dn = denom_mat[ti, :]
                nm = numer_mat[ti, :]
                for bi, b_name in enumerate(bet_names):
                    d = int(dn[bi])
                    if d > 0:
                        hit_den[(s_name, qlbl, t_name, b_name)] += d
                        hit_num[(s_name, qlbl, t_name, b_name)] += int(nm[bi])

            # --------- Vectorized long-ratio (per bet; independent of target) ---------
            long_den_vec = np.sum(B_fin, axis=0).astype(int)                        # (nb,)
            long_num_vec = np.sum((sgn_q > 0)[:, None] & B_fin, axis=0).astype(int) # (nb,)
            for bi, b_name in enumerate(bet_names):
                if long_den_vec[bi] > 0:
                    long_den[(s_name, qlbl, b_name)] += int(long_den_vec[bi])
                    long_num[(s_name, qlbl, b_name)] += int(long_num_vec[bi])

            # --------- Pooled pairs for r2/t, spearman, dcor (compact append) ---------
            # Keep the append tight: only small loops over (target, bet) to build pairs
            for bi in range(nb):
                b_ok = B_fin[:, bi]
                if not b_ok.any():
                    continue
                for ti in range(nt):
                    y_ok = Y_fin[:, ti]
                    m = b_ok & y_ok
                    if not m.any():
                        continue
                    xs = s_q[m].astype('float64', copy=False).ravel()
                    ys = Y_q[m, ti].astype('float64', copy=False).ravel()
                    key = (s_name, qlbl, tgt_names[ti], bet_names[bi])
                    pooled_pairs[key]['s'].append(xs)
                    pooled_pairs[key]['y'].append(ys)

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
    signal_cols   = _sanitize_list(signal_cols)
    target_cols   = _sanitize_list(target_cols)
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
    daily_ppd    = defaultdict(list)   # (signal, qlabel, target, bet) -> [ppd_day,...]
    pooled_pairs = defaultdict(lambda: {'s': [], 'y': []})
    hit_num      = defaultdict(int)
    hit_den      = defaultdict(int)
    long_num     = defaultdict(int)    # (signal, qlabel, bet)
    long_den     = defaultdict(int)

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
        # robust concatenation + alignment
        x = np.concatenate([np.asarray(a, dtype='float64').ravel() for a in bits['s']]) if bits['s'] else np.array([], dtype='float64')
        y = np.concatenate([np.asarray(a, dtype='float64').ravel() for a in bits['y']]) if bits['y'] else np.array([], dtype='float64')
        n = min(x.size, y.size)

        if n >= 3:
            if x.size != n: x = x[:n]
            if y.size != n: y = y[:n]
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
            else:
                sp = np.nan

            if add_dcor:
                try:
                    dc = float(_distance_correlation(x, y))
                except Exception:
                    dc = np.nan
            else:
                dc = np.nan
        else:
            r2 = np.nan; t_stat = np.nan; sp = np.nan; dc = np.nan

        s, ql, t, b = key_full
        out['r2'][s][ql][t][b] = r2 if np.isfinite(r2) else np.nan
        out['t_stat'][s][ql][t][b] = t_stat if np.isfinite(t_stat) else np.nan
        if add_spearman:
            out['spearman'][s][ql][t][b] = sp if np.isfinite(sp) else np.nan
        if add_dcor:
            out['dcor'][s][ql][t][b] = dc if np.isfinite(dc) else np.nan

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
