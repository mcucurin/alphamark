# =============================
# summary_stats.py
# =============================
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Sequence, List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import spearmanr

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

# We reuse the 5d nested-dict factory from daily_stats
# It must exist in your repo and return nested dicts:
# stats[stat_type][signal][qrank][target][bet] = value
from .daily_stats import create_5d_stats


# ----------------------------
# Helpers
# ----------------------------
def _qlabel(q: float) -> str:
    return f"qr_{int(round(q*100))}"

def _float_clean(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype="float64")
    out[~np.isfinite(out)] = np.nan
    return out

def _sanitize_list(cols: Sequence) -> List[str]:
    seen = set()
    out: List[str] = []
    for c in cols or []:
        if c is None or c is Ellipsis:
            continue
        if isinstance(c, str) and c not in seen:
            out.append(c); seen.add(c)
    return out

def _topk_mask_desc(abs_vals: np.ndarray, finite_mask: np.ndarray, q: float) -> np.ndarray:
    idx_fin = np.where(finite_mask)[0]
    out = np.zeros_like(finite_mask, dtype=bool)
    if idx_fin.size == 0 or q <= 0.0:
        return out
    if q >= 1.0:
        out[idx_fin] = True
        return out
    k = int(np.ceil(q * idx_fin.size))
    order = np.argsort(-abs_vals[idx_fin], kind="mergesort")
    choose = idx_fin[order[:k]]
    out[choose] = True
    return out


# ===================== INTERNAL CORE (single set of signals) ====================
def _compute_summary_stats_core(
    df: pd.DataFrame,
    date_col: str,
    signal_cols: Sequence[str],
    target_cols: Sequence[str],
    quantiles: Sequence[float],
    bet_size_cols: Sequence[str],
    type_quantile: str,
    add_spearman: bool,
    add_dcor: bool,
    spearman_sample_cap_per_key: int,
    random_state: int | None,
    spy_by_target: Optional[Dict[str, str]],          # map: target -> spy column (same horizon)
) -> Dict:
    """
    Compute strategy summary stats over all days.
    Returns nested dict: stats[stat_type][signal][qrank][target][bet] = value
    """
    out = create_5d_stats()

    # Sanitize lists
    signal_cols   = _sanitize_list(signal_cols)
    target_cols   = _sanitize_list(target_cols)
    bet_size_cols = _sanitize_list(bet_size_cols)

    if date_col not in df.columns:
        raise KeyError(f"[summary_stats] date_col '{date_col}' not found in DataFrame.")

    # Build the list we actually need present
    want = [date_col] + list(signal_cols) + list(target_cols) + list(bet_size_cols)
    if spy_by_target:
        want += [c for c in spy_by_target.values() if isinstance(c, str)]
    present = [c for c in want if c in df.columns]
    missing = [c for c in want if c not in df.columns]
    if missing:
        print(f"[WARN][summary_stats] Ignoring missing columns: {missing}")

    # Make an effective spy map limited to present columns
    effective_spy_map: Dict[str, str] = {}
    if spy_by_target:
        for t, sc in spy_by_target.items():
            if isinstance(t, str) and isinstance(sc, str) and (t in df.columns) and (sc in df.columns):
                effective_spy_map[t] = sc

    # Prepare df
    df = df[present].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if df.empty:
        return out

    # Pre-clean numeric columns
    numeric_cols = [c for c in present if c != date_col]
    for c in numeric_cols:
        df[c] = _float_clean(df[c].to_numpy())

    # Chronological groups
    grouped = df.sort_values(date_col).groupby(date_col, sort=True)

    # --------- Streaming accumulators ---------
    sqrt_252 = np.sqrt(252.0)
    rng = np.random.default_rng(random_state)

    # Welford stats for daily PPD
    from collections import defaultdict as _dd
    ppd_stats = _dd(lambda: [0, 0.0, 0.0])   # key=(s,q,t,b)

    # Regression pooled sufficient stats for r² / t
    reg = _dd(lambda: {'n':0, 'sx':0.0, 'sy':0.0, 'sxx':0.0, 'syy':0.0, 'sxy':0.0})

    # Hit / long ratios
    hit_num = _dd(int)         # key=(s,q,t,b)
    hit_den = _dd(int)
    long_num = _dd(int)        # key=(s,q,b)
    long_den = _dd(int)

    # Spearman/DCOR small reservoir (signal vs target per-row)
    class _Reservoir:
        def __init__(self, cap: int = 0, seed: int | None = 123):
            self.cap = int(cap) if cap and cap > 0 else 0
            self.rng = np.random.default_rng(seed)
            self.store: Dict[Tuple[str,str,str,str], Tuple[np.ndarray,np.ndarray,int]] = {}
        def add(self, key, xs: np.ndarray, ys: np.ndarray):
            if self.cap <= 0 or xs.size == 0:
                return
            m = min(xs.size, ys.size)
            if m == 0:
                return
            xs = xs[:m].astype('float64', copy=False)
            ys = ys[:m].astype('float64', copy=False)
            if key not in self.store:
                take = min(self.cap, m)
                idx = self.rng.choice(m, size=take, replace=False)
                self.store[key] = (xs[idx].copy(), ys[idx].copy(), m)
                return
            X, Y, seen = self.store[key]
            total = seen + m

            if X.size < self.cap:
                need = self.cap - X.size
                add = min(need, m)
                idx = self.rng.choice(m, size=add, replace=False)
                X = np.concatenate([X, xs[idx]])
                Y = np.concatenate([Y, ys[idx]])
                seen += m
                self.store[key] = (X, Y, seen)
                return

            if total > 0:
                p = self.cap / float(total)
                rcount = int(self.rng.binomial(m, p))
                if rcount > 0:
                    rep_new_idx = self.rng.choice(m, size=rcount, replace=False)
                    rep_old_idx = self.rng.choice(self.cap, size=rcount, replace=False)
                    X[rep_old_idx] = xs[rep_new_idx]
                    Y[rep_old_idx] = ys[rep_new_idx]

            seen += m
            self.store[key] = (X, Y, seen)

        def get(self, key):
            return self.store.get(key, (np.array([]), np.array([]), 0))[:2]

    sampler = _Reservoir(spearman_sample_cap_per_key if add_spearman or add_dcor else 0,
                         seed=random_state)

    # Daily totals accumulators (streamed sums over days)
    sum_pnl       = _dd(float)  # key=(s,q,t,b)
    sum_notional  = _dd(float)
    sum_nrInstr   = _dd(float)
    sum_ntrades   = _dd(float)

    # Per-key daily PnL vs SPY (same-horizon) series for Spearman
    # key -> (list_of_daily_pnl, list_of_daily_spy_ret)
    spy_pairs = _dd(lambda: ([], []))

    # --------- Stream each day ---------
    for dt, day in grouped:
        if day.empty:
            continue

        # Build matrices
        sig_names = [c for c in signal_cols if c in day.columns]
        tgt_names = [c for c in target_cols if c in day.columns]
        bet_names = [c for c in bet_size_cols if c in day.columns]
        if not sig_names or not tgt_names or not bet_names:
            continue

        S = np.column_stack([day[c].to_numpy() for c in sig_names])              # (n, ns)
        Y = np.column_stack([day[c].to_numpy() for c in tgt_names])              # (n, nt)
        B = np.column_stack([np.abs(day[c].to_numpy()) for c in bet_names])      # (n, nb)

        # For each target, pick the *same-horizon* SPY return for this day
        spy_val_by_t: Dict[str, float] = {}
        if effective_spy_map:
            for t_name in tgt_names:
                sc = effective_spy_map.get(t_name)
                if sc and sc in day.columns:
                    vals = np.asarray(day[sc].to_numpy(), dtype="float64")
                    v = np.nanmean(vals) if vals.size else np.nan
                    spy_val_by_t[t_name] = float(v) if np.isfinite(v) else np.nan

        _, nt = Y.shape
        _, nb = B.shape

        for si, s_name in enumerate(sig_names):
            s_all = S[:, si]
            m_fin = np.isfinite(s_all)
            if not m_fin.any():
                continue

            sgn = np.sign(s_all)
            abs_s = np.abs(s_all)

            # quantile bucket edges if needed
            if type_quantile != 'cumulative':
                sabs_fin = abs_s[m_fin]
                if sabs_fin.size:
                    K = len(quantiles)
                    probs = np.linspace(0.0, 1.0, K + 1)
                    edges = np.nanquantile(sabs_fin, probs)
                else:
                    edges = None
            else:
                edges = None

            for q in quantiles:
                qlbl = _qlabel(q)

                if type_quantile == 'cumulative':
                    mask_q = _topk_mask_desc(abs_s, m_fin, q)
                else:
                    if edges is None or not np.isfinite(edges).all():
                        mask_q = np.zeros_like(m_fin, dtype=bool)
                    else:
                        j = quantiles.index(q) + 1
                        lo, hi = edges[j-1], edges[j]
                        mask_q = m_fin & (abs_s >= lo) & (abs_s <= hi)

                if not mask_q.any():
                    continue

                # slice once
                s_q   = s_all[mask_q]
                sgn_q = sgn[mask_q]
                Y_q   = Y[mask_q, :]
                B_q   = B[mask_q, :]

                Y_fin = np.isfinite(Y_q)
                B_fin = np.isfinite(B_q)
                Yz    = np.where(Y_fin, Y_q, 0.0)
                Bz    = np.where(B_fin, B_q, 0.0)

                # ----- PnL / Notional (daily matrices) -----
                pnl_mat = ((Yz * sgn_q[:, None]).T @ Bz)       # (nt, nb)
                not_mat = (Y_fin.astype(float).T @ Bz)         # (nt, nb)

                # ----- PPD matrix (daily) -----
                ppd_mat = np.divide(pnl_mat, not_mat,
                                    out=np.full_like(pnl_mat, np.nan),
                                    where=(not_mat > 0))

                # ----- hit ratio counts -----
                y_sign   = np.sign(Y_q)
                nonzero  = (y_sign != 0.0) & Y_fin
                denom_hr = (nonzero.astype(float).T @ B_fin.astype(float))   # (nt, nb)
                eq_sign  = ((np.sign(s_q)[:, None] == y_sign) & nonzero)
                numer_hr = (eq_sign.astype(float).T @ B_fin.astype(float))   # (nt, nb)

                # ----- long ratio (per bet) -----
                long_den_vec = np.sum(B_fin, axis=0).astype(int)
                long_num_vec = np.sum((sgn_q > 0)[:, None] & B_fin, axis=0).astype(int)
                for bi, b_name in enumerate(bet_names):
                    if long_den_vec[bi] > 0:
                        long_den[(s_name, qlbl, b_name)] += int(long_den_vec[bi])
                        long_num[(s_name, qlbl, b_name)] += int(long_num_vec[bi])

                # ----- update all per (target, bet) -----
                for ti, t_name in enumerate(tgt_names):
                    row_ppd = ppd_mat[ti, :]
                    for bi, b_name in enumerate(bet_names):
                        key = (s_name, qlbl, t_name, b_name)

                        # Welford (daily PPD)
                        v = row_ppd[bi]
                        if np.isfinite(v):
                            n, mean, M2 = ppd_stats[key]
                            n += 1
                            delta = v - mean
                            mean += delta / n
                            M2 += delta * (v - mean)
                            ppd_stats[key] = [n, mean, M2]

                        # hit ratio counts
                        d = int(denom_hr[ti, bi])
                        if d > 0:
                            hit_den[key] += d
                            hit_num[key] += int(numer_hr[ti, bi])

                        # accumulate activity totals
                        p = pnl_mat[ti, bi]
                        ntn = not_mat[ti, bi]
                        if np.isfinite(p):
                            sum_pnl[key] += float(p)
                        if np.isfinite(ntn):
                            sum_notional[key] += float(ntn)

                        # nrInstr / n_trades: rows contributing today
                        b_ok = B_fin[:, bi]
                        y_ok = Y_fin[:, ti]
                        m = b_ok & y_ok
                        if m.any():
                            cnt = float(np.sum(m))
                            sum_nrInstr[key] += cnt
                            sum_ntrades[key] += cnt

                            # pooled regression sums + optional sample (signal vs target per-row)
                            xs = s_q[m]
                            ys = Y_q[m, ti]
                            nrows = xs.size
                            sx = float(xs.sum()); sy = float(ys.sum())
                            sxx = float((xs*xs).sum()); syy = float((ys*ys).sum())
                            sxy = float((xs*ys).sum())
                            st = reg[key]
                            st['n']  += nrows
                            st['sx'] += sx
                            st['sy'] += sy
                            st['sxx']+= sxx
                            st['syy']+= syy
                            st['sxy']+= sxy

                            if add_spearman or add_dcor:
                                cap = min(1024, spearman_sample_cap_per_key)
                                if xs.size > cap:
                                    idx = rng.choice(xs.size, size=cap, replace=False)
                                    xs = xs[idx]; ys = ys[idx]
                                try:
                                    sampler.add(key, xs, ys)
                                except Exception:
                                    pass

                        # Collect per-day PnL vs same-horizon SPY for Spearman
                        spy_v = spy_val_by_t.get(t_name, np.nan)
                        if np.isfinite(p) and np.isfinite(spy_v):
                            pnl_ser, spy_ser = spy_pairs[key]
                            pnl_ser.append(float(p))
                            spy_ser.append(float(spy_v))

    # --------- Finalize into nested output ---------
    out_nested = create_5d_stats()

    # Sharpe from daily PPD (Welford)
    for key, st in ppd_stats.items():
        n, mean, M2 = st
        mu, sd = (mean, np.sqrt(M2 / n)) if n > 1 else (mean, np.nan)
        sharpe = (mu / sd * np.sqrt(252.0)) if (np.isfinite(mu) and np.isfinite(sd) and sd > 0) else np.nan
        s, ql, t, b = key
        out_nested['sharpe'][s][ql][t][b] = float(sharpe) if np.isfinite(sharpe) else np.nan

    # r2 and t-stat from pooled sufficient stats (signal vs target per-row)
    eps = 1e-15
    for key, st in reg.items():
        n = st['n']
        if n >= 3:
            sx, sy, sxx, syy, sxy = st['sx'], st['sy'], st['sxx'], st['syy'], st['sxy']
            cov_xy = sxy - (sx * sy) / n
            var_x  = sxx - (sx * sx) / n
            var_y  = syy - (sy * sy) / n
            if var_x > eps and var_y > eps:
                r = cov_xy / np.sqrt(var_x * var_y)
                r = float(np.clip(r, -1.0, 1.0))
                r2 = r * r
                denom = max(eps, 1.0 - r2)
                t_stat = float(r * np.sqrt((n - 2) / denom))
            else:
                r2 = np.nan; t_stat = np.nan
        else:
            r2 = np.nan; t_stat = np.nan
        s, ql, t, b = key
        out_nested['r2'][s][ql][t][b] = r2 if np.isfinite(r2) else np.nan
        out_nested['t_stat'][s][ql][t][b] = t_stat if np.isfinite(t_stat) else np.nan

    # Optional: Spearman & DCOR on bounded samples (signal vs target per-row)
    if add_spearman or add_dcor:
        for key in reg.keys():  # compute only where we had data
            xs, ys = sampler.get(key)
            s, ql, t, b = key
            if add_spearman:
                if xs.size >= 3 and ys.size >= 3:
                    try:
                        sp = float(spearmanr(xs, ys, nan_policy='omit').correlation)
                    except Exception:
                        sp = np.nan
                else:
                    sp = np.nan
                out_nested['spearman'][s][ql][t][b] = sp if np.isfinite(sp) else np.nan
            if add_dcor:
                if xs.size >= 3 and ys.size >= 3:
                    try:
                        dc = float(_distance_correlation(xs, ys))
                    except Exception:
                        dc = np.nan
                else:
                    dc = np.nan
                out_nested['dcor'][s][ql][t][b] = dc if np.isfinite(dc) else np.nan

    # hit ratio
    for key, hn in hit_num.items():
        hd = hit_den.get(key, 0)
        s, ql, t, b = key
        out_nested['hit_ratio'][s][ql][t][b] = (hn / hd) if hd > 0 else np.nan

    # long ratio (per bet) -> broadcast to all targets we saw in keys
    seen_targets_per_sqb = defaultdict(set)
    for (s, ql, t, b) in ppd_stats.keys():
        seen_targets_per_sqb[(s, ql, b)].add(t)
    for (s, ql, b), ln in long_num.items():
        ld = long_den.get((s, ql, b), 0)
        val = (ln / ld) if ld > 0 else np.nan
        for t in seen_targets_per_sqb.get((s, ql, b), []):
            out_nested['long_ratio'][s][ql][t][b] = val

    # ===== Activity metrics to SUMMARY (totals + ratios) =====
    all_keys = set(sum_pnl) | set(sum_notional) | set(sum_nrInstr) | set(sum_ntrades)
    for key in all_keys:
        pnl_tot  = sum_pnl.get(key, 0.0)
        not_tot  = sum_notional.get(key, 0.0)
        nrin_tot = sum_nrInstr.get(key, 0.0)
        ntrd_tot = sum_ntrades.get(key, 0.0)

        ppd_val = (pnl_tot / not_tot) if (np.isfinite(pnl_tot) and np.isfinite(not_tot) and not_tot > 0) else np.nan

        s, ql, t, b = key
        out_nested['pnl'][s][ql][t][b]          = float(pnl_tot) if np.isfinite(pnl_tot) else np.nan
        out_nested['sizeNotional'][s][ql][t][b] = float(not_tot) if np.isfinite(not_tot) else np.nan
        out_nested['nrInstr'][s][ql][t][b]      = float(nrin_tot) if np.isfinite(nrin_tot) else np.nan
        out_nested['n_trades'][s][ql][t][b]     = float(ntrd_tot) if np.isfinite(ntrd_tot) else np.nan
        out_nested['ppd'][s][ql][t][b]          = float(ppd_val) if np.isfinite(ppd_val) else np.nan

    # ===== Spearman corr between per-day strategy PnL and same-horizon SPY return =====
    if effective_spy_map:
        for key, (pnl_series, spy_series) in spy_pairs.items():
            x = np.asarray(pnl_series, dtype=float)
            y = np.asarray(spy_series, dtype=float)
            if x.size >= 3 and y.size >= 3:
                try:
                    r = spearmanr(x, y, nan_policy='omit').correlation
                    r = float(r) if np.isfinite(r) else np.nan
                except Exception:
                    r = np.nan
            else:
                r = np.nan
            s, ql, t, b = key
            out_nested['market_corr'][s][ql][t][b] = r
            out_nested['spy_corr'][s][ql][t][b] = r  # backward compatibility
        # Ensure all observed (s, q, t, b) combos have an explicit market_corr entry,
        # even if the SPY mapping was sparse for a given horizon.
        for key in all_keys:
            s, ql, t, b = key
            if b not in out_nested['market_corr'][s][ql].get(t, {}):
                out_nested['market_corr'][s][ql][t][b] = np.nan
                out_nested['spy_corr'][s][ql][t][b] = np.nan

    return out_nested


def _merge_summary(dst: Dict, src: Dict) -> None:
    """Merge nested stats dicts whose SIGNAL subtrees are disjoint."""
    for stat_type, sig_tree in src.items():
        if stat_type not in dst:
            dst[stat_type] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for signal, q_tree in sig_tree.items():
            dst[stat_type][signal] = q_tree


# ===================== PUBLIC API ====================
def compute_summary_stats_over_days(
    df: pd.DataFrame,
    date_col: str,
    signal_cols: Sequence[str],
    target_cols: Sequence[str],
    quantiles: Sequence[float] = (1.0, 0.75, 0.5, 0.25),
    bet_size_cols: Sequence[str] = ('betsize_equal',),
    type_quantile: str = 'cumulative',   # 'cumulative' (>=thr) or 'quantEach' (exact bucket)
    add_spearman: bool = False,
    add_dcor: bool = False,
    n_jobs: int | None = None,           # threads across signals
    backend: str = "loky",               # kept for compat
    spearman_sample_cap_per_key: int = 10000,
    random_state: int | None = 123,
    spy_by_target: Optional[Dict[str, str]] = None,  # {target -> spy column}
    # NEW (optional) — per-id (e.g., per-ticker) corr & CCF dumps
    id_col: Optional[str] = None,
    dump_alpha_raw_corr_path: Optional[str] = None,
    dump_alpha_pnl_corr_path: Optional[str] = None,
    ccf_max_lag: int = 5,
    dump_alpha_raw_ccf_path: Optional[str] = None,
    dump_alpha_pnl_ccf_path: Optional[str] = None,
) -> Dict:
    """
    Returns ONE summary value per (signal, qrank, target, bet).

    If `spy_by_target` is provided as a dict mapping target column -> spy column that
    holds the *same-horizon* SPY return (broadcast per row), we compute:

        market_corr[signal][qrank][target][bet]
            = Spearman corr( daily strategy PnL, daily SPY return at that target's horizon ).
        spy_corr[signal][qrank][target][bet]
            = same value as market_corr (kept for backward compatibility).

    If `id_col` is provided and present in df, and any of the dump paths are provided,
    we also compute per-id Spearman correlations across days and dump them:
      - alpha_raw_spy_corr  (mean raw alpha per-id vs SPY)
      - alpha_pnl_spy_corr  (daily per-id PnL vs SPY)

    Additionally, if `ccf_max_lag > 0` and CCF dump paths are provided, we compute per-id
    cross-correlation functions (CCF) vs SPY across lags in [-ccf_max_lag, +ccf_max_lag]:
      - alpha_raw_spy_ccf
      - alpha_pnl_spy_ccf

    CCF is Spearman-based, using corr(x_t, spy_{t+lag}) with both series aligned on calendar dates.
    """
    signal_cols = _sanitize_list(signal_cols)
    if not signal_cols:
        return create_5d_stats()

    # Single-threaded path (common and deterministic)
    if not n_jobs or n_jobs <= 1 or len(signal_cols) == 1:
        out = _compute_summary_stats_core(
            df, date_col, signal_cols, target_cols, quantiles, bet_size_cols,
            type_quantile, add_spearman, add_dcor, spearman_sample_cap_per_key,
            random_state, spy_by_target
        )
    else:
        # Parallel across signals
        n_threads = min(len(signal_cols), int(n_jobs))
        out = create_5d_stats()

        def _chunks(lst, k):
            for i in range(k):
                yield [lst[j] for j in range(i, len(lst), k)]

        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futs = []
            rng = np.random.default_rng(random_state)
            for sub_signals in _chunks(signal_cols, n_threads):
                if not sub_signals:
                    continue
                futs.append(ex.submit(
                    _compute_summary_stats_core,
                    df,
                    date_col,
                    sub_signals,
                    target_cols,
                    quantiles,
                    bet_size_cols,
                    type_quantile,
                    add_spearman,
                    add_dcor,
                    spearman_sample_cap_per_key,
                    None if random_state is None else int(rng.integers(0, 2**31 - 1)),
                    spy_by_target
                ))
            for fut in as_completed(futs):
                _merge_summary(out, fut.result())

    # =================== NEW: Per-ID (e.g., per-ticker) corr & CCF dumps ===================
    try:
        do_per_id = (id_col is not None) and (isinstance(id_col, str)) and (id_col in df.columns)
        have_spy = isinstance(spy_by_target, dict) and (len(spy_by_target) > 0)
        want_corr = (dump_alpha_raw_corr_path is not None) or (dump_alpha_pnl_corr_path is not None)
        want_ccf  = (dump_alpha_raw_ccf_path is not None) or (dump_alpha_pnl_ccf_path is not None)

        if do_per_id and have_spy and (want_corr or want_ccf):
            import pickle as _p

            work = df.copy()
            work[date_col] = pd.to_datetime(work[date_col], errors='coerce')
            work = work.dropna(subset=[date_col, id_col])

            # Effective spy map
            eff_spy_map = {t: sc for t, sc in (spy_by_target or {}).items() if t in work.columns and sc in work.columns}
            if eff_spy_map:
                sigs = [c for c in signal_cols if c in work.columns]
                tgts = [c for c in target_cols if c in work.columns]
                bets = [c for c in bet_size_cols if c in work.columns]
                if sigs and tgts and bets:
                    recs_raw = [] if (dump_alpha_raw_corr_path or dump_alpha_raw_ccf_path) else None
                    recs_pnl = [] if (dump_alpha_pnl_corr_path or dump_alpha_pnl_ccf_path) else None

                    for dt, day in work.groupby(date_col, sort=True):
                        for s_name in sigs:
                            svals = np.asarray(day[s_name], float)
                            sfin  = np.isfinite(svals)
                            sgn   = np.sign(svals)
                            abs_s = np.abs(svals)

                            for q in quantiles:
                                qlbl = _qlabel(q)
                                if q >= 1.0:
                                    mask_q = sfin
                                else:
                                    idx_fin = np.where(sfin)[0]
                                    mask_q = np.zeros_like(sfin, bool)
                                    if idx_fin.size:
                                        k = int(np.ceil(q * idx_fin.size))
                                        order = np.argsort(-abs_s[idx_fin], kind="mergesort")
                                        choose = idx_fin[order[:k]]
                                        mask_q[choose] = True
                                if not mask_q.any():
                                    continue

                                s_q   = svals[mask_q]
                                ids_q = day.loc[mask_q, id_col].astype(str).values

                                for t_name in tgts:
                                    if t_name not in eff_spy_map:
                                        continue
                                    spy_col = eff_spy_map[t_name]
                                    spy_vals = np.asarray(day[spy_col], float)
                                    spy_v = np.nanmean(spy_vals) if spy_vals.size else np.nan
                                    if not np.isfinite(spy_v):
                                        continue

                                    y = np.asarray(day[t_name], float)[mask_q]
                                    yfin = np.isfinite(y)
                                    if not yfin.any():
                                        continue

                                    # RAW alpha (mean per id)
                                    if recs_raw is not None:
                                        dfraw = pd.DataFrame({id_col: ids_q, 'alpha_raw': s_q})
                                        raw_per_id = dfraw.groupby(id_col)['alpha_raw'].mean()
                                        for name_i, val in raw_per_id.items():
                                            if np.isfinite(val):
                                                recs_raw.append((
                                                    str(name_i), s_name, qlbl, t_name, "__RAW__", pd.Timestamp(dt), float(val), float(spy_v)
                                                ))

                                    # PNL per id (for each bet)
                                    if recs_pnl is not None:
                                        for b_name in bets:
                                            bcol = np.asarray(day[b_name], float)[mask_q]
                                            bcol = np.where(np.isfinite(bcol), np.abs(bcol), np.nan)
                                            pnl_row = y * np.sign(s_q) * bcol
                                            dfp = pd.DataFrame({id_col: ids_q, 'pnl': pnl_row})
                                            pnl_per_id = dfp.groupby(id_col)['pnl'].sum()
                                            for name_i, val in pnl_per_id.items():
                                                if np.isfinite(val):
                                                    recs_pnl.append((
                                                        str(name_i), s_name, qlbl, t_name, b_name, pd.Timestamp(dt), float(val), float(spy_v)
                                                    ))

                    def _collapse_and_dump(records, out_path, metric_name):
                        if (records is None) or (out_path is None) or (len(records) == 0):
                            return
                        cols = [id_col, 'signal', 'qrank', 'target', 'bet_size_col', 'date', 'series', 'spy']
                        dfrec = pd.DataFrame.from_records(records, columns=cols)
                        out_rows = []
                        for keys, grp in dfrec.groupby([id_col, 'signal', 'qrank', 'target', 'bet_size_col'], sort=False):
                            x = pd.to_numeric(grp['series'], errors='coerce')
                            y = pd.to_numeric(grp['spy'], errors='coerce')
                            m = x.notna() & y.notna()
                            if m.sum() >= 3 and x[m].nunique() >= 2 and y[m].nunique() >= 2:
                                r = spearmanr(x[m], y[m], nan_policy='omit').correlation
                                val = float(r) if np.isfinite(r) else np.nan
                            else:
                                val = np.nan
                            out_rows.append((*keys, metric_name, val))
                        dfout = pd.DataFrame.from_records(
                            out_rows,
                            columns=[id_col, 'signal', 'qrank', 'target', 'bet_size_col', 'stat_type', 'value']
                        )
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        with open(out_path + ".tmp", "wb") as _f:
                            _p.dump(dfout, _f, protocol=_p.HIGHEST_PROTOCOL)
                        os.replace(out_path + ".tmp", out_path)
                        print(f"[summary_stats] Wrote per-id corr: {out_path}  ({len(dfout)} rows)")

                    def _collapse_and_dump_ccf(records, out_path, metric_name, max_lag: int):
                        """
                        For each (id, signal, qrank, target, bet), build a date-indexed series
                        of 'series' vs 'spy' and compute Spearman cross-correlation at lags
                        in [-max_lag, +max_lag].  Writes columns:
                          [id_col, signal, qrank, target, bet_size_col, stat_type, lag, corr]
                        """
                        if (records is None) or (out_path is None) or (len(records) == 0):
                            return
                        if max_lag is None or int(max_lag) <= 0:
                            return
                        max_lag = int(max_lag)

                        cols = [id_col, 'signal', 'qrank', 'target', 'bet_size_col', 'date', 'series', 'spy']
                        dfrec = pd.DataFrame.from_records(records, columns=cols)
                        out_rows = []

                        for keys, grp in dfrec.groupby([id_col, 'signal', 'qrank', 'target', 'bet_size_col'], sort=False):
                            grp = grp.copy()
                            grp['date'] = pd.to_datetime(grp['date'], errors='coerce')
                            grp = grp.dropna(subset=['date'])
                            if grp.empty:
                                continue

                            grp = grp.sort_values('date')
                            x = pd.to_numeric(grp['series'], errors='coerce')
                            y = pd.to_numeric(grp['spy'], errors='coerce')
                            idx = grp['date']

                            sx = pd.Series(x.values, index=idx)
                            sy = pd.Series(y.values, index=idx)

                            for L in range(-max_lag, max_lag + 1):
                                # corr(sx_t, sy_{t+L}); implement by shifting sy
                                sy_shift = sy.shift(-L)
                                df_xy = pd.concat({'x': sx, 'y': sy_shift}, axis=1).dropna()
                                if df_xy.shape[0] < 5:
                                    continue
                                r = df_xy['x'].corr(df_xy['y'], method='spearman')
                                if np.isfinite(r):
                                    out_rows.append((*keys, metric_name, int(L), float(r)))

                        if not out_rows:
                            return

                        dfout = pd.DataFrame.from_records(
                            out_rows,
                            columns=[id_col, 'signal', 'qrank', 'target', 'bet_size_col', 'stat_type', 'lag', 'corr']
                        )
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        with open(out_path + ".tmp", "wb") as _f:
                            _p.dump(dfout, _f, protocol=_p.HIGHEST_PROTOCOL)
                        os.replace(out_path + ".tmp", out_path)
                        print(f"[summary_stats] Wrote per-id CCF: {out_path}  ({len(dfout)} rows)")

                    # Corr dumps
                    _collapse_and_dump(recs_raw, dump_alpha_raw_corr_path, "alpha_raw_spy_corr")
                    _collapse_and_dump(recs_pnl, dump_alpha_pnl_corr_path, "alpha_pnl_spy_corr")

                    # CCF dumps
                    _collapse_and_dump_ccf(recs_raw, dump_alpha_raw_ccf_path, "alpha_raw_spy_ccf", ccf_max_lag)
                    _collapse_and_dump_ccf(recs_pnl, dump_alpha_pnl_ccf_path, "alpha_pnl_spy_ccf", ccf_max_lag)

    except Exception as _e:
        print(f"[WARN][summary_stats] per-id corr/ccf dump failed: {_e}")

    return out


__all__ = [
    'compute_summary_stats_over_days',
    '_distance_correlation',
]
