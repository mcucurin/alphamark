# =============================
# summary_stats.py (fast, streaming, memory-light)
# =============================
from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Sequence, List, Dict, Tuple
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

# Reuse nested store
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

# ---------- Online stats ----------
def _welford_update(state, x):
    n, mean, M2 = state
    n += 1
    delta = x - mean
    mean += delta / n
    M2 += delta * (x - mean)
    return [n, mean, M2]

def _welford_finalize(n, mean, M2):
    if n <= 1:
        return mean, np.nan
    var = M2 / n  # population variance (ddof=0)
    return mean, float(np.sqrt(var))

# Small, bounded sampler for Spearman/DCOR to avoid O(N) memory
class _Reservoir:
    def __init__(self, cap: int = 0, seed: int | None = 123):
        self.cap = int(cap) if cap and cap > 0 else 0
        self.rng = np.random.default_rng(seed)
        self.store: Dict[Tuple[str,str,str,str], Tuple[np.ndarray,np.ndarray,int]] = {}
        # key -> (xs, ys, n_seen)

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

        # ensure arrays filled up to cap
        if X.size < self.cap:
            need = self.cap - X.size
            add = min(need, m)
            idx = self.rng.choice(m, size=add, replace=False)
            X = np.concatenate([X, xs[idx]])
            Y = np.concatenate([Y, ys[idx]])
            seen += m
            self.store[key] = (X, Y, seen)
            return

        # reservoir replacement
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


# ===================== SUMMARY over multiple days (single-pass) ====================
def compute_summary_stats_over_days(
    df,
    date_col: str,
    signal_cols: Sequence[str],
    target_cols: Sequence[str],
    quantiles: Sequence[float] = (1.0, 0.75, 0.5, 0.25),
    bet_size_cols: Sequence[str] = ('betsize_equal',),
    type_quantile: str = 'cumulative',   # 'cumulative' (>=thr) or 'quantEach' (exact bucket)
    add_spearman: bool = False,
    add_dcor: bool = False,
    n_jobs: int | None = None,           # kept for signature compatibility (unused)
    backend: str = "loky",               # kept for signature compatibility (unused)
    # ---- bounded sampling for Spearman/DCOR ----
    spearman_sample_cap_per_key: int = 10000,
    random_state: int | None = 123,
):
    """
    Returns ONE summary value per (signal, qrank, target, bet).
    Output access pattern:
      out[stat_type][signal][qrank][target][bet] = value

    Fast strategy:
      • Stream through dates once (no joblib/process pickling).
      • Welford for daily PPD Sharpe and efficiencyP.
      • r²/t-stat via pooled sufficient statistics (sx, sy, sxx, syy, sxy).
      • Spearman/DCOR optionally on a capped reservoir per key.
      • Accumulate totals for pnl / notional / instruments / trades, then derive ppd & ppt.
    """
    import pandas as pd

    out = create_5d_stats()

    # Sanitize lists
    signal_cols   = _sanitize_list(signal_cols)
    target_cols   = _sanitize_list(target_cols)
    bet_size_cols = _sanitize_list(bet_size_cols)

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

    # Pre-clean numeric columns once
    numeric_cols = [c for c in present if c != date_col]
    for c in numeric_cols:
        df[c] = _float_clean(df[c].to_numpy())

    # Group in chronological order
    grouped = df.sort_values(date_col).groupby(date_col, sort=True)

    # --------- Streaming accumulators ---------
    sqrt_252 = np.sqrt(252.0)
    rng = np.random.default_rng(random_state)

    # Welford stats for daily PPD and efficiencyP
    ppd_stats = defaultdict(lambda: [0, 0.0, 0.0])   # key=(s,q,t,b)
    effp_stats = defaultdict(lambda: [0, 0.0, 0.0])  # key=(s,q,t,b)

    # Regression pooled sufficient stats for r² / t
    reg = defaultdict(lambda: {'n':0, 'sx':0.0, 'sy':0.0, 'sxx':0.0, 'syy':0.0, 'sxy':0.0})

    # Hit / long ratios
    hit_num = defaultdict(int)         # key=(s,q,t,b)
    hit_den = defaultdict(int)
    long_num = defaultdict(int)        # key=(s,q,b)
    long_den = defaultdict(int)

    # Spearman/DCOR small reservoir
    sampler = _Reservoir(spearman_sample_cap_per_key if add_spearman or add_dcor else 0,
                         seed=random_state)

    # Daily totals accumulators (streamed sums over days)
    sum_pnl       = defaultdict(float)  # key=(s,q,t,b)
    sum_notional  = defaultdict(float)  # key=(s,q,t,b)
    sum_nrInstr   = defaultdict(float)  # key=(s,q,t,b)  (count summed over days)
    sum_ntrades   = defaultdict(float)  # key=(s,q,t,b)  (same definition here)

    # --------- Stream each day ---------
    for _, day in grouped:
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

        n, ns = S.shape
        _, nt = Y.shape
        _, nb = B.shape

        for si, s_name in enumerate(sig_names):
            s_all = S[:, si]
            m_fin = np.isfinite(s_all)
            if not m_fin.any():
                continue

            sgn = np.sign(s_all)
            abs_s = np.abs(s_all)

            # quantEach edges if needed
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
                s_q   = s_all[mask_q]                          # (m,)
                sgn_q = sgn[mask_q]                            # (m,)
                Y_q   = Y[mask_q, :]                           # (m, nt)
                B_q   = B[mask_q, :]                           # (m, nb)

                Y_fin = np.isfinite(Y_q)                       # (m, nt)
                B_fin = np.isfinite(B_q)                       # (m, nb)
                Yz    = np.where(Y_fin, Y_q, 0.0)              # (m, nt)
                Bz    = np.where(B_fin, B_q, 0.0)              # (m, nb)

                # ----- PnL / Notional (daily matrices) -----
                pnl_mat = ((Yz * sgn_q[:, None]).T @ Bz)       # (nt, nb)
                not_mat = (Y_fin.astype(float).T @ Bz)         # (nt, nb)

                # ----- PPD matrix (daily) -----
                ppd_mat = np.divide(pnl_mat, not_mat,
                                    out=np.full_like(pnl_mat, np.nan),
                                    where=(not_mat > 0))

                # ----- efficiencyP matrix (daily) -----
                denom_abs = (np.abs(Yz).T @ Bz)                # (nt, nb)
                effP_mat  = np.divide(np.abs(pnl_mat), denom_abs,
                                      out=np.full_like(pnl_mat, np.nan),
                                      where=(denom_abs > 0))

                # ----- hit ratio counts -----
                y_sign   = np.sign(Y_q)                        # (m, nt)
                nonzero  = (y_sign != 0.0) & Y_fin
                denom_hr = (nonzero.astype(float).T @ B_fin.astype(float))   # (nt, nb)
                eq_sign  = ((np.sign(s_q)[:, None] == y_sign) & nonzero)
                numer_hr = (eq_sign.astype(float).T @ B_fin.astype(float))   # (nt, nb)

                # ----- long ratio (per bet) -----
                long_den_vec = np.sum(B_fin, axis=0).astype(int)                        # (nb,)
                long_num_vec = np.sum((sgn_q > 0)[:, None] & B_fin, axis=0).astype(int) # (nb,)
                for bi, b_name in enumerate(bet_names):
                    if long_den_vec[bi] > 0:
                        long_den[(s_name, qlbl, b_name)] += int(long_den_vec[bi])
                        long_num[(s_name, qlbl, b_name)] += int(long_num_vec[bi])

                # ----- update all per (target, bet) -----
                for ti, t_name in enumerate(tgt_names):
                    row_ppd = ppd_mat[ti, :]
                    row_eff = effP_mat[ti, :]

                    for bi, b_name in enumerate(bet_names):
                        key = (s_name, qlbl, t_name, b_name)

                        # Welford (daily PPD / efficiencyP)
                        v = row_ppd[bi]
                        if np.isfinite(v):
                            ppd_stats[key] = _welford_update(ppd_stats[key], float(v))
                        ev = row_eff[bi]
                        if np.isfinite(ev):
                            effp_stats[key] = _welford_update(effp_stats[key], float(ev))

                        # hit ratio counts
                        d = int(denom_hr[ti, bi])
                        if d > 0:
                            hit_den[key] += d
                            hit_num[key] += int(numer_hr[ti, bi])

                        # NEW: accumulate activity totals (daily -> running sums)
                        p = pnl_mat[ti, bi]
                        ntn = not_mat[ti, bi]
                        if np.isfinite(p):
                            sum_pnl[key] += float(p)
                        if np.isfinite(ntn):
                            sum_notional[key] += float(ntn)

                        # nrInstr / n_trades: count of rows contributing to this (t,b) today
                        b_ok = B_fin[:, bi]
                        y_ok = Y_fin[:, ti]
                        m = b_ok & y_ok
                        if m.any():
                            cnt = float(np.sum(m))
                            sum_nrInstr[key] += cnt
                            sum_ntrades[key] += cnt  # same definition here

                        # pooled regression sums (per-row) + optional sample
                        if m.any():
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
                                sampler.add(key, xs, ys)

    # --------- Finalize into nested output ---------
    out_nested = create_5d_stats()

    # Sharpe from daily PPD (Welford)
    for key, st in ppd_stats.items():
        mu, sd = _welford_finalize(*st)
        sharpe = (mu / sd * sqrt_252) if (np.isfinite(mu) and np.isfinite(sd) and sd > 0) else np.nan
        s, ql, t, b = key
        out_nested['sharpe'][s][ql][t][b] = float(sharpe) if np.isfinite(sharpe) else np.nan

    # efficiencyP mean (daily)
    for key, st in effp_stats.items():
        n, mean, _ = st
        val = float(mean) if (n > 0 and np.isfinite(mean)) else np.nan
        s, ql, t, b = key
        out_nested['efficiencyP'][s][ql][t][b] = val

    # r2 and t-stat from pooled sufficient stats
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

    # Optional: Spearman & DCOR on bounded samples
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
        ppt_val = (pnl_tot / ntrd_tot) if (np.isfinite(pnl_tot) and np.isfinite(ntrd_tot) and ntrd_tot > 0) else np.nan

        s, ql, t, b = key
        out_nested['pnl'][s][ql][t][b]          = float(pnl_tot) if np.isfinite(pnl_tot) else np.nan
        out_nested['sizeNotional'][s][ql][t][b] = float(not_tot) if np.isfinite(not_tot) else np.nan
        out_nested['nrInstr'][s][ql][t][b]      = float(nrin_tot) if np.isfinite(nrin_tot) else np.nan
        out_nested['n_trades'][s][ql][t][b]     = float(ntrd_tot) if np.isfinite(ntrd_tot) else np.nan
        out_nested['ppd'][s][ql][t][b]          = float(ppd_val) if np.isfinite(ppd_val) else np.nan
        out_nested['ppt'][s][ql][t][b]          = float(ppt_val) if np.isfinite(ppt_val) else np.nan

    return out_nested


__all__ = [
    'compute_summary_stats_over_days',
    '_distance_correlation',
]
