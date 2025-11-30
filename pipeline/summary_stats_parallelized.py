# =============================
# summary_stats.py (fast, streaming, memory-light)
# Parallel-per-signal option; efficiencyP removed.
# =============================
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Sequence, List, Dict, Tuple, Optional
from scipy.stats import spearmanr
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    # NEW for per-ticker SPY correlations (saved to separate PKLs)
    id_col: str,
    run_alpha_raw_spy_corr: bool,
    run_alpha_pnl_spy_corr: bool,
    dump_alpha_raw_corr_path: Optional[str],
    dump_alpha_pnl_corr_path: Optional[str],
    min_days_per_ticker_corr: int = 30,
):
    out = create_5d_stats()

    # Sanitize lists
    signal_cols   = _sanitize_list(signal_cols)
    target_cols   = _sanitize_list(target_cols)
    bet_size_cols = _sanitize_list(bet_size_cols)

    # Column checks
    if date_col not in df.columns:
        raise KeyError(f"[summary_stats] date_col '{date_col}' not found in DataFrame.")
    have_id = id_col in df.columns
    if (run_alpha_raw_spy_corr or run_alpha_pnl_spy_corr) and not have_id:
        print(f"[WARN][summary_stats] id_col '{id_col}' not found; per-stock SPY correlations will be skipped.")

    # Build column wish-list
    want = [date_col] + list(signal_cols) + list(target_cols) + list(bet_size_cols)
    if spy_by_target:
        want += [c for c in spy_by_target.values() if isinstance(c, str)]
    if have_id:
        want.append(id_col)

    present = [c for c in want if c in df.columns]
    missing = [c for c in want if c not in df.columns]
    if missing:
        print(f"[WARN][summary_stats] Ignoring missing columns: {missing}")

    # Effective target->SPY mapping
    effective_spy_map: Dict[str, str] = {}
    if spy_by_target:
        for t, sc in spy_by_target.items():
            if isinstance(t, str) and isinstance(sc, str) and (t in df.columns) and (sc in df.columns):
                effective_spy_map[t] = sc

    df = df[present].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if df.empty:
        # still emit empty per-ticker PKLs if asked
        _maybe_dump_empty_per_ticker_pkls(dump_alpha_raw_corr_path, dump_alpha_pnl_corr_path)
        return out

    # Clean numeric once
    for c in [c for c in present if c not in (date_col, id_col)]:
        df[c] = _float_clean(df[c].to_numpy())

    grouped = df.sort_values(date_col).groupby(date_col, sort=True)

    # --------- Streaming accumulators (existing) ---------
    rng = np.random.default_rng(random_state)
    sqrt_252 = np.sqrt(252.0)

    # Welford stats for daily PPD
    ppd_stats = defaultdict(lambda: [0, 0.0, 0.0])   # key=(s,q,t,b)

    # Regression pooled sufficient stats
    reg = defaultdict(lambda: {'n':0, 'sx':0.0, 'sy':0.0, 'sxx':0.0, 'syy':0.0, 'sxy':0.0})

    # Hit / long ratios
    hit_num = defaultdict(int)         # key=(s,q,t,b)
    hit_den = defaultdict(int)
    long_num = defaultdict(int)        # key=(s,q,b)
    long_den = defaultdict(int)

    # Spearman/DCOR reservoir (signal vs target per-row)
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

    # Daily totals accumulators (summary totals over days)
    sum_pnl       = defaultdict(float)  # key=(s,q,t,b)
    sum_notional  = defaultdict(float)
    sum_nrInstr   = defaultdict(float)
    sum_ntrades   = defaultdict(float)

    # Per-key daily strategy PnL vs SPY (same-horizon)
    spy_pairs = defaultdict(lambda: ([], []))  # key=(s,q,t,b) -> (list pnl_day, list spy_day)

    # --------- NEW: per-stock across-days collections (no quantiles) ---------
    # raw alpha vs SPY: map (signal, target) -> id -> (alpha_series, spy_series)
    alpha_raw_pairs: Dict[Tuple[str, str], Dict[str, Tuple[List[float], List[float]]]] = \
        defaultdict(lambda: defaultdict(lambda: ([], [])))
    # per-stock PnL vs SPY: map (signal, target, bet) -> id -> (pnl_series, spy_series)
    pnl_stock_pairs: Dict[Tuple[str, str, str], Dict[str, Tuple[List[float], List[float]]]] = \
        defaultdict(lambda: defaultdict(lambda: ([], [])))

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

        # Per-day SPY value per target (same horizon) — use daily mean across rows
        spy_val_by_t: Dict[str, float] = {}
        if effective_spy_map:
            for t_name in tgt_names:
                sc = effective_spy_map.get(t_name)
                if sc and sc in day.columns:
                    vals = np.asarray(day[sc].to_numpy(), dtype="float64")
                    v = np.nanmean(vals) if vals.size else np.nan
                    spy_val_by_t[t_name] = float(v) if np.isfinite(v) else np.nan

        ids = day[id_col].astype(str).to_numpy() if have_id else None

        n, ns = S.shape
        _, nt = Y.shape
        _, nb = B.shape

        # ===== Existing quantile-based summary stream (unchanged) =====
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

                s_q   = s_all[mask_q]
                sgn_q = sgn[mask_q]
                Y_q   = Y[mask_q, :]
                B_q   = B[mask_q, :]

                Y_fin = np.isfinite(Y_q)
                B_fin = np.isfinite(B_q)
                Yz    = np.where(Y_fin, Y_q, 0.0)
                Bz    = np.where(B_fin, B_q, 0.0)

                # Daily PnL / Notional matrices
                pnl_mat = ((Yz * sgn_q[:, None]).T @ Bz)       # (nt, nb)
                not_mat = (Y_fin.astype(float).T @ Bz)         # (nt, nb)

                # PPD matrix (daily)
                ppd_mat = np.divide(pnl_mat, not_mat,
                                    out=np.full_like(pnl_mat, np.nan),
                                    where=(not_mat > 0))

                # hit ratio pieces
                y_sign   = np.sign(Y_q)
                nonzero  = (y_sign != 0.0) & Y_fin
                denom_hr = (nonzero.astype(float).T @ B_fin.astype(float))   # (nt, nb)
                eq_sign  = ((np.sign(s_q)[:, None] == y_sign) & nonzero)
                numer_hr = (eq_sign.astype(float).T @ B_fin.astype(float))   # (nt, nb)

                # long ratio (per bet)
                long_den_vec = np.sum(B_fin, axis=0).astype(int)
                long_num_vec = np.sum((sgn_q > 0)[:, None] & B_fin, axis=0).astype(int)
                for bi, b_name in enumerate(bet_names):
                    if long_den_vec[bi] > 0:
                        long_den[(s_name, qlbl, b_name)] += int(long_den_vec[bi])
                        long_num[(s_name, qlbl, b_name)] += int(long_num_vec[bi])

                # update all per (target, bet)
                for ti, t_name in enumerate(tgt_names):
                    row_ppd = ppd_mat[ti, :]
                    for bi, b_name in enumerate(bet_names):
                        key = (s_name, qlbl, t_name, b_name)

                        # Welford (daily PPD)
                        v = row_ppd[bi]
                        if np.isfinite(v):
                            n0, mean, M2 = ppd_stats[key]
                            n0 += 1
                            delta = v - mean
                            mean += delta / n0
                            M2 += delta * (v - mean)
                            ppd_stats[key] = [n0, mean, M2]

                        # hit ratio counts
                        d = int(denom_hr[ti, bi])
                        if d > 0:
                            hit_den[key] += d
                            hit_num[key] += int(numer_hr[ti, bi])

                        # activity totals
                        p = pnl_mat[ti, bi]
                        ntn = not_mat[ti, bi]
                        if np.isfinite(p):
                            sum_pnl[key] += float(p)
                        if np.isfinite(ntn):
                            sum_notional[key] += float(ntn)

                        # nrInstr / n_trades as contributing rows
                        b_ok = B_fin[:, bi]
                        y_ok = Y_fin[:, ti]
                        m = b_ok & y_ok
                        if m.any():
                            cnt = float(np.sum(m))
                            sum_nrInstr[key] += cnt
                            sum_ntrades[key] += cnt

                            # pooled regression + optional sample (signal vs target)
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

                        # per-day strat PnL vs SPY (same horizon)
                        spy_v = spy_val_by_t.get(t_name, np.nan)
                        if np.isfinite(p) and np.isfinite(spy_v):
                            pnl_ser, spy_ser = spy_pairs[key]
                            pnl_ser.append(float(p))
                            spy_ser.append(float(spy_v))

        # ===== NEW: per-ticker corr data capture (no quantiles) =====
        if have_id and effective_spy_map and (run_alpha_raw_spy_corr or run_alpha_pnl_spy_corr):
            id_arr = ids  # (n,)
            for ti, t_name in enumerate(tgt_names):
                spy_v = spy_val_by_t.get(t_name, np.nan)
                if not np.isfinite(spy_v):
                    continue

                # (2) Raw alpha vs SPY (per-signal)
                if run_alpha_raw_spy_corr:
                    for si, s_name in enumerate(sig_names):
                        x = S[:, si]
                        m = np.isfinite(x)
                        if not m.any():
                            continue
                        # append per id
                        for iid, xv in zip(id_arr[m], x[m]):
                            a, b = alpha_raw_pairs[(s_name, t_name)][iid]
                            a.append(float(xv))
                            b.append(float(spy_v))

                # (3) Per-stock PnL vs SPY (per-signal, per-bet)
                if run_alpha_pnl_spy_corr:
                    for si, s_name in enumerate(sig_names):
                        sgn_all = np.sign(S[:, si])
                        y = Y[:, ti]
                        if not np.isfinite(y).any():
                            continue
                        for bi, b_name in enumerate(bet_names):
                            betv = B[:, bi]
                            # pnl per row = sign(signal) * target * |bet|
                            pnl_row = sgn_all * y * betv
                            m = np.isfinite(pnl_row)
                            if not m.any():
                                continue
                            for iid, pv in zip(id_arr[m], pnl_row[m]):
                                a, b = pnl_stock_pairs[(s_name, t_name, b_name)][iid]
                                a.append(float(pv))
                                b.append(float(spy_v))

    # --------- Finalize into nested SUMMARY (existing) ---------
    # Sharpe from daily PPD
    sqrt_252 = np.sqrt(252.0)
    for key, st in ppd_stats.items():
        n0, mean, M2 = st
        mu, sd = (mean, np.sqrt(M2 / n0)) if n0 > 1 else (mean, np.nan)
        sharpe = (mu / sd * sqrt_252) if (np.isfinite(mu) and np.isfinite(sd) and sd > 0) else np.nan
        s, ql, t, b = key
        out['sharpe'][s][ql][t][b] = float(sharpe) if np.isfinite(sharpe) else np.nan

    # r2 and t-stat
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
        out['r2'][s][ql][t][b] = r2 if np.isfinite(r2) else np.nan
        out['t_stat'][s][ql][t][b] = t_stat if np.isfinite(t_stat) else np.nan

    # Optional: Spearman/DCOR samples
    if add_spearman or add_dcor:
        for key in reg.keys():
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
                out['spearman'][s][ql][t][b] = sp if np.isfinite(sp) else np.nan
            if add_dcor:
                if xs.size >= 3 and ys.size >= 3:
                    try:
                        dc = float(_distance_correlation(xs, ys))
                    except Exception:
                        dc = np.nan
                else:
                    dc = np.nan
                out['dcor'][s][ql][t][b] = dc if np.isfinite(dc) else np.nan

    # hit ratio
    for key, hn in hit_num.items():
        hd = hit_den.get(key, 0)
        s, ql, t, b = key
        out['hit_ratio'][s][ql][t][b] = (hn / hd) if hd > 0 else np.nan

    # long ratio (per bet) -> broadcast to all targets we saw
    seen_targets_per_sqb = defaultdict(set)
    for (s, ql, t, b) in ppd_stats.keys():
        seen_targets_per_sqb[(s, ql, b)].add(t)

    for (s, ql, b), ln in long_num.items():
        ld = long_den.get((s, ql, b), 0)
        val = (ln / ld) if ld > 0 else np.nan
        for t in seen_targets_per_sqb.get((s, ql, b), []):
            out['long_ratio'][s][ql][t][b] = val

    # Activity totals to SUMMARY (totals + ratios)
    all_keys = set(sum_pnl) | set(sum_notional) | set(sum_nrInstr) | set(sum_ntrades)
    for key in all_keys:
        pnl_tot  = sum_pnl.get(key, 0.0)
        not_tot  = sum_notional.get(key, 0.0)
        nrin_tot = sum_nrInstr.get(key, 0.0)
        ntrd_tot = sum_ntrades.get(key, 0.0)
        ppd_val = (pnl_tot / not_tot) if (np.isfinite(pnl_tot) and np.isfinite(not_tot) and not_tot > 0) else np.nan
        s, ql, t, b = key
        out['pnl'][s][ql][t][b]          = float(pnl_tot) if np.isfinite(pnl_tot) else np.nan
        out['sizeNotional'][s][ql][t][b] = float(not_tot) if np.isfinite(not_tot) else np.nan
        out['nrInstr'][s][ql][t][b]      = float(nrin_tot) if np.isfinite(nrin_tot) else np.nan
        out['n_trades'][s][ql][t][b]     = float(ntrd_tot) if np.isfinite(ntrd_tot) else np.nan
        out['ppd'][s][ql][t][b]          = float(ppd_val) if np.isfinite(ppd_val) else np.nan

    # SUMMARY: Spearman corr between per-day strategy PnL and same-horizon SPY (per (s,q,t,b))
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
            out['market_corr'][s][ql][t][b] = r
            out['spy_corr'][s][ql][t][b] = r  # backward compatibility
        # Guarantee entries exist for every observed (s, q, t, b), even if the SPY
        # horizon was missing on some days (so bars don't drop those targets).
        for key in all_keys:
            s, ql, t, b = key
            if b not in out['market_corr'][s][ql].get(t, {}):
                out['market_corr'][s][ql][t][b] = np.nan
                out['spy_corr'][s][ql][t][b] = np.nan

    # --------- NEW: finalize and dump per-ticker PKLs ---------
    if have_id and effective_spy_map:
        # (2) Raw alpha vs SPY
        if run_alpha_raw_spy_corr and dump_alpha_raw_corr_path:
            df_raw = _pairs_to_corr_df_alpha_raw(alpha_raw_pairs, min_days_per_ticker_corr)
            _safe_dump_pickle(df_raw, dump_alpha_raw_corr_path)

        # (3) Per-stock PnL vs SPY
        if run_alpha_pnl_spy_corr and dump_alpha_pnl_corr_path:
            df_pnl = _pairs_to_corr_df_pnl_stock(pnl_stock_pairs, min_days_per_ticker_corr)
            _safe_dump_pickle(df_pnl, dump_alpha_pnl_corr_path)

    return out


def _pairs_to_corr_df_alpha_raw(alpha_raw_pairs, min_days: int) -> pd.DataFrame:
    rows = []
    for (signal, target), bucket in alpha_raw_pairs.items():
        for iid, (xs, ys) in bucket.items():
            x = np.asarray(xs, dtype=float); y = np.asarray(ys, dtype=float)
            n = int(min(len(x), len(y)))
            r = np.nan
            if n >= 3 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
                try:
                    r = float(spearmanr(x, y, nan_policy='omit').correlation)
                except Exception:
                    r = np.nan
            rows.append((iid, signal, target, r, n))
    df = pd.DataFrame(rows, columns=["id","signal","target","corr","n_days"])
    # filter min_days if provided
    if min_days and min_days > 0:
        df = df[df["n_days"] >= int(min_days)]
    df.insert(0, "corr_type", "alpha_raw_spy")
    return df.reset_index(drop=True)


def _pairs_to_corr_df_pnl_stock(pnl_stock_pairs, min_days: int) -> pd.DataFrame:
    rows = []
    for (signal, target, bet), bucket in pnl_stock_pairs.items():
        for iid, (xs, ys) in bucket.items():
            x = np.asarray(xs, dtype=float); y = np.asarray(ys, dtype=float)
            n = int(min(len(x), len(y)))
            r = np.nan
            if n >= 3 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
                try:
                    r = float(spearmanr(x, y, nan_policy='omit').correlation)
                except Exception:
                    r = np.nan
            rows.append((iid, signal, target, bet, r, n))
    df = pd.DataFrame(rows, columns=["id","signal","target","bet_size_col","corr","n_days"])
    if min_days and min_days > 0:
        df = df[df["n_days"] >= int(min_days)]
    df.insert(0, "corr_type", "alpha_pnl_spy")
    return df.reset_index(drop=True)


def _safe_dump_pickle(df: pd.DataFrame, path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_pickle(path)
        print(f"[summary_stats] Wrote per-ticker corr PKL: {path}  rows={len(df)}")
    except Exception as e:
        print(f"[WARN][summary_stats] Failed to write {path}: {e}")


def _maybe_dump_empty_per_ticker_pkls(path_raw: Optional[str], path_pnl: Optional[str]) -> None:
    if path_raw:
        _safe_dump_pickle(pd.DataFrame(columns=["corr_type","id","signal","target","corr","n_days"]), path_raw)
    if path_pnl:
        _safe_dump_pickle(pd.DataFrame(columns=["corr_type","id","signal","target","bet_size_col","corr","n_days"]), path_pnl)


def _merge_summary(dst: Dict, src: Dict) -> None:
    """Merge nested stats dicts whose SIGNAL subtrees are disjoint."""
    for stat_type, sig_tree in src.items():
        if stat_type not in dst:
            dst[stat_type] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for signal, q_tree in sig_tree.items():
            dst[stat_type][signal] = q_tree


# ===================== PUBLIC API (optionally parallel) ====================
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
    n_jobs: int | None = None,           # threads per-signal
    backend: str = "loky",               # kept for compat
    spearman_sample_cap_per_key: int = 10000,
    random_state: int | None = 123,
    spy_by_target: Optional[Dict[str, str]] = None,  # {target -> spy column}

    # NEW controls/outputs
    id_col: str = "ticker",
    run_alpha_raw_spy_corr: bool = True,
    run_alpha_pnl_spy_corr: bool = True,
    dump_alpha_raw_corr_path: Optional[str] = None,
    dump_alpha_pnl_corr_path: Optional[str] = None,
    min_days_per_ticker_corr: int = 30,
):
    """
    Returns ONE summary value per (signal, qrank, target, bet). If `spy_by_target` is
    provided, also writes per-ticker correlation PKLs when the dump paths are given:

      * alpha_raw_spy PKL: columns = [corr_type, id, signal, target, corr, n_days]
      * alpha_pnl_spy PKL: columns = [corr_type, id, signal, target, bet_size_col, corr, n_days]

    These per-ticker files contain one row per (stock id, ...) with its across-days
    Spearman correlation vs same-horizon SPY.
    """
    signal_cols = _sanitize_list(signal_cols)
    if not signal_cols:
        return create_5d_stats()

    if not n_jobs or n_jobs <= 1 or len(signal_cols) == 1:
        return _compute_summary_stats_core(
            df, date_col, signal_cols, target_cols, quantiles, bet_size_cols,
            type_quantile, add_spearman, add_dcor, spearman_sample_cap_per_key,
            random_state, spy_by_target,
            id_col, run_alpha_raw_spy_corr, run_alpha_pnl_spy_corr,
            dump_alpha_raw_corr_path, dump_alpha_pnl_corr_path,
            min_days_per_ticker_corr
        )

    # Parallel across signals (we still emit single per-ticker files; each thread writes nothing,
    # and only the aggregator thread should write. To keep it simple and safe, when n_jobs>1 we
    # disable writing inside threads and only write from the merged result set — but per-ticker
    # pairs are built inside the core. To avoid complex cross-thread merging, we run single-thread
    # when dumps are requested.)
    if dump_alpha_raw_corr_path or dump_alpha_pnl_corr_path:
        print("[summary_stats] Per-ticker dump requested — forcing single-threaded run for correctness.")
        return _compute_summary_stats_core(
            df, date_col, signal_cols, target_cols, quantiles, bet_size_cols,
            type_quantile, add_spearman, add_dcor, spearman_sample_cap_per_key,
            random_state, spy_by_target,
            id_col, run_alpha_raw_spy_corr, run_alpha_pnl_spy_corr,
            dump_alpha_raw_corr_path, dump_alpha_pnl_corr_path,
            min_days_per_ticker_corr
        )

    # Otherwise safe to parallelize for the SUMMARY dict only
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
                spy_by_target,
                id_col,
                False,  # per-ticker off in threads
                False,
                None,
                None,
                min_days_per_ticker_corr
            ))
        for fut in as_completed(futs):
            _merge_summary(out, fut.result())

    return out


__all__ = [
    'compute_summary_stats_over_days',
    '_distance_correlation',
]
