# =============================
# summary_stats.py
#
# SPEC: Alpha_Mark__financial_analysis_benchmark_pipeline.pdf (Cucuringu, priority)
#       AlphaMark_Guide.pdf (Patel & Li, secondary)
#
# KEY METRIC DEFINITIONS:
#   PnL          — Σ sign(si)·fi·bi  (Eq.8)
#   sizeNotional — ΣB_t = Σ_t Σ_{i∈U_t^(q), bi finite} bi  (Eq.11, target-independent)
#   PPD          — ΣPnL / ΣB_t  (Eq.12); same B_t as sizeNotional so pnl/sizeNotional==ppd exactly
#   Sharpe       — mean(PnL)/std(PnL,ddof=1)*√252 over full T days incl. zero-PnL (Eq.7)
#   hit_ratio    — fraction instruments where sign(si)=sign(fi), fi≠0, bet-independent (§5.10)
#   long_ratio   — fraction instruments where sign(si)=1, bet-independent (§5.10)
#   nrInstr      — |U_t^(q)|, si≠0 only, bet-independent (§5.10)
#   n_trades     — Σ_T |U_t^(q)|  (AlphaMark guide §3.5)
#   market_corr  — Spearman(daily PnL_t, SPY_t)
#
# NOTE on PPD consistency:
#   B_t = Σ_{i: bi finite} bi (all portfolio members with finite bet, target-independent).
#   PnL = Σ_{i: bi,fi finite} sign(si)·fi·bi (only where both b and f are finite).
#   Instruments with NaN target contribute bi to B_t but 0 to PnL.
#   Therefore PPD = ΣPnL / ΣB_t and sizeNotional = ΣB_t use the same denominator,
#   so pnl / sizeNotional == ppd exactly — no split accumulator needed.
# =============================
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Sequence, List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import spearmanr

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

from .daily_stats import create_5d_stats


def _qlabel(q: float) -> str:
    return f"qr_{int(round(q * 100))}"


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
            out.append(c)
            seen.add(c)
    return out


def _topk_mask_desc(abs_vals: np.ndarray, finite_mask: np.ndarray, q: float) -> np.ndarray:
    idx_fin = np.where(finite_mask)[0]
    out     = np.zeros_like(finite_mask, dtype=bool)
    if idx_fin.size == 0 or q <= 0.0:
        return out
    if q >= 1.0:
        out[idx_fin] = True
        return out
    k      = int(np.ceil(q * idx_fin.size))
    order  = np.argsort(-abs_vals[idx_fin], kind="mergesort")
    out[idx_fin[order[:k]]] = True
    return out


def _welford_add(state: List[float], value: float) -> List[float]:
    n, mean, M2 = state
    n    += 1
    delta = value - mean
    mean += delta / n
    M2   += delta * (value - mean)
    return [n, mean, M2]


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
    spy_by_target: Optional[Dict[str, str]],
) -> Dict:
    out = create_5d_stats()

    signal_cols   = _sanitize_list(signal_cols)
    target_cols   = _sanitize_list(target_cols)
    bet_size_cols = _sanitize_list(bet_size_cols)

    if date_col not in df.columns:
        raise KeyError(f"[summary_stats] date_col '{date_col}' not found in DataFrame.")

    want    = [date_col] + list(signal_cols) + list(target_cols) + list(bet_size_cols)
    if spy_by_target:
        want += [c for c in spy_by_target.values() if isinstance(c, str)]
    present = [c for c in want if c in df.columns]
    missing = [c for c in want if c not in df.columns]
    if missing:
        print(f"[WARN][summary_stats] Ignoring missing columns: {missing}")

    effective_spy_map: Dict[str, str] = {}
    if spy_by_target:
        for t, sc in spy_by_target.items():
            if isinstance(t, str) and isinstance(sc, str) and (t in df.columns) and (sc in df.columns):
                effective_spy_map[t] = sc

    df = df[present].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if df.empty:
        return out

    for c in [col for col in present if col != date_col]:
        df[c] = _float_clean(df[c].to_numpy())

    grouped = df.sort_values(date_col).groupby(date_col, sort=True)

    rng = np.random.default_rng(random_state)

    from collections import defaultdict as _dd

    # Welford on daily PnL — for Sharpe (Eq.7)
    pnl_welford = _dd(lambda: [0, 0.0, 0.0])

    # Welford on daily PPD — diagnostic
    ppd_stats = _dd(lambda: [0, 0.0, 0.0])

    # Regression pooled sufficient stats
    reg = _dd(lambda: {'n': 0, 'sx': 0.0, 'sy': 0.0, 'sxx': 0.0, 'syy': 0.0, 'sxy': 0.0})

    # hit_ratio: per-instrument counts, bet-independent (§5.10)
    hit_num = _dd(int)
    hit_den = _dd(int)

    # long_ratio: per-instrument counts, bet-independent (§5.10)
    # keyed by (s, q) — broadcast to all (target, bet) at finalization
    long_num = _dd(int)
    long_den = _dd(int)

    # Spearman/DCOR reservoir
    class _Reservoir:
        def __init__(self, cap: int = 0, seed: int | None = 123):
            self.cap   = int(cap) if cap and cap > 0 else 0
            self.rng   = np.random.default_rng(seed)
            self.store: Dict = {}

        def add(self, key, xs: np.ndarray, ys: np.ndarray):
            if self.cap <= 0 or xs.size == 0:
                return
            m  = min(xs.size, ys.size)
            if m == 0:
                return
            xs = xs[:m].astype('float64', copy=False)
            ys = ys[:m].astype('float64', copy=False)
            if key not in self.store:
                take = min(self.cap, m)
                idx  = self.rng.choice(m, size=take, replace=False)
                self.store[key] = (xs[idx].copy(), ys[idx].copy(), m)
                return
            X, Y, seen = self.store[key]
            total      = seen + m
            if X.size < self.cap:
                need = self.cap - X.size
                idx  = self.rng.choice(m, size=min(need, m), replace=False)
                X    = np.concatenate([X, xs[idx]])
                Y    = np.concatenate([Y, ys[idx]])
                self.store[key] = (X, Y, seen + m)
                return
            if total > 0:
                p      = self.cap / float(total)
                rcount = int(self.rng.binomial(m, p))
                if rcount > 0:
                    rep_new = self.rng.choice(m, size=rcount, replace=False)
                    rep_old = self.rng.choice(self.cap, size=rcount, replace=False)
                    X[rep_old] = xs[rep_new]
                    Y[rep_old] = ys[rep_new]
            self.store[key] = (X, Y, seen + m)

        def get(self, key):
            return self.store.get(key, (np.array([]), np.array([]), 0))[:2]

    sampler = _Reservoir(
        spearman_sample_cap_per_key if add_spearman or add_dcor else 0,
        seed=random_state,
    )

    # Daily totals
    sum_pnl      = _dd(float)   # key=(s,q,t,b)
    # sum_notional: Σ_t B_t = Σ_t Σ_{i∈U_t^(q), bi finite} bi
    # Target-independent (Eq.11). Same value used for both sizeNotional and PPD denominator.
    # This means pnl/sizeNotional == ppd exactly — no split needed.
    sum_notional = _dd(float)
    sum_nrInstr      = _dd(float)
    sum_ntrades      = _dd(float)
    count_days       = _dd(int)   # days with finite B_t (for sizeNotional mean if needed)
    count_instr_days = _dd(int)   # days with any portfolio instruments (for nrInstr mean)

    spy_pairs = _dd(lambda: ([], []))

    for dt, day in grouped:
        if day.empty:
            continue

        sig_names = [c for c in signal_cols if c in day.columns]
        tgt_names = [c for c in target_cols if c in day.columns]
        bet_names = [c for c in bet_size_cols if c in day.columns]
        if not sig_names or not tgt_names or not bet_names:
            continue

        S = np.column_stack([day[c].to_numpy() for c in sig_names])
        Y = np.column_stack([day[c].to_numpy() for c in tgt_names])
        B = np.column_stack([np.abs(day[c].to_numpy()) for c in bet_names])

        spy_val_by_t: Dict[str, float] = {}
        if effective_spy_map:
            for t_name in tgt_names:
                sc = effective_spy_map.get(t_name)
                if sc and sc in day.columns:
                    vals = np.asarray(day[sc].to_numpy(), dtype="float64")
                    fin  = vals[np.isfinite(vals)]
                    v    = float(fin.mean()) if fin.size else np.nan
                    spy_val_by_t[t_name] = v

        _, nt = Y.shape
        _, nb = B.shape

        for si, s_name in enumerate(sig_names):
            s_all = S[:, si]
            m_fin = np.isfinite(s_all)
            if not m_fin.any():
                continue

            m_nz  = (s_all != 0.0)
            m_ok  = m_fin & m_nz   # portfolio universe: finite AND nonzero signal

            sgn   = np.sign(s_all)
            abs_s = np.abs(s_all)

            if type_quantile != 'cumulative':
                sabs_ok = abs_s[m_ok]
                if sabs_ok.size:
                    K     = len(quantiles)
                    probs = np.linspace(0.0, 1.0, K + 1)
                    edges = np.nanquantile(sabs_ok, probs)
                else:
                    edges = None
            else:
                edges = None

            for q in quantiles:
                qlbl = _qlabel(q)

                if type_quantile == 'cumulative':
                    mask_q = _topk_mask_desc(abs_s, m_ok, q)
                else:
                    if edges is None or not np.isfinite(edges).all():
                        mask_q = np.zeros_like(m_ok, dtype=bool)
                    else:
                        j      = quantiles.index(q) + 1
                        lo, hi = edges[j - 1], edges[j]
                        mask_q = m_ok & (abs_s >= lo) & (abs_s <= hi)

                if not mask_q.any():
                    # Zero-PnL day: include in Sharpe (Eq.7 over full T days)
                    for t_name in tgt_names:
                        for b_name in bet_names:
                            key = (s_name, qlbl, t_name, b_name)
                            pnl_welford[key] = _welford_add(pnl_welford[key], 0.0)
                    continue

                s_q   = s_all[mask_q]
                sgn_q = sgn[mask_q]
                Y_q   = Y[mask_q, :]
                B_q   = B[mask_q, :]

                Y_fin = np.isfinite(Y_q)
                B_fin = np.isfinite(B_q)
                Yz    = np.where(Y_fin, Y_q, 0.0)
                Bz    = np.where(B_fin, B_q, 0.0)

                # PnL matrix (nt, nb): Σ sign(si)·fi·bi over instruments with finite b and f
                # NaN targets zeroed so only finite-target instruments contribute
                pnl_mat = ((Yz * sgn_q[:, None]).T @ Bz)

                # B_t per bet (nb,): Σ bi over all portfolio instruments with finite bet
                # Target-independent (Eq.11) — used for both sizeNotional and PPD denominator
                bt_per_bet = Bz.sum(axis=0)

                # PPD matrix: pnl / B_t (Eq.12 / Eq.14)
                # Using same B_t as sizeNotional so pnl/sizeNotional == ppd exactly
                ppd_mat = np.divide(
                    pnl_mat,
                    bt_per_bet[np.newaxis, :].repeat(nt, axis=0),
                    out=np.full_like(pnl_mat, np.nan),
                    where=(bt_per_bet[np.newaxis, :] > 0).repeat(nt, axis=0),
                )

                # hit_ratio — bet-independent (§5.10)
                y_sign     = np.sign(Y_q)
                nonzero    = (y_sign != 0.0) & Y_fin
                eq_sign    = ((np.sign(s_q)[:, None] == y_sign) & nonzero)
                nonzero_ct = nonzero.sum(axis=0)   # (nt,)
                eq_sign_ct = eq_sign.sum(axis=0)   # (nt,)

                # long_ratio — bet-independent (§5.10), keyed by (s,q)
                sq_key    = (s_name, qlbl)
                long_num[sq_key] += int((sgn_q > 0).sum())
                long_den[sq_key] += int(mask_q.sum())

                # nrInstr — bet-independent (§5.10): |U_t^(q)|
                nr_instr_day = int(mask_q.sum())

                for ti, t_name in enumerate(tgt_names):
                    row_ppd = ppd_mat[ti, :]
                    row_pnl = pnl_mat[ti, :]
                    for bi, b_name in enumerate(bet_names):
                        key = (s_name, qlbl, t_name, b_name)

                        # Sharpe Welford on daily PnL (Eq.7)
                        pnl_v = row_pnl[bi]
                        if np.isfinite(pnl_v):
                            pnl_welford[key] = _welford_add(pnl_welford[key], float(pnl_v))

                        # PPD Welford (diagnostic)
                        v = row_ppd[bi]
                        if np.isfinite(v):
                            n, mean, M2    = ppd_stats[key]
                            n             += 1
                            delta          = v - mean
                            mean          += delta / n
                            M2            += delta * (v - mean)
                            ppd_stats[key] = [n, mean, M2]

                        # hit_ratio accumulator — bet-independent
                        d = int(nonzero_ct[ti])
                        if d > 0:
                            hit_den[key] += d
                            hit_num[key] += int(eq_sign_ct[ti])

                        # PnL total (for global PPD = ΣPnL/ΣB_t, Eq.12)
                        p = pnl_mat[ti, bi]
                        if np.isfinite(p):
                            sum_pnl[key] += float(p)

                        # sizeNotional: B_t = Σ bi (target-independent, Eq.11)
                        # PPD denominator uses the same B_t so pnl/sizeNotional == ppd
                        bt = bt_per_bet[bi]
                        if np.isfinite(bt):
                            sum_notional[key] += float(bt)
                            count_days[key]   += 1

                        # nrInstr and n_trades: bet-independent, broadcast same value
                        sum_nrInstr[key]      += nr_instr_day
                        sum_ntrades[key]      += nr_instr_day
                        count_instr_days[key] += 1   # unconditional day count for nrInstr mean

                        # Regression sufficient stats (over instruments with finite b AND f)
                        b_ok = B_fin[:, bi]
                        y_ok = Y_fin[:, ti]
                        m    = b_ok & y_ok
                        if m.any():
                            xs  = s_q[m]
                            ys  = Y_q[m, ti]
                            sx  = float(xs.sum());    sy  = float(ys.sum())
                            sxx = float((xs*xs).sum()); syy = float((ys*ys).sum())
                            sxy = float((xs*ys).sum())
                            st  = reg[key]
                            st['n']   += xs.size
                            st['sx']  += sx;  st['sy']  += sy
                            st['sxx'] += sxx; st['syy'] += syy
                            st['sxy'] += sxy

                            if add_spearman or add_dcor:
                                cap = min(1024, spearman_sample_cap_per_key)
                                if xs.size > cap:
                                    idx = rng.choice(xs.size, size=cap, replace=False)
                                    xs  = xs[idx]; ys = ys[idx]
                                try:
                                    sampler.add(key, xs, ys)
                                except Exception:
                                    pass

                        # market_corr: per-day PnL vs SPY
                        spy_v = spy_val_by_t.get(t_name, np.nan)
                        if np.isfinite(p) and np.isfinite(spy_v):
                            pnl_ser, spy_ser = spy_pairs[key]
                            pnl_ser.append(float(p))
                            spy_ser.append(float(spy_v))

    # --------- Finalize ---------
    out_nested = create_5d_stats()

    # Sharpe — Eq.7: mean(PnL)/std(PnL,ddof=1)*√252 over full T-day series
    for key, st in pnl_welford.items():
        n, mean, M2 = st
        if n > 1:
            sd     = np.sqrt(M2 / (n - 1))
            sharpe = (mean / sd * np.sqrt(252.0)) if (np.isfinite(mean) and np.isfinite(sd) and sd > 0) else np.nan
        else:
            sharpe = np.nan
        s, ql, t, b = key
        out_nested['sharpe_ratio'][s][ql][t][b] = float(sharpe) if np.isfinite(sharpe) else np.nan

    # R² and t-stat from pooled regression sufficient stats
    eps = 1e-15
    for key, st in reg.items():
        n = st['n']
        if n >= 3:
            sx, sy, sxx, syy, sxy = st['sx'], st['sy'], st['sxx'], st['syy'], st['sxy']
            cov_xy = sxy - (sx * sy) / n
            var_x  = sxx - (sx * sx) / n
            var_y  = syy - (sy * sy) / n
            if var_x > eps and var_y > eps:
                r      = float(np.clip(cov_xy / np.sqrt(var_x * var_y), -1.0, 1.0))
                r2     = r * r
                denom  = max(eps, 1.0 - r2)
                t_stat = float(r * np.sqrt((n - 2) / denom))
            else:
                r2 = np.nan; t_stat = np.nan
        else:
            r2 = np.nan; t_stat = np.nan
        s, ql, t, b = key
        out_nested['r2'][s][ql][t][b]     = r2     if np.isfinite(r2)     else np.nan
        out_nested['t_stat'][s][ql][t][b] = t_stat if np.isfinite(t_stat) else np.nan

    # Spearman & DCOR
    if add_spearman or add_dcor:
        for key in reg.keys():
            xs, ys = sampler.get(key)
            s, ql, t, b = key
            if add_spearman:
                sp = float(spearmanr(xs, ys, nan_policy='omit').correlation) if xs.size >= 3 else np.nan
                out_nested['spearman'][s][ql][t][b] = sp if np.isfinite(sp) else np.nan
            if add_dcor:
                dc = float(_distance_correlation(xs, ys)) if xs.size >= 3 else np.nan
                out_nested['dcor'][s][ql][t][b] = dc if np.isfinite(dc) else np.nan

    # hit_ratio — bet-independent per-instrument fraction (§5.10)
    for key, hn in hit_num.items():
        hd = hit_den.get(key, 0)
        s, ql, t, b = key
        out_nested['hit_ratio'][s][ql][t][b] = (hn / hd) if hd > 0 else np.nan

    # Activity metrics
    all_keys = set(sum_pnl) | set(sum_notional) | set(sum_nrInstr) | set(sum_ntrades)

    # long_ratio — bet-independent, broadcast to all (target, bet)
    # Use all_keys (union of sum_pnl, sum_notional, etc.) to avoid missing keys
    # where ppd happened to always be NaN (e.g. zero notional days only)
    seen_tb: Dict[tuple, set] = defaultdict(set)
    for (s, ql, t, b) in all_keys:
        seen_tb[(s, ql)].add((t, b))
    for sq_key, ln in long_num.items():
        ld  = long_den.get(sq_key, 0)
        val = (ln / ld) if ld > 0 else np.nan
        s, ql = sq_key
        for (t, b) in seen_tb.get(sq_key, []):
            out_nested['long_ratio'][s][ql][t][b] = val
    for key in all_keys:
        pnl_tot  = sum_pnl.get(key, 0.0)
        not_tot  = sum_notional.get(key, 0.0)   # ΣB_t — used for BOTH sizeNotional and PPD
        nrin_tot = sum_nrInstr.get(key, 0.0)    # Σ_T |U_t^(q)| — divide by T for mean daily count
        ntrd_tot = sum_ntrades.get(key, 0.0)    # Σ_T |U_t^(q)| — total (NTrades, guide §3.5)
        T_bt   = count_days.get(key, 0)       # days with finite B_t
        T_instr = count_instr_days.get(key, 0)  # days with portfolio instruments (for nrInstr)

        # PPD = ΣPnL / ΣB_t (Eq.12); same B_t as sizeNotional so pnl/sizeNotional == ppd exactly
        ppd_val = (pnl_tot / not_tot) if (np.isfinite(pnl_tot) and np.isfinite(not_tot) and not_tot > 0) else np.nan

        # nrInstr summary = mean daily |U_t^(q)| (Cucuringu §5.10 — daily count averaged over T)
        # n_trades summary = Σ_T |U_t^(q)| (AlphaMark guide §3.5 — total across all days)
        nrin_mean = (nrin_tot / T_instr) if T_instr > 0 else np.nan

        s, ql, t, b = key
        out_nested['pnl'][s][ql][t][b]          = float(pnl_tot)   if np.isfinite(pnl_tot)   else np.nan
        out_nested['size_notional'][s][ql][t][b] = float(not_tot)   if np.isfinite(not_tot)   else np.nan
        out_nested['nr_instr'][s][ql][t][b]      = float(nrin_mean) if np.isfinite(nrin_mean) else np.nan
        out_nested['nr_trades'][s][ql][t][b]     = float(ntrd_tot)  if np.isfinite(ntrd_tot)  else np.nan
        out_nested['ppd'][s][ql][t][b]          = float(ppd_val)   if np.isfinite(ppd_val)   else np.nan

    # market_corr — Spearman(PnL_t, SPY_t)
    if effective_spy_map:
        for key, (pnl_series, spy_series) in spy_pairs.items():
            x = np.asarray(pnl_series, dtype=float)
            y = np.asarray(spy_series, dtype=float)
            if x.size >= 3 and y.size >= 3 and len(np.unique(x[np.isfinite(x)])) > 1 and len(np.unique(y[np.isfinite(y)])) > 1:
                try:
                    r = float(spearmanr(x, y, nan_policy='omit').correlation)
                    r = r if np.isfinite(r) else np.nan
                except Exception:
                    r = np.nan
            else:
                r = np.nan
            s, ql, t, b = key
            out_nested['market_corr'][s][ql][t][b] = r
            out_nested['spy_corr'][s][ql][t][b]    = r
        for key in all_keys:
            s, ql, t, b = key
            if b not in out_nested['market_corr'][s][ql].get(t, {}):
                out_nested['market_corr'][s][ql][t][b] = np.nan
                out_nested['spy_corr'][s][ql][t][b]    = np.nan

    return out_nested


def _merge_summary(dst: Dict, src: Dict) -> None:
    for stat_type, sig_tree in src.items():
        if stat_type not in dst:
            dst[stat_type] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for signal, q_tree in sig_tree.items():
            dst[stat_type][signal] = q_tree


def compute_summary_stats_over_days(
    df: pd.DataFrame,
    date_col: str,
    signal_cols: Sequence[str],
    target_cols: Sequence[str],
    quantiles: Sequence[float] = (1.0, 0.75, 0.5, 0.25),
    bet_size_cols: Sequence[str] = ('betsize_equal',),
    type_quantile: str = 'cumulative',
    add_spearman: bool = False,
    add_dcor: bool = False,
    n_jobs: int | None = None,
    backend: str = "loky",
    spearman_sample_cap_per_key: int = 10000,
    random_state: int | None = 123,
    spy_by_target: Optional[Dict[str, str]] = None,
    id_col: Optional[str] = None,
    dump_alpha_raw_corr_path: Optional[str] = None,
    dump_alpha_pnl_corr_path: Optional[str] = None,
    ccf_max_lag: int = 5,
    dump_alpha_raw_ccf_path: Optional[str] = None,
    dump_alpha_pnl_ccf_path: Optional[str] = None,
) -> Dict:
    signal_cols = _sanitize_list(signal_cols)
    if not signal_cols:
        return create_5d_stats()

    if not n_jobs or n_jobs <= 1 or len(signal_cols) == 1:
        out = _compute_summary_stats_core(
            df, date_col, signal_cols, target_cols, quantiles, bet_size_cols,
            type_quantile, add_spearman, add_dcor, spearman_sample_cap_per_key,
            random_state, spy_by_target,
        )
    else:
        n_threads = min(len(signal_cols), int(n_jobs))
        out       = create_5d_stats()

        def _chunks(lst, k):
            for i in range(k):
                yield [lst[j] for j in range(i, len(lst), k)]

        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futs = []
            rng  = np.random.default_rng(random_state)
            for sub_signals in _chunks(signal_cols, n_threads):
                if not sub_signals:
                    continue
                futs.append(ex.submit(
                    _compute_summary_stats_core,
                    df, date_col, sub_signals, target_cols, quantiles, bet_size_cols,
                    type_quantile, add_spearman, add_dcor, spearman_sample_cap_per_key,
                    None if random_state is None else int(rng.integers(0, 2**31 - 1)),
                    spy_by_target,
                ))
            for fut in as_completed(futs):
                _merge_summary(out, fut.result())

    # Per-ID corr & CCF dumps
    try:
        do_per_id = (id_col is not None) and (isinstance(id_col, str)) and (id_col in df.columns)
        have_spy  = isinstance(spy_by_target, dict) and (len(spy_by_target) > 0)
        want_corr = (dump_alpha_raw_corr_path is not None) or (dump_alpha_pnl_corr_path is not None)
        want_ccf  = (dump_alpha_raw_ccf_path  is not None) or (dump_alpha_pnl_ccf_path  is not None)

        if do_per_id and have_spy and (want_corr or want_ccf):
            import pickle as _p

            work = df.copy()
            work[date_col] = pd.to_datetime(work[date_col], errors='coerce')
            work = work.dropna(subset=[date_col, id_col])

            eff_spy_map = {t: sc for t, sc in (spy_by_target or {}).items()
                           if t in work.columns and sc in work.columns}
            if eff_spy_map:
                sigs = [c for c in signal_cols if c in work.columns]
                tgts = [c for c in target_cols  if c in work.columns]
                bets = [c for c in bet_size_cols if c in work.columns]
                if sigs and tgts and bets:
                    recs_raw = [] if (dump_alpha_raw_corr_path or dump_alpha_raw_ccf_path) else None
                    recs_pnl = [] if (dump_alpha_pnl_corr_path or dump_alpha_pnl_ccf_path) else None

                    for dt, day in work.groupby(date_col, sort=True):
                        for s_name in sigs:
                            svals = np.asarray(day[s_name], float)
                            sfin  = np.isfinite(svals) & (svals != 0.0)
                            abs_s = np.abs(svals)

                            for q in quantiles:
                                qlbl   = _qlabel(q)
                                idx_ok = np.where(sfin)[0]
                                mask_q = np.zeros_like(sfin, bool)
                                if idx_ok.size:
                                    if q >= 1.0:
                                        mask_q[idx_ok] = True
                                    else:
                                        k      = int(np.ceil(q * idx_ok.size))
                                        order  = np.argsort(-abs_s[idx_ok], kind="mergesort")
                                        mask_q[idx_ok[order[:k]]] = True
                                if not mask_q.any():
                                    continue

                                s_q   = svals[mask_q]
                                # Use iloc-style indexing: mask_q is a positional bool array;
                                # day.loc with a bool array is unreliable on non-default indexes.
                                ids_q = day[id_col].to_numpy().astype(str)[mask_q]

                                for t_name in tgts:
                                    if t_name not in eff_spy_map:
                                        continue
                                    spy_col  = eff_spy_map[t_name]
                                    spy_vals = np.asarray(day[spy_col], float)
                                    fin_spy  = spy_vals[np.isfinite(spy_vals)]
                                    spy_v    = float(fin_spy.mean()) if fin_spy.size else np.nan
                                    if not np.isfinite(spy_v):
                                        continue
                                    y    = np.asarray(day[t_name], float)[mask_q]
                                    yfin = np.isfinite(y)
                                    if not yfin.any():
                                        continue

                                    if recs_raw is not None:
                                        dfraw      = pd.DataFrame({id_col: ids_q, 'alpha_raw': s_q})
                                        raw_per_id = dfraw.groupby(id_col)['alpha_raw'].mean()
                                        for name_i, val in raw_per_id.items():
                                            if np.isfinite(val):
                                                recs_raw.append((str(name_i), s_name, qlbl, t_name,
                                                                  "__RAW__", pd.Timestamp(dt), float(val), float(spy_v)))

                                    if recs_pnl is not None:
                                        for b_name in bets:
                                            bcol    = np.asarray(day[b_name], float)[mask_q]
                                            bcol    = np.where(np.isfinite(bcol), np.abs(bcol), np.nan)
                                            pnl_row = y * np.sign(s_q) * bcol
                                            dfp     = pd.DataFrame({id_col: ids_q, 'pnl': pnl_row})
                                            for name_i, val in dfp.groupby(id_col)['pnl'].sum().items():
                                                if np.isfinite(val):
                                                    recs_pnl.append((str(name_i), s_name, qlbl, t_name,
                                                                      b_name, pd.Timestamp(dt), float(val), float(spy_v)))

                    def _dump_corr(records, out_path, metric_name):
                        if (records is None) or (out_path is None) or (len(records) == 0):
                            return
                        cols  = [id_col, 'signal', 'qrank', 'target', 'bet_size_col', 'date', 'series', 'spy']
                        dfrec = pd.DataFrame.from_records(records, columns=cols)
                        rows  = []
                        for keys, grp in dfrec.groupby([id_col, 'signal', 'qrank', 'target', 'bet_size_col'], sort=False):
                            x = pd.to_numeric(grp['series'], errors='coerce')
                            y = pd.to_numeric(grp['spy'],    errors='coerce')
                            m = x.notna() & y.notna()
                            if m.sum() >= 3 and x[m].nunique() >= 2 and y[m].nunique() >= 2:
                                r   = spearmanr(x[m], y[m], nan_policy='omit').correlation
                                val = float(r) if np.isfinite(r) else np.nan
                            else:
                                val = np.nan
                            rows.append((*keys, metric_name, val))
                        dfout = pd.DataFrame.from_records(
                            rows, columns=[id_col, 'signal', 'qrank', 'target', 'bet_size_col', 'stat_type', 'value'])
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        with open(out_path + ".tmp", "wb") as f:
                            _p.dump(dfout, f, protocol=_p.HIGHEST_PROTOCOL)
                        os.replace(out_path + ".tmp", out_path)
                        print(f"[summary_stats] Wrote per-id corr: {out_path}  ({len(dfout)} rows)")

                    def _dump_ccf(records, out_path, metric_name, max_lag: int):
                        if (records is None) or (out_path is None) or (len(records) == 0) or int(max_lag) <= 0:
                            return
                        max_lag = int(max_lag)
                        cols    = [id_col, 'signal', 'qrank', 'target', 'bet_size_col', 'date', 'series', 'spy']
                        dfrec   = pd.DataFrame.from_records(records, columns=cols)
                        rows    = []
                        for keys, grp in dfrec.groupby([id_col, 'signal', 'qrank', 'target', 'bet_size_col'], sort=False):
                            grp  = grp.copy()
                            grp['date'] = pd.to_datetime(grp['date'], errors='coerce')
                            grp  = grp.dropna(subset=['date']).sort_values('date')
                            if grp.empty:
                                continue
                            x  = pd.to_numeric(grp['series'], errors='coerce')
                            y  = pd.to_numeric(grp['spy'],    errors='coerce')
                            sx = pd.Series(x.values, index=grp['date'])
                            sy = pd.Series(y.values, index=grp['date'])
                            for L in range(-max_lag, max_lag + 1):
                                df_xy = pd.concat({'x': sx, 'y': sy.shift(-L)}, axis=1).dropna()
                                if df_xy.shape[0] < 5:
                                    continue
                                r = df_xy['x'].corr(df_xy['y'], method='spearman')
                                if np.isfinite(r):
                                    rows.append((*keys, metric_name, int(L), float(r)))
                        if not rows:
                            return
                        dfout = pd.DataFrame.from_records(
                            rows, columns=[id_col, 'signal', 'qrank', 'target', 'bet_size_col', 'stat_type', 'lag', 'corr'])
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        with open(out_path + ".tmp", "wb") as f:
                            _p.dump(dfout, f, protocol=_p.HIGHEST_PROTOCOL)
                        os.replace(out_path + ".tmp", out_path)
                        print(f"[summary_stats] Wrote per-id CCF: {out_path}  ({len(dfout)} rows)")

                    _dump_corr(recs_raw, dump_alpha_raw_corr_path, "alpha_raw_spy_corr")
                    _dump_corr(recs_pnl, dump_alpha_pnl_corr_path, "alpha_pnl_spy_corr")
                    _dump_ccf(recs_raw, dump_alpha_raw_ccf_path, "alpha_raw_spy_ccf", ccf_max_lag)
                    _dump_ccf(recs_pnl, dump_alpha_pnl_ccf_path, "alpha_pnl_spy_ccf", ccf_max_lag)

    except Exception as _e:
        print(f"[WARN][summary_stats] per-id corr/ccf dump failed: {_e}")

    return out


__all__ = [
    'compute_summary_stats_over_days',
    '_distance_correlation',
]