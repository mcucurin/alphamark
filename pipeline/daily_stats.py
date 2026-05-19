# =============================
# daily_stats.py — stateless n_trades = Σ_T |U_t^(q)|, no carry/position-diff.
#
# SPEC: Alpha_Mark__financial_analysis_benchmark_pipeline.pdf (Cucuringu, priority)
#       AlphaMark_Guide.pdf (Patel & Li, secondary — followed where no contradiction)
#
# KEY METRIC DEFINITIONS (Cucuringu §5.10, priority):
#   PnL          — Σ sign(si) · fi · bi  (instruments with finite s, b, f)
#   sizeNotional — B_t = Σ bi over U_t^(q) with finite bet (Eq.11, target-independent)
#   ppd          — PnL / B_t  (Eq.14 daily local PPD; same B_t as sizeNotional)
#   nrInstr      — |U_t^(q)|, si≠0 only, bet-independent
#   hit_ratio    — fraction where sign(si)=sign(fi), fi≠0, bet-independent
#   long_ratio   — fraction where sign(si)=1, bet-independent
#   Sharpe       — cross-sectional proxy only; true Sharpe (Eq.7) in summary_stats
#   n_trades     — Σ_T |U_t^(q)| (simple portfolio size, spec definition)
#
# NOTE on PPD consistency:
#   sizeNotional = B_t = Σ_{i: bi finite} bi (all portfolio members with finite bet).
#   ppd          = PnL / B_t  using the same B_t.
#   Instruments with missing target (fi=NaN) contribute bi to B_t but 0 to PnL
#   (NaN targets are zeroed in the PnL sum). This means ppd = pnl / sizeNotional
#   exactly, so verification checks pnl/sizeNotional == ppd will pass.
# =============================
from __future__ import annotations

import math
import numpy as np
from typing import Dict, Sequence
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


def create_5d_stats():
    return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))


def _label_for_quantile(q: float) -> str:
    return f'qr_{int(round(q * 100))}'


def _topk_mask_desc(abs_vals: np.ndarray, valid_mask: np.ndarray, q: float) -> np.ndarray:
    """Top 100q% by |signal| from valid_mask. Stable (mergesort) for deterministic ties."""
    idx = np.where(valid_mask)[0]
    out = np.zeros_like(valid_mask, dtype=bool)
    if idx.size == 0 or q <= 0.0:
        return out
    if q >= 1.0:
        out[idx] = True
        return out
    k = int(np.ceil(q * idx.size))
    order = np.argsort(-abs_vals[idx], kind="mergesort")
    out[idx[order[:k]]] = True
    return out


def _merge_signal_branch(dst: Dict, src: Dict, signal: str) -> None:
    for stat_type in src:
        if stat_type not in dst:
            dst[stat_type] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        dst[stat_type][signal] = src[stat_type][signal]


def _compute_daily_stats_for_one_signal(
    signal: str,
    df_np: Dict[str, np.ndarray],
    id_arr: np.ndarray | None,
    target_cols: Sequence[str],
    quantiles: Sequence[float],
    bet_size_cols: Sequence[str],
    type_quantile: str,
    enable_distributions: bool,
    max_dist_samples_per_series: int,
    rng_state: int | None,
) -> Dict:
    rng   = np.random.default_rng(rng_state)
    stats = create_5d_stats()

    s = df_np.get(signal)
    if s is None or s.size == 0:
        return stats

    m_fin = np.isfinite(s)
    m_nz  = (s != 0.0)
    # Portfolio universe: finite AND nonzero signal (Cucuringu §5.10)
    m_ok  = m_fin & m_nz
    if not m_fin.any():
        return stats

    sgn   = np.sign(s)
    abs_s = np.abs(s)

    bet_abs: Dict[str, np.ndarray] = {}
    for b_name in bet_size_cols:
        x = df_np.get(b_name)
        bet_abs[b_name] = np.abs(x) if x is not None else np.full(s.size, np.nan, dtype='float64')

    if type_quantile == 'quantEach':
        K     = len(quantiles)
        probs = np.linspace(0.0, 1.0, K + 1, dtype='float64')
        edges = np.nanquantile(abs_s[m_ok], probs) if m_ok.any() else np.array([np.nan, np.nan])
    else:
        edges = None

    alpha_written = set()

    for q in quantiles:
        qlabel = _label_for_quantile(q)

        if type_quantile == 'cumulative':
            mask_q = _topk_mask_desc(abs_s, m_ok, q)
        else:
            if edges is None or not np.isfinite(edges).all():
                mask_q = np.zeros_like(m_ok, dtype=bool)
            else:
                j  = quantiles.index(q) + 1
                lo = edges[j - 1]
                hi = edges[j - 1] if j == len(quantiles) else edges[j]
                mask_q = (m_ok & (abs_s >= lo)) if j == len(quantiles) else (m_ok & (abs_s >= lo) & (abs_s <= hi))

        alpha_sum_today = float(np.nansum(s[mask_q])) if mask_q.any() else 0.0
        if qlabel not in alpha_written:
            stats['alpha_sum'][signal][qlabel]['__ALL__']['__ALL__'] = alpha_sum_today
            alpha_written.add(qlabel)

        # nrInstr — bet-independent: count si≠0 (Cucuringu §5.10)
        if id_arr is not None:
            if mask_q.any():
                ids = id_arr[mask_q]
                nr_instr_today = int(np.unique(ids[ids == ids]).size)
            else:
                nr_instr_today = 0
        else:
            nr_instr_today = int(mask_q.sum()) if mask_q.any() else 0

        # long_ratio — bet-independent: fraction sign(si)=1 (Cucuringu §5.10)
        long_ratio_q = float(np.nanmean(sgn[mask_q] > 0)) if mask_q.any() else np.nan

        for bet in bet_size_cols:
            if bet == "__ALL__":
                continue
            b = bet_abs.get(bet)
            if b is None:
                continue
            b_fin   = np.isfinite(b)
            mask_qb = mask_q & b_fin   # portfolio with finite bet

            # B_t = Σ_{i∈mask_qb} bi — target-independent (Eq.11)
            # Used for both sizeNotional and as PPD denominator so pnl/sizeNotional == ppd exactly.
            Bt = float(np.nansum(b[mask_qb]))

            # n_trades = |U_t^(q)| — simple portfolio size per spec definition
            n_trades_today = float(mask_qb.sum())

            for target in target_cols:
                if target == "__ALL__":
                    continue
                y = df_np.get(target)
                if y is None:
                    continue
                y_fin = np.isfinite(y)

                # m: finite bet + finite target → used for PnL, regression
                m = mask_qb & y_fin

                # m_hit: finite signal(≠0) + finite target, NO bet — for hit_ratio only.
                m_hit = mask_q & y_fin

                # n_trades per (target, bet) = |m| — simple portfolio size
                n_trades_tgt = float(m.sum())

                # -------------------------------------------------------
                # PnL: Σ sign(si)·fi·bi over instruments with finite s, b, f.
                # sizeNotional: B_t = Σ bi over mask_qb (all finite-bet instruments),
                # target-independent per Eq.11. Same B_t used as PPD denominator so
                # pnl / sizeNotional == ppd exactly (verification-safe).
                # -------------------------------------------------------
                if m.any():
                    pnl_vec  = sgn[m] * y[m] * b[m]
                    pnl      = float(np.nansum(pnl_vec))
                else:
                    pnl_vec  = np.array([], dtype='float64')
                    pnl      = 0.0

                # PPD = PnL / B_t (Eq.14). B_t = Bt (computed above, target-independent).
                ppd = (pnl / Bt) if Bt > 0 else np.nan

                # hit_ratio — bet-independent (Cucuringu §5.10)
                if m_hit.any():
                    y_hit     = y[m_hit]
                    s_hit     = sgn[m_hit]
                    nonzero_y = y_hit != 0.0
                    hit_ratio = float(np.mean(s_hit[nonzero_y] == np.sign(y_hit[nonzero_y]))) if nonzero_y.any() else np.nan
                else:
                    hit_ratio = np.nan

                if m.any():
                    # Cross-sectional regression (y on s) over bet-filtered portfolio
                    n      = int(m.sum())
                    s_vals = s[m].astype('float64', copy=False)
                    y_vals = y[m].astype('float64', copy=False)
                    s_mean = np.nanmean(s_vals); y_mean = np.nanmean(y_vals)
                    s_dev  = s_vals - s_mean;    y_dev  = y_vals - y_mean
                    s_var  = float(np.nanvar(s_vals, ddof=0)) if n > 1 else np.nan
                    cov    = float(np.nanmean(s_dev * y_dev)) if n > 0 else np.nan
                    s_std  = math.sqrt(s_var) if np.isfinite(s_var) and s_var > 0 else np.nan
                    y_std  = float(np.nanstd(y_vals, ddof=0)) if n > 1 else np.nan
                    r      = (cov / (s_std * y_std)) if (np.isfinite(cov) and s_std and y_std and s_std > 0 and y_std > 0) else np.nan
                    r2     = float(r * r) if np.isfinite(r) else np.nan
                    t_stat = (r * math.sqrt(n - 2) / math.sqrt(max(1e-15, 1.0 - r * r))) if (np.isfinite(r) and n > 2 and (1.0 - r * r) > 0) else np.nan
                    # Cross-sectional Sharpe proxy only — true annualised Sharpe (Eq.7) in summary_stats
                    sharpe = (np.nanmean(pnl_vec) / np.nanstd(pnl_vec, ddof=1)) if (pnl_vec.size > 1 and np.nanstd(pnl_vec, ddof=1) > 0) else np.nan
                else:
                    r2 = np.nan; t_stat = np.nan; sharpe = np.nan

                stats['pnl'][signal][qlabel][target][bet]          = pnl
                stats['ppd'][signal][qlabel][target][bet]          = ppd
                stats['sizeNotional'][signal][qlabel][target][bet] = Bt
                stats['nrInstr'][signal][qlabel][target][bet]      = nr_instr_today
                stats['n_trades'][signal][qlabel][target][bet]     = n_trades_tgt
                stats['hit_ratio'][signal][qlabel][target][bet]    = hit_ratio
                stats['long_ratio'][signal][qlabel][target][bet]   = long_ratio_q
                stats['r2'][signal][qlabel][target][bet]           = r2
                stats['t_stat'][signal][qlabel][target][bet]       = t_stat
                stats['sharpe'][signal][qlabel][target][bet]       = sharpe

            # __ALL__ target aggregates
            stats['nrInstr'][signal][qlabel]['__ALL__'][bet]  = nr_instr_today
            stats['n_trades'][signal][qlabel]['__ALL__'][bet] = n_trades_today

    if enable_distributions:
        for target in target_cols:
            y = df_np.get(target)
            if y is None:
                continue
            y = y[np.isfinite(y)]
            if y.size == 0:
                continue
            if y.size > max_dist_samples_per_series:
                y = y[rng.choice(y.size, size=max_dist_samples_per_series, replace=False)]
            stats['fret_value'][f'__S__{signal}']['__ALL__'][target]['__ALL__'] = float(np.nanmean(y))
        for bet in bet_size_cols:
            b = df_np.get(bet)
            if b is None:
                continue
            x = np.abs(b[np.isfinite(b)])
            if x.size == 0:
                continue
            if x.size > max_dist_samples_per_series:
                x = x[rng.choice(x.size, size=max_dist_samples_per_series, replace=False)]
            stats['betsize_value'][f'__B__{bet}']['__ALL__']['__ALL__'][bet] = float(np.nanmean(x))

    return stats


def compute_daily_stats(
    df,
    signal_cols: Sequence[str],
    target_cols: Sequence[str],
    quantiles: Sequence[float] = (1.0, 0.75, 0.5, 0.25),
    bet_size_cols: Sequence[str] = ('betsize_equal',),
    type_quantile: str = 'cumulative',
    enable_distributions: bool = False,
    max_dist_samples_per_series: int = 50_000,
    random_state=None,
    n_jobs: int = 1,
):
    import pandas as pd

    rng   = np.random.default_rng(random_state)
    stats = create_5d_stats()

    id_col = 'ticker' if 'ticker' in df.columns else None
    want_numeric = (set(signal_cols) | set(target_cols) | set(bet_size_cols)) - ({id_col} if id_col else set())
    df_np: Dict[str, np.ndarray] = {}

    for col in want_numeric:
        if col not in df.columns:
            continue
        arr = pd.to_numeric(df[col], errors='coerce').to_numpy().astype('float64', copy=False)
        arr[~np.isfinite(arr)] = np.nan
        df_np[col] = arr

    id_arr = df[id_col].to_numpy() if id_col else None

    if len(df) == 0:
        return stats

    signals   = [s for s in signal_cols if s in df_np]
    n_threads = min(max(1, int(n_jobs or 1)), len(signals) or 1)

    if n_threads == 1:
        for signal in signals:
            stats_sig = _compute_daily_stats_for_one_signal(
                signal=signal, df_np=df_np, id_arr=id_arr,
                target_cols=target_cols, quantiles=quantiles,
                bet_size_cols=bet_size_cols, type_quantile=type_quantile,
                enable_distributions=enable_distributions,
                max_dist_samples_per_series=max_dist_samples_per_series,
                rng_state=None if random_state is None else int(rng.integers(0, 2**31 - 1)),
            )
            _merge_signal_branch(stats, stats_sig, signal)
    else:
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futs = []
            for signal in signals:
                futs.append(ex.submit(
                    _compute_daily_stats_for_one_signal,
                    signal, df_np, id_arr, target_cols, quantiles, bet_size_cols,
                    type_quantile, enable_distributions, max_dist_samples_per_series,
                    None if random_state is None else int(rng.integers(0, 2**31 - 1)),
                ))
            for fut in as_completed(futs):
                stats_sig = fut.result()
                try:
                    sig_name = next(iter(next(iter(stats_sig.values())).keys()))
                except Exception:
                    sig_name = signals[0] if signals else "__SIG__"
                _merge_signal_branch(stats, stats_sig, sig_name)

    return stats


def compute_series_continuous(df_sorted_by_date, date_col: str, **kwargs):
    import pandas as pd
    kwargs.pop('prev_state', None)   # accepted for back-compat, ignored
    out = []
    for d, df_day in df_sorted_by_date.sort_values(date_col).groupby(date_col):
        out.append((pd.Timestamp(d), compute_daily_stats(df_day, **kwargs)))
    return out


__all__ = [
    'compute_daily_stats',
    'compute_series_continuous',
    'create_5d_stats',
]
