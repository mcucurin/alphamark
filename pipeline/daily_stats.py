# =============================
# daily_stats.py — year-aware n_trades (carry + year-open override), no PPT
# Parallel-per-signal (threads) with truthful state merging.
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
#   n_trades     — position-diff diagnostic
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
import os
import pickle
import numpy as np
from typing import Dict, MutableMapping, Sequence, Tuple, List
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


def create_5d_stats():
    return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))


_GLOBAL_PREV_STATE: Dict = {}


def get_trading_state():
    return _GLOBAL_PREV_STATE


def reset_trading_state():
    pass


def save_trading_state(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(_GLOBAL_PREV_STATE, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_trading_state(path: str, strict: bool = False):
    if not os.path.isfile(path):
        if strict:
            raise FileNotFoundError(f"No trading state at {path}")
        return
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        _GLOBAL_PREV_STATE.clear()
        _GLOBAL_PREV_STATE.update(obj)


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
    empty_day_policy: str,
    report_empty_trades_as_nan: bool,
    prev_state_slice: MutableMapping,
) -> Tuple[Dict, Dict]:
    rng   = np.random.default_rng(rng_state)
    stats = create_5d_stats()

    s = df_np.get(signal)
    if s is None or s.size == 0:
        return stats, prev_state_slice

    m_fin = np.isfinite(s)
    m_nz  = (s != 0.0)
    # Portfolio universe: finite AND nonzero signal (Cucuringu §5.10)
    m_ok  = m_fin & m_nz
    if not m_fin.any():
        return stats, prev_state_slice

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

            key_sb   = (signal, qlabel, bet)
            prev     = prev_state_slice.get(key_sb, {})
            prev_map = prev.get('pos_map', {}) if isinstance(prev.get('pos_map', {}), dict) else {}

            # B_t = Σ_{i∈mask_qb} bi — target-independent (Eq.11)
            # Used for both sizeNotional and as PPD denominator so pnl/sizeNotional == ppd exactly.
            Bt = float(np.nansum(b[mask_qb]))

            # n_trades: position-diff diagnostic
            if id_arr is not None:
                if mask_qb.any():
                    pos_today = (sgn[mask_qb] * b[mask_qb]).astype('float64', copy=False)
                    ids_today = id_arr[mask_qb]
                    pos_map_today: Dict = {}
                    for inst, pos in zip(ids_today, pos_today):
                        pos_map_today[inst] = float(pos)
                    if not prev_map:
                        day_trades = len(pos_map_today)
                    else:
                        day_trades = sum(1 for inst, pos in pos_map_today.items()
                                         if pos != float(prev_map.get(inst, 0.0)))
                        day_trades += len(set(prev_map.keys()) - set(pos_map_today.keys()))
                    n_trades_today = float(day_trades)
                    prev_state_slice[key_sb] = {
                        'Bt': Bt, 'mean_bet': float(np.nanmean(b[mask_qb])), 'pos_map': pos_map_today}
                else:
                    if empty_day_policy == "close":
                        n_trades_today = float(len(prev_map))
                        prev_state_slice[key_sb] = {'Bt': 0.0, 'mean_bet': 0.0, 'pos_map': {}}
                    elif empty_day_policy == "carry":
                        n_trades_today = np.nan if report_empty_trades_as_nan else 0.0
                    else:
                        n_trades_today = np.nan
            else:
                if mask_qb.any():
                    mean_bet = float(np.nanmean(b[mask_qb]))
                    prev_Bt  = float(prev.get('Bt', np.nan)) if prev and 'Bt' in prev else np.nan
                    prev_mb  = float(prev.get('mean_bet', np.nan)) if prev and 'mean_bet' in prev else np.nan
                    if np.isfinite(prev_Bt):
                        dBt   = abs(Bt - prev_Bt)
                        denom = mean_bet if mean_bet > 0 else (prev_mb if np.isfinite(prev_mb) and prev_mb > 0 else np.nan)
                        n_trades_today = (dBt / denom) if (np.isfinite(denom) and denom > 0) else (np.nan if report_empty_trades_as_nan else 0.0)
                    else:
                        n_trades_today = (Bt / mean_bet) if mean_bet > 0 else (np.nan if report_empty_trades_as_nan else 0.0)
                    prev_state_slice[key_sb] = {'Bt': Bt, 'mean_bet': mean_bet, 'pos_map': {}}
                else:
                    if empty_day_policy == "close":
                        prev_Bt = float(prev.get('Bt', 0.0) or 0.0)
                        prev_mb = float(prev.get('mean_bet', 0.0) or 0.0)
                        n_trades_today = (prev_Bt / prev_mb) if prev_mb > 0 else 0.0
                        prev_state_slice[key_sb] = {'Bt': 0.0, 'mean_bet': 0.0, 'pos_map': {}}
                    elif empty_day_policy == "carry":
                        n_trades_today = np.nan if report_empty_trades_as_nan else 0.0
                    else:
                        n_trades_today = np.nan

            for target in target_cols:
                if target == "__ALL__":
                    continue
                y = df_np.get(target)
                if y is None:
                    continue
                y_fin = np.isfinite(y)

                # m: finite bet + finite target → used for PnL, regression, n_trades per target
                m = mask_qb & y_fin

                # m_hit: finite signal(≠0) + finite target, NO bet — for hit_ratio only.
                # nrInstr = |U_t^(q)| is target-independent; use nr_instr_today (from mask_q).
                m_hit = mask_q & y_fin

                # n_trades per (target, bet): position-diff diagnostic
                key_sqtb     = (signal, qlabel, target, bet)
                prev_tgt     = prev_state_slice.get(key_sqtb, {})
                prev_map_tgt = prev_tgt.get('pos_map', {}) if isinstance(prev_tgt.get('pos_map', {}), dict) else {}

                if id_arr is not None:
                    if m.any():
                        pos_today_t = (sgn[m] * b[m]).astype('float64', copy=False)
                        ids_today_t = id_arr[m]
                        pos_map_t   = {inst: float(pos) for inst, pos in zip(ids_today_t, pos_today_t)}
                        if not prev_map_tgt:
                            day_trades_t = len(pos_map_t)
                        else:
                            day_trades_t = sum(1 for inst, pos in pos_map_t.items()
                                               if pos != float(prev_map_tgt.get(inst, 0.0)))
                            day_trades_t += len(set(prev_map_tgt.keys()) - set(pos_map_t.keys()))
                        n_trades_tgt = float(day_trades_t)
                        prev_state_slice[key_sqtb] = {
                            'Bt': float(np.nansum(b[m])), 'mean_bet': float(np.nanmean(b[m])), 'pos_map': pos_map_t}
                    else:
                        if empty_day_policy == "close":
                            n_trades_tgt = float(len(prev_map_tgt))
                            prev_state_slice[key_sqtb] = {'Bt': 0.0, 'mean_bet': 0.0, 'pos_map': {}}
                        elif empty_day_policy == "carry":
                            n_trades_tgt = np.nan if report_empty_trades_as_nan else 0.0
                        else:
                            n_trades_tgt = np.nan
                else:
                    if m.any():
                        Bt_t     = float(np.nansum(b[m]))
                        mb_t     = float(np.nanmean(b[m]))
                        prev_Bt_t = float(prev_tgt.get('Bt', np.nan)) if prev_tgt and 'Bt' in prev_tgt else np.nan
                        prev_mb_t = float(prev_tgt.get('mean_bet', np.nan)) if prev_tgt and 'mean_bet' in prev_tgt else np.nan
                        if np.isfinite(prev_Bt_t):
                            dBt_t   = abs(Bt_t - prev_Bt_t)
                            denom_t = mb_t if mb_t > 0 else (prev_mb_t if np.isfinite(prev_mb_t) and prev_mb_t > 0 else np.nan)
                            n_trades_tgt = (dBt_t / denom_t) if (np.isfinite(denom_t) and denom_t > 0) else (np.nan if report_empty_trades_as_nan else 0.0)
                        else:
                            n_trades_tgt = (Bt_t / mb_t) if mb_t > 0 else (np.nan if report_empty_trades_as_nan else 0.0)
                        prev_state_slice[key_sqtb] = {'Bt': Bt_t, 'mean_bet': mb_t, 'pos_map': {}}
                    else:
                        if empty_day_policy == "close":
                            prev_Bt_t = float(prev_tgt.get('Bt', 0.0) or 0.0)
                            prev_mb_t = float(prev_tgt.get('mean_bet', 0.0) or 0.0)
                            n_trades_tgt = (prev_Bt_t / prev_mb_t) if prev_mb_t > 0 else 0.0
                            prev_state_slice[key_sqtb] = {'Bt': 0.0, 'mean_bet': 0.0, 'pos_map': {}}
                        elif empty_day_policy == "carry":
                            n_trades_tgt = np.nan if report_empty_trades_as_nan else 0.0
                        else:
                            n_trades_tgt = np.nan

                # -------------------------------------------------------
                # PnL: Σ sign(si)·fi·bi over instruments with finite s, b, f.
                # NaN targets are excluded (m already requires y_fin).
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
                # Computed from m_hit (mask_q & y_fin), NOT from m (mask_qb & y_fin).
                # Must be outside the if m.any() block so it is computed whenever
                # there are instruments with finite signal and target, regardless of bets.
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
                    # hit_ratio already computed above from m_hit (bet-independent)
                    r2 = np.nan; t_stat = np.nan; sharpe = np.nan

                stats['pnl'][signal][qlabel][target][bet]          = pnl
                stats['ppd'][signal][qlabel][target][bet]          = ppd
                # sizeNotional = B_t (Eq.11, target-independent); ppd = pnl/sizeNotional exactly
                stats['sizeNotional'][signal][qlabel][target][bet] = Bt
                # nrInstr = |U_t^(q)|: bet-independent AND target-independent (Cucuringu §5.10)
                # Use nr_instr_today (from mask_q only), not nr_instr_tgt (which conditions on y_fin)
                stats['nrInstr'][signal][qlabel][target][bet]      = nr_instr_today
                stats['n_trades'][signal][qlabel][target][bet]     = float(n_trades_tgt) if np.isfinite(n_trades_tgt) else np.nan
                stats['hit_ratio'][signal][qlabel][target][bet]    = hit_ratio
                stats['long_ratio'][signal][qlabel][target][bet]   = long_ratio_q
                stats['r2'][signal][qlabel][target][bet]           = r2
                stats['t_stat'][signal][qlabel][target][bet]       = t_stat
                stats['sharpe'][signal][qlabel][target][bet]       = sharpe

            # __ALL__ target aggregates
            stats['nrInstr'][signal][qlabel]['__ALL__'][bet]  = nr_instr_today
            stats['n_trades'][signal][qlabel]['__ALL__'][bet] = float(n_trades_today) if np.isfinite(n_trades_today) else np.nan

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

    return stats, prev_state_slice


def compute_daily_stats(
    df,
    signal_cols: Sequence[str],
    target_cols: Sequence[str],
    quantiles: Sequence[float] = (1.0, 0.75, 0.5, 0.25),
    bet_size_cols: Sequence[str] = ('betsize_equal',),
    prev_state: MutableMapping | None = None,
    type_quantile: str = 'cumulative',
    enable_distributions: bool = False,
    max_dist_samples_per_series: int = 50_000,
    random_state=None,
    empty_day_policy: str = "carry",
    report_empty_trades_as_nan: bool = True,
    n_jobs: int = 1,
):
    import pandas as pd

    if prev_state is None:
        prev_state = _GLOBAL_PREV_STATE

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

    def _slice_state(sig: str) -> Dict:
        return {k: v for k, v in prev_state.items()
                if isinstance(k, tuple) and len(k) >= 3 and k[0] == sig}

    signals   = [s for s in signal_cols if s in df_np]
    n_threads = min(max(1, int(n_jobs or 1)), len(signals) or 1)

    if n_threads == 1:
        for signal in signals:
            stats_sig, st_sig = _compute_daily_stats_for_one_signal(
                signal=signal, df_np=df_np, id_arr=id_arr,
                target_cols=target_cols, quantiles=quantiles,
                bet_size_cols=bet_size_cols, type_quantile=type_quantile,
                enable_distributions=enable_distributions,
                max_dist_samples_per_series=max_dist_samples_per_series,
                rng_state=None if random_state is None else int(rng.integers(0, 2**31 - 1)),
                empty_day_policy=empty_day_policy,
                report_empty_trades_as_nan=report_empty_trades_as_nan,
                prev_state_slice=_slice_state(signal),
            )
            _merge_signal_branch(stats, stats_sig, signal)
            prev_state.update(st_sig)
    else:
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futs = []
            for signal in signals:
                futs.append(ex.submit(
                    _compute_daily_stats_for_one_signal,
                    signal, df_np, id_arr, target_cols, quantiles, bet_size_cols,
                    type_quantile, enable_distributions, max_dist_samples_per_series,
                    None if random_state is None else int(rng.integers(0, 2**31 - 1)),
                    empty_day_policy, report_empty_trades_as_nan, _slice_state(signal),
                ))
            for fut in as_completed(futs):
                stats_sig, st_sig = fut.result()
                try:
                    sig_name = next(iter(next(iter(stats_sig.values())).keys()))
                except Exception:
                    sig_name = signals[0] if signals else "__SIG__"
                _merge_signal_branch(stats, stats_sig, sig_name)
                prev_state.update(st_sig)

    return stats


def _snapshot_prev_book_counts(prev_state: MutableMapping) -> Dict[tuple, int]:
    out = {}
    for key_sb, obj in prev_state.items():
        if not isinstance(key_sb, tuple) or len(key_sb) != 3:
            continue
        pos_map = obj.get('pos_map', {})
        if isinstance(pos_map, dict) and len(pos_map) > 0:
            out[key_sb] = len(pos_map)
        else:
            Bt = float(obj.get('Bt', 0.0) or 0.0)
            mb = float(obj.get('mean_bet', 0.0) or 0.0)
            out[key_sb] = int(round(Bt / mb)) if mb > 0 else 0
    return out


def _apply_year_opening_override(stats: Dict, prev_counts: Dict[tuple, int], override_if: str = "zero_or_nan"):
    def _should(x):
        if override_if == "always":         return True
        if override_if == "zero_or_nan":    return (x is None) or (not math.isfinite(x)) or (x == 0.0)
        if override_if == "nan_only":       return (x is None) or (not math.isfinite(x))
        return False

    for signal, qdict in stats.get('n_trades', {}).items():
        for qlabel, tdict in qdict.items():
            for _, bdict in tdict.items():
                for bet, ntr_val in list(bdict.items()):
                    k = (signal, qlabel, bet)
                    if k in prev_counts and _should(ntr_val):
                        bdict[bet] = float(prev_counts[k])


def compute_series_continuous(df_sorted_by_date, date_col: str, **kwargs):
    import pandas as pd
    prev = kwargs.pop('prev_state', None)
    if prev is None:
        prev = _GLOBAL_PREV_STATE
    out = []
    for d, df_day in df_sorted_by_date.sort_values(date_col).groupby(date_col):
        out.append((pd.Timestamp(d), compute_daily_stats(df_day, prev_state=prev, **kwargs)))
    return out


def compute_series_continuous_yearaware(
    df_sorted_by_date, date_col: str,
    *, override_if: str = "zero_or_nan",
    **kwargs
):
    import pandas as pd
    prev      = kwargs.pop('prev_state', None)
    if prev is None:
        prev = get_trading_state()
    out       = []
    prev_year = None

    for d, df_day in df_sorted_by_date.sort_values(date_col).groupby(date_col):
        ts           = pd.Timestamp(d)
        year_changed = (prev_year is not None) and (ts.year != prev_year)
        prev_counts  = _snapshot_prev_book_counts(prev) if year_changed else None
        stats        = compute_daily_stats(df_day, prev_state=prev, **kwargs)
        if year_changed and prev_counts:
            _apply_year_opening_override(stats, prev_counts, override_if=override_if)
        out.append((ts, stats))
        prev_year = ts.year

    return out


__all__ = [
    'compute_daily_stats',
    'compute_series_continuous',
    'compute_series_continuous_yearaware',
    'create_5d_stats',
    'get_trading_state',
    'save_trading_state',
    'load_trading_state',
    'reset_trading_state',
]