# =============================
# daily_stats.py — year-aware n_trades (carry + year-open override), no PPT
# Parallel-per-signal (threads) with truthful state merging.
# =============================
from __future__ import annotations

import math
import os
import pickle
import numpy as np
from typing import Dict, MutableMapping, Sequence, Tuple, List
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# -----------------------------------------------------------------------------
# Nested metrics store:
# stats[stat_type][signal][qrank][target][bet] -> value
# -----------------------------------------------------------------------------
def create_5d_stats():
    return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

# -----------------------------------------------------------------------------
# GLOBAL PERSISTENT STATE (kept in-memory + optional on-disk persistence)
# -----------------------------------------------------------------------------
_GLOBAL_PREV_STATE: Dict = {}

def get_trading_state():
    return _GLOBAL_PREV_STATE

def reset_trading_state():
    """Kept for API compatibility (no-op)."""
    pass

def save_trading_state(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(_GLOBAL_PREV_STATE, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_trading_state(path: str, strict: bool=False):
    if not os.path.isfile(path):
        if strict:
            raise FileNotFoundError(f"No trading state at {path}")
        return
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        _GLOBAL_PREV_STATE.clear()
        _GLOBAL_PREV_STATE.update(obj)

# ========================= DAILY (single-date snapshot) =======================

def _label_for_quantile(q: float) -> str:
    return f'qr_{int(round(q * 100))}'

def _topk_mask_desc(abs_vals: np.ndarray, valid_mask: np.ndarray, q: float) -> np.ndarray:
    """
    Pick exactly ceil(q * N) largest values from abs_vals among indices where valid_mask == True.
    Stable ordering for deterministic ties (mergesort).
    """
    idx = np.where(valid_mask)[0]
    out = np.zeros_like(valid_mask, dtype=bool)
    if idx.size == 0 or q <= 0.0:
        return out
    if q >= 1.0:
        out[idx] = True
        return out
    k = int(np.ceil(q * idx.size))
    order = np.argsort(-abs_vals[idx], kind="mergesort")  # descending, stable
    choose = idx[order[:k]]
    out[choose] = True
    return out

def _merge_signal_branch(dst: Dict, src: Dict, signal: str) -> None:
    """Per-signal subtrees are disjoint, so we can assign directly."""
    for stat_type, sig_dict in src.items():
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
    """Worker: compute metrics for ONE signal (thread)."""
    rng = np.random.default_rng(rng_state)
    stats = create_5d_stats()

    s = df_np.get(signal)
    if s is None or s.size == 0:
        return stats, prev_state_slice

    m_fin = np.isfinite(s)
    m_nz  = (s != 0.0)
    m_ok  = m_fin & m_nz
    if not m_fin.any():
        return stats, prev_state_slice

    sgn = np.sign(s)
    abs_s = np.abs(s)

    # Precompute absolute bet arrays
    bet_abs: Dict[str, np.ndarray] = {}
    for b in bet_size_cols:
        x = df_np.get(b)
        bet_abs[b] = np.abs(x) if x is not None else np.full(s.size, np.nan, dtype='float64')

    # Optional bucket edges
    if type_quantile == 'quantEach':
        K = len(quantiles)
        probs = np.linspace(0.0, 1.0, K + 1, dtype='float64')
        edges = np.nanquantile(abs_s[m_ok], probs) if m_ok.any() else np.array([np.nan, np.nan])
    else:
        edges = None

    alpha_written = set()

    for q in quantiles:
        qlabel = _label_for_quantile(q)

        # Selection mask
        if type_quantile == 'cumulative':
            mask_q = _topk_mask_desc(abs_s, m_ok, q)
        else:
            if edges is None or not np.isfinite(edges).all():
                mask_q = np.zeros_like(m_ok, dtype=bool)
            else:
                j = quantiles.index(q) + 1  # 1..K
                lo, hi = edges[j-1], (edges[j-1] if j == len(quantiles) else edges[j])
                mask_q = m_ok & (abs_s >= lo) if j == len(quantiles) else (m_ok & (abs_s >= lo) & (abs_s <= hi))

        # alpha_sum (emit once per (signal,q))
        alpha_sum_today = float(np.nansum(s[mask_q])) if mask_q.any() else 0.0
        if qlabel not in alpha_written:
            stats['alpha_sum'][signal][qlabel]['__ALL__']['__ALL__'] = alpha_sum_today
            alpha_written.add(qlabel)

        # nrInstr (bet-independent)
        if id_arr is not None:
            if mask_q.any():
                ids = id_arr[mask_q]
                ids = ids[ids == ids]  # drop NaN-ish
                nr_instr_today = int(np.unique(ids).size)
            else:
                nr_instr_today = 0
        else:
            nr_instr_today = int(mask_q.sum()) if mask_q.any() else 0

        for bet in bet_size_cols:
            if bet == "__ALL__":
                continue
            b = bet_abs.get(bet)
            if b is None:
                continue
            b_fin = np.isfinite(b)
            mask_qb = mask_q & b_fin

            key_sb = (signal, qlabel, bet)
            prev = prev_state_slice.get(key_sb, {})
            prev_map = prev.get('pos_map', {}) if isinstance(prev.get('pos_map', {}), dict) else {}

            # n_trades (truthful empties)
            if id_arr is not None:
                if mask_qb.any():
                    pos_today = (sgn[mask_qb] * b[mask_qb]).astype('float64', copy=False)
                    ids_today = id_arr[mask_qb]
                    pos_map_today: Dict = {}
                    for inst, pos in zip(ids_today, pos_today):
                        pos_map_today[inst] = float(pos)

                    if not prev_map:
                        day_trades = len(pos_map_today)  # openers
                    else:
                        day_trades = 0
                        for inst, pos in pos_map_today.items():
                            prev_pos = float(prev_map.get(inst, 0.0))
                            if pos != prev_pos:
                                day_trades += 1
                        day_trades += len(set(prev_map.keys()) - set(pos_map_today.keys()))
                    n_trades_today = float(day_trades)
                    prev_state_slice[key_sb] = {
                        'Bt': float(np.nansum(b[mask_qb])),
                        'mean_bet': float(np.nanmean(b[mask_qb])),
                        'pos_map': pos_map_today,
                    }
                else:
                    if empty_day_policy == "close":
                        n_trades_today = float(len(prev_map))
                        prev_state_slice[key_sb] = {'Bt': 0.0, 'mean_bet': 0.0, 'pos_map': {}}
                    elif empty_day_policy == "carry":
                        n_trades_today = (np.nan if report_empty_trades_as_nan else 0.0)
                    else:  # skip
                        n_trades_today = np.nan
            else:
                # proxy mode (no IDs)
                if mask_qb.any():
                    Bt_today = float(np.nansum(b[mask_qb]))
                    mean_bet = float(np.nanmean(b[mask_qb]))
                    prev_Bt = float(prev.get('Bt', np.nan)) if prev and 'Bt' in prev else np.nan
                    prev_mb = float(prev.get('mean_bet', np.nan)) if prev and 'mean_bet' in prev else np.nan

                    if np.isfinite(prev_Bt):
                        dBt = abs(Bt_today - prev_Bt)
                        denom = mean_bet if mean_bet > 0 else (prev_mb if np.isfinite(prev_mb) and prev_mb > 0 else np.nan)
                        n_trades_today = (dBt / denom) if (np.isfinite(denom) and denom > 0) else (np.nan if report_empty_trades_as_nan else 0.0)
                    else:
                        n_trades_today = (Bt_today / mean_bet) if (mean_bet and mean_bet > 0) else (np.nan if report_empty_trades_as_nan else 0.0)

                    prev_state_slice[key_sb] = {'Bt': Bt_today, 'mean_bet': mean_bet, 'pos_map': {}}
                else:
                    if empty_day_policy == "close":
                        prev_Bt = float(prev.get('Bt', 0.0) or 0.0)
                        prev_mb = float(prev.get('mean_bet', 0.0) or 0.0)
                        n_trades_today = (prev_Bt / prev_mb) if prev_mb > 0 else 0.0
                        prev_state_slice[key_sb] = {'Bt': 0.0, 'mean_bet': 0.0, 'pos_map': {}}
                    elif empty_day_policy == "carry":
                        n_trades_today = (np.nan if report_empty_trades_as_nan else 0.0)
                    else:
                        n_trades_today = np.nan

            # per-target metrics
            for target in target_cols:
                if target == "__ALL__":
                    # Do not emit n_trades / nrInstr for pseudo targets
                    continue
                y = df_np.get(target)
                if y is None:
                    continue
                y_fin = np.isfinite(y)
                m = mask_qb & y_fin

                # nrInstr and n_trades are target-specific
                if id_arr is not None:
                    nr_instr_today = int(np.unique(id_arr[m]).size) if m.any() else 0
                else:
                    nr_instr_today = int(m.sum()) if m.any() else 0

                key_sqtb = (signal, qlabel, target, bet)
                prev = prev_state_slice.get(key_sqtb, {})
                prev_map = prev.get('pos_map', {}) if isinstance(prev.get('pos_map', {}), dict) else {}

                if id_arr is not None:
                    if m.any():
                        pos_today = (sgn[m] * b[m]).astype('float64', copy=False)
                        ids_today = id_arr[m]
                        pos_map_today: Dict = {inst: float(pos) for inst, pos in zip(ids_today, pos_today)}

                        if not prev_map:
                            day_trades = len(pos_map_today)
                        else:
                            day_trades = 0
                            for inst, pos in pos_map_today.items():
                                prev_pos = float(prev_map.get(inst, 0.0))
                                if pos != prev_pos:
                                    day_trades += 1
                            day_trades += len(set(prev_map.keys()) - set(pos_map_today.keys()))
                        n_trades_today = float(day_trades)
                        prev_state_slice[key_sqtb] = {
                            'Bt': float(np.nansum(b[m])),
                            'mean_bet': float(np.nanmean(b[m])),
                            'pos_map': pos_map_today,
                        }
                    else:
                        if empty_day_policy == "close":
                            n_trades_today = float(len(prev_map))
                            prev_state_slice[key_sqtb] = {'Bt': 0.0, 'mean_bet': 0.0, 'pos_map': {}}
                        elif empty_day_policy == "carry":
                            n_trades_today = (np.nan if report_empty_trades_as_nan else 0.0)
                        else:
                            n_trades_today = np.nan
                else:
                    if m.any():
                        Bt_today = float(np.nansum(b[m]))
                        mean_bet = float(np.nanmean(b[m]))
                        prev_Bt = float(prev.get('Bt', np.nan)) if prev and 'Bt' in prev else np.nan
                        prev_mb = float(prev.get('mean_bet', np.nan)) if prev and 'mean_bet' in prev else np.nan

                        if np.isfinite(prev_Bt):
                            dBt = abs(Bt_today - prev_Bt)
                            denom = mean_bet if mean_bet > 0 else (prev_mb if np.isfinite(prev_mb) and prev_mb > 0 else np.nan)
                            n_trades_today = (dBt / denom) if (np.isfinite(denom) and denom > 0) else (np.nan if report_empty_trades_as_nan else 0.0)
                        else:
                            n_trades_today = (Bt_today / mean_bet) if (mean_bet and mean_bet > 0) else (np.nan if report_empty_trades_as_nan else 0.0)

                        prev_state_slice[key_sqtb] = {'Bt': Bt_today, 'mean_bet': mean_bet, 'pos_map': {}}
                    else:
                        if empty_day_policy == "close":
                            prev_Bt = float(prev.get('Bt', 0.0) or 0.0)
                            prev_mb = float(prev.get('mean_bet', 0.0) or 0.0)
                            n_trades_today = (prev_Bt / prev_mb) if prev_mb > 0 else 0.0
                            prev_state_slice[key_sqtb] = {'Bt': 0.0, 'mean_bet': 0.0, 'pos_map': {}}
                        elif empty_day_policy == "carry":
                            n_trades_today = (np.nan if report_empty_trades_as_nan else 0.0)
                        else:
                            n_trades_today = np.nan

                if m.any():
                    pnl_vec = sgn[m] * y[m] * b[m]
                    pnl = float(np.nansum(pnl_vec))
                    notional = float(np.nansum(b[m]))
                    ppd = (pnl / notional) if notional > 0 else np.nan

                    # Extras
                    hit_ratio = float(np.nanmean(pnl_vec > 0)) if pnl_vec.size else np.nan
                    long_ratio = float(np.nanmean(sgn[m] > 0)) if sgn[m].size else np.nan
                    n = int(m.sum())
                    # Simple cross-sectional regression stats (y on s)
                    s_vals = s[m].astype('float64', copy=False)
                    y_vals = y[m].astype('float64', copy=False)
                    s_mean = np.nanmean(s_vals)
                    y_mean = np.nanmean(y_vals)
                    s_dev = s_vals - s_mean
                    y_dev = y_vals - y_mean
                    s_var = float(np.nanvar(s_vals, ddof=1)) if n > 1 else np.nan
                    cov = float(np.nanmean(s_dev * y_dev)) if n > 0 else np.nan
                    beta = (cov / s_var) if (np.isfinite(cov) and np.isfinite(s_var) and s_var > 0) else np.nan
                    s_std = math.sqrt(s_var) if np.isfinite(s_var) and s_var > 0 else np.nan
                    y_std = float(np.nanstd(y_vals, ddof=1)) if n > 1 else np.nan
                    r = (cov / (s_std * y_std)) if (np.isfinite(cov) and np.isfinite(s_std) and np.isfinite(y_std) and s_std > 0 and y_std > 0) else np.nan
                    r2 = float(r * r) if np.isfinite(r) else np.nan
                    t_stat = (r * math.sqrt(n - 2) / math.sqrt(1 - r * r)) if (np.isfinite(r) and n > 2 and (1 - r * r) > 0) else np.nan
                    sharpe = (np.nanmean(pnl_vec) / np.nanstd(pnl_vec, ddof=1)) if (pnl_vec.size > 1 and np.nanstd(pnl_vec, ddof=1) > 0) else np.nan
                else:
                    pnl = 0.0
                    notional = 0.0
                    ppd = np.nan
                    hit_ratio = np.nan
                    long_ratio = np.nan
                    r2 = np.nan
                    t_stat = np.nan
                    sharpe = np.nan

                stats['pnl'][signal][qlabel][target][bet]          = pnl
                stats['ppd'][signal][qlabel][target][bet]          = ppd
                stats['sizeNotional'][signal][qlabel][target][bet] = notional
                stats['nrInstr'][signal][qlabel][target][bet]      = nr_instr_today
                ntr = float(n_trades_today) if np.isfinite(n_trades_today) else np.nan
                stats['n_trades'][signal][qlabel][target][bet]     = ntr
                stats['hit_ratio'][signal][qlabel][target][bet]    = hit_ratio
                stats['long_ratio'][signal][qlabel][target][bet]   = long_ratio
                stats['r2'][signal][qlabel][target][bet]           = r2
                stats['t_stat'][signal][qlabel][target][bet]       = t_stat
                stats['sharpe'][signal][qlabel][target][bet]       = sharpe

            # Store nrInstr and n_trades once per (signal, qrank, bet) under target="__ALL__"
            stats['nrInstr'][signal][qlabel]['__ALL__'][bet] = nr_instr_today
            ntr = float(n_trades_today) if np.isfinite(n_trades_today) else np.nan
            stats['n_trades'][signal][qlabel]['__ALL__'][bet] = ntr

    # Optional raw distributions (thin sampling) per signal
    if enable_distributions:
        # targets
        for target in target_cols:
            y = df_np.get(target)
            if y is None:
                continue
            y = y[np.isfinite(y)]
            if y.size == 0:
                continue
            if y.size > max_dist_samples_per_series:
                idx = rng.choice(y.size, size=max_dist_samples_per_series, replace=False)
                y = y[idx]
            stats['fret_value'][f'__S__{signal}']['__ALL__'][target]['__ALL__'] = float(np.nanmean(y))
        # bet columns
        for bet in bet_size_cols:
            b = df_np.get(bet)
            if b is None:
                continue
            x = np.abs(b[np.isfinite(b)])
            if x.size == 0:
                continue
            if x.size > max_dist_samples_per_series:
                idx = rng.choice(x.size, size=max_dist_samples_per_series, replace=False)
                x = x[idx]
            stats['betsize_value'][f'__B__{bet}']['__ALL__']['__ALL__'][bet] = float(np.nanmean(x))

    return stats, prev_state_slice


def compute_daily_stats(
    df,
    signal_cols: Sequence[str],
    target_cols: Sequence[str],
    quantiles: Sequence[float] = (1.0, 0.75, 0.5, 0.25),
    bet_size_cols: Sequence[str] = ('betsize_equal',),
    prev_state: MutableMapping | None = None,
    type_quantile: str = 'cumulative',   # 'cumulative' (top-K) or 'quantEach' (bucket)
    # ---- RAW distribution rows (for plotting histograms) ----
    enable_distributions: bool = False,
    max_dist_samples_per_series: int = 50_000,
    random_state=None,
    # ---- Empty-slice handling (default 'carry' so state persists across empties)
    empty_day_policy: str = "carry",     # 'close' | 'carry' | 'skip'
    # ---- Reporting tweak on empties under "carry"
    report_empty_trades_as_nan: bool = True,
    # ---- Parallelism
    n_jobs: int = 1,                     # >1 => threads per-signal
):
    """Per-day cross-section stats (single date), optionally parallelized per signal."""
    import pandas as pd

    if prev_state is None:
        prev_state = _GLOBAL_PREV_STATE

    rng = np.random.default_rng(random_state)
    stats = create_5d_stats()

    # ----------------------------
    # Column preparation (FAST)
    # ----------------------------
    id_col = 'ticker' if 'ticker' in df.columns else None

    want_numeric = (set(signal_cols) | set(target_cols) | set(bet_size_cols)) - ({id_col} if id_col else set())
    df_np: Dict[str, np.ndarray] = {}

    for col in want_numeric:
        if col not in df.columns:
            continue
        arr = pd.to_numeric(df[col], errors='coerce').to_numpy()
        arr = arr.astype('float64', copy=False)
        arr[~np.isfinite(arr)] = np.nan
        df_np[col] = arr

    id_arr = df[id_col].to_numpy() if id_col else None

    n = len(df)
    if n == 0:
        return stats

    # Prepare per-signal prev_state slices
    def _slice_state_for_signal(sig: str) -> Dict:
        out = {}
        for k, v in prev_state.items():
            if isinstance(k, tuple) and len(k) == 3 and k[0] == sig:
                out[k] = v
        return out

    signals = [s for s in signal_cols if s in df_np]
    n_threads = min(max(1, int(n_jobs or 1)), len(signals) or 1)

    if n_threads == 1:
        for signal in signals:
            stats_sig, st_sig = _compute_daily_stats_for_one_signal(
                signal=signal,
                df_np=df_np,
                id_arr=id_arr,
                target_cols=target_cols,
                quantiles=quantiles,
                bet_size_cols=bet_size_cols,
                type_quantile=type_quantile,
                enable_distributions=enable_distributions,
                max_dist_samples_per_series=max_dist_samples_per_series,
                rng_state=None if random_state is None else int(rng.integers(0, 2**31 - 1)),
                empty_day_policy=empty_day_policy,
                report_empty_trades_as_nan=report_empty_trades_as_nan,
                prev_state_slice=_slice_state_for_signal(signal),
            )
            _merge_signal_branch(stats, stats_sig, signal)
            prev_state.update(st_sig)
    else:
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futs = []
            for signal in signals:
                futs.append(ex.submit(
                    _compute_daily_stats_for_one_signal,
                    signal,
                    df_np,
                    id_arr,
                    target_cols,
                    quantiles,
                    bet_size_cols,
                    type_quantile,
                    enable_distributions,
                    max_dist_samples_per_series,
                    None if random_state is None else int(rng.integers(0, 2**31 - 1)),
                    empty_day_policy,
                    report_empty_trades_as_nan,
                    _slice_state_for_signal(signal),
                ))
            for fut in as_completed(futs):
                stats_sig, st_sig = fut.result()
                # find signal name inside branch (fallbacks safe)
                try:
                    sig_name = next(iter(next(iter(stats_sig.values())).keys()))
                except Exception:
                    sig_name = signals[0] if signals else "__SIG__"
                _merge_signal_branch(stats, stats_sig, sig_name)
                prev_state.update(st_sig)

    return stats

# ========================= YEAR-AWARE OVERRIDE HELPERS ========================

def _snapshot_prev_book_counts(prev_state: MutableMapping) -> Dict[tuple, int]:
    """(signal, qlabel, bet) -> count of open names in carried book."""
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
    """
    On the first trading day of a year, for each (signal, qlabel, bet),
    if n_trades meets the condition, set to prev_counts[(signal, qlabel, bet)].
    """
    def _should_override(x):
        if override_if == "always":
            return True
        if override_if == "zero_or_nan":
            return (x is None) or (not math.isfinite(x)) or (x == 0.0)
        if override_if == "nan_only":
            return (x is None) or (not math.isfinite(x))
        return False

    ntr_tree = stats.get('n_trades', {})
    for signal, qdict in ntr_tree.items():
        for qlabel, tdict in qdict.items():
            for _, bdict in tdict.items():
                for bet, ntr_val in list(bdict.items()):
                    k = (signal, qlabel, bet)
                    if k in prev_counts and _should_override(ntr_val):
                        bdict[bet] = float(prev_counts[k])

# ========================= SERIES RUNNERS (with override) =====================

def compute_series_continuous(df_sorted_by_date, date_col: str, **kwargs):
    """Original continuous runner (no year-aware override)."""
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
    *, override_if: str = "zero_or_nan",  # "zero_or_nan" | "nan_only" | "always"
    **kwargs
):
    """
    Continuous runner that, on the first trading day of each calendar year, replaces
    n_trades with the size of yesterday's carried book when n_trades would be 0/NaN.
    """
    import pandas as pd

    prev = kwargs.pop('prev_state', None)
    if prev is None:
        prev = get_trading_state()

    out = []
    prev_year = None

    for d, df_day in df_sorted_by_date.sort_values(date_col).groupby(date_col):
        ts = pd.Timestamp(d)
        year_changed = (prev_year is not None) and (ts.year != prev_year)

        prev_counts = _snapshot_prev_book_counts(prev) if year_changed else None

        stats = compute_daily_stats(df_day, prev_state=prev, **kwargs)

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
