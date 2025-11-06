# =============================
# daily_stats.py — year-aware n_trades (carry + year-open override), no PPT
# =============================
from __future__ import annotations

import os
import pickle
import numpy as np
from typing import Dict, MutableMapping, Sequence
from collections import defaultdict

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
    report_empty_trades_as_nan: bool = True,  # True: NaN on empty days (prevents fake 0 dips)
):
    """
    Per-day cross-section stats (single date snapshot).

    Truthfulness choices here:
      * Quantiles are exact top-K by |signal|, excluding zeros.
      * If a (signal, q, bet) slice is empty:
          - 'close': treat as closing all prior positions (n_trades = #prior names),
                     state is reset to empty; emit rows with notional=0, pnl=0, ratios=NaN.
          - 'carry': keep prior state; emit rows with notional=0, pnl=0; by default
                     n_trades=NaN (report_empty_trades_as_nan=True) to avoid fake dips.
          - 'skip' : do not update state; emit minimal rows with n_trades=NaN.
    """
    import pandas as pd

    rng = np.random.default_rng(random_state)
    stats = create_5d_stats()

    if prev_state is None:
        prev_state = _GLOBAL_PREV_STATE

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

    # Precompute absolute bet arrays
    bet_abs: Dict[str, np.ndarray] = {}
    for b in bet_size_cols:
        if b in df_np:
            bet_abs[b] = np.abs(df_np[b])
        else:
            bet_abs[b] = np.full(n, np.nan, dtype='float64')

    # ----------------------------
    # Main loops (signals -> q -> bets -> targets)
    # ----------------------------
    for signal in signal_cols:
        if signal not in df_np:
            continue

        s = df_np[signal]                       # float64
        m_fin = np.isfinite(s)
        m_nz  = (s != 0.0)
        m_ok  = m_fin & m_nz                    # finite AND non-zero only
        if not m_fin.any():
            continue

        sgn = np.sign(s)
        abs_s = np.abs(s)

        # Prepare optional bucket edges (rarely used)
        if type_quantile == 'quantEach':
            K = len(quantiles)
            probs = np.linspace(0.0, 1.0, K + 1, dtype='float64')
            edges = np.nanquantile(abs_s[m_ok], probs) if m_ok.any() else np.array([np.nan, np.nan])
        else:
            edges = None

        alpha_written = set()

        for q in quantiles:
            qlabel = _label_for_quantile(q)

            # ---- Selection mask (exact-size or bucket) ----
            if type_quantile == 'cumulative':
                mask_q = _topk_mask_desc(abs_s, m_ok, q)
            else:
                if edges is None or not np.isfinite(edges).all():
                    mask_q = np.zeros_like(m_ok, dtype=bool)
                else:
                    j = quantiles.index(q) + 1  # 1..K
                    lo, hi = edges[j-1], j == len(quantiles) and edges[j-1] or edges[j]
                    # Inclusive on hi to keep coverage; still limited by non-zero filter above
                    mask_q = m_ok & (abs_s >= lo) & (abs_s <= hi)

            # -------- alpha_sum (always emit; 0 if empty) --------
            alpha_sum_today = float(np.nansum(s[mask_q])) if mask_q.any() else 0.0
            if qlabel not in alpha_written:
                stats['alpha_sum'][signal][qlabel]['__ALL__']['__ALL__'] = alpha_sum_today
                alpha_written.add(qlabel)

            # -------- nrInstr (bet-independent; 0 if empty) --------
            if id_arr is not None:
                nr_instr_today = int(pd.Series(id_arr[mask_q]).dropna().nunique()) if mask_q.any() else 0
            else:
                nr_instr_today = int(np.sum(mask_q)) if mask_q.any() else 0

            # ----------------------------
            # Per-bet work (trades, sizeNotional part of PPD)
            # ----------------------------
            for bet in bet_size_cols:
                b = bet_abs.get(bet)
                if b is None:
                    continue

                b_fin = np.isfinite(b)
                mask_qb = mask_q & b_fin

                key_sb = (signal, qlabel, bet)
                prev = prev_state.get(key_sb, {})
                prev_map = prev.get('pos_map', {}) if isinstance(prev.get('pos_map', {}), dict) else {}

                # ---- TRADES & STATE (truthful empty-day handling) ----
                if id_arr is not None:
                    if mask_qb.any():
                        # exact trades with IDs
                        pos_today = (sgn[mask_qb] * b[mask_qb]).astype('float64', copy=False)
                        ids_today = id_arr[mask_qb]
                        pos_map_today: Dict = {}
                        for inst, pos in zip(ids_today, pos_today):
                            pos_map_today[inst] = float(pos)

                        if not prev_map:
                            day_trades = len(pos_map_today)  # opening trades on first observed day
                        else:
                            day_trades = 0
                            # changes & opens
                            for inst, pos in pos_map_today.items():
                                prev_pos = float(prev_map.get(inst, 0.0))
                                if pos != prev_pos:
                                    day_trades += 1
                            # closes
                            day_trades += len(set(prev_map.keys()) - set(pos_map_today.keys()))

                        n_trades_today = float(day_trades)

                        prev_state[key_sb] = {
                            'Bt': float(np.nansum(b[mask_qb])),
                            'mean_bet': float(np.nanmean(b[mask_qb])),
                            'pos_map': pos_map_today,
                        }
                    else:
                        # EMPTY slice today
                        if empty_day_policy == "close":
                            # close everything that was open
                            n_trades_today = float(len(prev_map))
                            prev_state[key_sb] = {'Bt': 0.0, 'mean_bet': 0.0, 'pos_map': {}}
                        elif empty_day_policy == "carry":
                            # carry the book; don't fake a zero in outputs
                            n_trades_today = (np.nan if report_empty_trades_as_nan else 0.0)
                            # keep prev_state as-is
                        else:  # 'skip'
                            n_trades_today = np.nan
                            # do not modify prev_state
                else:
                    # proxy mode (no IDs) using |ΔBt| / mean_bet
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
                            # first observed day → approx number of names
                            n_trades_today = (Bt_today / mean_bet) if (mean_bet and mean_bet > 0) else (np.nan if report_empty_trades_as_nan else 0.0)

                        prev_state[key_sb] = {'Bt': Bt_today, 'mean_bet': mean_bet, 'pos_map': {}}
                    else:
                        if empty_day_policy == "close":
                            prev_Bt = float(prev.get('Bt', 0.0)) if prev else 0.0
                            prev_mb = float(prev.get('mean_bet', 0.0)) if prev else 0.0
                            n_trades_today = (prev_Bt / prev_mb) if prev_mb > 0 else 0.0
                            prev_state[key_sb] = {'Bt': 0.0, 'mean_bet': 0.0, 'pos_map': {}}
                        elif empty_day_policy == "carry":
                            n_trades_today = (np.nan if report_empty_trades_as_nan else 0.0)
                            # keep prev_state
                        else:
                            n_trades_today = np.nan
                            # do not modify prev_state

                # ----------------------------
                # Per-target metrics (emit even if empty)
                # ----------------------------
                for target in target_cols:
                    y = df_np.get(target)
                    if y is None:
                        continue

                    y_fin = np.isfinite(y)
                    m = mask_qb & y_fin

                    if m.any():
                        pnl_vec = sgn[m] * y[m] * b[m]
                        pnl = float(np.nansum(pnl_vec))
                        notional = float(np.nansum(b[m]))
                        ppd = (pnl / notional) if notional > 0 else np.nan
                    else:
                        pnl = 0.0
                        notional = 0.0
                        ppd = np.nan

                    # write all metrics (no PPT here)
                    stats['pnl'][signal][qlabel][target][bet]          = pnl
                    stats['ppd'][signal][qlabel][target][bet]          = ppd
                    stats['sizeNotional'][signal][qlabel][target][bet] = notional
                    stats['nrInstr'][signal][qlabel][target][bet]      = nr_instr_today

                    ntr = float(n_trades_today) if np.isfinite(n_trades_today) else np.nan
                    stats['n_trades'][signal][qlabel][target][bet]     = ntr

        # ----------------------------
        # Optional raw distributions (thin sampling)
        # ----------------------------
        if enable_distributions:
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
            for bet in bet_size_cols:
                b = bet_abs.get(bet)
                if b is None:
                    continue
                x = b[np.isfinite(b)]
                if x.size == 0:
                    continue
                if x.size > max_dist_samples_per_series:
                    idx = rng.choice(x.size, size=max_dist_samples_per_series, replace=False)
                    x = x[idx]
                stats['betsize_value'][f'__B__{bet}']['__ALL__']['__ALL__'][bet] = float(np.nanmean(x))

    return stats

# ========================= YEAR-AWARE OVERRIDE HELPERS ========================

def _snapshot_prev_book_counts(prev_state: MutableMapping) -> Dict[tuple, int]:
    """
    Build a map (signal, qlabel, bet) -> count of open names in the carried book.
    Works with ID mode (pos_map) and proxy mode (Bt/mean_bet).
    """
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
    Mutates `stats['n_trades']` in place:
    On the first trading day of a year, for each (signal, qlabel, bet),
    if n_trades meets the condition, set to prev_counts[(signal, qlabel, bet)].
    """
    import math

    def _should_override(x):
        if override_if == "always":
            return True
        if override_if == "zero_or_nan":
            return (x is None) or (not math.isfinite(x)) or (x == 0.0)
        if override_if == "nan_only":
            return (x is None) or (not math.isfinite(x))
        return False

    ntr_tree = stats.get('n_trades', {})

    for _, qdict in ntr_tree.items():
        for _, tdict in qdict.items():
            for _, bdict in tdict.items():
                for bet, ntr_val in list(bdict.items()):
                    # try to infer (signal, qlabel, bet) from the traversal path is messy here,
                    # so we iterate the full stats dict below for safety.
                    pass

    # Safe pass: iterate full path
    for signal, qdict in ntr_tree.items():
        for qlabel, tdict in qdict.items():
            for target, bdict in tdict.items():
                for bet, ntr_val in list(bdict.items()):
                    k = (signal, qlabel, bet)
                    if k in prev_counts and _should_override(ntr_val):
                        new_ntr = float(prev_counts[k])
                        bdict[bet] = new_ntr  # write n_trades

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
    Continuous runner that, on the *first trading day of each calendar year*, replaces
    n_trades with the size of yesterday's carried book when n_trades would be 0/NaN.

    Recommended: keep empty_day_policy="carry" so the book truly persists across empties.
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

        # Snapshot previous day's carried book sizes BEFORE computing today's stats
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
