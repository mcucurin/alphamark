# =============================
# daily_stats.py (fast, safe)
# =============================
from __future__ import annotations

import os
import pickle
import numpy as np
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence
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
    """No-op to avoid unintended resets; kept for API compatibility."""
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

def compute_daily_stats(
    df,
    signal_cols: Sequence[str],
    target_cols: Sequence[str],
    quantiles: Sequence[float] = (1.0, 0.75, 0.5, 0.25),
    bet_size_cols: Sequence[str] = ('betsize_equal',),
    prev_state: MutableMapping | None = None,
    type_quantile: str = 'cumulative',  # 'cumulative' (>=thr) or 'quantEach' (bucket)
    # ---- RAW distribution rows (for plotting histograms) ----
    enable_distributions: bool = False,
    max_dist_samples_per_series: int = 50_000,
    random_state=None,
):
    """
    Per-day cross-section stats (single date snapshot).

    Metrics written per (signal, qrank, target, bet):
      - pnl, ppd, sizeNotional, nrInstr (bet-independent), n_trades, ppt

    Alpha metric stored once per (signal, qrank) under target='__ALL__', bet='__ALL__':
      - alpha_sum : sum(signal) over the selected quantile (signed)

    nrInstr:
      Count of instruments with finite and non-zero s_i within the quantile portfolio,
      independent of bet caps. If column `ticker` exists, we count unique tickers;
      otherwise, we count rows.
    """
    import pandas as pd

    rng = np.random.default_rng(random_state)
    stats = create_5d_stats()

    if prev_state is None:
        prev_state = _GLOBAL_PREV_STATE

    # ----------------------------
    # Column preparation (FAST)
    # ----------------------------
    # Identify instrument id (optional, for exact trade counting)
    id_col = 'ticker' if 'ticker' in df.columns else None

    # Convert only numeric metric columns to float64 once; exclude the ID column.
    want_numeric = (set(signal_cols) | set(target_cols) | set(bet_size_cols)) - ({id_col} if id_col else set())
    df_np: Dict[str, np.ndarray] = {}

    for col in want_numeric:
        if col not in df.columns:
            continue
        # Force float64 so we can safely assign NaN; then scrub non-finite.
        arr = pd.to_numeric(df[col], errors='coerce').to_numpy()
        arr = arr.astype('float64', copy=False)
        # Replace +/-inf with NaN (and keep existing NaNs)
        arr[~np.isfinite(arr)] = np.nan
        df_np[col] = arr

    # Keep the instrument IDs as-is (object/str); do NOT coerce to numeric.
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

        s = df_np[signal]                              # float64
        m_s_fin = np.isfinite(s)
        if not m_s_fin.any():
            continue

        s_abs = np.abs(s[m_s_fin])
        if s_abs.size == 0:
            continue

        # Prepare quantile logic for this signal/day
        if type_quantile == 'cumulative':
            # thresholds dict: qlabel -> cut
            thresholds = {
                _label_for_quantile(q): float(np.nanquantile(s_abs, 1.0 - q))
                for q in quantiles
            }
            # Edges unused in cumulative mode
            edges = None
        else:
            # quantEach: split |s| into K equal buckets
            K = len(quantiles)
            probs = np.linspace(0.0, 1.0, K + 1, dtype='float64')
            edges = np.nanquantile(s_abs, probs)
            thresholds = None  # unused

        # Cache sign(s) for PnL
        sgn = np.sign(s)

        # We will store alpha_sum once per (signal, qlabel)
        alpha_written = set()

        # Iterate quantiles in the given order
        for q in quantiles:
            qlabel = _label_for_quantile(q)

            if type_quantile == 'cumulative':
                thr = thresholds[qlabel]
                # mask_q: finite s and abs(s) >= thr
                # NOTE: include zeros only if thr==0 (rare); consistent with original.
                mask_q = m_s_fin & (np.abs(s) >= thr)
            else:
                # 'quantEach' — place into K buckets; this q is bucket j
                j = quantiles.index(q) + 1  # 1..K
                lo, hi = edges[j-1], edges[j]
                mask_q = m_s_fin & (np.abs(s) >= lo) & (np.abs(s) <= hi)

            if not mask_q.any():
                continue

            # -------- alpha_sum (store once per (signal, q)) --------
            if qlabel not in alpha_written:
                alpha_sum_today = float(np.nansum(s[mask_q]))
                stats['alpha_sum'][signal][qlabel]['__ALL__']['__ALL__'] = alpha_sum_today
                alpha_written.add(qlabel)

            # -------- nrInstr (bet-independent) --------
            m_nz = mask_q & (s != 0.0)
            if id_arr is not None:
                # unique instrument count among non-zero positions
                nr_instr_today = int(pd.Series(id_arr[m_nz]).dropna().nunique())
            else:
                nr_instr_today = int(np.sum(m_nz))

            # ----------------------------
            # Per-bet work (trades, sizeNotional part of PPD/PPT)
            # ----------------------------
            for bet in bet_size_cols:
                b = bet_abs.get(bet)
                if b is None:
                    continue

                b_fin = np.isfinite(b)
                mask_qb = mask_q & b_fin

                # ---- Trades counting (exact with IDs, fallback proxy) ----
                key_sb = (signal, qlabel, bet)
                prev = prev_state.get(key_sb, {})

                if id_arr is not None:
                    # exact: build today's pos per instrument within the quantile
                    pos_today = (sgn[mask_qb] * b[mask_qb]).astype('float64', copy=False)
                    ids_today = id_arr[mask_qb]

                    # Build map of id -> position (last one wins if duplicates)
                    pos_map_today = {}
                    # small, fast loop (vectorizing dict assignment is not worth it)
                    for inst, pos in zip(ids_today, pos_today):
                        pos_map_today[inst] = float(pos)

                    prev_map = prev.get('pos_map', {}) if isinstance(prev.get('pos_map', {}), dict) else {}

                    # Count changes vs. previous map
                    day_trades = 0
                    for inst, pos in pos_map_today.items():
                        prev_pos = float(prev_map.get(inst, 0.0))
                        if pos != prev_pos:
                            day_trades += 1
                    # positions that disappeared
                    if prev_map:
                        day_trades += len(set(prev_map.keys()) - set(pos_map_today.keys()))

                    n_trades_today = float(day_trades)

                    # Update state
                    prev_state[key_sb] = {
                        'Bt': float(np.nansum(b[mask_qb])) if mask_qb.any() else 0.0,
                        'mean_bet': float(np.nanmean(b[mask_qb])) if mask_qb.any() else 0.0,
                        'pos_map': pos_map_today,
                    }
                else:
                    # proxy using |ΔBt| / mean_bet
                    Bt_today = float(np.nansum(b[mask_qb])) if mask_qb.any() else 0.0
                    mean_bet = float(np.nanmean(b[mask_qb])) if mask_qb.any() else 0.0

                    prev_Bt = float(prev.get('Bt', np.nan)) if prev and 'Bt' in prev else np.nan
                    prev_mb = float(prev.get('mean_bet', np.nan)) if prev and 'mean_bet' in prev else np.nan

                    if np.isfinite(prev_Bt):
                        dBt = abs(Bt_today - prev_Bt)
                        denom = mean_bet if mean_bet > 0 else (prev_mb if np.isfinite(prev_mb) and prev_mb > 0 else np.nan)
                        n_trades_today = (dBt / denom) if np.isfinite(denom) and denom > 0 else 0.0
                    else:
                        n_trades_today = (Bt_today / mean_bet) if mean_bet > 0 else 0.0

                    prev_state[key_sb] = {
                        'Bt': Bt_today,
                        'mean_bet': mean_bet,
                        'pos_map': {},  # not used in proxy mode
                    }

                # ----------------------------
                # Per-target metrics
                # ----------------------------
                # Shared pieces for all targets for this (signal, q, bet)
                sign_s = sgn
                for target in target_cols:
                    y = df_np.get(target)
                    if y is None:
                        continue

                    y_fin = np.isfinite(y)
                    m = mask_qb & y_fin
                    if not m.any():
                        # Write nrInstr (bet-independent), but mark trades as NA for this target-day
                        # so PPT (pnl/trades) isn't computed on a mismatched universe.
                        stats['nrInstr'][signal][qlabel][target][bet]  = nr_instr_today
                        stats['n_trades'][signal][qlabel][target][bet] = np.nan  # <<< key change
                        # (Optionally also write ppt=np.nan explicitly if you want)
                        # stats['ppt'][signal][qlabel][target][bet] = np.nan
                        continue


                    # pnl = sum(sign(s) * y * |bet|)
                    pnl_vec = sign_s[m] * y[m] * b[m]
                    pnl = float(np.nansum(pnl_vec))
                    notional = float(np.nansum(b[m]))

                    stats['pnl'][signal][qlabel][target][bet]          = pnl
                    stats['ppd'][signal][qlabel][target][bet]          = (pnl / notional) if notional > 0 else np.nan
                    stats['sizeNotional'][signal][qlabel][target][bet] = notional
                    stats['nrInstr'][signal][qlabel][target][bet]      = nr_instr_today

                    ntr = float(n_trades_today) if np.isfinite(n_trades_today) else np.nan
                    stats['n_trades'][signal][qlabel][target][bet]     = ntr
                    stats['ppt'][signal][qlabel][target][bet]          = (pnl / ntr) if (np.isfinite(ntr) and ntr > 1e-12) else np.nan

        # ----------------------------
        # Optional raw distributions (thin sampling)
        # ----------------------------
        if enable_distributions:
            # fret_* raw values by target (independent of quantiles/bets)
            for target in target_cols:
                if target not in df_np:
                    continue
                y = df_np[target]
                y = y[np.isfinite(y)]
                if y.size == 0:
                    continue
                if y.size > max_dist_samples_per_series:
                    idx = rng.choice(y.size, size=max_dist_samples_per_series, replace=False)
                    y = y[idx]
                # Place under synthetic keys, so downstream can recognize histograms
                # We store each day's values as a small summary (mean) to keep store tiny;
                # switch to storing arrays if your downstream expects per-sample rows.
                stats['fret_value'][f'__S__{signal}']['__ALL__'][target]['__ALL__'] = float(np.nanmean(y))
            # betsize_* raw |bet|
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


# Convenience: iterate across multiple days with continuous state
def compute_series_continuous(df_sorted_by_date, date_col: str, **kwargs):
    import pandas as pd
    prev = kwargs.pop('prev_state', None)
    if prev is None:
        prev = _GLOBAL_PREV_STATE
    out = []
    for d, df_day in df_sorted_by_date.sort_values(date_col).groupby(date_col):
        out.append((pd.Timestamp(d), compute_daily_stats(df_day, prev_state=prev, **kwargs)))
    return out


__all__ = [
    'compute_daily_stats',
    'compute_series_continuous',
    'create_5d_stats',
    'get_trading_state',
    'save_trading_state',
    'load_trading_state',
    'reset_trading_state',
]
