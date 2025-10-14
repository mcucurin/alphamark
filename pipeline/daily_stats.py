# =============================
# daily_stats.py (no dcor here)
# =============================
import os
import pickle
import numpy as np
from collections import defaultdict


def create_5d_stats():
    return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

# -----------------------------------------------------------------------------
# GLOBAL PERSISTENT STATE (kept in-memory + optional on-disk persistence)
# -----------------------------------------------------------------------------
_GLOBAL_PREV_STATE = {}

def get_trading_state():
    return _GLOBAL_PREV_STATE

def reset_trading_state():
    """No-op to avoid unintended resets; kept for API compatibility."""
    pass

# ---- Persistence helpers (optional) ----

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

def compute_daily_stats(
    df,
    signal_cols,
    target_cols,
    quantiles=[1.0, 0.75, 0.5, 0.25],
    bet_size_cols=['betsize_equal'],
    prev_state=None,
    type_quantile='cumulative'  # 'cumulative' (>=thr) or 'quantEach' (bucket)
):
    """
    Per-day cross-section stats (single date snapshot).

    Metrics written per (signal, qrank, target, bet):
      - pnl, ppd, sizeNotional, nrInstr (bet-independent), n_trades, ppt

    Continuity: pass the same `prev_state` dict across all days/years.
    If None, a process-global dict `_GLOBAL_PREV_STATE` is used.

    nrInstr definition:
      Count of instruments with finite and non-zero s_i within the quantile portfolio
      (based on |s| thresholds), independent of bet caps. If a `ticker` column exists,
      we count unique tickers meeting the criterion; otherwise, count rows.
    """
    import pandas as pd
    stats = create_5d_stats()

    if prev_state is None:
        prev_state = _GLOBAL_PREV_STATE

    # instrument id (optional, for exact trade counting)
    id_col = 'ticker' if 'ticker' in df.columns else None

    # Clean but DO NOT drop rows globally (we'll drop per-need later)
    df = df.replace([np.inf, -np.inf], np.nan).copy()

    trades_cache = {}

    for signal in signal_cols:
        if signal not in df.columns:
            continue

        s_all = df[signal].to_numpy(float)
        m_fin = np.isfinite(s_all)
        sabs_all = np.abs(s_all[m_fin])
        if sabs_all.size == 0:
            continue

        # thresholds/buckets for this signal on this day
        day_thresholds = {f'qr_{int(q*100)}': np.nanquantile(sabs_all, 1.0 - q) for q in quantiles}
        if type_quantile == 'quantEach':
            K = len(quantiles)
            probs = [i / K for i in range(K + 1)]
            edges = np.nanquantile(sabs_all, probs)

        for bet in bet_size_cols:
            if bet not in df.columns:
                continue

            for q in quantiles:
                qlabel = f'qr_{int(q*100)}'
                if type_quantile == 'cumulative':
                    mask = m_fin & (np.abs(s_all) >= day_thresholds[qlabel])
                else:  # 'quantEach'
                    j = quantiles.index(q) + 1
                    lo, hi = edges[j-1], edges[j]
                    mask = m_fin & (np.abs(s_all) >= lo) & (np.abs(s_all) <= hi)

                if not np.any(mask):
                    continue

                # -------- nrInstr (bet-independent) --------
                m_nz = mask & (s_all != 0.0)
                if id_col:
                    nr_instr_today = int(df.loc[m_nz, id_col].dropna().nunique())
                else:
                    nr_instr_today = int(np.sum(m_nz))

                # -------- base arrays for trade counting (needs bet) --------
                base_cols = [signal, bet] + ([id_col] if id_col else [])
                sub_base = df.loc[mask, base_cols].dropna(how='any')
                if not sub_base.empty:
                    s_base = sub_base[signal].to_numpy(float)
                    b_abs  = np.abs(sub_base[bet].to_numpy(float))
                    ids    = sub_base[id_col].to_numpy() if id_col else None
                else:
                    s_base = np.array([], float)
                    b_abs  = np.array([], float)
                    ids    = None

                key_sb = (signal, qlabel, bet)
                if key_sb not in trades_cache:
                    Bt_today = float(b_abs.sum()) if b_abs.size else 0.0
                    mean_bet = float(np.nanmean(b_abs)) if b_abs.size else 0.0

                    prev = prev_state.get(key_sb, {})
                    prev_Bt = prev.get('Bt', np.nan)
                    prev_mb = prev.get('mean_bet', np.nan)
                    prev_map = prev.get('pos_map', {}) if isinstance(prev.get('pos_map', {}), dict) else {}

                    if id_col and ids is not None and ids.size:
                        # exact per-instrument changed-position count
                        pos_today = (np.sign(s_base) * b_abs).astype(float)
                        pos_map_today = {inst: float(pos) for inst, pos in zip(ids, pos_today)}

                        day_trades = 0
                        for inst, pos in pos_map_today.items():
                            prev_pos = float(prev_map.get(inst, 0.0))
                            if abs(pos - prev_pos) > 0.0:
                                day_trades += 1
                        if prev_map:
                            gone = set(prev_map.keys()) - set(pos_map_today.keys())
                            day_trades += len(gone)

                        n_trades_today = float(day_trades)
                        trades_cache[key_sb] = {
                            'n_trades': n_trades_today,
                            'Bt_today': Bt_today,
                            'pos_map_today': pos_map_today,
                            'mean_bet_today': mean_bet,
                        }
                        prev_state[key_sb] = {
                            'Bt': Bt_today,
                            'mean_bet': mean_bet,
                            'pos_map': pos_map_today,
                        }
                    else:
                        # proxy using |ΔBt| / mean_bet
                        if np.isfinite(prev_Bt):
                            dBt = abs(Bt_today - prev_Bt)
                            denom = mean_bet if mean_bet > 0 else (prev_mb if np.isfinite(prev_mb) and prev_mb > 0 else np.nan)
                            n_trades_today = (dBt / denom) if np.isfinite(denom) and denom > 0 else 0.0
                        else:
                            n_trades_today = (Bt_today / mean_bet) if mean_bet > 0 else 0.0

                        trades_cache[key_sb] = {
                            'n_trades': float(n_trades_today) if np.isfinite(n_trades_today) else np.nan,
                            'Bt_today': Bt_today,
                            'pos_map_today': {},
                            'mean_bet_today': mean_bet,
                        }
                        prev_state[key_sb] = {
                            'Bt': Bt_today,
                            'mean_bet': mean_bet,
                            'pos_map': {},
                        }

                n_trades_today = trades_cache[key_sb]['n_trades']

                # ----- per-target metrics -----
                for target in target_cols:
                    if target not in df.columns:
                        continue
                    cols = [signal, target, bet]
                    if id_col: cols.append(id_col)
                    sub = df.loc[mask, cols].dropna(how='any')
                    if sub.empty:
                        continue

                    s = sub[signal].to_numpy(float)
                    y = sub[target].to_numpy(float)
                    b_abs = np.abs(sub[bet].to_numpy(float))

                    pnl_vec = np.sign(s) * y * b_abs
                    pnl = float(pnl_vec.sum())
                    Bt_today = float(b_abs.sum())

                    stats['pnl'][signal][qlabel][target][bet] = pnl
                    stats['ppd'][signal][qlabel][target][bet] = (pnl / Bt_today) if Bt_today > 0 else np.nan
                    stats['sizeNotional'][signal][qlabel][target][bet] = Bt_today
                    stats['nrInstr'][signal][qlabel][target][bet] = nr_instr_today
                    stats['n_trades'][signal][qlabel][target][bet] = float(n_trades_today) if np.isfinite(n_trades_today) else np.nan
                    ppt_today = (pnl / n_trades_today) if (np.isfinite(n_trades_today) and n_trades_today > 1e-12) else np.nan
                    stats['ppt'][signal][qlabel][target][bet] = float(ppt_today) if np.isfinite(ppt_today) else np.nan

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
]