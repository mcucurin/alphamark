#!/usr/bin/env python3
# debug/early_inflation_debug.py
#
# Generates diagnostics to investigate early inflation in PPD (cum_pnl / cum_gross_notional)
# and PPT (cum_pnl / cum_trades). Outputs a single CSV:
#   output/early_inflation_debug.csv
#
# Run:
#   python debug/early_inflation_debug.py

import os
import glob
import math
import pandas as pd
import numpy as np

# -------------------------
# Tunables / thresholds
# -------------------------
EARLY_WINDOW_DAYS   = 60   # compare first 60d vs the next 60d
LATE_WINDOW_DAYS    = 60
GROSS_DEN_MIN       = 1e-6 # cum gross notional <= this -> effectively zero
TRADES_DEN_MIN      = 1    # cum trades must be >= this to be “valid”
INFLATION_FACTOR    = 2.0  # if early > INFLATION_FACTOR * later, flag inflation

DEFAULT_COLS = ['date','signal','target','qrank','stat_type','bet_size_col','value']

# -------------------------
# Data loading (self-contained)
# -------------------------
def _load_stats_df_new():
    daily_dir   = os.path.join("output", "DAILY_STATS")
    summary_dir = os.path.join("output", "SUMMARY_STATS")

    if not os.path.isdir(daily_dir):
        raise FileNotFoundError("Expected 'output/DAILY_STATS' with per-day pickles (stats_YYYYMMDD.pkl).")

    daily_paths = sorted(glob.glob(os.path.join(daily_dir, "stats_*.pkl")))
    if not daily_paths:
        raise FileNotFoundError("No daily files found in 'output/DAILY_STATS'.")

    daily_frames = [pd.read_pickle(p) for p in daily_paths]
    stats_daily  = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(columns=DEFAULT_COLS)

    if os.path.isdir(summary_dir):
        summary_paths = sorted(glob.glob(os.path.join(summary_dir, "summary_stats_*.pkl")))
    else:
        summary_paths = []

    stats_summary = pd.read_pickle(summary_paths[-1]) if summary_paths else pd.DataFrame(columns=DEFAULT_COLS)

    parts = [df for df in (stats_daily, stats_summary) if not df.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=DEFAULT_COLS)

# -------------------------
# Helpers
# -------------------------
def _merge_three_stats(sub_q: pd.DataFrame) -> pd.DataFrame:
    """Return date-indexed frame with pnl, notional, trades, and cumulative/gross/ratios."""
    pnl_daily = (sub_q[sub_q['stat_type']=='pnl'][['date','value']]
                 .rename(columns={'value':'pnl'}))
    ntl_daily = (sub_q[sub_q['stat_type']=='sizeNotional'][['date','value']]
                 .rename(columns={'value':'notional'}))
    trd_daily = (sub_q[sub_q['stat_type']=='n_trades'][['date','value']]
                 .rename(columns={'value':'trades'}))

    # Union of dates to keep alignment transparent
    dates = pd.to_datetime(pd.concat([
        pnl_daily['date'], ntl_daily['date'], trd_daily['date']
    ]).dropna().unique())
    merged = pd.DataFrame({'date': dates}).sort_values('date')

    merged = merged.merge(pnl_daily, on='date', how='left')
    merged = merged.merge(ntl_daily, on='date', how='left')
    merged = merged.merge(trd_daily, on='date', how='left')
    merged[['pnl','notional','trades']] = merged[['pnl','notional','trades']].fillna(0.0)

    # Derived
    merged['gross_notional']      = merged['notional'].abs()
    merged['cum_pnl']             = merged['pnl'].cumsum()
    merged['cum_notional']        = merged['notional'].cumsum()
    merged['cum_gross_notional']  = merged['gross_notional'].cumsum()
    merged['cum_trades']          = merged['trades'].cumsum()

    # Ratios (guarded)
    merged['cum_ppd'] = np.where(merged['cum_gross_notional'] > 0,
                                 merged['cum_pnl'] / merged['cum_gross_notional'],
                                 np.nan)
    merged['cum_ppt'] = np.where(merged['cum_trades'] > 0,
                                 merged['cum_pnl'] / merged['cum_trades'],
                                 np.nan)
    return merged

def _first_valid_idx(s: pd.Series, thresh: float, strictly_greater=True):
    """First index (timestamp) where s crosses threshold."""
    if s.empty:
        return None
    if strictly_greater:
        mask = s > thresh
    else:
        mask = s >= thresh
    if not mask.any():
        return None
    # return label (index is datetime index later)
    return mask.idxmax()

def _window_avg(series: pd.Series, start_ts, days: int):
    """Average of series between [start_ts, start_ts+days)."""
    if start_ts is None or pd.isna(start_ts):
        return np.nan
    end_ts = start_ts + pd.Timedelta(days=days)
    win = series[(series.index >= start_ts) & (series.index < end_ts)]
    return float(win.mean()) if len(win) else np.nan

def _safe_ratio(a, b):
    if b is None or (isinstance(b, float) and math.isnan(b)) or abs(b) == 0:
        return np.nan
    return a / b

# -------------------------
# Main
# -------------------------
def main():
    os.makedirs("output", exist_ok=True)

    stats_df = _load_stats_df_new()
    if stats_df.empty:
        raise RuntimeError("Loaded empty stats_df; nothing to debug.")

    # Basic hygiene
    stats_df['date'] = pd.to_datetime(stats_df['date'], errors='coerce')
    stats_df = stats_df.dropna(subset=['date', 'value'])

    # Limit to rows that can feed diagnostics
    needed = stats_df['stat_type'].isin(['pnl', 'sizeNotional', 'n_trades'])
    if not needed.any():
        raise RuntimeError("No rows with stat_type in ['pnl','sizeNotional','n_trades'] found.")

    # Iterate only existing combos to avoid empty work
    combos = (stats_df.loc[needed, ['target','signal','bet_size_col','qrank']]
              .dropna()
              .drop_duplicates()
              .itertuples(index=False, name=None))

    debug_rows = []

    for tgt, sig, bet, q in combos:
        sub = stats_df[
            (stats_df['target'] == tgt) &
            (stats_df['signal'] == sig) &
            (stats_df['bet_size_col'] == bet) &
            (stats_df['qrank'] == q) &
            (stats_df['stat_type'].isin(['pnl','sizeNotional','n_trades']))
        ]
        if sub.empty:
            continue

        merged = _merge_three_stats(sub).set_index('date')

        # Date alignment diagnostics
        pnl_dates = set(sub[sub['stat_type']=='pnl']['date'].dt.normalize())
        ntl_dates = set(sub[sub['stat_type']=='sizeNotional']['date'].dt.normalize())
        trd_dates = set(sub[sub['stat_type']=='n_trades']['date'].dt.normalize())

        pnl_only_days = len(pnl_dates - ntl_dates)
        ntl_only_days = len(ntl_dates - pnl_dates)

        # First valid denominators
        first_gross_ts  = _first_valid_idx(merged['cum_gross_notional'], GROSS_DEN_MIN, strictly_greater=True)
        first_trades_ts = _first_valid_idx(merged['cum_trades'], TRADES_DEN_MIN - 1e-9, strictly_greater=True)

        # Early vs next windows
        ppd_early = _window_avg(merged['cum_ppd'], first_gross_ts, EARLY_WINDOW_DAYS)
        ppd_next  = _window_avg(
            merged['cum_ppd'],
            (first_gross_ts + pd.Timedelta(days=EARLY_WINDOW_DAYS)) if first_gross_ts is not None else None,
            LATE_WINDOW_DAYS
        )

        ppt_early = _window_avg(merged['cum_ppt'], first_trades_ts, EARLY_WINDOW_DAYS)
        ppt_next  = _window_avg(
            merged['cum_ppt'],
            (first_trades_ts + pd.Timedelta(days=EARLY_WINDOW_DAYS)) if first_trades_ts is not None else None,
            LATE_WINDOW_DAYS
        )

        # Early denominator magnitudes (snapshot near start)
        gross_start = _window_avg(merged['cum_gross_notional'], first_gross_ts, 1)
        trades_start = _window_avg(merged['cum_trades'], first_trades_ts, 1)

        # Flags
        ppd_inflated = (
            (not math.isnan(ppd_early) and not math.isnan(ppd_next) and abs(ppd_next) > 0 and
             abs(ppd_early) > INFLATION_FACTOR * abs(ppd_next))
            or (gross_start is not None and not math.isnan(gross_start) and gross_start <= GROSS_DEN_MIN)
        )

        ppt_inflated = (
            (not math.isnan(ppt_early) and not math.isnan(ppt_next) and abs(ppt_next) > 0 and
             abs(ppt_early) > INFLATION_FACTOR * abs(ppt_next))
            or (trades_start is not None and not math.isnan(trades_start) and trades_start < TRADES_DEN_MIN)
        )

        debug_rows.append({
            'target': tgt,
            'signal': sig,
            'bet_size_col': bet,
            'qrank': q,

            # Coverage / alignment
            'pnl_only_days': int(pnl_only_days),
            'notional_only_days': int(ntl_only_days),
            'trade_days': int(len(trd_dates)),

            # First valid dates
            'first_gross_notional_date': pd.NaT if first_gross_ts is None else pd.Timestamp(first_gross_ts),
            'first_trade_date':          pd.NaT if first_trades_ts is None else pd.Timestamp(first_trades_ts),

            # Early vs next windows
            f'PPD_mean_first_{EARLY_WINDOW_DAYS}d': ppd_early,
            f'PPD_mean_next_{LATE_WINDOW_DAYS}d':  ppd_next,
            f'PPT_mean_first_{EARLY_WINDOW_DAYS}d': ppt_early,
            f'PPT_mean_next_{LATE_WINDOW_DAYS}d':  ppt_next,

            # Early denominator magnitudes
            'cum_gross_notional_at_start': gross_start,
            'cum_trades_at_start':         trades_start,

            # Flags
            'PPD_suspected_early_inflation': bool(ppd_inflated),
            'PPT_suspected_early_inflation': bool(ppt_inflated),
        })

    debug_df = pd.DataFrame(debug_rows)

    # Sort: suspicious first, then by keys
    if not debug_df.empty:
        debug_df = debug_df.sort_values(
            ['PPD_suspected_early_inflation','PPT_suspected_early_inflation','target','signal','bet_size_col','qrank'],
            ascending=[False, False, True, True, True, True]
        )

    out_path = os.path.join("output", "early_inflation_debug.csv")
    debug_df.to_csv(out_path, index=False)
    print(f"[DEBUG] Wrote diagnostics to {out_path} (rows={len(debug_df)})")

if __name__ == "__main__":
    main()
