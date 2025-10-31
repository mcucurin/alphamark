# runner.py
import os
import pandas as pd
import numpy as np
from tempfile import NamedTemporaryFile
import shutil

from pipeline.loader import load_daily_files
from pipeline.features import generate_signals_and_targets
from pipeline.daily_stats import compute_daily_stats, get_trading_state
from pipeline.summary_stats import compute_summary_stats_over_days
from pipeline.outliers_stats import compute_outliers, save_outliers  # PKL

RAW_DATA_DIR = 'data/RAW_DATA'

# === output layout (no sidecars / heatmaps) ===
OUTPUT_ROOT       = 'output'
DAILY_STATS_DIR   = os.path.join(OUTPUT_ROOT, 'DAILY_STATS')     # one file per day
SUMMARY_STATS_DIR = os.path.join(OUTPUT_ROOT, 'SUMMARY_STATS')   # one summary file
OUTLIERS_DIR      = os.path.join(OUTPUT_ROOT, 'OUTLIERS')        # outliers (PKL)
os.makedirs(DAILY_STATS_DIR, exist_ok=True)
os.makedirs(SUMMARY_STATS_DIR, exist_ok=True)
os.makedirs(OUTLIERS_DIR, exist_ok=True)

# Outliers config
OUTLIER_METRICS  = ['pnl', 'ppd', 'sizeNotional', 'nrInstr', 'n_trades', 'ppt']
OUTLIER_Z_THRESH = 3.0
OUTLIER_TOP_K    = 5

def _atomic_pickle_dump(df: pd.DataFrame, path: str) -> None:
    """Atomic write to avoid partial files."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as tmp:
        df.to_pickle(tmp.name)
        tmp_name = tmp.name
    shutil.move(tmp_name, path)

# ---------- year-aware helpers (override Jan-1 dips) ----------

def _snapshot_prev_book_counts(prev_state) -> dict:
    """
    Build a map (signal, qlabel, bet) -> count of open names in the carried book.
    Works with ID mode (pos_map) and proxy mode (Bt/mean_bet).
    """
    out = {}
    for key_sb, obj in prev_state.items():
        if not isinstance(key_sb, tuple) or len(key_sb) != 3:
            continue
        pos_map = obj.get('pos_map', {})
        if isinstance(pos_map, dict) and pos_map:
            out[key_sb] = len(pos_map)
        else:
            Bt = float(obj.get('Bt', 0.0) or 0.0)
            mb = float(obj.get('mean_bet', 0.0) or 0.0)
            out[key_sb] = int(round(Bt / mb)) if mb > 0 else 0
    return out

def _apply_year_opening_override(stats: dict, prev_counts: dict, override_if="zero_or_nan"):
    """
    Mutate `stats['n_trades']` (and `ppt`) in place on the first trading day of a year:
    if a key's n_trades is 0/NaN, set it to the carried book size from prev day.
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
    pnl_tree = stats.get('pnl', {})
    ppt_tree = stats.get('ppt', {})

    for signal, qdict in ntr_tree.items():
        for qlabel, tdict in qdict.items():
            for target, bdict in tdict.items():
                for bet, ntr_val in list(bdict.items()):
                    key_sb = (signal, qlabel, bet)
                    if key_sb not in prev_counts or not _should_override(ntr_val):
                        continue
                    new_ntr = float(prev_counts[key_sb])
                    # write n_trades
                    bdict[bet] = new_ntr
                    # recompute ppt = pnl / n_trades
                    pnl_val = ((((pnl_tree.get(signal, {})
                                           .get(qlabel, {})
                                           .get(target, {}))
                                           .get(bet, 0.0))))
                    if new_ntr > 1e-12:
                        (((ppt_tree.setdefault(signal, {})
                                     .setdefault(qlabel, {})
                                     .setdefault(target, {})))[bet]) = float(pnl_val) / new_ntr
                    else:
                        (((ppt_tree.setdefault(signal, {})
                                     .setdefault(qlabel, {})
                                     .setdefault(target, {})))[bet]) = np.nan

# ---------------------------------------------------------------

def run_pipeline():
    # 1) Load raw & add features (signals/targets/bets, etc.)
    daily_data = load_daily_files(RAW_DATA_DIR)
    enriched_data = generate_signals_and_targets(daily_data)

    # --- ensure chronological order (CRITICAL for stateful trades) ---
    items = []
    for fname, df in enriched_data:
        # filename is expected to start with YYYYMMDD
        day_str = fname[:8]
        day_dt = pd.to_datetime(day_str, format='%Y%m%d', errors='coerce')
        if pd.isna(day_dt):
            # fallback: if your loader gives a date column, use that; else skip
            if 'date' in df.columns:
                day_dt = pd.to_datetime(df['date'].iloc[0])
            else:
                continue
        items.append((day_dt, day_str, fname, df))
    items.sort(key=lambda x: x[0])  # sort by date ascending

    raw_days_for_summary = []   # keep raw per-day frames (with 'date') for summary pass
    all_signal_cols = set()
    all_target_cols = set()
    all_bet_cols    = set()

    daily_stats_frames = []     # for outliers
    per_day_index_rows = []     # optional index of written daily files

    prev_year = None
    # We'll read the live state from daily_stats' global container
    prev_state = get_trading_state()

    # 2) Per-day loop -> compute & save DAILY_STATS
    for day_dt, day_str, fname, df in items:
        # infer columns; keep this simple & explicit
        signal_cols = [c for c in df.columns if c.startswith('pret_')]
        target_cols = [c for c in df.columns if c.startswith('fret_')]
        bet_size_cols_all = ['betsize_equal', 'betsize_cap200k', 'betsize_cap250k']
        bet_size_cols = [b for b in bet_size_cols_all if b in df.columns]

        if not signal_cols or not target_cols or not bet_size_cols:
            print(f"Skipping {fname}: missing signals/targets/bets")
            continue

        # keep for summary
        raw_days_for_summary.append(df.assign(date=day_dt))
        all_signal_cols.update(signal_cols)
        all_target_cols.update(target_cols)
        all_bet_cols.update(bet_size_cols)

        # ---- YEAR-OPEN snapshot (before computing today's stats) ----
        year_changed = (prev_year is not None) and (day_dt.year != prev_year)
        prev_counts = _snapshot_prev_book_counts(prev_state) if year_changed else None

        # compute daily stats with CARRY behavior
        stats = compute_daily_stats(
            df,
            signal_cols=signal_cols,
            target_cols=target_cols,
            quantiles=[1.0, 0.75, 0.5, 0.25],
            bet_size_cols=bet_size_cols,
            type_quantile='cumulative',
            empty_day_policy='carry',
            report_empty_trades_as_nan=True,
        )

        # ---- YEAR-OPEN override: kill 0/NaN dips on first trading day ----
        if year_changed and prev_counts:
            _apply_year_opening_override(stats, prev_counts, override_if="zero_or_nan")

        # flatten nested dict -> rows
        rows = []
        for stat_type, sig_dict in stats.items():
            for s, qd in sig_dict.items():
                for q, td in qd.items():
                    for t, bd in td.items():
                        for b, v in bd.items():
                            rows.append((day_str, s, t, q, stat_type, b, v))

        if not rows:
            print(f"Skipping {fname}: no stats produced")
            continue

        cols = ['date','signal','target','qrank','stat_type','bet_size_col','value']
        day_df = pd.DataFrame(rows, columns=cols)

        out_path = os.path.join(DAILY_STATS_DIR, f'stats_{day_str}.pkl')
        _atomic_pickle_dump(day_df, out_path)
        per_day_index_rows.append({'date': day_str, 'path': out_path, 'n_rows': len(day_df)})
        daily_stats_frames.append(day_df)

        print(f"📦 Saved daily stats for {day_str} -> {out_path} ({len(day_df)} rows)")

        prev_year = day_dt.year  # advance year tracker

    # 3) SUMMARY_STATS over all days (single file)
    summary_path = None
    if raw_days_for_summary:
        big_df = pd.concat(raw_days_for_summary, ignore_index=True)
        sig_list = sorted([c for c in all_signal_cols if c in big_df.columns])
        tgt_list = sorted([c for c in all_target_cols if c in big_df.columns])
        bet_list = sorted([c for c in all_bet_cols    if c in big_df.columns])

        if sig_list and tgt_list and bet_list:
            # NOTE: this used to pass `df`; that was a bug. Use big_df.
            summary = compute_summary_stats_over_days(
                big_df,
                date_col="date",
                signal_cols=["pret_1_MR","pret_1_RR","pret_3_MR","pret_3_RR"],
                target_cols=["fret_1_MR","fret_1_RR","fret_3_MR","fret_3_RR"],
                bet_size_cols=["betsize_equal","betsize_cap200k","betsize_cap250k"],
                quantiles=[1.0,0.75,0.5,0.25],
                type_quantile="cumulative",
                add_spearman=True,
                add_dcor=True,
                n_jobs=-1,                # <- parallel
                backend="loky",           # process-based
            )

            # flatten summary -> rows; tag with date range
            all_days = pd.to_datetime(big_df['date'], errors='coerce')
            first_date = pd.to_datetime(all_days.min())
            last_date  = pd.to_datetime(all_days.max())
            date_tag   = f"{first_date:%Y%m%d}_{last_date:%Y%m%d}" if pd.notna(first_date) and pd.notna(last_date) else "summary"

            s_rows = []
            for stat_type, sig_dict in summary.items():
                for s, qd in sig_dict.items():
                    for q, td in qd.items():
                        for t, bd in td.items():
                            for b, v in bd.items():
                                s_rows.append((f"{last_date:%Y%m%d}" if pd.notna(last_date) else None, s, t, q, stat_type, b, v))

            if s_rows:
                cols = ['date','signal','target','qrank','stat_type','bet_size_col','value']
                summary_df = pd.DataFrame(s_rows, columns=cols)
                summary_path = os.path.join(SUMMARY_STATS_DIR, f'summary_stats_{date_tag}.pkl')
                _atomic_pickle_dump(summary_df, summary_path)
                print(f"✅ Saved summary stats -> {summary_path} ({len(summary_df)} rows)")
            else:
                print("Summary produced no rows; not saving.")
        else:
            print("No valid columns for summary stats; skipping summary save.")
    else:
        print("No daily data collected; skipping summary computation.")

    # 4) OUTLIERS over all daily stats
    outliers_path = None
    if daily_stats_frames:
        stats_all = pd.concat(daily_stats_frames, ignore_index=True)
        dates = pd.to_datetime(stats_all['date'], errors='coerce')
        first_date = pd.to_datetime(dates.min())
        last_date  = pd.to_datetime(dates.max())
        date_tag   = f"{first_date:%Y%m%d}_{last_date:%Y%m%d}" if pd.notna(first_date) and pd.notna(last_date) else "all"

        odf = compute_outliers(
            stats_all,
            stats_list=OUTLIER_METRICS,
            z_thresh=OUTLIER_Z_THRESH,
            top_k=OUTLIER_TOP_K,
        )
        outliers_path = os.path.join(OUTLIERS_DIR, f'outliers_{date_tag}.pkl')
        save_outliers(odf, outliers_path)
        print(f"⚠️  Saved outliers -> {outliers_path} ({len(odf)} rows)")
    else:
        print("No daily stats frames accumulated; skipping outlier computation.")

    # 5) Optional: write an index of daily stat files
    index_path = None
    if per_day_index_rows:
        index_df = pd.DataFrame(per_day_index_rows).sort_values('date')
        index_path = os.path.join(DAILY_STATS_DIR, '_index.csv')
        with NamedTemporaryFile(dir=os.path.dirname(index_path), delete=False, mode='w', newline='') as tmp:
            index_df.to_csv(tmp.name, index=False)
            tmp_name = tmp.name
        shutil.move(tmp_name, index_path)
        print(f"🧭 Wrote daily index -> {index_path}")

    # Return paths for downstream use
    return {
        'daily_dir': DAILY_STATS_DIR,
        'summary_path': summary_path,
        'outliers_path': outliers_path,
        'index_path': index_path,
    }

if __name__ == "__main__":
    run_pipeline()
