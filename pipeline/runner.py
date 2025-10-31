# runner.py — fast daily loop + lean summary (no year-open overrides)
import os
import pandas as pd
import numpy as np
from tempfile import NamedTemporaryFile
import shutil

from pipeline.loader import load_daily_files
from pipeline.features import generate_signals_and_targets
from pipeline.daily_stats import compute_daily_stats
from pipeline.summary_stats import compute_summary_stats_over_days
from pipeline.outliers_stats import compute_outliers, save_outliers  # PKL

# ======== CONFIG ========
RAW_DATA_DIR = 'data/RAW_DATA'

OUTPUT_ROOT       = 'output'
DAILY_STATS_DIR   = os.path.join(OUTPUT_ROOT, 'DAILY_STATS')
SUMMARY_STATS_DIR = os.path.join(OUTPUT_ROOT, 'SUMMARY_STATS')
OUTLIERS_DIR      = os.path.join(OUTPUT_ROOT, 'OUTLIERS')
os.makedirs(DAILY_STATS_DIR, exist_ok=True)
os.makedirs(SUMMARY_STATS_DIR, exist_ok=True)
os.makedirs(OUTLIERS_DIR, exist_ok=True)

# Outliers: z-score based (top-K rule removed in compute_outliers per your change)
OUTLIER_METRICS  = ['pnl', 'ppd', 'sizeNotional', 'nrInstr', 'n_trades', 'ppt']
OUTLIER_Z_THRESH = 3.0

# ======== UTIL ========
def _atomic_pickle_dump(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as tmp:
        df.to_pickle(tmp.name)
        tmp_name = tmp.name
    shutil.move(tmp_name, path)

# ============================== PIPELINE ==============================
def run_pipeline():
    # 1) Load & feature
    daily_data = load_daily_files(RAW_DATA_DIR)
    enriched_data = generate_signals_and_targets(daily_data)

    # chronological order
    items = []
    for fname, df in enriched_data:
        day_str = fname[:8]
        day_dt = pd.to_datetime(day_str, format='%Y%m%d', errors='coerce')
        if pd.isna(day_dt):
            if 'date' in df.columns:
                day_dt = pd.to_datetime(df['date'].iloc[0])
            else:
                continue
        items.append((day_dt, day_str, fname, df))
    items.sort(key=lambda x: x[0])

    daily_stats_frames = []     # for outliers
    per_day_index_rows = []

    # Quantile config
    quantiles = [1.0, 0.75, 0.5, 0.25]
    type_quantile = 'cumulative'

    # Collect minimal per-day frames for summary (only needed columns)
    raw_days_for_summary = []
    needed_sig = set()
    needed_tgt = set()
    needed_bet = set()

    # 2) Per-day loop
    for day_dt, day_str, fname, df in items:
        signal_cols = [c for c in df.columns if c.startswith('pret_')]
        target_cols = [c for c in df.columns if c.startswith('fret_')]
        bet_size_cols_all = ['betsize_equal', 'betsize_cap200k', 'betsize_cap250k']
        bet_size_cols = [b for b in bet_size_cols_all if b in df.columns]

        if not signal_cols or not target_cols or not bet_size_cols:
            print(f"Skipping {fname}: missing signals/targets/bets")
            continue

        needed_sig.update(signal_cols)
        needed_tgt.update(target_cols)
        needed_bet.update(bet_size_cols)

        # DAILY STATS (stateful carry lives inside compute_daily_stats)
        stats = compute_daily_stats(
            df,
            signal_cols=signal_cols,
            target_cols=target_cols,
            quantiles=quantiles,
            bet_size_cols=bet_size_cols,
            type_quantile=type_quantile,
            empty_day_policy='carry',
            report_empty_trades_as_nan=True,
        )

        # SAVE daily stats
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

        day_df_stats = pd.DataFrame(
            rows,
            columns=['date','signal','target','qrank','stat_type','bet_size_col','value']
        )
        out_path = os.path.join(DAILY_STATS_DIR, f'stats_{day_str}.pkl')
        _atomic_pickle_dump(day_df_stats, out_path)
        per_day_index_rows.append({'date': day_str, 'path': out_path, 'n_rows': len(day_df_stats)})
        daily_stats_frames.append(day_df_stats)
        print(f"📦 Saved daily stats for {day_str} -> {out_path} ({len(day_df_stats)} rows)")

        # Keep a LEAN frame for summary (only needed columns + date)
        keep_cols = ['date'] + signal_cols + target_cols + bet_size_cols
        raw_days_for_summary.append(
            df[signal_cols + target_cols + bet_size_cols].assign(date=day_dt)[keep_cols]
        )

    # 3) FINALIZE SUMMARY using optimized summary_stats.py
    summary_path = None
    if raw_days_for_summary:
        big_df = pd.concat(raw_days_for_summary, ignore_index=True, copy=False)

        sig_list = sorted([c for c in needed_sig if c in big_df.columns])
        tgt_list = sorted([c for c in needed_tgt if c in big_df.columns])
        bet_list = sorted([c for c in needed_bet if c in big_df.columns])

        if sig_list and tgt_list and bet_list:
            # Fast: disable expensive correlations unless needed
            summary = compute_summary_stats_over_days(
                big_df,
                date_col="date",
                signal_cols=sig_list,
                target_cols=tgt_list,
                bet_size_cols=bet_list,
                quantiles=quantiles,
                type_quantile=type_quantile,
                add_spearman=False,
                add_dcor=False,
                # If your summary_stats supports these knobs, keep them; else remove:
                spearman_sample_cap_per_key=10000,
                random_state=123,
            )

            # flatten summary -> rows; tag with date range
            all_days = pd.to_datetime(big_df['date'], errors='coerce')
            first_date = pd.to_datetime(all_days.min())
            last_date  = pd.to_datetime(all_days.max())
            date_tag   = (
                f"{first_date:%Y%m%d}_{last_date:%Y%m%d}"
                if pd.notna(first_date) and pd.notna(last_date) else "summary"
            )

            s_rows = []
            for stat_type, sig_dict in summary.items():
                for s, qd in sig_dict.items():
                    for q, td in qd.items():
                        for t, bd in td.items():
                            for b, v in bd.items():
                                s_rows.append((
                                    f"{last_date:%Y%m%d}" if pd.notna(last_date) else None,
                                    s, t, q, stat_type, b, v
                                ))

            if s_rows:
                summary_df = pd.DataFrame(
                    s_rows,
                    columns=['date','signal','target','qrank','stat_type','bet_size_col','value']
                )
                summary_path = os.path.join(SUMMARY_STATS_DIR, f'summary_stats_{date_tag}.pkl')
                _atomic_pickle_dump(summary_df, summary_path)
                print(f"✅ Saved summary stats -> {summary_path} ({len(summary_df)} rows)")
            else:
                print("Summary produced no rows; not saving.")
        else:
            print("No valid columns for summary stats; skipping summary save.")
    else:
        print("No daily data collected; skipping summary computation.")

    # 4) OUTLIERS from all daily stat frames (z-score only in compute_outliers)
    outliers_path = None
    if daily_stats_frames:
        stats_all = pd.concat(daily_stats_frames, ignore_index=True, copy=False)
        dates = pd.to_datetime(stats_all['date'], errors='coerce')
        first_date = pd.to_datetime(dates.min())
        last_date  = pd.to_datetime(dates.max())
        date_tag   = (
            f"{first_date:%Y%m%d}_{last_date:%Y%m%d}"
            if pd.notna(first_date) and pd.notna(last_date) else "all"
        )

        odf = compute_outliers(
            stats_all,
            stats_list=OUTLIER_METRICS,
            z_thresh=OUTLIER_Z_THRESH,
        )
        outliers_path = os.path.join(OUTLIERS_DIR, f'outliers_{date_tag}.pkl')
        save_outliers(odf, outliers_path)
        print(f"⚠️  Saved outliers -> {outliers_path} ({len(odf)} rows)")
    else:
        print("No daily stats frames accumulated; skipping outlier computation.")

    # 5) Index of daily files
    index_path = None
    if per_day_index_rows:
        index_df = pd.DataFrame(per_day_index_rows).sort_values('date')
        index_path = os.path.join(DAILY_STATS_DIR, '_index.csv')
        with NamedTemporaryFile(dir=os.path.dirname(index_path), delete=False, mode='w', newline='') as tmp:
            index_df.to_csv(tmp.name, index=False)
            tmp_name = tmp.name
        shutil.move(tmp_name, index_path)
        print(f"🧭 Wrote daily index -> {index_path}")

    return {
        'daily_dir': DAILY_STATS_DIR,
        'summary_path': summary_path,
        'outliers_path': outliers_path,
        'index_path': index_path,
    }

if __name__ == "__main__":
    run_pipeline()
