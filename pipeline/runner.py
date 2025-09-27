import os
import pandas as pd
import pickle
import numpy as np  # (optional) move import up

from pipeline.loader import load_daily_files
from pipeline.features import generate_signals_and_targets
from pipeline.stats import compute_daily_stats, compute_summary_stats_over_days  # NEW

RAW_DATA_DIR = 'data/RAW_DATA'
DAILY_SUMMARIES_DIR = 'output/DAILY_SUMMARIES'
os.makedirs(DAILY_SUMMARIES_DIR, exist_ok=True)

def run_pipeline():
    daily_data = load_daily_files(RAW_DATA_DIR)
    enriched_data = generate_signals_and_targets(daily_data)

    stats_dfs = []
    # NEW: keep a list of raw day DataFrames (with a date column) and unions of cols
    raw_days_for_summary = []  # NEW
    all_signal_cols = set()    # NEW
    all_target_cols = set()    # NEW
    all_bet_cols    = set()    # NEW

    for fname, df in enriched_data:
        # infer per-file columns
        signal_cols = [col for col in df.columns if col.startswith('pret_')]
        target_cols = [col for col in df.columns if col.startswith('fret_')]

        bet_size_cols = ['betsize_equal', 'betsize_cap200k', 'betsize_cap250k']
        valid_bet_size_cols = [b for b in bet_size_cols if b in df.columns]
        if not valid_bet_size_cols:
            print(f"Skipping {fname}: no valid bet size columns")
            continue  # <-- minimal guard

        # NEW: track unions and keep raw df with date for summary pass
        day_str = fname[:8]  # assumes YYYYMMDD in first 8 chars
        day_dt = pd.to_datetime(day_str, format='%Y%m%d', errors='coerce')
        raw_days_for_summary.append(df.assign(date=day_dt))  # NEW
        all_signal_cols.update(signal_cols)                  # NEW
        all_target_cols.update(target_cols)                  # NEW
        all_bet_cols.update(valid_bet_size_cols)             # NEW

        # Pass all valid bet sizes at once for 5D aggregation (DAILY metrics)
        stats = compute_daily_stats(
            df,
            signal_cols=signal_cols,
            target_cols=target_cols,
            quantiles=[1.0, 0.75, 0.5, 0.25],
            bet_size_cols=valid_bet_size_cols
        )

        # ---- MINIMAL FIX: define stat_type and sig_dict from stats.items() ----
        gen = (
            (day_str, s, t, q, stat_type, b, v)
            for stat_type, sig_dict in stats.items()
            for s, qd in sig_dict.items()
            for q, td in qd.items()
            for t, bd in td.items()
            for b, v in bd.items()
        )

        arr = np.array(list(gen), dtype=object)
        cols = ['date','signal','target','qrank','stat_type','bet_size_col','value']
        rows = [dict(zip(cols, rec)) for rec in arr]

        flat_df = pd.DataFrame(rows)
        stats_dfs.append(flat_df)
        print(f"Processed {fname}")

    # Combine all daily rows first
    stats_df = pd.concat(stats_dfs, ignore_index=True) if stats_dfs else pd.DataFrame(
        columns=['date','signal','target','qrank','stat_type','bet_size_col','value']
    )

    # ---------- NEW: compute SUMMARY metrics once across all days ----------
    if raw_days_for_summary:
        big_df = pd.concat(raw_days_for_summary, ignore_index=True)
        # Build sorted lists of columns available in the concatenated df
        sig_list = sorted([c for c in all_signal_cols if c in big_df.columns])
        tgt_list = sorted([c for c in all_target_cols if c in big_df.columns])
        bet_list = sorted([c for c in all_bet_cols    if c in big_df.columns])

        if sig_list and tgt_list and bet_list:
            summary = compute_summary_stats_over_days(
                big_df,
                date_col='date',
                signal_cols=sig_list,
                target_cols=tgt_list,
                quantiles=[1.0, 0.75, 0.5, 0.25],
                bet_size_cols=bet_list,
                type_quantile='cumulative',   # keep semantics minimal-change
                add_spearman=True,
                add_dcor=False
            )

            # Flatten nested summary dict to rows, set date to last day (single value per combo)
            last_date = pd.to_datetime(stats_df['date'], errors='coerce').max() if not stats_df.empty else big_df['date'].max()

            gen_sum = (
                (last_date.strftime('%Y%m%d') if pd.notna(last_date) else None, s, t, q, stat_type, b, v)
                for stat_type, sig_dict in summary.items()
                for s, qd in sig_dict.items()
                for q, td in qd.items()
                for t, bd in td.items()
                for b, v in bd.items()
            )
            arr_sum = np.array(list(gen_sum), dtype=object)
            if arr_sum.size > 0:
                cols = ['date','signal','target','qrank','stat_type','bet_size_col','value']
                rows_sum = [dict(zip(cols, rec)) for rec in arr_sum]
                summary_df = pd.DataFrame(rows_sum)
                # Append to the daily stats df
                stats_df = pd.concat([stats_df, summary_df], ignore_index=True)

    # Final tidy order
    stats_df = stats_df[['date', 'signal', 'target', 'qrank', 'stat_type', 'bet_size_col', 'value']]

    # Save
    pickle_path = os.path.join(DAILY_SUMMARIES_DIR, 'stats_tensor.pkl')
    stats_df.to_pickle(pickle_path)

    print(f"✅ Saved tensor stats (daily + summary) to {pickle_path}")
    return stats_df
