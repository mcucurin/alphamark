import os
import pandas as pd
import pickle

from pipeline.loader import load_daily_files
from pipeline.features import generate_signals_and_targets
from pipeline.stats import compute_daily_stats

RAW_DATA_DIR = 'data/RAW_DATA'
DAILY_SUMMARIES_DIR = 'output/DAILY_SUMMARIES'
os.makedirs(DAILY_SUMMARIES_DIR, exist_ok=True)

def run_pipeline():
    daily_data = load_daily_files(RAW_DATA_DIR)
    enriched_data = generate_signals_and_targets(daily_data)

    stats_dfs = []
    for fname, df in enriched_data:
        signal_cols = [col for col in df.columns if col.startswith('pret_')]
        target_cols = [col for col in df.columns if col.startswith('fret_')]

        bet_size_cols = ['betsize_equal', 'betsize_cap200k', 'betsize_cap250k']
        valid_bet_size_cols = [b for b in bet_size_cols if b in df.columns]

        # Pass all valid bet sizes at once for 5D aggregation
        stats = compute_daily_stats(
            df,
            signal_cols=signal_cols,
            target_cols=target_cols,
            quantiles=[1.0, 0.75, 0.5, 0.25],
            bet_size_col=valid_bet_size_cols
        )

        # Flatten the 5D stats
        rows = []
        for stat_type, sig_dict in stats.items():
            for signal, qrank_dict in sig_dict.items():
                for qrank, tgt_dict in qrank_dict.items():
                    for target, bsz_dict in tgt_dict.items():
                        for bet_size_col, value in bsz_dict.items():
                            rows.append({
                                'date': fname[:8],
                                'signal': signal,
                                'target': target,
                                'qrank': qrank,
                                'stat_type': stat_type,
                                'bet_size_col': bet_size_col,
                                'value': value
                            })

        flat_df = pd.DataFrame(rows)
        stats_dfs.append(flat_df)
        print(f"Processed {fname}")

    # Combine and save all days
    stats_df = pd.concat(stats_dfs, ignore_index=True)
    stats_df = stats_df[['date', 'signal', 'target', 'qrank', 'stat_type', 'bet_size_col', 'value']]

    pickle_path = os.path.join(DAILY_SUMMARIES_DIR, 'stats_tensor.pkl')
    stats_df.to_pickle(pickle_path)

    print(f"✅ Saved tensor stats with bet sizes to {pickle_path}")
    return stats_df
