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

    all_stats = []
    for fname, df in enriched_data:
        signal_cols = [col for col in df.columns if col.startswith('signal_')]
        target_cols = [col for col in df.columns if col.startswith('fret_')]
        stats = compute_daily_stats(df, signal_cols, target_cols)

        for stat_type, sig_dict in stats.items():
            for signal, qrank_dict in sig_dict.items():
                for qrank, tgt_dict in qrank_dict.items():
                    for target, value in tgt_dict.items():
                        stat_row = {
                            'date': fname[:8],
                            'signal': signal,
                            'target': target,
                            'qrank': qrank,
                            'stat_type': stat_type,
                            'value': value
                        }
                        all_stats.append(stat_row)

        print(f"Processed {fname}")

    stats_df = pd.DataFrame(all_stats)
    pickle_path = os.path.join(DAILY_SUMMARIES_DIR, 'stats_tensor.pkl')
    stats_df.to_pickle(pickle_path)
    print(f"Saved tensor stats to {pickle_path}")
    return stats_df
