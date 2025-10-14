# main.py
from pipeline.runner import run_pipeline
import pickle as pkl
import pandas as pd
import os, glob

DEFAULT_COLS = ['date','signal','target','qrank','stat_type','bet_size_col','value']

if __name__ == '__main__':
    result = run_pipeline()

    # If the runner returns a dict (new behavior), reconstruct a combined DataFrame.
    if isinstance(result, dict):
        daily_dir    = result.get('daily_dir')
        summary_path = result.get('summary_path')

        # Gather all daily frames
        daily_paths = sorted(glob.glob(os.path.join(daily_dir, 'stats_*.pkl'))) if daily_dir else []
        daily_frames = [pd.read_pickle(p) for p in daily_paths]
        stats_daily = (
            pd.concat(daily_frames, ignore_index=True)
            if daily_frames else pd.DataFrame(columns=DEFAULT_COLS)
        )

        # Read summary (optional)
        if summary_path and os.path.exists(summary_path):
            stats_summary = pd.read_pickle(summary_path)
        else:
            stats_summary = pd.DataFrame(columns=DEFAULT_COLS)

        # Combined DataFrame for convenience/backwards-compat
        parts = [df for df in (stats_daily, stats_summary) if not df.empty]
        stats_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=DEFAULT_COLS)

    else:
        # Old behavior: runner returned the DataFrame directly
        stats_df = result if isinstance(result, pd.DataFrame) else pd.DataFrame(columns=DEFAULT_COLS)

    print(f"\n📦 Loaded stats_df with shape: {stats_df.shape}")
    print("📄 Columns:", stats_df.columns.tolist())
    print("\n🔍 Preview of stats_df:")
    print(stats_df.head(10))

    # --- Backwards-compatible outputs in DAILY_SUMMARIES ---
    compat_dir = "./output/DAILY_SUMMARIES"
    os.makedirs(compat_dir, exist_ok=True)

    # Save pickle
    compat_pkl = os.path.join(compat_dir, "stats_tensor.pkl")
    stats_df.to_pickle(compat_pkl)

    # Also save CSV
    compat_csv = os.path.join(compat_dir, "stats_tensor.csv")
    stats_df.to_csv(compat_csv, index=False)

    # If you still want to demonstrate reloading the pickle:
    with open(compat_pkl, "rb") as f:
        obj = pkl.load(f)
    # obj is already a DataFrame; write again to CSV just to match your original flow
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(compat_csv, index=False)
