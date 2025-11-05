# main.py
from pipeline.runner import run_pipeline
import pickle as pkl
import pandas as pd
import os, glob
import time

start = time.perf_counter()

DEFAULT_COLS = ['date','signal','target','qrank','stat_type','bet_size_col','value']

# --- NEW: NumPy 1.x/2.x compatible pickle loader ---
def read_pickle_compat(path: str):
    """Unpickle objects across NumPy 1.x/2.x by remapping numpy._core -> numpy.core."""
    class NPCompatUnpickler(pkl.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core")
            return super().find_class(module, name)
    with open(path, "rb") as f:
        return NPCompatUnpickler(f).load()

if __name__ == '__main__':
    result = run_pipeline()

    # If the runner returns a dict (new behavior), reconstruct a combined DataFrame.
    if isinstance(result, dict):
        daily_dir    = result.get('daily_dir')
        summary_path = result.get('summary_path')

        # Gather all daily frames
        daily_paths = sorted(glob.glob(os.path.join(daily_dir, 'stats_*.pkl'))) if daily_dir else []
        # CHANGED: use compat loader instead of pd.read_pickle
        daily_frames = [read_pickle_compat(p) for p in daily_paths]
        stats_daily = (
            pd.concat(daily_frames, ignore_index=True)
            if daily_frames else pd.DataFrame(columns=DEFAULT_COLS)
        )

        # Read summary (optional)
        if summary_path and os.path.exists(summary_path):
            # CHANGED: use compat loader instead of pd.read_pickle
            stats_summary = read_pickle_compat(summary_path)
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

end = time.perf_counter()
print(f"Time taken: {end - start} seconds")
