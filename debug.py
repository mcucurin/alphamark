"""
debug_pnl.py — inspect what's actually in the daily PKL rows
Run from: /Users/dhruvpatel/financial_pipeline/
"""
import glob, os, pickle, numpy as np, pandas as pd

DAILY_DIR = "output/DAILY_STATS"

class _NPCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)

def load_pkl(path):
    with open(path, "rb") as f:
        return _NPCompatUnpickler(f).load()

paths = sorted(glob.glob(os.path.join(DAILY_DIR, "stats_*.pkl")))
print(f"Total PKL files: {len(paths)}")
print(f"Sample filenames: {[os.path.basename(p) for p in paths[:3]]}")
print()

# Check year_trading_days parsing
from datetime import datetime
year_counts = {}
for p in paths:
    stem = os.path.basename(p).replace("stats_","").replace(".pkl","")
    try:
        dt = datetime.strptime(stem, "%Y%m%d")
        year_counts[dt.year] = year_counts.get(dt.year, 0) + 1
    except:
        print(f"  PARSE FAIL: {stem}")
print("Trading days per year (from filenames):")
for y in sorted(year_counts):
    print(f"  {y}: {year_counts[y]}")
print()

# Load one PKL and inspect structure
p = paths[100]
df = load_pkl(p)
print(f"Sample PKL: {os.path.basename(p)}")
print(f"  Type: {type(df)}")
if isinstance(df, pd.DataFrame):
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Sample rows:")
    print(df.head(10).to_string())
    print()
    # Check if there are rows with pnl=0
    pnl_rows = df[df["stat_type"]=="pnl"] if "stat_type" in df.columns else pd.DataFrame()
    print(f"  PNL rows: {len(pnl_rows)}")
    if not pnl_rows.empty:
        print(f"  PNL values sample: {pnl_rows['value'].head(10).tolist()}")
        print(f"  Any zero PNL: {(pnl_rows['value']==0).any()}")
        print(f"  Any NaN PNL:  {pnl_rows['value'].isna().any()}")
print()

# Now check year 2000: how many PKL files vs how many rows loaded
print("Year 2000 check:")
y2000 = [p for p in paths if "stats_2000" in p]
print(f"  PKL files for 2000: {len(y2000)}")
frames_2000 = []
for p in y2000[:5]:  # just first 5
    df = load_pkl(p)
    if isinstance(df, pd.DataFrame):
        pnl = df[
            (df["stat_type"]=="pnl") &
            (df.get("target", pd.Series(dtype=str)) == "fret_OPCL_MR") &
            (df.get("bet_size_col", pd.Series(dtype=str)) == "bs_m025_c100k") &
            (df.get("signal", pd.Series(dtype=str)) == "pret_dcsf_base")
        ] if "stat_type" in df.columns else pd.DataFrame()
        has_signal = len(pnl) > 0
        print(f"  {os.path.basename(p)}: has_signal={has_signal}, pnl_rows={len(pnl)}")
        if has_signal:
            print(f"    value={pnl['value'].values}")