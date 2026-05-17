"""
Annual Sharpe ratio breakdown.
Source: DAILY_STATS/ PKLs (one file per trading day).
For each (signal, target, qrank, bet_size_col) combination, per year:
    SR = mean(PnL_t) / std(PnL_t, ddof=1) * sqrt(252)   [benchmark Eq. 7]
where PnL_t includes ALL trading days in that year, with 0 for inactive days.
Active days (PnL != 0) are reported separately for transparency.
Filter: target=fret_OPCL_MR, bet_size_col=bs_m025_c100k
"""
import glob
import os
import pickle
import numpy as np
import pandas as pd

DAILY_DIR    = "output/DAILY_STATS"
TARGET_FILTER = "fret_OPCL_MR"
BET_FILTER   = "bs_m025_c100k"
TRADING_DAYS = 252
MIN_DAYS     = 20


# ── NumPy-compatible loader ───────────────────────────────────────────────────
class _NPCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)

def load_pkl(path: str):
    with open(path, "rb") as f:
        return _NPCompatUnpickler(f).load()


# ── Load all daily PKLs ───────────────────────────────────────────────────────
paths = sorted(glob.glob(os.path.join(DAILY_DIR, "stats_*.pkl")))
print(f"Loading {len(paths):,} daily PKL files from {DAILY_DIR}/ …")
frames = []
for p in paths:
    try:
        df = load_pkl(p)
        if isinstance(df, pd.DataFrame) and not df.empty:
            frames.append(df)
    except Exception as e:
        print(f"  [WARN] skipped {os.path.basename(p)}: {e}")

raw = pd.concat(frames, ignore_index=True)
total_calendar_days = len(frames)   # 6,261 — used for full-period Sharpe
print(f"  Loaded: {len(raw):,} rows across {total_calendar_days:,} days\n")


# ── Filter to PnL rows ───────────────────────────────────────────────────────
pnl = raw[
    (raw["stat_type"]    == "pnl")
  & (raw["target"]       == TARGET_FILTER)
  & (raw["bet_size_col"] == BET_FILTER)
  & (raw["target"]       != "__ALL__")
  & (raw["bet_size_col"] != "__ALL__")
  & (raw["qrank"]        != "__ALL__")
].copy()

pnl["date"]  = pd.to_datetime(pnl["date"], errors="coerce")
pnl["value"] = pd.to_numeric(pnl["value"], errors="coerce")
pnl          = pnl.dropna(subset=["date"])
pnl["year"]  = pnl["date"].dt.year

# Count trading days per calendar year from the loaded files
year_trading_days = (
    pd.Series([pd.Timestamp(os.path.basename(p).split("_")[1].split(".")[0])
               for p in paths])
    .dt.year.value_counts().to_dict()
)

GROUP_COLS = ["signal", "qrank"]


# ── Sharpe helper ─────────────────────────────────────────────────────────────
def sharpe(series: pd.Series, n_calendar_days: int) -> float:
    """
    SR = mean(PnL) / std(PnL, ddof=1) * sqrt(252).
    Inactive days (NaN or missing) are filled with 0 before computation.
    Pads to n_calendar_days if fewer rows than expected.
    Returns NaN if fewer than MIN_DAYS observations.
    """
    # Fill NaN with 0 — inactive days contribute zero PnL
    s = series.fillna(0.0)

    # Pad to full calendar length if signal has no row at all for some days
    if len(s) < n_calendar_days:
        padding = pd.Series(np.zeros(n_calendar_days - len(s)))
        s = pd.concat([s, padding], ignore_index=True)

    if len(s) < MIN_DAYS:
        return np.nan
    sd = s.std(ddof=1)
    if sd == 0:
        return np.nan
    return float(s.mean() / sd * np.sqrt(TRADING_DAYS))


# ── Print ─────────────────────────────────────────────────────────────────────
combos = pnl.groupby(GROUP_COLS, observed=True)

print(f"{'='*72}")
print(f"  Annual Sharpe  |  Target: {TARGET_FILTER}  |  Bet: {BET_FILTER}")
print(f"{'='*72}\n")

for (signal, qrank), grp in combos:
    print(f"  Signal: {signal}   QRank: {qrank}")
    print(f"  {'-'*66}")
    print(f"  {'Year':>6}  {'Total Days':>11}  {'Active Days':>12}  {'Sharpe':>10}")
    print(f"  {'-'*66}")

    for year, yr_grp in grp.groupby("year"):
        n_cal = year_trading_days.get(int(year), TRADING_DAYS)
        total  = int(yr_grp["value"].notna().sum())
        active = int((yr_grp["value"].fillna(0) != 0).sum())
        sr     = sharpe(yr_grp["value"], n_cal)
        sr_str = f"{sr:+.3f}" if np.isfinite(sr) else "   n/a"
        print(f"  {int(year):>6}  {total:>11}  {active:>12}  {sr_str:>10}")

    print(f"  {'-'*66}")

    # Full period: use actual total calendar days loaded
    full_total  = int(grp["value"].notna().sum())
    full_active = int((grp["value"].fillna(0) != 0).sum())
    full_sr     = sharpe(grp["value"], total_calendar_days)
    full_str    = f"{full_sr:+.3f}" if np.isfinite(full_sr) else "   n/a"
    print(f"  {'FULL':>6}  {full_total:>11}  {full_active:>12}  {full_str:>10}")
    print()