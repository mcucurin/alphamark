# pipeline/outliers_stats.py
from __future__ import annotations
import os
import pickle
import numpy as np
import pandas as pd
from tempfile import NamedTemporaryFile
from typing import Iterable, List

def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    m = s.mean(skipna=True)
    v = s.std(skipna=True, ddof=1)
    if not np.isfinite(v) or v == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - m) / v

def compute_outliers(
    stats_df: pd.DataFrame,
    stats_list: Iterable[str],
    z_thresh: float = 3.0,  # ignored (kept for signature compatibility)
) -> pd.DataFrame:
    """
    Compute outliers per requested stat_type using global z-scores on 'value'.
    Expects a 'long' frame with columns:
      ['date','signal','target','qrank','stat_type','bet_size_col','value'] (at minimum).

    Returns a concatenated DataFrame that includes all rows (not just the extremes),
    with columns:
      date, signal, target, qrank, stat_type, bet_size_col, value, z, abs_z, is_outlier, rule
    """
    if stats_df is None or len(stats_df) == 0:
        return pd.DataFrame()

    # Make sure required columns exist
    need = {'date','signal','target','qrank','stat_type','bet_size_col','value'}
    missing_cols = need - set(stats_df.columns)
    if missing_cols:
        raise ValueError(f"[outliers] stats_df missing required columns: {sorted(missing_cols)}")

    # Normalize dtypes
    df = stats_df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    # Requested metrics vs actually present in data
    req = [str(x) for x in (stats_list or [])]
    present = sorted(df['stat_type'].dropna().astype(str).unique().tolist())
    use = [m for m in req if m in present]

    print(f"[outliers] requested: {req}")
    print(f"[outliers] present  : {present}")
    print(f"[outliers] using    : {use}  (skipped: {[m for m in req if m not in present]})")

    if not use:
        # Nothing to compute — return empty but well-shaped frame
        return pd.DataFrame(columns=list(need) + ['z','abs_z','is_outlier','rule'])

    frames: List[pd.DataFrame] = []
    for metric in use:
        sub = df[df['stat_type'] == metric].copy()
        # Keep rows that have numeric 'value'
        sub = sub.dropna(subset=['value'])
        if sub.empty:
            continue

        z = _zscore(sub['value'])
        sub['z'] = z
        sub['abs_z'] = z.abs()
        # No hard threshold: flag everything and let downstream pages take the top |z|
        sub['is_outlier'] = True
        sub['rule'] = "ranked by |z| (no threshold)"
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=list(need) + ['z','abs_z','is_outlier','rule'])

    # Keep all rows (not only outliers) so your tables can select highs/lows flexibly
    out = pd.concat(frames, ignore_index=True)
    # Drop rule column to keep output compact; plotting ranks by |z| directly
    if 'rule' in out.columns:
        out = out.drop(columns=['rule'])
    # Stable ordering: by stat, then by date
    out = out.sort_values(['stat_type','date','signal','target','bet_size_col','qrank'], kind='mergesort')
    return out

def save_outliers(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with NamedTemporaryFile(dir=os.path.dirname(path) or ".", delete=False) as tmp:
        pickle.dump(df, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_name = tmp.name
    os.replace(tmp_name, path)
    print(f"[outliers] saved -> {path}  ({len(df)} rows)")
