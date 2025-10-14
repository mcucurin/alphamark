# pipeline/outliers_stats.py
import os
from typing import List, Tuple

import numpy as np
import pandas as pd


def compute_outliers(
    stats_df: pd.DataFrame,
    stats_list: List[str],                      # required (no defaults)
    z_thresh: float = 3.0,
    top_k: int = 5,
    group_keys: Tuple[str, ...] = ('target', 'signal', 'bet_size_col', 'qrank', 'stat_type'),
) -> pd.DataFrame:
    """
    Flag outliers within each (target, signal, bet_size_col, qrank, stat_type) group over time.
    Expects columns: ['date','value','stat_type','signal','target','bet_size_col','qrank'].

    Rules:
      - Statistical: |z| >= z_thresh, computed within group.
      - Extremes: top_k largest |value| within group.

    Returns only rows flagged as outliers with diagnostics.
    """
    if not stats_list:
        raise ValueError("compute_outliers: 'stats_list' must be a non-empty list of stat_type names.")

    use = stats_df[stats_df['stat_type'].isin(stats_list)].copy()
    use['value'] = pd.to_numeric(use['value'], errors='coerce')
    use = use[np.isfinite(use['value'])]
    if use.empty:
        cols = list(set(['date', 'value'] + list(group_keys))) + [
            'mean', 'std', 'z', 'abs_val', 'rank_abs', 'is_outlier', 'rule'
        ]
        return pd.DataFrame(columns=cols)

    # Ensure datetime for sorting/plotting later
    use['date'] = pd.to_datetime(use['date'], errors='coerce')

    # Group-wise stats
    gkeys = list(group_keys)
    use['mean'] = use.groupby(gkeys)['value'].transform('mean')
    use['std']  = use.groupby(gkeys)['value'].transform('std')
    use['z']    = (use['value'] - use['mean']) / use['std'].replace(0, np.nan)

    # Magnitudes + ranks
    use['abs_val']  = use['value'].abs()
    use['rank_abs'] = use.groupby(gkeys)['abs_val'].rank(ascending=False, method='first')

    # Flags
    by_z   = use['z'].abs() >= z_thresh
    by_top = use['rank_abs'] <= top_k
    use['is_outlier'] = by_z | by_top
    use.loc[by_z & ~by_top, 'rule']  = f'|z|>={z_thresh}'
    use.loc[by_top & ~by_z, 'rule']  = f'top_{top_k}_abs'
    use.loc[by_top & by_z,  'rule']  = f'|z|>={z_thresh}+top_{top_k}'

    out = use[use['is_outlier']].copy()
    out = out.sort_values(['abs_val', 'date'], ascending=[False, False])
    return out


def save_outliers(odf: pd.DataFrame, path: str) -> None:
    """Save outliers as a PKL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    odf.to_pickle(path)


def load_outliers(path: str) -> pd.DataFrame:
    """Load outliers from a PKL file."""
    return pd.read_pickle(path)
