import os
from typing import List, Tuple

import numpy as np
import pandas as pd


def compute_outliers(
    stats_df: pd.DataFrame,
    stats_list: List[str],
    z_thresh: float = 3.0,
    top_k: int = 5,  # kept for backward compatibility; UNUSED
    group_keys: Tuple[str, ...] = ('target', 'signal', 'bet_size_col', 'qrank', 'stat_type'),
) -> pd.DataFrame:
    """
    Flag outliers *only* by |z| within each (target, signal, bet_size_col, qrank, stat_type) group over time.
    Expects columns: ['date','value','stat_type','signal','target','bet_size_col','qrank'].

    Rule:
      - Statistical only: |z| >= z_thresh (computed within group across dates).

    Notes:
      - 'top_k' is ignored (kept to avoid breaking callers).
      - Output is sorted by |z| descending, then date descending.
    """
    if not stats_list:
        raise ValueError("compute_outliers: 'stats_list' must be a non-empty list of stat_type names.")

    use = stats_df[stats_df['stat_type'].isin(stats_list)].copy()
    use['value'] = pd.to_numeric(use['value'], errors='coerce')
    use = use[np.isfinite(use['value'])]
    if use.empty:
        cols = list(set(['date', 'value'] + list(group_keys))) + [
            'mean', 'std', 'z', 'abs_z', 'is_outlier', 'rule'
        ]
        return pd.DataFrame(columns=cols)

    # Ensure datetime
    use['date'] = pd.to_datetime(use['date'], errors='coerce')

    # Group-wise mean/std/z
    gkeys = list(group_keys)
    use['mean'] = use.groupby(gkeys)['value'].transform('mean')
    use['std']  = use.groupby(gkeys)['value'].transform('std')
    use['z']    = (use['value'] - use['mean']) / use['std'].replace(0, np.nan)
    use['abs_z'] = use['z'].abs()

    # Flag by z only
    use['is_outlier'] = use['abs_z'] >= float(z_thresh)
    use.loc[use['is_outlier'], 'rule'] = f'|z|>={z_thresh}'

    out = use[use['is_outlier']].copy()

    # Sort strictly by |z| (desc), then by date (desc)
    out = out.sort_values(['abs_z', 'date'], ascending=[False, False])

    return out


def save_outliers(odf: pd.DataFrame, path: str) -> None:
    """Save outliers as a PKL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    odf.to_pickle(path)


def load_outliers(path: str) -> pd.DataFrame:
    """Load outliers from a PKL file."""
    return pd.read_pickle(path)
