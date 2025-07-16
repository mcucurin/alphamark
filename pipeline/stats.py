import numpy as np
from collections import defaultdict

from pipeline.pnl import comp_pnl_all_quantiles


def create_4d_stats():
    """Returns a nested dictionary structure for storing stats."""
    return defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))


def compute_daily_stats(df, signal_cols, target_cols, quantiles=[1.0, 0.75, 0.5, 0.25]):
    """
    Compute PnL quantile statistics for each (signal, target) pair.
    Returns a 4-level dict: stats[metric][signal][quantile_label][target] = value
    """
    stats = create_4d_stats()

    for signal, target in zip(signal_cols, target_cols):
        df_clean = df[[signal, target]].replace([np.inf, -np.inf], np.nan)
        s = df_clean[signal]
        f = df_clean[target]

        pct_rank = s.abs().rank(pct=True)
        n_q = len(quantiles)
        qranks = (
            np.ceil(pct_rank * n_q)
            .fillna(0)
            .astype(int)
            .to_frame(signal)
        )

        fut_rets = f.to_frame(signal)

        horizon = target.split('_')[-1]
        bs_col = f"betsize_{horizon}"
        bs = df[bs_col].to_frame(signal) if bs_col in df.columns else None

        out = comp_pnl_all_quantiles(
            qranks,
            fut_rets,
            bet_size=bs,
            probs=quantiles,
            add_corr=True,
            add_long_ratio=True,
            add_hit_ratio=True,
            add_stdev=False,
            add_r2tval=True,
        )

        for label, info_df in out.items():
            for metric in info_df.index:
                stats[metric][signal][label][target] = info_df.at[metric, signal]

    return stats

