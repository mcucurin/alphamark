import numpy as np
from scipy.stats import spearmanr, linregress
from collections import defaultdict
import dcor
from itertools import product

def create_5d_stats():
    return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

def compute_daily_stats(df, signal_cols, target_cols, quantiles=[1.0, 0.75, 0.5, 0.25], bet_size_col=['betsize_equal']):
    stats = create_5d_stats()

    # Pre-clean data for all required columns once
    required_all = set(signal_cols + target_cols + bet_size_col)
    df_clean = df[list(required_all)].replace([np.inf, -np.inf], np.nan).dropna()

    if df_clean.empty:
        return stats

    # Precompute quantile masks per signal
    quantile_masks = {
        signal: {
            f'qr_{int(q * 100)}': df_clean[signal].abs() >= df_clean[signal].abs().quantile(1 - q)
            for q in quantiles
        } for signal in signal_cols
    }

    for signal, target, bet_col in product(signal_cols, target_cols, bet_size_col):
        if not all(col in df_clean.columns for col in [signal, target, bet_col]):
            continue

        for q, mask in quantile_masks[signal].items():
            portfolio = df_clean[mask][[signal, target, bet_col]].dropna()
            if portfolio.empty:
                continue

            sig = portfolio[signal].values
            tgt = portfolio[target].values
            bsz = portfolio[bet_col].values

            signed = np.sign(sig)
            pnl = signed * tgt * bsz
            notional = np.abs(sig * bsz).sum()
            std = pnl.std()

            stats['pnl'][signal][q][target][bet_col] = pnl.sum()
            stats['ppd'][signal][q][target][bet_col] = pnl.sum() / notional if notional > 0 else np.nan
            stats['sharpe'][signal][q][target][bet_col] = (pnl.sum() / std * np.sqrt(252)) if std > 0 else np.nan
            stats['hit_ratio'][signal][q][target][bet_col] = (np.sign(sig) == np.sign(tgt)).mean()
            stats['long_ratio'][signal][q][target][bet_col] = (signed == 1).mean()
            stats['bet_size'][signal][q][target][bet_col] = bsz.mean()
            stats['n'][signal][q][target][bet_col] = len(sig)
            stats['sizeNotional'][signal][q][target][bet_col] = notional
            stats['nrInstr'][signal][q][target][bet_col] = len(sig)

            if np.unique(sig).size > 1 and np.unique(tgt).size > 1:
                stats['spearman'][signal][q][target][bet_col] = spearmanr(sig, tgt)[0]
            else:
                stats['spearman'][signal][q][target][bet_col] = np.nan

            if np.unique(sig).size > 1:
                slope, intercept, r_val, p_val, stderr = linregress(sig, tgt)
                stats['r2'][signal][q][target][bet_col] = r_val ** 2
                stats['t_stat'][signal][q][target][bet_col] = slope / stderr if stderr > 0 else np.nan
            else:
                stats['r2'][signal][q][target][bet_col] = np.nan
                stats['t_stat'][signal][q][target][bet_col] = np.nan

            try:
                stats['dcor'][signal][q][target][bet_col] = dcor.distance_correlation(sig, tgt)
            except Exception:
                stats['dcor'][signal][q][target][bet_col] = np.nan

    return stats
