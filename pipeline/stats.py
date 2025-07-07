import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict

def create_4d_stats():
    return defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

def compute_daily_stats(df, signal_cols, target_cols, quantiles=[1.0, 0.75, 0.5, 0.25]):
    stats = create_4d_stats()
    for signal in signal_cols:
        for target in target_cols:
            df_filtered = df[[signal, target]].dropna()
            abs_signal = df_filtered[signal].abs()
            for q in quantiles:
                threshold = abs_signal.quantile(1 - q)
                portfolio = df_filtered[abs_signal >= threshold]
                if portfolio.empty:
                    continue
                pnl = np.sign(portfolio[signal]) * portfolio[target]
                key_qrank = f'qr_{int(q * 100)}'

                stats['pnl'][signal][key_qrank][target] = pnl.sum()
                stats['ppd'][signal][key_qrank][target] = pnl.mean()
                stats['sharpe'][signal][key_qrank][target] = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() > 0 else np.nan
                stats['hit_ratio'][signal][key_qrank][target] = (np.sign(portfolio[signal]) == np.sign(portfolio[target])).mean()
                stats['long_ratio'][signal][key_qrank][target] = (np.sign(portfolio[signal]) == 1).mean()
                stats['spearman'][signal][key_qrank][target] = spearmanr(portfolio[signal], portfolio[target])[0]
                stats['n'][signal][key_qrank][target] = len(portfolio)
    return stats
