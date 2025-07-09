import numpy as np
from scipy.stats import spearmanr, linregress
from collections import defaultdict
import dcor

def create_4d_stats():
    return defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

def compute_daily_stats(df, signal_cols, target_cols, quantiles=[1.0, 0.75, 0.5, 0.25]):
    stats = create_4d_stats()
    for signal in signal_cols:
        for target in target_cols:
            df_filtered = df[[signal, target]].replace([np.inf, -np.inf], np.nan).dropna()
            abs_signal = df_filtered[signal].abs()
            for q in quantiles:
                threshold = abs_signal.quantile(1 - q)
                portfolio = df_filtered[abs_signal >= threshold]
                if portfolio.empty:
                    continue

                key_qrank = f'qr_{int(q * 100)}'

                # Metrics
                sig = portfolio[signal]
                tgt = portfolio[target]
                pnl = np.sign(sig) * tgt
                notional = np.abs(sig).sum()
                nr_instr = np.isfinite(sig) & (sig != 0)
                ppd = pnl.mean()
                std = pnl.std()

                stats['pnl'][signal][key_qrank][target] = pnl.sum()
                stats['ppd'][signal][key_qrank][target] = ppd
                stats['sharpe'][signal][key_qrank][target] = ppd / std * np.sqrt(252) if std > 0 else np.nan
                stats['hit_ratio'][signal][key_qrank][target] = (np.sign(sig) == np.sign(tgt)).mean()
                stats['long_ratio'][signal][key_qrank][target] = (np.sign(sig) == 1).mean()
                stats['spearman'][signal][key_qrank][target] = spearmanr(sig, tgt)[0]
                stats['n'][signal][key_qrank][target] = nr_instr.sum()
                stats['sizeNotional'][signal][key_qrank][target] = notional
                stats['nrInstr'][signal][key_qrank][target] = nr_instr.sum()

                # R2 and t-stat
                slope, intercept, r_val, p_val, stderr = linregress(sig, tgt)
                stats['r2'][signal][key_qrank][target] = r_val**2
                stats['t_stat'][signal][key_qrank][target] = slope / stderr if stderr > 0 else np.nan

                # Distance correlation
                try:
                    stats['dcor'][signal][key_qrank][target] = dcor.distance_correlation(sig.values, tgt.values)
                except Exception:
                    stats['dcor'][signal][key_qrank][target] = np.nan
    return stats
