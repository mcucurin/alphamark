def generate_signals_and_targets(daily_data_by_date, horizons=[1, 3, 5]):
    import numpy as np
    import pandas as pd

    result = []
    for i in range(len(daily_data_by_date)):
        fname, df = daily_data_by_date[i]
        date = fname[:8]
        df = df.copy()

        # past returns (signals)
        for h in horizons:
            if i - h < 0:
                continue
            prev_df = daily_data_by_date[i - h][1]
            merged = df[['ticker', 'prevAdjClose']].merge(
                prev_df[['ticker', 'prevAdjClose']], on='ticker', suffixes=('', f'_past{h}'))
            valid = (merged['prevAdjClose'] > 0) & (merged[f'prevAdjClose_past{h}'] > 0)
            with np.errstate(invalid='ignore', divide='ignore'):
                merged['signal_RR'] = np.where(valid,
                                              np.log(merged['prevAdjClose'] / merged[f'prevAdjClose_past{h}']),
                                              np.nan)
            merged = merged[['ticker', 'signal_RR']].rename(columns={'signal_RR': f'signal_RR_{h}d'})
            df = df.merge(merged, on='ticker', how='left')
            df[f'signal_MR_{h}d'] = df[f'signal_RR_{h}d'] - df['SPpvCLCL']

        # future returns (targets)
        for h in horizons:
            if i + h >= len(daily_data_by_date):
                continue
            future_df = daily_data_by_date[i + h][1]
            future_prices = future_df[['ticker', 'prevAdjClose']].rename(columns={'prevAdjClose': f'future_adj_close_{h}d'})
            merged = df[['ticker', 'prevAdjClose']].merge(future_prices, on='ticker', how='left')
            valid = (merged['prevAdjClose'] > 0) & (merged[f'future_adj_close_{h}d'] > 0)
            with np.errstate(invalid='ignore', divide='ignore'):
                merged['fret_RR'] = np.where(valid,
                                            np.log(merged[f'future_adj_close_{h}d'] / merged['prevAdjClose']),
                                            np.nan)
            merged = merged[['ticker', 'fret_RR']].rename(columns={'fret_RR': f'fret_RR_{h}d'})
            df = df.merge(merged, on='ticker', how='left')
            df[f'fret_MR_{h}d'] = df[f'fret_RR_{h}d'] - df['SPpvCLCL']

        df['date'] = date
        result.append((fname, df))
    return result
