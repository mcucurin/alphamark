def generate_signals_and_targets(daily_data_by_date, horizons=[1, 3, 5]):
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
            rr = (merged['prevAdjClose'] / merged[f'prevAdjClose_past{h}']) - 1
            df[f'signal_RR_{h}d'] = rr
            df[f'signal_MR_{h}d'] = rr - df['SPpvCLCL']

        # future returns (targets)
        for h in horizons:
            if i + h >= len(daily_data_by_date):
                continue
            future_df = daily_data_by_date[i + h][1]
            future_prices = future_df[['ticker', 'prevAdjClose']].rename(columns={'prevAdjClose': f'future_adj_close_{h}d'})
            merged = df[['ticker', 'prevAdjClose']].merge(future_prices, on='ticker', how='left')
            fr = (merged[f'future_adj_close_{h}d'] / merged['prevAdjClose']) - 1
            df[f'fret_RR_{h}d'] = fr
            df[f'fret_MR_{h}d'] = fr - df['SPpvCLCL']

        df['date'] = date
        result.append((fname, df))
    return result
