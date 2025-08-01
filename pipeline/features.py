def generate_signals_and_targets(daily_data_by_date, horizons=[1, 3]):
    import numpy as np
    import pandas as pd

    result = []
    for i in range(len(daily_data_by_date)):
        fname, df = daily_data_by_date[i]
        date = fname[:8]
        df = df.copy()

        # === Bet sizing strategies ===
        df['betsize_equal'] = 1.0  # Equal weight strategy

        # Compute 21-day median dollar volume (MDV21)
        if 'volume' in df.columns and 'prevAdjClose' in df.columns:
            mdv_data = []
            for j in range(max(0, i - 20), i + 1):
                _, df_j = daily_data_by_date[j]
                if 'volume' in df_j.columns and 'prevAdjClose' in df_j.columns:
                    temp = df_j[['ticker', 'volume', 'prevAdjClose']].copy()
                    temp['dollar_vol'] = temp['volume'] * temp['prevAdjClose']
                    temp = temp[['ticker', 'dollar_vol']]
                    temp['date_index'] = j
                    mdv_data.append(temp)

            if mdv_data:
                mdv_all = pd.concat(mdv_data)
                mdv21 = (
                    mdv_all.groupby('ticker')['dollar_vol']
                    .median()
                    .rename('mdv21')
                    .reset_index()
                )
                df = df.merge(mdv21, on='ticker', how='left')

                df['betsize_cap200k'] = np.minimum(200000, 0.005 * df['mdv21'])
                df['betsize_cap150k'] = np.minimum(150000, 0.005 * df['mdv21'])
                df['betsize_cap250k'] = np.minimum(250000, 0.005 * df['mdv21'])
            else:
                df['betsize_cap200k'] = 1.0
                df['betsize_cap150k'] = 1.0
                df['betsize_cap250k'] = 1.0
        else:
            df['betsize_cap200k'] = 1.0
            df['betsize_cap150k'] = 1.0
            df['betsize_cap250k'] = 1.0

        # === Past returns (signals) - Close to Close ===
        for h in horizons:
            if i - h < 0:
                continue
            prev_df = daily_data_by_date[i - h][1]
            merged = df[['ticker', 'prevAdjClose']].merge(
                prev_df[['ticker', 'prevAdjClose']], on='ticker', suffixes=('', f'_past{h}')
            )
            valid = (merged['prevAdjClose'] > 0) & (merged[f'prevAdjClose_past{h}'] > 0)
            with np.errstate(invalid='ignore', divide='ignore'):
                merged['pret_CLCL'] = np.where(
                    valid,
                    np.log(merged['prevAdjClose'].values / merged[f'prevAdjClose_past{h}'].values),
                    np.nan
                )
            merged = merged[['ticker', 'pret_CLCL']].rename(columns={'pret_CLCL': f'pret_{h}_CLCL'})
            df = df.merge(merged, on='ticker', how='left')

        # === Past returns - Open to Close (1-day only) ===
        if i - 1 >= 0:
            prev_df = daily_data_by_date[i - 1][1]
            if 'open' in prev_df.columns:
                merged = df[['ticker', 'prevAdjClose']].merge(
                    prev_df[['ticker', 'open']], on='ticker', suffixes=('', '_past1')
                )
                open_col = 'open_past1' if 'open_past1' in merged.columns else 'open'
                if open_col in merged.columns:
                    valid = (merged['prevAdjClose'] > 0) & (merged[open_col] > 0)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        merged['pret_OPCL'] = np.where(
                            valid,
                            np.log(merged['prevAdjClose'].values / merged[open_col].values),
                            np.nan
                        )
                    merged = merged[['ticker', 'pret_OPCL']].rename(columns={'pret_OPCL': 'pret_1_OPCL'})
                    df = df.merge(merged, on='ticker', how='left')

        # === Future returns (targets) - Close to Close ===
        for h in horizons:
            if i + h >= len(daily_data_by_date):
                continue
            future_df = daily_data_by_date[i + h][1]
            future_prices = future_df[['ticker', 'prevAdjClose']].rename(
                columns={'prevAdjClose': f'future_adj_close_{h}d'}
            )
            merged = df[['ticker', 'prevAdjClose']].merge(future_prices, on='ticker', how='left')

            merged['prevAdjClose'] = pd.to_numeric(merged['prevAdjClose'], errors='coerce')
            merged[f'future_adj_close_{h}d'] = pd.to_numeric(merged[f'future_adj_close_{h}d'], errors='coerce')

            valid = (merged['prevAdjClose'] > 0) & (merged[f'future_adj_close_{h}d'] > 0)
            with np.errstate(invalid='ignore', divide='ignore'):
                merged['fret'] = np.where(
                    valid,
                    np.log(merged[f'future_adj_close_{h}d'].values / merged['prevAdjClose'].values),
                    np.nan
                )
            merged = merged[['ticker', 'fret']].rename(columns={'fret': f'fret_{h}d'})
            df = df.merge(merged, on='ticker', how='left')

        df['date'] = date
        result.append((fname, df))

    return result
