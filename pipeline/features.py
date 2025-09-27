def generate_signals_and_targets(
    daily_data_by_date,
    horizons=[1, 3],
    include_bet_caps=True,   # keep your cap bet sizes
):
    """
    Produces:
      - Signals: pret_{h}_RR, pret_{h}_MR   (RR = log(P_t / P_{t-h}); MR = RR - sum_{k=t-h+1..t} SPpvCLCL_k)
      - Targets: fret_{h}_RR, fret_{h}_MR   (RR = log(P_{t+h} / P_t); MR = RR - sum_{k=t+1..t+h} SPpvCLCL_k)
    Also builds: betsize_equal (+ optional MDV21-based caps).
    Assumes each day's df has columns: ['ticker','prevAdjClose'] and (optionally) 'SPpvCLCL' (per-day market log return).
    """
    import numpy as np
    import pandas as pd

    def _sp_scalar(df):
        """Get the day's market log return (scalar) from SPpvCLCL column, if present."""
        if 'SPpvCLCL' not in df.columns:
            return np.nan
        s = pd.to_numeric(df['SPpvCLCL'], errors='coerce').dropna()
        return float(s.iloc[0]) if not s.empty else np.nan

    # Pre-extract day-level SP returns to avoid re-scanning frames
    # idx -> scalar SPpvCLCL (can be NaN if missing)
    sp_by_idx = {}
    for j, (_, dfx) in enumerate(daily_data_by_date):
        sp_by_idx[j] = _sp_scalar(dfx)

    result = []
    for i in range(len(daily_data_by_date)):
        fname, df = daily_data_by_date[i]
        date = fname[:8]
        df = df.copy()

        # ---- Bet sizing ----
        df['betsize_equal'] = 1.0
        if include_bet_caps and ('volume' in df.columns) and ('prevAdjClose' in df.columns):
            mdv_data = []
            for j in range(max(0, i - 20), i + 1):
                _, df_j = daily_data_by_date[j]
                if ('volume' in df_j.columns) and ('prevAdjClose' in df_j.columns):
                    tmp = df_j[['ticker', 'volume', 'prevAdjClose']].copy()
                    tmp['dollar_vol'] = tmp['volume'] * tmp['prevAdjClose']
                    mdv_data.append(tmp[['ticker', 'dollar_vol']])
            if mdv_data:
                mdv_all = pd.concat(mdv_data, ignore_index=True)
                mdv21 = (mdv_all.groupby('ticker')['dollar_vol']
                                  .median()
                                  .rename('mdv21')
                                  .reset_index())
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

        # Ensure numeric
        if 'prevAdjClose' in df.columns:
            df['prevAdjClose'] = pd.to_numeric(df['prevAdjClose'], errors='coerce')

        # ---- SIGNALS: RR (past) & MR (horizon-matched) ----
        for h in horizons:
            if i - h < 0:
                continue
            prev_df = daily_data_by_date[i - h][1]
            prev_df = prev_df[['ticker', 'prevAdjClose']].copy()
            prev_df.rename(columns={'prevAdjClose': f'past_{h}'}, inplace=True)

            cur = df[['ticker', 'prevAdjClose']].merge(prev_df, on='ticker', how='left')
            cur['prevAdjClose'] = pd.to_numeric(cur['prevAdjClose'], errors='coerce')
            cur[f'past_{h}'] = pd.to_numeric(cur[f'past_{h}'], errors='coerce')

            valid = (cur['prevAdjClose'] > 0) & (cur[f'past_{h}'] > 0)
            with np.errstate(invalid='ignore', divide='ignore'):
                rr = np.where(valid, np.log(cur['prevAdjClose'].values / cur[f'past_{h}'].values), np.nan)

            df = df.merge(
                pd.DataFrame({'ticker': cur['ticker'].values, f'pret_{h}_RR': rr}),
                on='ticker', how='left'
            )

            # sum SP returns over days (i-h+1 .. i)
            sp_sum = 0.0
            ok = True
            for k in range(i - h + 1, i + 1):
                spv = sp_by_idx.get(k, np.nan)
                if not np.isfinite(spv):
                    ok = False
                    break
                sp_sum += spv
            if ok:
                df[f'pret_{h}_MR'] = df[f'pret_{h}_RR'] - sp_sum
            else:
                df[f'pret_{h}_MR'] = np.nan

        # ---- TARGETS: RR (future) & MR (horizon-matched) ----
        for h in horizons:
            if i + h >= len(daily_data_by_date):
                continue
            fut_df = daily_data_by_date[i + h][1]
            fut_df = fut_df[['ticker', 'prevAdjClose']].copy()
            fut_df.rename(columns={'prevAdjClose': f'future_{h}'}, inplace=True)

            cur = df[['ticker', 'prevAdjClose']].merge(fut_df, on='ticker', how='left')
            cur['prevAdjClose'] = pd.to_numeric(cur['prevAdjClose'], errors='coerce')
            cur[f'future_{h}'] = pd.to_numeric(cur[f'future_{h}'], errors='coerce')

            valid = (cur['prevAdjClose'] > 0) & (cur[f'future_{h}'] > 0)
            with np.errstate(invalid='ignore', divide='ignore'):
                rr = np.where(valid, np.log(cur[f'future_{h}'].values / cur['prevAdjClose'].values), np.nan)

            df = df.merge(
                pd.DataFrame({'ticker': cur['ticker'].values, f'fret_{h}_RR': rr}),
                on='ticker', how='left'
            )

            # sum SP returns over days (i+1 .. i+h)
            sp_sum = 0.0
            ok = True
            for k in range(i + 1, i + h + 1):
                spv = sp_by_idx.get(k, np.nan)
                if not np.isfinite(spv):
                    ok = False
                    break
                sp_sum += spv
            if ok:
                df[f'fret_{h}_MR'] = df[f'fret_{h}_RR'] - sp_sum
            else:
                df[f'fret_{h}_MR'] = np.nan

        # (optional) downcast floats to save memory
        float_cols = df.select_dtypes(include=['float64', 'float32']).columns
        if len(float_cols):
            df[float_cols] = df[float_cols].astype('float32')

        df['date'] = date
        result.append((fname, df))

    return result
