# pipeline/features.py
import numpy as np
import pandas as pd


def generate_signals_and_targets(
    daily_data_by_date,
    horizons=[1, 3],
    include_bet_caps=True,   # keep cap bet sizes
):
    """
    Produces per-day frames with:
      Signals: pret_{h}_RR, pret_{h}_MR   (RR = log(P_t / P_{t-h}))
               MR = RR - sum_{k=t-h+1..t} SPpvCLCL_k   (if SPpvCLCL exists; else == RR)

      Targets: fret_{h}_RR, fret_{h}_MR   (RR = log(P_{t+h} / P_t))
               MR = RR - sum_{k=t+1..t+h} SPpvCLCL_k   (if SPpvCLCL exists; else == RR)

      Bet sizing:
        - betsize_equal = 1.0
        - optional MDV21-based caps: betsize_cap150k, betsize_cap200k, betsize_cap250k
          computed as min(cap, 0.005 * rolling_median_21d(dollar_vol)), per ticker.

    Key change vs. old version: we build a single panel sorted by ['ticker','date']
    and use groupby().shift to compute lookback/forward returns. This eliminates
    start-of-year dips that came from resetting the lookback index.

    Parameters
    ----------
    daily_data_by_date : list[(fname, df)]
        df must include columns: 'ticker', 'prevAdjClose'
        optional: 'SPpvCLCL' (market log return per date), 'volume'
        fname must begin with YYYYMMDD (used as the date stamp).
    horizons : list[int]
        Lookback/forward horizons (in trading days) for signals/targets.
    include_bet_caps : bool
        Whether to compute MDV21-based bet caps.

    Returns
    -------
    list of (fname, enriched_df) in the same order as input.
    """

    # --------- 1) Build a continuous panel (no per-year reset) ----------
    # Normalize input → attach a 'date' column parsed from fname[:8]
    records = []
    for fname, df in daily_data_by_date:
        if df is None or df.empty:
            continue
        d = str(fname)[:8]
        date = pd.to_datetime(d, format="%Y%m%d", errors="coerce")
        block = df.copy()
        block["date"] = date
        records.append(block)

    if not records:
        return []

    panel = pd.concat(records, ignore_index=True)

    # Basic hygiene
    # Keep essential columns; preserve all others to merge back later.
    must_have = ["ticker", "date", "prevAdjClose"]
    for col in must_have:
        if col not in panel.columns:
            raise ValueError(f"features.generate_signals_and_targets: required column '{col}' missing.")

    # Force types
    panel["ticker"] = panel["ticker"].astype(str)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["prevAdjClose"] = pd.to_numeric(panel["prevAdjClose"], errors="coerce")

    # Optional columns
    has_sp = "SPpvCLCL" in panel.columns
    if has_sp:
        panel["SPpvCLCL"] = pd.to_numeric(panel["SPpvCLCL"], errors="coerce")
    has_vol = "volume" in panel.columns
    if has_vol:
        panel["volume"] = pd.to_numeric(panel["volume"], errors="coerce")

    panel = panel.sort_values(["ticker", "date"], kind="mergesort")

    # --------- 2) Signals (RR/MR) using past prices and market sums ----------
    # Market sums by date (scalar per day)
    if has_sp:
        # Sum of last h days INCLUDING today (t-h+1..t) for signals MR
        mkt_rolling = (
            panel.drop_duplicates("date")[["date", "SPpvCLCL"]]
            .set_index("date")
            .sort_index()["SPpvCLCL"]
        )
        # For signals MR: inclusive back-rolling window of length h
        # For targets MR: sum of next h days EXCLUDING today => shift(-1).rolling(h)
    else:
        mkt_rolling = None

    # Helper: safe log ratio
    def _log_ratio(num, den):
        num = pd.to_numeric(num, errors="coerce")
        den = pd.to_numeric(den, errors="coerce")
        ok = (num > 0) & (den > 0)
        out = pd.Series(np.nan, index=num.index, dtype="float64")
        with np.errstate(invalid="ignore", divide="ignore"):
            out[ok] = np.log(num[ok].values / den[ok].values)
        return out

    # Compute signals/targets per horizon with groupby().shift
    by_ticker = panel.groupby("ticker", sort=False, group_keys=False)
    for h in horizons:
        # SIGNALS: past h-day log return
        price_t   = panel["prevAdjClose"]
        price_t_h = by_ticker["prevAdjClose"].shift(h)
        panel[f"pret_{h}_RR"] = _log_ratio(price_t, price_t_h)

        if has_sp:
            # Sum SPpvCLCL from t-h+1..t (inclusive)
            sp_back_sum = (
                mkt_rolling.rolling(h, min_periods=h).sum()
                .reindex(panel["date"].values)
                .to_numpy()
            )
            panel[f"pret_{h}_MR"] = panel[f"pret_{h}_RR"] - sp_back_sum
        else:
            panel[f"pret_{h}_MR"] = panel[f"pret_{h}_RR"]

        # TARGETS: forward h-day log return
        price_t_ph = by_ticker["prevAdjClose"].shift(-h)
        panel[f"fret_{h}_RR"] = _log_ratio(price_t_ph, price_t)

        if has_sp:
            # Sum SPpvCLCL from t+1..t+h (exclude today)
            sp_fwd_sum = (
                mkt_rolling.shift(-1).rolling(h, min_periods=h).sum()
                .reindex(panel["date"].values)
                .to_numpy()
            )
            panel[f"fret_{h}_MR"] = panel[f"fret_{h}_RR"] - sp_fwd_sum
        else:
            panel[f"fret_{h}_MR"] = panel[f"fret_{h}_RR"]

    # --------- 3) Bet sizes ----------
    panel["betsize_equal"] = 1.0

    if include_bet_caps and has_vol and ("prevAdjClose" in panel.columns):
        # Dollar volume and 21-day rolling median per ticker
        panel["dollar_vol"] = panel["volume"] * panel["prevAdjClose"]
        mdv21 = (
            by_ticker["dollar_vol"]
            .rolling(window=21, min_periods=1)
            .median()
            .reset_index(level=0, drop=True)
        )
        panel["mdv21"] = mdv21.astype("float64")

        def _cap(colname, cap_amt):
            panel[colname] = np.minimum(cap_amt, 0.005 * panel["mdv21"])
            # If mdv21 is NaN, fall back to 1.0 so the column exists and is usable
            panel[colname] = panel[colname].where(panel["mdv21"].notna(), 1.0)

        _cap("betsize_cap150k", 150000.0)
        _cap("betsize_cap200k", 200000.0)
        _cap("betsize_cap250k", 250000.0)
    else:
        # Provide the columns with 1.0 so downstream code is not surprised
        panel["betsize_cap150k"] = 1.0
        panel["betsize_cap200k"] = 1.0
        panel["betsize_cap250k"] = 1.0

    # --------- 4) (Optional) downcast floats to save memory ----------
    float_cols = panel.select_dtypes(include=["float64", "float32"]).columns
    if len(float_cols):
        panel[float_cols] = panel[float_cols].astype("float32")

    # --------- 5) Split back into original per-day DataFrames ----------
    # We maintain the original order and row set of each input df, and left-merge the new columns.
    result = []
    # Collect the set of new columns we created
    new_cols = [c for c in panel.columns if c.startswith("pret_") or c.startswith("fret_") or c.startswith("betsize_")]
    new_cols = sorted(set(new_cols))  # stable order

    # Quick index by date for fast slicing
    panel_idx = panel.set_index("date")

    for fname, df_orig in daily_data_by_date:
        d = pd.to_datetime(str(fname)[:8], format="%Y%m%d", errors="coerce")
        # Subset panel for that date with only (ticker + new columns)
        try:
            sub = panel_idx.loc[d].reset_index()
        except KeyError:
            # No data for this date (e.g., holiday); just attach defaults
            sub = pd.DataFrame(columns=["date", "ticker"] + new_cols)

        keep = ["ticker"] + new_cols
        sub = sub[keep] if not sub.empty else pd.DataFrame(columns=keep)

        # Merge onto original df to preserve all other columns and row order
        enriched = df_orig.merge(sub, on="ticker", how="left")

        # Always echo the 'date' as a string (YYYYMMDD) for downstream usage
        enriched["date"] = str(fname)[:8]

        result.append((fname, enriched))

    return result
