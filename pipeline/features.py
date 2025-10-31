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

    Key points:
      * Single continuous panel across years (no calendar resets).
      * Drop duplicate (ticker, date); keep the last.
      * Guard against backward date jumps within each ticker.
      * Mask first/last h rows per ticker so boundary artifacts (incl. year turn)
        don’t enter signals/targets used by downstream quantile selection.
    """

    # ---------- 1) Build a continuous panel (no per-year reset) ----------
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

    # Hygiene & types
    must_have = ["ticker", "date", "prevAdjClose"]
    for col in must_have:
        if col not in panel.columns:
            raise ValueError(f"features.generate_signals_and_targets: required column '{col}' missing.")

    panel["ticker"] = panel["ticker"].astype(str)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["prevAdjClose"] = pd.to_numeric(panel["prevAdjClose"], errors="coerce")

    has_sp = "SPpvCLCL" in panel.columns
    if has_sp:
        panel["SPpvCLCL"] = pd.to_numeric(panel["SPpvCLCL"], errors="coerce")
    has_vol = "volume" in panel.columns
    if has_vol:
        panel["volume"] = pd.to_numeric(panel["volume"], errors="coerce")

    # Sort, drop rows with bad dates, and de-duplicate (ticker, date)
    panel = panel.dropna(subset=["date"]).sort_values(["ticker", "date"], kind="mergesort")
    panel = panel.drop_duplicates(subset=["ticker", "date"], keep="last")

    # Guard against true backward date jumps (allow equal dates)
    neg_jump = (
        panel.groupby("ticker", sort=False)["date"]
        .apply(lambda s: (s.diff().dt.days < 0).any())
    )
    if bool(neg_jump.any()):
        bad_tickers = neg_jump[neg_jump].index.tolist()[:5]
        raise AssertionError(
            f"Found backward date jumps within tickers (e.g., {bad_tickers}). "
            "Upstream loader may be mixing calendars."
        )

    # ---------- 2) Market series (if provided) ----------
    # Work at the date level, then map back to panel rows
    if has_sp:
        mkt_rolling = (
            panel[["date", "SPpvCLCL"]]
            .dropna(subset=["date"])
            .drop_duplicates("date")
            .set_index("date")
            .sort_index()["SPpvCLCL"]
        )
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

    by_ticker = panel.groupby("ticker", sort=False, group_keys=False)
    # Positions within each ticker to mask first/last h rows (prevents boundary artifacts)
    pos_in_group = by_ticker.cumcount()
    n_in_group   = by_ticker["date"].transform("size")

    # ---------- 3) Signals/Targets (RR & MR) ----------
    for h in horizons:
        valid_back = pos_in_group >= h
        valid_fwd  = (n_in_group - pos_in_group - 1) >= h

        # SIGNALS: past h-day log return
        price_t   = panel["prevAdjClose"]
        price_t_h = by_ticker["prevAdjClose"].shift(h)
        sig_rr = _log_ratio(price_t, price_t_h).where(valid_back)
        panel[f"pret_{h}_RR"] = sig_rr

        if has_sp:
            # Sum SPpvCLCL from t-h+1..t (inclusive)
            sp_back_sum = (
                mkt_rolling.rolling(h, min_periods=h).sum()
                .reindex(panel["date"].values)
                .to_numpy()
            )
            sig_mr = (sig_rr - sp_back_sum).where(valid_back)
            panel[f"pret_{h}_MR"] = sig_mr
        else:
            panel[f"pret_{h}_MR"] = sig_rr

        # TARGETS: forward h-day log return
        price_t_ph = by_ticker["prevAdjClose"].shift(-h)
        tgt_rr = _log_ratio(price_t_ph, price_t).where(valid_fwd)
        panel[f"fret_{h}_RR"] = tgt_rr

        if has_sp:
            # Sum SPpvCLCL from t+1..t+h (exclude today)
            sp_fwd_sum = (
                mkt_rolling.shift(-1).rolling(h, min_periods=h).sum()
                .reindex(panel["date"].values)
                .to_numpy()
            )
            tgt_mr = (tgt_rr - sp_fwd_sum).where(valid_fwd)
            panel[f"fret_{h}_MR"] = tgt_mr
        else:
            panel[f"fret_{h}_MR"] = tgt_rr

    # ---------- 4) Bet sizes ----------
    panel["betsize_equal"] = 1.0

    if include_bet_caps and has_vol and ("prevAdjClose" in panel.columns):
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
            # If mdv21 is NaN (early warmup), fall back to 1.0 so column is usable
            panel[colname] = panel[colname].where(panel["mdv21"].notna(), 1.0)

        _cap("betsize_cap150k", 150000.0)
        _cap("betsize_cap200k", 200000.0)
        _cap("betsize_cap250k", 250000.0)
    else:
        panel["betsize_cap150k"] = 1.0
        panel["betsize_cap200k"] = 1.0
        panel["betsize_cap250k"] = 1.0

    # ---------- 5) Downcast floats to save memory ----------
    float_cols = panel.select_dtypes(include=["float64", "float32"]).columns
    if len(float_cols):
        panel[float_cols] = panel[float_cols].astype("float32")

    # ---------- 6) Split back to per-day frames ----------
    result = []
    new_cols = [c for c in panel.columns if c.startswith("pret_") or c.startswith("fret_") or c.startswith("betsize_")]
    new_cols = sorted(set(new_cols))
    panel_idx = panel.set_index("date")

    for fname, df_orig in daily_data_by_date:
        d = pd.to_datetime(str(fname)[:8], format="%Y%m%d", errors="coerce")
        try:
            sub = panel_idx.loc[d].reset_index()
        except KeyError:
            sub = pd.DataFrame(columns=["date", "ticker"] + new_cols)

        keep = ["ticker"] + new_cols
        sub = sub[keep] if not sub.empty else pd.DataFrame(columns=keep)

        # Merge onto original df to preserve other columns and row order
        enriched = df_orig.merge(sub, on="ticker", how="left")
        enriched["date"] = str(fname)[:8]  # always store YYYYMMDD string
        result.append((fname, enriched))

    return result
