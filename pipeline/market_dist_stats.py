"""
Market Distribution Stats (per-id corr/CCF vs SPY)

Builds per-identifier (e.g., ticker) distributions for:
  - alpha_raw vs SPY correlation
  - alpha_pnl vs SPY correlation (per bet)
  - alpha_raw vs SPY CCF
  - alpha_pnl vs SPY CCF (per bet)

Outputs live in a dedicated directory (e.g., output/MDS_STATS) to keep them
separate from DAILY/SUMMARY stats.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Dict, Sequence, Optional
from scipy.stats import spearmanr


def _qlabel(q: float) -> str:
    return f"qr_{int(round(q * 100))}"


def _sanitize_list(cols: Sequence) -> list[str]:
    seen = set()
    out: list[str] = []
    for c in cols or []:
        if c is None or c is Ellipsis:
            continue
        if isinstance(c, str) and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _safe_dump(df: pd.DataFrame, path: str, desc: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_pickle(path)
    print(f"[market_dist] Wrote {desc}: {path}  ({len(df)} rows)")


def _quantile_mask(values: np.ndarray, q: float, finite_mask: np.ndarray) -> np.ndarray:
    """Fast selection of top-|values| quantile using argpartition (no full sort)."""
    if q >= 1.0:
        return finite_mask
    idx = np.where(finite_mask)[0]
    if idx.size == 0:
        return np.zeros_like(finite_mask, dtype=bool)
    k = int(np.ceil(q * idx.size))
    if k <= 0:
        return np.zeros_like(finite_mask, dtype=bool)
    abs_vals = np.abs(values[idx])
    if k >= idx.size:
        chosen = idx
    else:
        part = np.argpartition(-abs_vals, kth=k - 1)[:k]
        chosen = idx[part]
    mask = np.zeros_like(finite_mask, dtype=bool)
    mask[chosen] = True
    return mask


def _collapse_corr(records: list[tuple], id_col: str, out_path: Optional[str], metric_name: str):
    if not records or out_path is None:
        return None
    cols = [id_col, "signal", "qrank", "target", "bet_size_col", "date", "series", "spy"]
    dfrec = pd.DataFrame.from_records(records, columns=cols)
    out_rows = []
    for keys, grp in dfrec.groupby([id_col, "signal", "qrank", "target", "bet_size_col"], sort=False):
        x = pd.to_numeric(grp["series"], errors="coerce")
        y = pd.to_numeric(grp["spy"], errors="coerce")
        m = x.notna() & y.notna()
        if m.sum() >= 3 and x[m].nunique() >= 2 and y[m].nunique() >= 2:
            r = spearmanr(x[m], y[m], nan_policy="omit").correlation
            val = float(r) if np.isfinite(r) else np.nan
        else:
            val = np.nan
        out_rows.append((*keys, metric_name, val))
    dfout = pd.DataFrame.from_records(
        out_rows,
        columns=[id_col, "signal", "qrank", "target", "bet_size_col", "stat_type", "value"],
    )
    _safe_dump(dfout, out_path, f"per-id {metric_name}")
    return out_path


def _collapse_ccf(records: list[tuple], id_col: str, out_path: Optional[str], metric_name: str, max_lag: int):
    if not records or out_path is None or max_lag is None or int(max_lag) <= 0:
        return None
    max_lag = int(max_lag)
    cols = [id_col, "signal", "qrank", "target", "bet_size_col", "date", "series", "spy"]
    dfrec = pd.DataFrame.from_records(records, columns=cols)
    out_rows = []

    for keys, grp in dfrec.groupby([id_col, "signal", "qrank", "target", "bet_size_col"], sort=False):
        grp = grp.copy()
        grp["date"] = pd.to_datetime(grp["date"], errors="coerce")
        grp = grp.dropna(subset=["date"])
        if grp.empty:
            continue

        grp = grp.sort_values("date")
        x = pd.to_numeric(grp["series"], errors="coerce")
        y = pd.to_numeric(grp["spy"], errors="coerce")
        idx = grp["date"]

        sx = pd.Series(x.values, index=idx)
        sy = pd.Series(y.values, index=idx)

        for lag in range(-max_lag, max_lag + 1):
            sy_shift = sy.shift(-lag)
            df_xy = pd.concat({"x": sx, "y": sy_shift}, axis=1).dropna()
            if df_xy.shape[0] < 5:
                continue
            r = df_xy["x"].corr(df_xy["y"], method="spearman")
            if np.isfinite(r):
                out_rows.append((*keys, metric_name, int(lag), float(r)))

    if not out_rows:
        return None

    dfout = pd.DataFrame.from_records(
        out_rows,
        columns=[id_col, "signal", "qrank", "target", "bet_size_col", "stat_type", "lag", "corr"],
    )
    _safe_dump(dfout, out_path, f"per-id {metric_name}")
    return out_path


def compute_market_dist_stats(
    df: pd.DataFrame,
    *,
    date_col: str,
    id_col: str = "ticker",
    signal_cols: Sequence[str],
    target_cols: Sequence[str],
    bet_size_cols: Sequence[str],
    quantiles: Sequence[float],
    spy_by_target: Dict[str, str],
    output_dir: str,
    ccf_enable: bool = True,
    ccf_max_lag: int = 5,
) -> Dict[str, Optional[str]]:
    """
    Build per-id corr/CCF files vs SPY into `output_dir`.
    Returns dict of generated paths (may be None if not written).
    """
    signal_cols = _sanitize_list(signal_cols)
    target_cols = _sanitize_list(target_cols)
    bet_size_cols = _sanitize_list(bet_size_cols)

    if not signal_cols or not target_cols or not bet_size_cols:
        print("[market_dist] Missing required columns; skipping market distribution stats.")
        return {"raw_corr": None, "pnl_corr": None, "raw_ccf": None, "pnl_ccf": None}

    if id_col not in df.columns:
        print(f"[market_dist] id_col '{id_col}' not in DataFrame; skipping.")
        return {"raw_corr": None, "pnl_corr": None, "raw_ccf": None, "pnl_ccf": None}

    eff_spy_map = {t: sc for t, sc in (spy_by_target or {}).items() if t in df.columns and sc in df.columns}
    if not eff_spy_map:
        print("[market_dist] No usable SPY columns found; skipping.")
        return {"raw_corr": None, "pnl_corr": None, "raw_ccf": None, "pnl_ccf": None}

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col, id_col])
    if work.empty:
        print("[market_dist] Empty input after cleaning; skipping.")
        return {"raw_corr": None, "pnl_corr": None, "raw_ccf": None, "pnl_ccf": None}

    all_days = work[date_col]
    first_day = pd.to_datetime(all_days.min())
    last_day = pd.to_datetime(all_days.max())
    date_tag = (
        f"{first_day:%Y%m%d}_{last_day:%Y%m%d}"
        if pd.notna(first_day) and pd.notna(last_day)
        else "range"
    )

    os.makedirs(output_dir, exist_ok=True)
    raw_corr_path = os.path.join(output_dir, f"mds_alpha_raw_spy_corr_{date_tag}.pkl")
    pnl_corr_path = os.path.join(output_dir, f"mds_alpha_pnl_spy_corr_{date_tag}.pkl")
    raw_ccf_path = os.path.join(output_dir, f"mds_alpha_raw_spy_ccf_{date_tag}.pkl") if ccf_enable else None
    pnl_ccf_path = os.path.join(output_dir, f"mds_alpha_pnl_spy_ccf_{date_tag}.pkl") if ccf_enable else None

    recs_raw: list[tuple] = []
    recs_pnl: list[tuple] = []

    grouped = work.sort_values(date_col).groupby(date_col, sort=True)

    for dt, day in grouped:
        sigs = [c for c in signal_cols if c in day.columns]
        tgts = [c for c in target_cols if c in day.columns]
        bets = [c for c in bet_size_cols if c in day.columns]
        if not sigs or not tgts or not bets:
            continue

        ids_all = day[id_col].astype(str).to_numpy()

        for s_name in sigs:
            svals = np.asarray(day[s_name], float)
            finite_s = np.isfinite(svals)
            if not finite_s.any():
                continue
            sign_s = np.sign(svals)

            for q in quantiles:
                mask_q = _quantile_mask(svals, q, finite_s)
                if not mask_q.any():
                    continue

                ids_q = ids_all[mask_q]
                s_q = svals[mask_q]

                for t_name in tgts:
                    spy_col = eff_spy_map.get(t_name)
                spy_col = eff_spy_map.get(t_name)
                if not spy_col:
                    continue
                spy_vals = np.asarray(day[spy_col], float)
                spy_v = np.nanmean(spy_vals) if spy_vals.size else np.nan
                if not np.isfinite(spy_v):
                    continue

                y = np.asarray(day[t_name], float)[mask_q]
                if not np.isfinite(y).any():
                    continue

                df_base = pd.DataFrame({
                    id_col: ids_q,
                    "alpha_raw": s_q,
                    "pnl_raw": y * sign_s[mask_q],
                })

                if recs_raw is not None:
                    raw_per_id = df_base.groupby(id_col, sort=False)["alpha_raw"].mean()
                    recs_raw.extend([
                        (str(name_i), s_name, _qlabel(q), t_name, "__RAW__", pd.Timestamp(dt), float(val), float(spy_v))
                        for name_i, val in raw_per_id.items() if np.isfinite(val)
                    ])

                if recs_pnl is not None:
                    for b_name in bets:
                        bcol = np.asarray(day[b_name], float)[mask_q]
                        bcol = np.where(np.isfinite(bcol), np.abs(bcol), np.nan)
                        if not np.isfinite(bcol).any():
                            continue
                        df_p = df_base.copy()
                        df_p["pnl"] = df_p["pnl_raw"] * bcol
                        pnl_per_id = df_p.groupby(id_col, sort=False)["pnl"].sum()
                        recs_pnl.extend([
                            (str(name_i), s_name, _qlabel(q), t_name, b_name, pd.Timestamp(dt), float(val), float(spy_v))
                            for name_i, val in pnl_per_id.items() if np.isfinite(val)
                        ])

    paths = {
        "raw_corr": _collapse_corr(recs_raw, id_col, raw_corr_path, "alpha_raw_spy_corr"),
        "pnl_corr": _collapse_corr(recs_pnl, id_col, pnl_corr_path, "alpha_pnl_spy_corr"),
        "raw_ccf": _collapse_ccf(recs_raw, id_col, raw_ccf_path, "alpha_raw_spy_ccf", ccf_max_lag) if ccf_enable else None,
        "pnl_ccf": _collapse_ccf(recs_pnl, id_col, pnl_ccf_path, "alpha_pnl_spy_ccf", ccf_max_lag) if ccf_enable else None,
    }
    return paths


__all__ = ["compute_market_dist_stats"]
