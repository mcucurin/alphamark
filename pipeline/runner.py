# pipeline/runner.py — consume daily feature PKLs -> produce stats/summary/outliers (all PKL)
# All user-facing config now lives in main.py and is passed in as a dict to run_pipeline(cfg).

import os
import re
import json
import pickle
import shutil
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict

import pandas as pd

from pipeline.daily_stats import compute_daily_stats
from pipeline.summary_stats import compute_summary_stats_over_days
from pipeline.outliers_stats import compute_outliers, save_outliers


def _dirpaths(output_root: str):
    """Derive canonical subdirectories under the output root."""
    daily = os.path.join(output_root, "DAILY_STATS")
    summary = os.path.join(output_root, "SUMMARY_STATS")
    outliers = os.path.join(output_root, "OUTLIERS")  # NOTE: fixed spelling
    per_ticker = os.path.join(output_root, "PER_TICKER")
    return daily, summary, outliers, per_ticker


# ===================== UTILS =====================
def _atomic_pickle_dump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as tmp:
        pickle.dump(obj, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def _extract_date_str(name: str) -> str | None:
    m = re.search(r"(\d{8})", name)
    return m.group(1) if m else None


def _read_pickle_compat(path: Path):
    class NPCompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core")
            return super().find_class(module, name)

    with path.open("rb") as f:
        return NPCompatUnpickler(f).load()


def _ensure_dirs(output_root: str):
    daily, summary, outliers, per_ticker = _dirpaths(output_root)
    os.makedirs(daily, exist_ok=True)
    os.makedirs(summary, exist_ok=True)
    os.makedirs(outliers, exist_ok=True)
    os.makedirs(per_ticker, exist_ok=True)
    return daily, summary, outliers, per_ticker


def _split_list_arg(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_quantiles(s: Optional[str]):
    if s is None:
        return None
    out = []
    for tok in _split_list_arg(s):
        v = float(tok)
        out.append(v / 100.0 if v > 1.0 else v)
    return out


def _parse_interval(start_s: Optional[str], end_s: Optional[str]) -> Tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Parse user provided interval strings into normalized (date-only) pd.Timestamps."""
    def _coerce(x):
        if x is None:
            return None
        try:
            dt = pd.to_datetime(x, errors="coerce")
            if pd.isna(dt):
                return None
            return pd.Timestamp(dt).normalize()
        except Exception:
            return None

    s = _coerce(start_s)
    e = _coerce(end_s)
    return s, e


# ===================== PIPELINE =====================
def run_pipeline(cfg: Dict) -> Dict[str, Optional[str]]:
    """
    Run the pipeline.

    All configuration must be passed in from main.py via the `cfg` dict.
    This function does *not* define defaults; it assumes the caller populates
    all keys that used to live in DEFAULT_CONFIG.
    """
    t0 = time.perf_counter()
    local_cfg = dict(cfg)  # shallow copy to be safe

    # unpack required config
    (features_input_dir, features_glob, output_root, signal_prefix, target_prefix, bet_prefix,
     signal_regex, target_regex, bet_regex, spy_ticker, spy_col_base, spy_single_name,
     quantiles, type_quantile, do_daily, do_summary, do_outliers, add_spearman, add_dcor,
     spearman_sample_cap_per_key, dump_alpha_raw_per_id, dump_alpha_pnl_per_id,
     ccf_enable, ccf_max_lag, dump_alpha_raw_ccf_per_id, dump_alpha_pnl_ccf_per_id,
     outlier_metrics, outlier_z_thresh, empty_day_policy,
     report_empty_trades_as_nan, n_jobs_io, n_jobs_daily, n_jobs_summary, random_state,
     interval_start, interval_end) = (
        local_cfg["features_input_dir"], local_cfg["features_glob"], local_cfg["output_root"],
        local_cfg["signal_prefix"], local_cfg["target_prefix"], local_cfg["bet_prefix"],
        local_cfg["signal_regex"], local_cfg["target_regex"], local_cfg["bet_regex"],
        local_cfg["spy_ticker"], local_cfg["spy_col_base"], local_cfg["spy_single_name"],
        local_cfg["quantiles"], local_cfg["type_quantile"],
        local_cfg["do_daily"], local_cfg["do_summary"], local_cfg["do_outliers"],
        local_cfg["add_spearman"], local_cfg["add_dcor"], local_cfg["spearman_sample_cap_per_key"],
        local_cfg["dump_alpha_raw_per_id"], local_cfg["dump_alpha_pnl_per_id"],
        local_cfg["ccf_enable"], local_cfg["ccf_max_lag"],
        local_cfg["dump_alpha_raw_ccf_per_id"], local_cfg["dump_alpha_pnl_ccf_per_id"],
        local_cfg["outlier_metrics"], local_cfg["outlier_z_thresh"], local_cfg["empty_day_policy"],
        local_cfg["report_empty_trades_as_nan"], local_cfg["n_jobs_io"], local_cfg["n_jobs_daily"],
        local_cfg["n_jobs_summary"], local_cfg["random_state"],
        local_cfg["interval_start"], local_cfg["interval_end"]
    )

    DAILY_STATS_DIR, SUMMARY_STATS_DIR, OUTLIERS_DIR, PER_TICKER_DIR = _ensure_dirs(output_root)

    # Parse interval (inclusive)
    START_DT, END_DT = _parse_interval(interval_start, interval_end)
    if START_DT and END_DT and END_DT < START_DT:
        # swap if user reversed them
        START_DT, END_DT = END_DT, START_DT
    if START_DT:
        print(f"[info] Interval start (inclusive): {START_DT:%Y-%m-%d}")
    if END_DT:
        print(f"[info] Interval end   (inclusive): {END_DT:%Y-%m-%d}")

    # 1) Discover and load feature PKLs (one per day) — parallel I/O
    features_dir = Path(features_input_dir)
    files = sorted(features_dir.glob(features_glob), key=lambda p: p.name)
    if not files:
        print(f"[stop] No feature PKLs matching {features_dir / features_glob}")
        elapsed = time.perf_counter() - t0
        return {
            "daily_dir": DAILY_STATS_DIR,
            "summary_path": None,
            "summary_dir": SUMMARY_STATS_DIR,
            "outliers_path": None,
            "outliers_dir": OUTLIERS_DIR,
            "per_ticker_dir": PER_TICKER_DIR,
            "index_csv": None,
            "index_pkl": None,
            "elapsed_sec": elapsed,
        }

    def _pick_cols(df: pd.DataFrame, prefix: str, regex: Optional[str]) -> List[str]:
        pool = [c for c in df.columns if isinstance(c, str)]
        if regex:
            import re as _re
            r = _re.compile(regex)
            return [c for c in pool if r.search(c)]
        return [c for c in pool if c.startswith(prefix)]

    def _spy_col_for_target(target_name: str) -> str:
        # final spy column per target, stable naming
        return f"{spy_col_base}__{target_name}"

    def _load_one(p: Path) -> Optional[Tuple[pd.Timestamp, str, str, pd.DataFrame, Dict[str, str]]]:
        day_str = _extract_date_str(p.name)
        if not day_str:
            print(f"[skip] cannot parse date from {p.name}")
            return None
        try:
            df = _read_pickle_compat(p)
        except Exception as e:
            print(f"[skip] failed to read {p}: {e}")
            return None
        if "ticker" not in df.columns:
            print(f"[skip] {p.name}: missing 'ticker'")
            return None

        signal_cols = _pick_cols(df, signal_prefix, signal_regex)
        target_cols = _pick_cols(df, target_prefix, target_regex)
        bet_cols = _pick_cols(df, bet_prefix, bet_regex)

        if not signal_cols or not target_cols or not bet_cols:
            print(f"[skip] {p.name}: missing features ({signal_prefix}*, {target_prefix}*, {bet_prefix}*)")
            return None

        keep = ["ticker"] + sorted(set(signal_cols + target_cols + bet_cols))
        df_use = df[keep].copy()

        # Derive per-target SPY columns: for each target, take SPY row's target value and broadcast it
        spy_map_for_day: Dict[str, str] = {}
        if spy_ticker:
            try:
                has_spy_row = (df_use['ticker'] == spy_ticker).any()
            except Exception:
                has_spy_row = False

            if has_spy_row:
                spy_sub = df_use.loc[df_use['ticker'] == spy_ticker, target_cols]
                for tcol in target_cols:
                    col_name = _spy_col_for_target(tcol)  # e.g., "spy__fret_5D"
                    try:
                        v = pd.to_numeric(spy_sub[tcol], errors='coerce').dropna()
                        spy_val = float(v.mean()) if not v.empty else None
                    except Exception:
                        spy_val = None
                    if spy_val is not None:
                        df_use[col_name] = spy_val
                        spy_map_for_day[tcol] = col_name

        day_dt = pd.to_datetime(day_str, format="%Y%m%d", errors="coerce")
        if pd.isna(day_dt):
            print(f"[skip] bad date {day_str} from {p.name}")
            return None

        # Apply interval filter (inclusive) here so later steps only see requested window
        if START_DT and (day_dt.normalize() < START_DT):
            return None
        if END_DT and (day_dt.normalize() > END_DT):
            return None

        return (day_dt, day_str, str(p), df_use, spy_map_for_day)

    items: List[Tuple[pd.Timestamp, str, str, pd.DataFrame, Dict[str, str]]] = []
    if n_jobs_io <= 1:
        for p in files:
            rec = _load_one(p)
            if rec is not None:
                items.append(rec)
    else:
        with ThreadPoolExecutor(max_workers=int(n_jobs_io)) as ex:
            futs = [ex.submit(_load_one, p) for p in files]
            for fut in as_completed(futs):
                rec = fut.result()
                if rec is not None:
                    items.append(rec)

    # sort chronologically
    items.sort(key=lambda x: x[0])
    if not items:
        print("[stop] No usable feature files after filtering (interval may have excluded all).")
        elapsed = time.perf_counter() - t0
        return {
            "daily_dir": DAILY_STATS_DIR,
            "summary_path": None,
            "summary_dir": SUMMARY_STATS_DIR,
            "outliers_path": None,
            "outliers_dir": OUTLIERS_DIR,
            "per_ticker_dir": PER_TICKER_DIR,
            "index_csv": None,
            "index_pkl": None,
            "elapsed_sec": elapsed,
        }

    # 2) Per-day stats (PKL), and build inputs for summary/outliers
    daily_stats_frames: List[pd.DataFrame] = []
    per_day_index_rows = []
    raw_days_for_summary = []
    needed_sig, needed_tgt, needed_bet = set(), set(), set()

    # collect union mapping {target -> spy_col} across all days (only names; presence checked later)
    spy_by_target_global: Dict[str, str] = {}

    for day_dt, day_str, src_path, df, spy_map_for_day in items:
        signal_cols = [c for c in df.columns if c.startswith(signal_prefix)]
        target_cols = [c for c in df.columns if c.startswith(target_prefix)]
        bet_size_cols = [c for c in df.columns if c.startswith(bet_prefix)]

        needed_sig.update(signal_cols)
        needed_tgt.update(target_cols)
        needed_bet.update(bet_size_cols)

        # track spy mapping (names)
        for t, sc in spy_map_for_day.items():
            spy_by_target_global[t] = sc

        if do_daily:
            stats = compute_daily_stats(
                df,
                signal_cols=signal_cols,
                target_cols=target_cols,
                quantiles=quantiles,
                bet_size_cols=bet_size_cols,
                type_quantile=type_quantile,
                empty_day_policy=empty_day_policy,
                report_empty_trades_as_nan=report_empty_trades_as_nan,
                n_jobs=n_jobs_daily,
                random_state=random_state,
            )

            # Flatten nested dict -> rows
            rows = []
            for stat_type, sig_dict in stats.items():
                for s, qd in sig_dict.items():
                    for q, td in qd.items():
                        for t, bd in td.items():
                            for b, v in bd.items():
                                rows.append((day_str, s, t, q, stat_type, b, v))

            if rows:
                day_df_stats = pd.DataFrame(
                    rows,
                    columns=["date", "signal", "target", "qrank", "stat_type", "bet_size_col", "value"],
                )
                out_path = os.path.join(DAILY_STATS_DIR, f"stats_{day_str}.pkl")
                _atomic_pickle_dump(day_df_stats, out_path)
                per_day_index_rows.append({"date": day_str, "path": out_path, "n_rows": len(day_df_stats)})
                daily_stats_frames.append(day_df_stats)
                print(f"📦 Saved daily stats PKL for {day_str} -> {out_path} ({len(day_df_stats)} rows)")
            else:
                print(f"[skip] {day_str}: no stats produced")

        if do_summary:
            # include any spy columns we created this day
            spy_cols_today = list(spy_map_for_day.values())

            # Always include ticker for potential per-id dumps
            base_cols: List[str] = []
            if "ticker" in df.columns:
                base_cols.append("ticker")

            keep_cols = ["date"] + base_cols + signal_cols + target_cols + bet_size_cols + spy_cols_today
            raw_days_for_summary.append(
                df[base_cols + signal_cols + target_cols + bet_size_cols + spy_cols_today]
                .assign(date=day_dt)[keep_cols]
            )

    # 3) Summary stats over all days (PKL)
    summary_path = None
    if do_summary and raw_days_for_summary:
        big_df = pd.concat(raw_days_for_summary, ignore_index=True, copy=False)

        sig_list = sorted([c for c in needed_sig if c in big_df.columns])
        tgt_list = sorted([c for c in needed_tgt if c in big_df.columns])
        bet_list = sorted([c for c in needed_bet if c in big_df.columns])

        # Limit spy map to columns actually present in big_df
        spy_by_target_effective = {
            t: sc for t, sc in spy_by_target_global.items()
            if (t in big_df.columns and sc in big_df.columns)
        }

        if sig_list and tgt_list and bet_list:
            # Date range labels
            all_days = pd.to_datetime(big_df["date"], errors="coerce")
            first_day = pd.to_datetime(all_days.min())
            last_day = pd.to_datetime(all_days.max())

            # Prepare optional per-ticker dump paths (only if we actually have spy map)

            # (1) Per-id corr dumps (already existed)
            dump_raw = (
                os.path.join(
                    PER_TICKER_DIR,
                    f"per_ticker_alpha_raw_spy_corr_{first_day:%Y%m%d}_{last_day:%Y%m%d}.pkl",
                )
                if (spy_by_target_effective and dump_alpha_raw_per_id) else None
            )
            dump_pnl = (
                os.path.join(
                    PER_TICKER_DIR,
                    f"per_ticker_alpha_pnl_spy_corr_{first_day:%Y%m%d}_{last_day:%Y%m%d}.pkl",
                )
                if (spy_by_target_effective and dump_alpha_pnl_per_id) else None
            )

            # (2) NEW: per-id CCF dumps
            dump_raw_ccf = (
                os.path.join(
                    PER_TICKER_DIR,
                    f"per_ticker_alpha_raw_spy_ccf_{first_day:%Y%m%d}_{last_day:%Y%m%d}.pkl",
                )
                if (spy_by_target_effective and ccf_enable and dump_alpha_raw_ccf_per_id) else None
            )
            dump_pnl_ccf = (
                os.path.join(
                    PER_TICKER_DIR,
                    f"per_ticker_alpha_pnl_spy_ccf_{first_day:%Y%m%d}_{last_day:%Y%m%d}.pkl",
                )
                if (spy_by_target_effective and ccf_enable and dump_alpha_pnl_ccf_per_id) else None
            )

            summary = compute_summary_stats_over_days(
                big_df,
                date_col="date",
                signal_cols=sig_list,
                target_cols=tgt_list,
                bet_size_cols=bet_list,
                quantiles=quantiles,
                type_quantile=type_quantile,
                add_spearman=add_spearman,
                add_dcor=add_dcor,
                n_jobs=n_jobs_summary,  # parallel across signals
                spearman_sample_cap_per_key=spearman_sample_cap_per_key,
                random_state=random_state,
                spy_by_target=spy_by_target_effective if spy_by_target_effective else None,
                # per-ID dumps
                id_col="ticker",
                dump_alpha_raw_corr_path=dump_raw,
                dump_alpha_pnl_corr_path=dump_pnl,
                # NEW: per-ID CCF dumps
                dump_alpha_raw_ccf_path=dump_raw_ccf,
                dump_alpha_pnl_ccf_path=dump_pnl_ccf,
                ccf_max_lag=ccf_max_lag if ccf_enable else 0,
            )

            # flatten summary -> rows; tag with date range
            date_tag = (
                f"{first_day:%Y%m%d}_{last_day:%Y%m%d}"
                if pd.notna(first_day) and pd.notna(last_day) else "summary"
            )

            s_rows = []
            for stat_type, sig_dict in summary.items():
                for s, qd in sig_dict.items():
                    for q, td in qd.items():
                        for t, bd in td.items():
                            for b, v in bd.items():
                                s_rows.append((
                                    f"{last_day:%Y%m%d}" if pd.notna(last_day) else None,
                                    s, t, q, stat_type, b, v
                                ))

            if s_rows:
                summary_df = pd.DataFrame(
                    s_rows,
                    columns=["date", "signal", "target", "qrank", "stat_type", "bet_size_col", "value"],
                )
                summary_path = os.path.join(SUMMARY_STATS_DIR, f"summary_stats_{date_tag}.pkl")
                _atomic_pickle_dump(summary_df, summary_path)
                print(f"✅ Saved summary stats PKL -> {summary_path} ({len(summary_df)} rows)")
            else:
                print("[info] Summary produced no rows; not saving.")
        else:
            print("[info] No valid columns for summary stats; skipping summary save.")
    elif do_summary:
        print("[info] No daily data collected; skipping summary computation.")

    # 4) Outliers across all daily-stat frames (PKL)
    outliers_path = None
    if do_outliers and daily_stats_frames:
        stats_all = pd.concat(daily_stats_frames, ignore_index=True, copy=False)
        dates = pd.to_datetime(stats_all["date"], errors="coerce")
        first_day = pd.to_datetime(dates.min())
        last_day = pd.to_datetime(dates.max())
        date_tag = (
            f"{first_day:%Y%m%d}_{last_day:%Y%m%d}"
            if pd.notna(first_day) and pd.notna(last_day) else "all"
        )

        odf = compute_outliers(
            stats_all,
            stats_list=outlier_metrics,
            z_thresh=outlier_z_thresh,
        )
        print(f"[info] outlier metrics requested: {outlier_metrics}")
        print(f"[info] stat_types present in DAILY: {sorted(stats_all['stat_type'].dropna().astype(str).unique().tolist())}")
        outliers_path = os.path.join(OUTLIERS_DIR, f"outliers_{date_tag}.pkl")
        save_outliers(odf, outliers_path)
        print(f"⚠️  Saved outliers PKL -> {outliers_path} ({len(odf)} rows)")
    elif do_outliers:
        print("[info] No daily stats frames accumulated; skipping outlier computation.")

    # 5) Write index for daily stats (CSV + PKL)
    index_csv = None
    index_pkl = None
    if do_daily and per_day_index_rows:
        index_df = pd.DataFrame(per_day_index_rows).sort_values("date")
        index_csv = os.path.join(DAILY_STATS_DIR, "_index.csv")
        with NamedTemporaryFile(dir=os.path.dirname(index_csv), delete=False, mode="w", newline="") as tmp:
            index_df.to_csv(tmp.name, index=False)
            tmp_name = tmp.name
        shutil.move(tmp_name, index_csv)

        index_pkl = os.path.join(DAILY_STATS_DIR, "_index.pkl")
        _atomic_pickle_dump(index_df, index_pkl)
        print(f"🧭 Wrote daily index -> {index_csv} and {index_pkl}")

    elapsed = time.perf_counter() - t0
    print(f"⏱️  Pipeline finished in {elapsed:.3f} seconds.")
    return {
        "daily_dir": DAILY_STATS_DIR,
        "summary_path": summary_path,
        "summary_dir": SUMMARY_STATS_DIR,
        "outliers_path": outliers_path,
        "outliers_dir": OUTLIERS_DIR,
        "per_ticker_dir": PER_TICKER_DIR,
        "index_csv": index_csv,
        "index_pkl": index_pkl,
        "elapsed_sec": elapsed,
    }
