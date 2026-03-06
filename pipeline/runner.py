# pipeline/runner.py — consume daily feature PKLs -> produce stats/summary/outliers (all PKL)
# All user-facing config now lives in main.py and is passed in as a dict to run_pipeline(cfg).
# Supports three separate input directories for signals, targets, and bet-sizes
# (or a single combined directory — the default).

import os
import re
import json
import pickle
import shutil
import time
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from pipeline.daily_stats import compute_daily_stats
from pipeline.summary_stats import compute_summary_stats_over_days
from pipeline.outliers_stats import compute_outliers, save_outliers


def _dirpaths(output_root: str):
    """Derive canonical subdirectories under the output root."""
    daily = os.path.join(output_root, "DAILY_STATS")
    summary = os.path.join(output_root, "SUMMARY_STATS")
    outliers = os.path.join(output_root, "OUTLIERS")  # NOTE: fixed spelling
    per_ticker = os.path.join(output_root, "MDS_STATS")  # Market Distribution Stats (per-id corr/CCF)
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


# ===================== VERIFICATION =====================
def _verify_daily_stats(stats: Dict, day_str: str, signal_cols: List[str],
                        target_cols: List[str], bet_size_cols: List[str],
                        quantiles: List[float]) -> List[str]:
    """
    Run sanity checks on the daily stats dict for one day.
    Returns a list of warning strings (empty = all checks passed).
    """
    warnings_list: List[str] = []
    expected_qlabels = [f"qr_{int(round(q * 100))}" for q in quantiles]

    for stat_type in ['pnl', 'ppd', 'sizeNotional']:
        if stat_type not in stats:
            warnings_list.append(f"  [{day_str}] Missing stat_type '{stat_type}' entirely.")
            continue
        for sig in signal_cols:
            if sig not in stats[stat_type]:
                continue
            for ql in expected_qlabels:
                if ql not in stats[stat_type][sig]:
                    continue
                for tgt in target_cols:
                    if tgt == "__ALL__":
                        continue
                    if tgt not in stats[stat_type][sig][ql]:
                        continue
                    for bet in bet_size_cols:
                        val = stats[stat_type][sig][ql].get(tgt, {}).get(bet)
                        if val is None:
                            warnings_list.append(
                                f"  [{day_str}] {stat_type} missing for "
                                f"sig={sig}, q={ql}, tgt={tgt}, bet={bet}"
                            )

    # Check PnL = sum(sign(s) * fret * betsize) consistency
    pnl_tree = stats.get('pnl', {})
    ppd_tree = stats.get('ppd', {})
    not_tree = stats.get('sizeNotional', {})
    for sig in signal_cols:
        for ql in expected_qlabels:
            for tgt in target_cols:
                if tgt == "__ALL__":
                    continue
                for bet in bet_size_cols:
                    pnl_v = pnl_tree.get(sig, {}).get(ql, {}).get(tgt, {}).get(bet)
                    ppd_v = ppd_tree.get(sig, {}).get(ql, {}).get(tgt, {}).get(bet)
                    not_v = not_tree.get(sig, {}).get(ql, {}).get(tgt, {}).get(bet)
                    if (pnl_v is not None and not_v is not None and ppd_v is not None
                            and np.isfinite(pnl_v) and np.isfinite(not_v) and not_v > 0
                            and np.isfinite(ppd_v)):
                        recomputed_ppd = pnl_v / not_v
                        if abs(recomputed_ppd - ppd_v) > 1e-10:
                            warnings_list.append(
                                f"  [{day_str}] PPD mismatch for sig={sig}, q={ql}, tgt={tgt}, bet={bet}: "
                                f"pnl/notional={recomputed_ppd:.8f} vs stored ppd={ppd_v:.8f}"
                            )
    return warnings_list


def _verify_summary_stats(summary: Dict, signal_cols: List[str],
                          target_cols: List[str], bet_size_cols: List[str]) -> List[str]:
    """
    Run sanity checks on the aggregated summary stats dict.
    Returns a list of warning strings.
    """
    warnings_list: List[str] = []

    # Check Sharpe is computed from daily PPD mean/std
    sharpe_tree = summary.get('sharpe', {})
    ppd_tree = summary.get('ppd', {})
    for sig, q_dict in sharpe_tree.items():
        for ql, t_dict in q_dict.items():
            for tgt, b_dict in t_dict.items():
                for bet, sr_val in b_dict.items():
                    if sr_val is not None and np.isfinite(sr_val):
                        ppd_val = ppd_tree.get(sig, {}).get(ql, {}).get(tgt, {}).get(bet)
                        if ppd_val is not None and np.isfinite(ppd_val):
                            if ppd_val == 0.0 and sr_val != 0.0:
                                warnings_list.append(
                                    f"  [summary] Sharpe={sr_val:.4f} but PPD=0 for "
                                    f"sig={sig}, q={ql}, tgt={tgt}, bet={bet}"
                                )

    # Check PPD = PnL / SizeNotional consistency
    pnl_tree = summary.get('pnl', {})
    not_tree = summary.get('sizeNotional', {})
    for sig, q_dict in ppd_tree.items():
        for ql, t_dict in q_dict.items():
            for tgt, b_dict in t_dict.items():
                for bet, ppd_v in b_dict.items():
                    pnl_v = pnl_tree.get(sig, {}).get(ql, {}).get(tgt, {}).get(bet)
                    not_v = not_tree.get(sig, {}).get(ql, {}).get(tgt, {}).get(bet)
                    if (ppd_v is not None and pnl_v is not None and not_v is not None
                            and np.isfinite(ppd_v) and np.isfinite(pnl_v)
                            and np.isfinite(not_v) and not_v > 0):
                        expected = pnl_v / not_v
                        if abs(expected - ppd_v) > 1e-8:
                            warnings_list.append(
                                f"  [summary] PPD mismatch: pnl/notional={expected:.8f} "
                                f"vs stored ppd={ppd_v:.8f} for "
                                f"sig={sig}, q={ql}, tgt={tgt}, bet={bet}"
                            )

    return warnings_list


# ===================== MULTI-DIR LOADING =====================
def _load_category_files(
    category_dir: Optional[str],
    glob_pattern: Optional[str],
    fallback_dir: str,
    fallback_glob: str,
    label: str,
) -> List[Path]:
    """Discover PKL files for a given category (signals/targets/betsizes)."""
    d = category_dir if category_dir else fallback_dir
    g = glob_pattern if glob_pattern else fallback_glob
    p = Path(d)
    if not p.is_dir():
        print(f"[WARN] {label} directory does not exist: {d}")
        return []
    files = sorted(p.glob(g), key=lambda x: x.name)
    print(f"[info] {label} directory: {d}  ({len(files)} files matching '{g}')")
    return files
def run_pipeline(cfg: Dict) -> Dict[str, Optional[str]]:
    """
    Run the pipeline.

    All configuration must be passed in from main.py via the `cfg` dict.
    This function does *not* define defaults; it assumes the caller populates
    all keys that used to live in DEFAULT_CONFIG.

    Supports three separate input directories for signals, targets, and
    bet-sizes.  When all three point to the same path (or are None), the
    pipeline behaves identically to the original single-directory mode.
    """
    t0 = time.perf_counter()
    local_cfg = dict(cfg)  # shallow copy to be safe

    # ---- Unpack input directories (new dict-based config) ----
    def _input_spec(key: str, fallback_dir=".", fallback_glob="*.pkl"):
        """Extract (dir, glob) from a config entry that is either a dict or a legacy string."""
        v = local_cfg.get(key)
        if isinstance(v, dict):
            return v.get("dir", fallback_dir), v.get("glob") or fallback_glob
        # Legacy: bare string = dir path
        if isinstance(v, str):
            return v, fallback_glob
        return fallback_dir, fallback_glob

    sig_dir, sig_glob   = _input_spec("signals_input")
    tgt_dir, tgt_glob   = _input_spec("targets_input")
    bet_dir, bet_glob   = _input_spec("betsizes_input")

    # Legacy fallback: if old-style features_input_dir is set but new keys aren't,
    # use it for all three.
    legacy_dir = local_cfg.get("features_input_dir")
    legacy_glob = local_cfg.get("features_glob", "*.pkl")
    if legacy_dir and not local_cfg.get("signals_input"):
        sig_dir, sig_glob = legacy_dir, legacy_glob
    if legacy_dir and not local_cfg.get("targets_input"):
        tgt_dir, tgt_glob = legacy_dir, legacy_glob
    if legacy_dir and not local_cfg.get("betsizes_input"):
        bet_dir, bet_glob = legacy_dir, legacy_glob

    output_root = local_cfg["output_root"]

    # ---- Column discovery ----
    signal_prefix = local_cfg["signal_prefix"]
    target_prefix = local_cfg["target_prefix"]
    bet_prefix    = local_cfg["bet_prefix"]
    signal_regex  = local_cfg.get("signal_regex")
    target_regex  = local_cfg.get("target_regex")
    bet_regex     = local_cfg.get("bet_regex")

    # ---- SPY ----
    spy_ticker      = local_cfg["spy_ticker"]
    spy_col_base    = local_cfg["spy_col_base"]
    spy_single_name = local_cfg["spy_single_name"]

    # ---- Quantiles ----
    quantiles      = local_cfg["quantiles"]
    type_quantile  = local_cfg["type_quantile"]

    # ---- Stage toggles ----
    do_daily    = local_cfg["do_daily"]
    do_summary  = local_cfg["do_summary"]
    do_outliers = local_cfg["do_outliers"]

    # ---- Summary extras ----
    add_spearman = local_cfg["add_spearman"]
    add_dcor     = local_cfg["add_dcor"]
    spearman_sample_cap_per_key = local_cfg["spearman_sample_cap_per_key"]

    # ---- CCF (replaces the 4 old dump_ booleans) ----
    ccf_enable           = local_cfg.get("ccf_enable", False)
    ccf_max_lag          = local_cfg.get("ccf_max_lag", 5)
    ccf_dump_per_ticker  = local_cfg.get("ccf_dump_per_ticker", False)

    # ---- Outlier / daily behaviour ----
    outlier_metrics            = local_cfg["outlier_metrics"]
    empty_day_policy           = local_cfg["empty_day_policy"]
    report_empty_trades_as_nan = local_cfg["report_empty_trades_as_nan"]

    # ---- Parallelism ----
    n_jobs_io      = local_cfg["n_jobs_io"]
    n_jobs_daily   = local_cfg["n_jobs_daily"]
    n_jobs_summary = local_cfg["n_jobs_summary"]
    random_state   = local_cfg["random_state"]

    # ---- Date range ----
    interval_start = local_cfg["interval_start"]
    interval_end   = local_cfg["interval_end"]

    # Determine multi-directory mode
    _real = lambda d: os.path.realpath(d)
    multi_dir_mode = not (_real(sig_dir) == _real(tgt_dir) == _real(bet_dir))
    if multi_dir_mode:
        print("[info] Multi-directory mode: signals, targets, and bet-sizes loaded from separate directories.")
        print(f"       signals  : {sig_dir}  (glob: {sig_glob})")
        print(f"       targets  : {tgt_dir}  (glob: {tgt_glob})")
        print(f"       betsizes : {bet_dir}  (glob: {bet_glob})")
    else:
        print(f"[info] Single-directory mode: {sig_dir}  (glob: {sig_glob})")

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
    # In single-dir mode, use sig_dir/sig_glob as the canonical source
    features_dir = Path(sig_dir)
    files = sorted(features_dir.glob(sig_glob), key=lambda p: p.name)

    # In multi-dir mode, also discover per-category files
    if multi_dir_mode:
        sig_files = _load_category_files(sig_dir, sig_glob, sig_dir, sig_glob, "Signals")
        tgt_files = _load_category_files(tgt_dir, tgt_glob, sig_dir, sig_glob, "Targets")
        bet_files = _load_category_files(bet_dir, bet_glob, sig_dir, sig_glob, "Betsizes")
        # Build date -> path maps for each category
        def _date_map(file_list):
            out = {}
            for p in file_list:
                ds = _extract_date_str(p.name)
                if ds:
                    out[ds] = p
            return out
        sig_by_date = _date_map(sig_files)
        tgt_by_date = _date_map(tgt_files)
        bet_by_date = _date_map(bet_files)
        # Union of all dates across categories
        all_dates = sorted(set(sig_by_date) | set(tgt_by_date) | set(bet_by_date))
        print(f"[info] Multi-dir: {len(all_dates)} unique dates across all category directories.")
    else:
        sig_by_date = tgt_by_date = bet_by_date = None
        all_dates = None

    if not files and not multi_dir_mode:
        print(f"[stop] No feature PKLs matching {features_dir / sig_glob}")
        elapsed = time.perf_counter() - t0
        return {
            "daily_dir": DAILY_STATS_DIR,
            "summary_path": None,
            "summary_dir": SUMMARY_STATS_DIR,
            "outliers_path": None,
            "outliers_dir": OUTLIERS_DIR,
            "per_ticker_dir": PER_TICKER_DIR,
            "market_dist_dir": PER_TICKER_DIR,
            "market_dist_paths": {},
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

    def _load_and_merge_multi(day_str: str) -> Optional[pd.DataFrame]:
        """Load signal/target/betsize from separate directories and merge on ticker."""
        frames = {}
        col_sets = {}
        for label, by_date, prefix, regex in [
            ("signal", sig_by_date, signal_prefix, signal_regex),
            ("target", tgt_by_date, target_prefix, target_regex),
            ("betsize", bet_by_date, bet_prefix, bet_regex),
        ]:
            p = by_date.get(day_str) if by_date else None
            if p is None or not p.exists():
                continue
            try:
                df = _read_pickle_compat(p)
            except Exception as e:
                print(f"[skip] failed to read {p}: {e}")
                continue
            if "ticker" not in df.columns:
                print(f"[skip] {p.name}: missing 'ticker'")
                continue
            cols = _pick_cols(df, prefix, regex)
            if not cols:
                continue
            frames[label] = df[["ticker"] + cols].copy()
            col_sets[label] = cols

        if not frames:
            return None

        # Start with whichever frame has ticker
        base = None
        for k in ("signal", "target", "betsize"):
            if k in frames:
                if base is None:
                    base = frames[k]
                else:
                    base = base.merge(frames[k], on="ticker", how="inner")

        if base is None or "ticker" not in base.columns:
            return None

        return base

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

    def _load_one_multi(day_str: str) -> Optional[Tuple[pd.Timestamp, str, str, pd.DataFrame, Dict[str, str]]]:
        """Multi-directory loader: merge signal/target/betsize files for one day."""
        df_use = _load_and_merge_multi(day_str)
        if df_use is None:
            return None

        signal_cols = _pick_cols(df_use, signal_prefix, signal_regex)
        target_cols = _pick_cols(df_use, target_prefix, target_regex)
        bet_cols = _pick_cols(df_use, bet_prefix, bet_regex)

        if not signal_cols or not target_cols or not bet_cols:
            print(f"[skip] {day_str}: missing features after merge ({signal_prefix}*, {target_prefix}*, {bet_prefix}*)")
            return None

        # Derive per-target SPY columns
        spy_map_for_day: Dict[str, str] = {}
        if spy_ticker:
            try:
                has_spy_row = (df_use['ticker'] == spy_ticker).any()
            except Exception:
                has_spy_row = False
            if has_spy_row:
                spy_sub = df_use.loc[df_use['ticker'] == spy_ticker, target_cols]
                for tcol in target_cols:
                    col_name = _spy_col_for_target(tcol)
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
            print(f"[skip] bad date {day_str}")
            return None
        if START_DT and (day_dt.normalize() < START_DT):
            return None
        if END_DT and (day_dt.normalize() > END_DT):
            return None
        return (day_dt, day_str, "multi-dir", df_use, spy_map_for_day)

    # Load items — choose single-dir or multi-dir path
    items: List[Tuple[pd.Timestamp, str, str, pd.DataFrame, Dict[str, str]]] = []
    if multi_dir_mode and all_dates:
        if n_jobs_io <= 1:
            for ds in all_dates:
                rec = _load_one_multi(ds)
                if rec is not None:
                    items.append(rec)
        else:
            with ThreadPoolExecutor(max_workers=int(n_jobs_io)) as ex:
                futs = [ex.submit(_load_one_multi, ds) for ds in all_dates]
                for fut in as_completed(futs):
                    rec = fut.result()
                    if rec is not None:
                        items.append(rec)
    else:
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
            "market_dist_dir": PER_TICKER_DIR,
            "market_dist_paths": {},
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
        signal_cols = _pick_cols(df, signal_prefix, signal_regex)
        target_cols = _pick_cols(df, target_prefix, target_regex)
        bet_size_cols = _pick_cols(df, bet_prefix, bet_regex)

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
                print(f"Saved daily stats PKL for {day_str} -> {out_path} ({len(day_df_stats)} rows)")

                # Verification: check internal consistency of daily stats
                vwarns = _verify_daily_stats(stats, day_str, signal_cols, target_cols,
                                             bet_size_cols, quantiles)
                if vwarns:
                    for w in vwarns[:5]:  # cap at 5 warnings per day
                        print(f"⚠️  VERIFY {w}")
                    if len(vwarns) > 5:
                        print(f"⚠️  VERIFY  ... and {len(vwarns) - 5} more warnings for {day_str}")
                else:
                    print(f"Verification passed for {day_str}")
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

            # Optional per-id CCF dump paths (stored in MDS_STATS/per_ticker_dir)
            dump_raw_ccf = (
                os.path.join(
                    PER_TICKER_DIR,
                    f"mds_alpha_raw_spy_ccf_{first_day:%Y%m%d}_{last_day:%Y%m%d}.pkl",
                )
                if (spy_by_target_effective and ccf_enable and ccf_dump_per_ticker)
                else None
            )
            dump_pnl_ccf = (
                os.path.join(
                    PER_TICKER_DIR,
                    f"mds_alpha_pnl_spy_ccf_{first_day:%Y%m%d}_{last_day:%Y%m%d}.pkl",
                )
                if (spy_by_target_effective and ccf_enable and ccf_dump_per_ticker)
                else None
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
                # per-ID CCF dumps (corr dumps remain disabled unless added back later)
                id_col="ticker",
                dump_alpha_raw_corr_path=None,
                dump_alpha_pnl_corr_path=None,
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
                print(f"Saved summary stats PKL -> {summary_path} ({len(summary_df)} rows)")

                # Verification: check internal consistency of summary stats
                vwarns = _verify_summary_stats(summary, sig_list, tgt_list, bet_list)
                if vwarns:
                    for w in vwarns[:10]:
                        print(f"⚠️  VERIFY {w}")
                    if len(vwarns) > 10:
                        print(f"⚠️  VERIFY  ... and {len(vwarns) - 10} more summary warnings")
                else:
                    print(f"Summary verification passed")

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
        print(f"Wrote daily index -> {index_csv} and {index_pkl}")

    elapsed = time.perf_counter() - t0
    print(f"Pipeline finished in {elapsed:.3f} seconds.")
    return {
        "daily_dir": DAILY_STATS_DIR,
        "summary_path": summary_path,
        "summary_dir": SUMMARY_STATS_DIR,
        "outliers_path": outliers_path,
        "outliers_dir": OUTLIERS_DIR,
        "per_ticker_dir": PER_TICKER_DIR,
        "market_dist_dir": PER_TICKER_DIR,
        "market_dist_paths": {},
        "index_csv": index_csv,
        "index_pkl": index_pkl,
        "elapsed_sec": elapsed,
    }