# pipeline/runner.py — consume daily feature PKLs -> produce stats/summary/outliers (all PKL)
# DEFAULT_CONFIG + optional JSON/CLI overrides; parallel I/O; truthful state.
# Now: run_pipeline() can be called with NO ARGS (uses DEFAULT_CONFIG), and we time it.
import os
import re
import json
import pickle
import shutil
import argparse
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict

import pandas as pd

from pipeline.daily_stats import compute_daily_stats
from pipeline.summary_stats import compute_summary_stats_over_days
from pipeline.outliers_stats import compute_outliers, save_outliers

# ===================== DEFAULT CONFIG (edit me) =====================
# HOW THIS CONFIG IS USED
# - You can edit values here, or supply a JSON via --config, or use CLI flags.
# - Precedence: DEFAULT_CONFIG  <  JSON --config  <  CLI flags (only when provided).
# - Any value set to None means “disabled” or “use the fallback behavior” (see notes).

DEFAULT_CONFIG: Dict = {
    # -------- I/O: where to read feature PKLs and where to write outputs --------
    "features_input_dir": "input/DAILY_FEATURES_PKL",   # str: folder containing feature *.pkl files (one per day)
    "features_glob": "features_*.pkl",                  # str: filename pattern for daily PKLs (e.g., features_20240103.pkl)
    "output_root": "output",                            # str: root folder; subfolders DAILY_STATS/, SUMMARY_STATS/, OUTLIERS/ are created

    # -------- Column selection: how to find signals, targets, bet sizes in your PKLs --------
    "signal_prefix": "pret_",                           # e.g., pret_alpha, pret_xxx
    "target_prefix": "fret_",                           # e.g., fret_1D, fret_5D
    "bet_prefix":    "betsize_",                        # nonnegative sizing
    "signal_regex":  None,                              # optional regex overrides
    "target_regex":  None,
    "bet_regex":     None,

    # -------- Market proxy: derive per-day SPY return *per target horizon* --------
    # We create one SPY column per target (e.g., target 'fret_5D' -> 'spy__fret_5D'),
    # by pulling the SPY row for that target and broadcasting its value to all rows.
    "spy_ticker": "SPY",                                # Optional[str]; set None to disable market correlation entirely
    "spy_col_base": "spy",                              # base name for derived columns, final col is f"{spy_col_base}__{target}"
    # Back-compat convenience: if you still want a single-column SPY for a specific target,
    # set this to that target name; the runner will also write {spy_col_name} with that horizon.
    "spy_col_name": "spy_ret",                          # legacy single SPY column (optional)
    "prefer_target_for_spy": "fret_1D",                 # which target to mirror into the single-column legacy name

    # -------- Stats knobs: how to slice the cross-section by signal strength --------
    "quantiles": [1.0, 0.75, 0.5, 0.25],               # use [1.0] for “all”, or e.g. [1.0, 0.2, 0.1]
    "type_quantile": "cumulative",                      # "cumulative" | "quantEach"

    # -------- Compute toggles --------
    "do_daily": True,
    "do_summary": True,
    "do_outliers": True,

    # -------- Summary extras --------
    "add_spearman": False,
    "add_dcor": False,
    "spearman_sample_cap_per_key": 10000,

    # -------- Outliers --------
    "outlier_metrics": ["pnl", "ppd", "sizeNotional", "nrInstr", "n_trades"],
    "outlier_z_thresh": 3.0,

    # -------- Daily behavior --------
    "empty_day_policy": "carry",                        # "carry" | "close" | "skip"
    "report_empty_trades_as_nan": True,

    # -------- Parallelism --------
    "n_jobs_io": 1,
    "n_jobs_daily": 3,
    "n_jobs_summary": 3,

    # -------- Reproducibility --------
    "random_state": 123,
}

def _dirpaths(output_root: str):
    daily = os.path.join(output_root, "DAILY_STATS")
    summary = os.path.join(output_root, "SUMMARY_STATS")
    outliers = os.path.join(output_root, "OUTLIERS")
    return daily, summary, outliers

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
    daily, summary, outliers = _dirpaths(output_root)
    os.makedirs(daily, exist_ok=True)
    os.makedirs(summary, exist_ok=True)
    os.makedirs(outliers, exist_ok=True)
    return daily, summary, outliers

def _split_list_arg(s: Optional[str]) -> List[str]:
    if not s: return []
    return [x.strip() for x in s.split(",") if x.strip()]

def _parse_quantiles(s: Optional[str]):
    if s is None: return None
    out = []
    for tok in _split_list_arg(s):
        v = float(tok)
        out.append(v/100.0 if v > 1.0 else v)
    return out

def _merge_config(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = v
    return out

# ===================== PIPELINE =====================
def run_pipeline(**cfg) -> Dict[str, Optional[str]]:
    """
    Run the pipeline. If called with no kwargs, uses DEFAULT_CONFIG.
    Any provided kwargs override DEFAULT_CONFIG keys.
    Returns a dict that also includes 'elapsed_sec'.
    """
    if not cfg:
        cfg = dict(DEFAULT_CONFIG)
    else:
        cfg = _merge_config(DEFAULT_CONFIG, cfg)

    t0 = time.perf_counter()

    # unpack
    (features_input_dir, features_glob, output_root, signal_prefix, target_prefix, bet_prefix,
     signal_regex, target_regex, bet_regex, spy_ticker, spy_col_base, spy_col_name, prefer_target_for_spy,
     quantiles, type_quantile, do_daily, do_summary, do_outliers, add_spearman, add_dcor,
     spearman_sample_cap_per_key, outlier_metrics, outlier_z_thresh, empty_day_policy,
     report_empty_trades_as_nan, n_jobs_io, n_jobs_daily, n_jobs_summary, random_state) = (
        cfg["features_input_dir"], cfg["features_glob"], cfg["output_root"],
        cfg["signal_prefix"], cfg["target_prefix"], cfg["bet_prefix"],
        cfg["signal_regex"], cfg["target_regex"], cfg["bet_regex"],
        cfg["spy_ticker"], cfg["spy_col_base"], cfg["spy_col_name"], cfg["prefer_target_for_spy"],
        cfg["quantiles"], cfg["type_quantile"], cfg["do_daily"], cfg["do_summary"], cfg["do_outliers"],
        cfg["add_spearman"], cfg["add_dcor"], cfg["spearman_sample_cap_per_key"],
        cfg["outlier_metrics"], cfg["outlier_z_thresh"], cfg["empty_day_policy"],
        cfg["report_empty_trades_as_nan"], cfg["n_jobs_io"], cfg["n_jobs_daily"],
        cfg["n_jobs_summary"], cfg["random_state"]
    )

    DAILY_STATS_DIR, SUMMARY_STATS_DIR, OUTLIERS_DIR = _ensure_dirs(output_root)

    # 1) Discover and load feature PKLs (one per day) — parallel I/O
    features_dir = Path(features_input_dir)
    files = sorted(features_dir.glob(features_glob), key=lambda p: p.name)
    if not files:
        print(f"[stop] No feature PKLs matching {features_dir / features_glob}")
        elapsed = time.perf_counter() - t0
        return {"daily_dir": DAILY_STATS_DIR, "summary_path": None, "outliers_path": None,
                "index_csv": None, "index_pkl": None, "elapsed_sec": elapsed}

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

    def _load_one(p: Path) -> Optional[Tuple[pd.Timestamp, str, str, pd.DataFrame, Dict[str,str]]]:
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
        bet_cols    = _pick_cols(df, bet_prefix, bet_regex)

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

                # Optional legacy single-column SPY for a chosen target
                if spy_col_name and prefer_target_for_spy in target_cols:
                    try:
                        v = pd.to_numeric(spy_sub[prefer_target_for_spy], errors='coerce').dropna()
                        legacy_val = float(v.mean()) if not v.empty else None
                    except Exception:
                        legacy_val = None
                    if legacy_val is not None:
                        df_use[spy_col_name] = legacy_val

        day_dt = pd.to_datetime(day_str, format="%Y%m%d", errors="coerce")
        if pd.isna(day_dt):
            print(f"[skip] bad date {day_str} from {p.name}")
            return None

        return (day_dt, day_str, str(p), df_use, spy_map_for_day)

    items: List[Tuple[pd.Timestamp, str, str, pd.DataFrame, Dict[str,str]]] = []
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
        print("[stop] No usable feature files after filtering.")
        elapsed = time.perf_counter() - t0
        return {"daily_dir": DAILY_STATS_DIR, "summary_path": None, "outliers_path": None,
                "index_csv": None, "index_pkl": None, "elapsed_sec": elapsed}

    # 2) Per-day stats (PKL), and build inputs for summary/outliers
    daily_stats_frames: List[pd.DataFrame] = []
    per_day_index_rows = []
    raw_days_for_summary = []
    needed_sig, needed_tgt, needed_bet = set(), set(), set()

    # collect union mapping {target -> spy_col} across all days (only names; presence checked later)
    spy_by_target_global: Dict[str, str] = {}

    for day_dt, day_str, src_path, df, spy_map_for_day in items:
        signal_cols    = [c for c in df.columns if c.startswith(signal_prefix)]
        target_cols    = [c for c in df.columns if c.startswith(target_prefix)]
        bet_size_cols  = [c for c in df.columns if c.startswith(bet_prefix)]

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
                out_path = os.path.join(_dirpaths(output_root)[0], f"stats_{day_str}.pkl")
                _atomic_pickle_dump(day_df_stats, out_path)
                per_day_index_rows.append({"date": day_str, "path": out_path, "n_rows": len(day_df_stats)})
                daily_stats_frames.append(day_df_stats)
                print(f"📦 Saved daily stats PKL for {day_str} -> {out_path} ({len(day_df_stats)} rows)")
            else:
                print(f"[skip] {day_str}: no stats produced")

        if do_summary:
            # include any spy columns we created this day
            spy_cols_today = list(spy_map_for_day.values())
            keep_cols = ["date"] + signal_cols + target_cols + bet_size_cols + spy_cols_today
            raw_days_for_summary.append(
                df[signal_cols + target_cols + bet_size_cols + spy_cols_today]
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
            t: sc for t, sc in spy_by_target_global.items() if (t in big_df.columns and sc in big_df.columns)
        }

        if sig_list and tgt_list and bet_list:
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
            )

            # flatten summary -> rows; tag with date range
            all_days  = pd.to_datetime(big_df["date"], errors="coerce")
            first_day = pd.to_datetime(all_days.min())
            last_day  = pd.to_datetime(all_days.max())
            date_tag  = (
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
                summary_path = os.path.join(_dirpaths(output_root)[1], f"summary_stats_{date_tag}.pkl")
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
        last_day  = pd.to_datetime(dates.max())
        date_tag  = (
            f"{first_day:%Y%m%d}_{last_day:%Y%m%d}"
            if pd.notna(first_day) and pd.notna(last_day) else "all"
        )

        odf = compute_outliers(
            stats_all,
            stats_list=outlier_metrics,
            z_thresh=outlier_z_thresh,
        )
        outliers_path = os.path.join(_dirpaths(output_root)[2], f"outliers_{date_tag}.pkl")
        save_outliers(odf, outliers_path)
        print(f"⚠️  Saved outliers PKL -> {outliers_path} ({len(odf)} rows)")
    elif do_outliers:
        print("[info] No daily stats frames accumulated; skipping outlier computation.")

    # 5) Write index for daily stats (CSV + PKL)
    index_csv = None
    index_pkl = None
    if do_daily and per_day_index_rows:
        index_df = pd.DataFrame(per_day_index_rows).sort_values("date")
        index_csv = os.path.join(_dirpaths(output_root)[0], "_index.csv")
        with NamedTemporaryFile(dir=os.path.dirname(index_csv), delete=False, mode="w", newline="") as tmp:
            index_df.to_csv(tmp.name, index=False)
            tmp_name = tmp.name
        shutil.move(tmp_name, index_csv)

        index_pkl = os.path.join(_dirpaths(output_root)[0], "_index.pkl")
        _atomic_pickle_dump(index_df, index_pkl)
        print(f"🧭 Wrote daily index -> {index_csv} and {index_pkl}")

    elapsed = time.perf_counter() - t0
    print(f"⏱️  Pipeline finished in {elapsed:.3f} seconds.")
    return {
        "daily_dir": _dirpaths(output_root)[0],
        "summary_path": summary_path,
        "outliers_path": outliers_path,
        "index_csv": index_csv,
        "index_pkl": index_pkl,
        "elapsed_sec": elapsed,
    }

# ===================== CLI (optional) =====================
def _load_json_config(path: Optional[str]) -> Dict:
    if not path:
        return {}
    with open(path, "r") as f:
        return json.load(f)

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Daily/Summary/Outliers pipeline (fast, truthful, parallel).")
    p.add_argument("--config", help="JSON file to override DEFAULT_CONFIG")
    p.add_argument("--write-active-config", help="Path to dump the final merged config", default=None)
    # Lightweight overrides (optional)
    p.add_argument("--features-input-dir"); p.add_argument("--features-glob"); p.add_argument("--output-root")
    p.add_argument("--signal-prefix"); p.add_argument("--target-prefix"); p.add_argument("--bet-prefix")
    p.add_argument("--signal-regex"); p.add_argument("--target-regex"); p.add_argument("--bet-regex")
    p.add_argument("--spy-ticker"); p.add_argument("--spy-col-base")
    p.add_argument("--spy-col-name"); p.add_argument("--prefer-target-for-spy")
    p.add_argument("--quantiles"); p.add_argument("--type-quantile", choices=["cumulative", "quantEach"])
    p.add_argument("--no-daily", action="store_true"); p.add_argument("--no-summary", action="store_true"); p.add_argument("--no-outliers", action="store_true")
    p.add_argument("--add-spearman", action="store_true"); p.add_argument("--add-dcor", action="store_true")
    p.add_argument("--spearman-sample-cap-per-key", type=int); p.add_argument("--outlier-metrics")
    p.add_argument("--outlier-z-thresh", type=float)
    p.add_argument("--empty-day-policy", choices=["carry","close","skip"])
    p.add_argument("--report-empty-trades-as-nan", action="store_true")
    p.add_argument("--n-jobs-io", type=int); p.add_argument("--n-jobs-daily", type=int); p.add_argument("--n-jobs-summary", type=int)
    p.add_argument("--random-state", type=int)
    return p

def _apply_cli_overrides(cfg: Dict, args: argparse.Namespace) -> Dict:
    # only set keys user provided
    def maybe(k, v):
        if v is not None: cfg[k] = v
    maybe("features_input_dir", args.features_input_dir)
    maybe("features_glob", args.features_glob)
    maybe("output_root", args.output_root)
    maybe("signal_prefix", args.signal_prefix)
    maybe("target_prefix", args.target_prefix)
    maybe("bet_prefix", args.bet_prefix)
    maybe("signal_regex", args.signal_regex)
    maybe("target_regex", args.target_regex)
    maybe("bet_regex", args.bet_regex)
    maybe("spy_ticker", args.spy_ticker)
    maybe("spy_col_base", args.spy_col_base)
    maybe("spy_col_name", args.spy_col_name)
    maybe("prefer_target_for_spy", args.prefer_target_for_spy)
    if args.quantiles: cfg["quantiles"] = [float(x)/100.0 if float(x)>1 else float(x) for x in args.quantiles.split(",")]
    maybe("type_quantile", args.type_quantile)
    if args.no_daily: cfg["do_daily"] = False
    if args.no_summary: cfg["do_summary"] = False
    if args.no_outliers: cfg["do_outliers"] = False
    if args.add_spearman: cfg["add_spearman"] = True
    if args.add_dcor: cfg["add_dcor"] = True
    maybe("spearman_sample_cap_per_key", args.spearman_sample_cap_per_key)
    if args.outlier_metrics: cfg["outlier_metrics"] = [s.strip() for s in args.outlier_metrics.split(",") if s.strip()]
    maybe("outlier_z_thresh", args.outlier_z_thresh)
    maybe("empty_day_policy", args.empty_day_policy)
    if args.report_empty_trades_as_nan: cfg["report_empty_trades_as_nan"] = True
    maybe("n_jobs_io", args.n_jobs_io); maybe("n_jobs_daily", args.n_jobs_daily); maybe("n_jobs_summary", args.n_jobs_summary)
    maybe("random_state", args.random_state)
    return cfg

def main():
    ap = _build_arg_parser()
    args = ap.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    json_cfg = _load_json_config(args.config)
    cfg = _merge_config(cfg, json_cfg)
    cfg = _apply_cli_overrides(cfg, args)

    if args.write_active_config:
        os.makedirs(os.path.dirname(args.write_active_config), exist_ok=True)
        with open(args.write_active_config, "w") as f:
            json.dump(cfg, f, indent=2, sort_keys=True)
        print(f"[info] Wrote active config -> {args.write_active_config}")

    run_pipeline(**cfg)

# ===================== ENTRY =====================
if __name__ == "__main__":
    main()
