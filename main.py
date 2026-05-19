# main.py
from pipeline.runner import run_pipeline
from plotting.plot_quantile_bars import generate_quantile_report

import argparse
import pickle as pkl
import pandas as pd
import os, glob, json
import time

start = time.perf_counter()

DEFAULT_COLS = ['date', 'signal', 'target', 'qrank', 'stat_type', 'bet_size_col', 'value']


# --- NumPy 1.x/2.x compatible pickle loader ---
def read_pickle_compat(path: str):
    """Unpickle objects across NumPy 1.x/2.x by remapping numpy._core -> numpy.core."""
    class NPCompatUnpickler(pkl.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core")
            return super().find_class(module, name)
    with open(path, "rb") as f:
        return NPCompatUnpickler(f).load()


def _load_json(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _env_or_none(name: str):
    v = os.getenv(name, "")
    return v if v.strip() else None


def _parse_int_tuple(s: str | None):
    if not s:
        return None
    try:
        parts = [int(x.strip()) for x in str(s).split(",") if x.strip()]
        if len(parts) == 2:
            return tuple(parts)
    except Exception:
        return None
    return None


def _parse_list(s: str | None):
    if not s:
        return None
    return [x.strip() for x in str(s).split(",") if x.strip()]


# =====================================================================
#                        CENTRALIZED CONFIG
# =====================================================================

# ---- Runner / pipeline config ----
DEFAULT_RUNNER_CONFIG = {
    # ========== I/O Configuration ==========
    "signals_input":  {"dir": "input/DAILY_FEATURES_PKL", "glob": "features_*.pkl"},
    "targets_input":  {"dir": "input/DAILY_FEATURES_PKL", "glob": "features_*.pkl"},
    "betsizes_input": {"dir": "input/DAILY_FEATURES_PKL", "glob": "features_*.pkl"},

    "output_root": "output",

    # ========== Column Discovery ==========
    "signal_prefix": "pret_",       "signal_regex": None,       "signal_list": None,
    "target_prefix": "fret_",       "target_regex": None,       "target_list": None,
    "bet_prefix":    "betsize_",    "bet_regex":    None,       "bet_list": None,

    # ========== Market Proxy (SPY) ==========
    "spy_ticker":      "SPY",
    "spy_col_base":    "spy",
    "spy_single_name": "spy_ret",

    # ========== Quantile Configuration ==========
    "quantiles": [1.0, 0.75, 0.5, 0.25],
    "type_quantile": "cumulative",

    # ========== Summary Statistics Extras ==========
    "add_spearman": False,
    "add_dcor": False,
    "spearman_sample_cap_per_key": 10000,

    # ========== CCF (Cross-Correlation vs Market Proxy) ==========
    "ccf_enable": False,
    "ccf_max_lag": 5,
    "ccf_dump_per_ticker": False,

    # ========== Outlier Detection ==========
    "outlier_metrics": ["pnl", "ppd", "sizeNotional", "n_trades"],

    # ========== Daily Processing Behavior ==========

    # ========== Parallelism Configuration ==========
    "n_jobs_io": 1,
    "n_jobs_daily": 3,
    "n_jobs_summary": 3,

    # ========== Reproducibility ==========
    "random_state": 123,

    # ========== Date Range Filter (Inclusive) ==========
    "interval_start": "2024-01-01",
    "interval_end":   "2024-01-31",
}

# ---- Plotting / report config ----
DEFAULT_PLOT_CONFIG = {
    # ========== Quantile Display Configuration ==========
    "qranks": ["qr_100", "qr_75", "qr_50", "qr_25"],

    # ========== Heatmap Filter Configuration (H2/H3) ==========
    "H2_targets": "AUTO",
    "H2_bets":    "AUTO",
    "H3_targets": "AUTO",
    "H3_bets":    "AUTO",

    # ========== Temporal Line Smoothing Windows ==========
    "roll_h1_lines": 30,
    "roll_h2_lines": 30,
    "roll_h3_lines": 1,

    # ========== Rolling Windows for Temporal Panels ==========
    # Set to 1 for no smoothing; the plotting module applies an adaptive
    # data-length-aware minimum so short intervals are never blanked out.
    "roll_nrinstr":       1,
    "roll_trades":        1,
    "roll_size_notional": 1,
    "roll_sharpe":        60,
    "roll_hit_ratio":     1,

    # ========== Temporal Plot Configuration ==========
    "variables_temporal_plot": ["pnl", "ppd", "n_trades", "sizeNotional"],
    "arrayDim_temporal_plot":  (2, 2),

    # ========== Bar Plot Configuration (SUMMARY Data Only) ==========
    "bar_page_vars":        ["target", "bet_size_col"],
    "bar_x_vars":           ["signal"],
    "bar_metrics": [
        "pnl", "ppd", "sharpe", "hit_ratio", "long_ratio",
        "sizeNotional", "r2", "t_stat", "n_trades", "market_corr"
    ],
    "aspect_ratio_barplots": 16 / 9,

    # ========== Outlier Table Configuration ==========
    "outlier_metrics_for_tables": ["pnl", "ppd", "sizeNotional", "n_trades"],
    "outlier_top_k":           3,
    "outlier_tables_per_page": 2,

    # ========== Plot Styling ==========
    "line_style":  "-",

    # Quantile color palette — professional, colorblind-safe.
    # Override individual keys as needed, e.g. {"qr_100": "#E31A1C"}.
    "quantile_colors": {
        "qr_100": "#2166AC",   # steel blue
        "qr_75":  "#4DAC26",   # muted green
        "qr_50":  "#D6604D",   # muted coral
        "qr_25":  "#9970AB",   # muted purple
    },

    # ========== Layout and Metadata ==========
    # Custom footer text. If None, auto-generated from the date window.
    "meta_text": None,
}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="AlphaMark benchmarking pipeline")
    ap.add_argument(
        "--plot-only", action="store_true",
        help=(
            "Skip all pipeline computation and regenerate the PDF report "
            "directly from existing PKLs in output/DAILY_STATS, "
            "output/SUMMARY_STATS, and output/OUTLIERS."
        ),
    )
    args = ap.parse_args()

    # -----------------------------------------------------------------
    # 1) Build centralized configs for runner + plotting
    # -----------------------------------------------------------------
    runner_cfg = dict(DEFAULT_RUNNER_CONFIG)
    plot_cfg   = dict(DEFAULT_PLOT_CONFIG)

    # ---- Optional JSON overrides ----
    cfg_path = _env_or_none("FP_CONFIG")
    if cfg_path:
        runner_cfg.update(_load_json(cfg_path))
        plot_cfg.update(_load_json(cfg_path))

    plot_cfg_path = _env_or_none("FP_PLOT_CONFIG")
    if plot_cfg_path:
        plot_cfg.update(_load_json(plot_cfg_path))

    # ---- Env overrides (interval, I/O) ----
    env_start = _env_or_none("FP_INTERVAL_START")
    env_end   = _env_or_none("FP_INTERVAL_END")
    if env_start is not None:
        runner_cfg["interval_start"] = env_start
    if env_end is not None:
        runner_cfg["interval_end"] = env_end

    env_features_dir = _env_or_none("FP_FEATURES_DIR")
    if env_features_dir is not None:
        for k in ("signals_input", "targets_input", "betsizes_input"):
            runner_cfg[k] = {"dir": env_features_dir, "glob": runner_cfg[k].get("glob")}

    for env_key, cfg_key in [
        ("FP_SIGNALS_DIR",  "signals_input"),
        ("FP_TARGETS_DIR",  "targets_input"),
        ("FP_BETSIZES_DIR", "betsizes_input"),
    ]:
        v = _env_or_none(env_key)
        if v is not None:
            runner_cfg[cfg_key] = {"dir": v, "glob": runner_cfg[cfg_key].get("glob")}

    env_output_root = _env_or_none("FP_OUTPUT_ROOT")
    if env_output_root is not None:
        runner_cfg["output_root"] = env_output_root

    # ---- Optional env overrides for temporal plot grid ----
    env_temp_grid = _env_or_none("FP_TEMPORAL_GRID")
    parsed_grid   = _parse_int_tuple(env_temp_grid)
    if parsed_grid:
        plot_cfg["arrayDim_temporal_plot"] = parsed_grid

    # ---- Optional env overrides for H2/H3 filters ----
    env_h2_targets = _parse_list(_env_or_none("FP_H2_TARGETS"))
    env_h2_bets    = _parse_list(_env_or_none("FP_H2_BETS"))
    env_h3_targets = _parse_list(_env_or_none("FP_H3_TARGETS"))
    env_h3_bets    = _parse_list(_env_or_none("FP_H3_BETS"))
    if env_h2_targets is not None: plot_cfg["H2_targets"] = env_h2_targets
    if env_h2_bets    is not None: plot_cfg["H2_bets"]    = env_h2_bets
    if env_h3_targets is not None: plot_cfg["H3_targets"] = env_h3_targets
    if env_h3_bets    is not None: plot_cfg["H3_bets"]    = env_h3_bets

    # ---- Apply explicit list overrides (list → exact-match regex) ----
    for list_key, regex_key in [
        ("signal_list", "signal_regex"),
        ("target_list", "target_regex"),
        ("bet_list",    "bet_regex"),
    ]:
        lst = runner_cfg.get(list_key)
        if lst:
            runner_cfg[regex_key] = "^(" + "|".join(lst) + ")$"

    # ---- Keep plotting interval in sync with runner ----
    plot_cfg["interval_start"] = runner_cfg.get("interval_start")
    plot_cfg["interval_end"]   = runner_cfg.get("interval_end")

    # ---- Keep CCF settings in sync ----
    plot_cfg["ccf_enable"]  = runner_cfg.get("ccf_enable", False)
    plot_cfg["ccf_max_lag"] = runner_cfg.get("ccf_max_lag", 5)

    output_root = runner_cfg["output_root"]

    # -----------------------------------------------------------------
    # 2) Run pipeline OR use existing PKLs (--plot-only)
    # -----------------------------------------------------------------
    if args.plot_only:
        # Derive all dirs from output_root — no computation at all.
        print("[INFO] --plot-only: skipping pipeline, reading existing PKLs.")
        daily_dir       = os.path.join(output_root, "DAILY_STATS")
        summary_dir     = os.path.join(output_root, "SUMMARY_STATS")
        outliers_dir    = os.path.join(output_root, "OUTLIERS")
        market_dist_dir = os.path.join(output_root, "MDS_STATS")

        for d in (daily_dir, summary_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(
                    f"[--plot-only] Expected directory not found: {d}\n"
                    f"Run without --plot-only first to generate the PKLs."
                )
    else:
        result = run_pipeline(runner_cfg)

        daily_dir       = result.get('daily_dir')
        summary_dir     = result.get('summary_dir')
        summary_path    = result.get('summary_path')
        outliers_dir    = result.get('outliers_dir')
        market_dist_dir = result.get('market_dist_dir') or result.get('per_ticker_dir')

        # ---- Build combined stats_df (backwards compatibility) ----
        if isinstance(result, dict) and daily_dir:
            daily_paths  = sorted(glob.glob(os.path.join(daily_dir, 'stats_*.pkl')))
            daily_frames = [read_pickle_compat(p) for p in daily_paths]
            stats_daily  = (
                pd.concat(daily_frames, ignore_index=True)
                if daily_frames else pd.DataFrame(columns=DEFAULT_COLS)
            )
            if summary_path and os.path.exists(summary_path):
                stats_summary = read_pickle_compat(summary_path)
            else:
                stats_summary = pd.DataFrame(columns=DEFAULT_COLS)
            parts    = [df for df in (stats_daily, stats_summary) if not df.empty]
            stats_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=DEFAULT_COLS)
        else:
            stats_df = pd.DataFrame(columns=DEFAULT_COLS)

        print(f"\nLoaded stats_df with shape: {stats_df.shape}")
        print("Columns:", stats_df.columns.tolist())
        print("\nPreview of stats_df:")
        print(stats_df.head(10))

        # ---- Backwards-compatible outputs ----
        compat_dir = os.path.join(output_root, "DAILY_SUMMARIES")
        os.makedirs(compat_dir, exist_ok=True)
        compat_pkl = os.path.join(compat_dir, "stats_tensor.pkl")
        compat_csv = os.path.join(compat_dir, "stats_tensor.csv")
        stats_df.to_pickle(compat_pkl)
        stats_df.to_csv(compat_csv, index=False)
        with open(compat_pkl, "rb") as f:
            obj = pkl.load(f)
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(compat_csv, index=False)

    # -----------------------------------------------------------------
    # 3) Generate PDF report (always runs)
    # -----------------------------------------------------------------
    plot_cfg["daily_dir"]      = daily_dir
    plot_cfg["summary_dir"]    = summary_dir
    plot_cfg["per_ticker_dir"] = market_dist_dir
    plot_cfg["outliers_dir"]   = outliers_dir
    plot_cfg["output_pdf"]     = os.path.join(output_root, "Quantile_Combined_Report.pdf")

    generate_quantile_report(plot_cfg)

    end = time.perf_counter()
    print(f"\nTotal time: {end - start:.3f} seconds")