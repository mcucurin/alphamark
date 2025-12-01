# main.py
from pipeline.runner import run_pipeline
from plotting.plot_quantile_bars import generate_quantile_report

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


# =====================================================================
#                        CENTRALIZED CONFIG
# =====================================================================

# ---- Runner / pipeline config (moved from runner.DEFAULT_CONFIG) ----
DEFAULT_RUNNER_CONFIG = {
    # -------- I/O --------
    "features_input_dir": "input/DAILY_FEATURES_PKL",
    "features_glob": "features_*.pkl",
    "output_root": "output",

    # -------- Column selection --------
    "signal_prefix": "pret_",
    "target_prefix": "fret_",
    "bet_prefix": "betsize_",
    "signal_regex": None,
    "target_regex": None,
    "bet_regex": None,

    # -------- Market proxy (per-target SPY columns) --------
    "spy_ticker": "SPY",
    "spy_col_base": "spy",              # per-target columns are f"{spy_col_base}__{target}"
    "spy_single_name": "spy_ret",

    # -------- Quantiles --------
    "quantiles": [1.0, 0.75, 0.5, 0.25],
    "type_quantile": "cumulative",      # "cumulative" | "quantEach"

    # -------- Toggles --------
    "do_daily": True,
    "do_summary": True,
    "do_outliers": True,

    # -------- Summary extras --------
    "add_spearman": False,
    "add_dcor": False,
    "spearman_sample_cap_per_key": 10000,

    # -------- Per-ID (ticker) corr dump toggles --------
    "dump_alpha_raw_per_id": False,
    "dump_alpha_pnl_per_id": False,

    # -------- Per-ID CCF (alpha vs SPY) dump toggles --------
    # These are consumed inside the runner when calling compute_summary_stats_over_days
    # and will generate per_ticker_alpha_*_spy_ccf_*.pkl in per_ticker_dir.
    "ccf_enable": False,
    "ccf_max_lag": 5,
    "dump_alpha_raw_ccf_per_id": False,
    "dump_alpha_pnl_ccf_per_id": False,

    # -------- Outliers --------
    "outlier_metrics": ["pnl", "ppd", "sizeNotional", "nrInstr", "n_trades"],
    "outlier_z_thresh": None,  # ignored; outliers are ranked by |z| with no threshold

    # -------- Daily behavior --------
    "empty_day_policy": "carry",        # "carry" | "close" | "skip"
    "report_empty_trades_as_nan": True,

    # -------- Parallelism --------
    "n_jobs_io": 1,
    "n_jobs_daily": 3,
    "n_jobs_summary": 3,

    # -------- Reproducibility --------
    "random_state": 123,

    # -------- Interval filter (inclusive) --------
    # Accepts many formats: "2021-01-01", "01/01/2021", "20210101"
    "interval_start": None,
    "interval_end": None,
}

# ---- Plotting / report config (moved from plot_quantile_bars.CONFIG) ----
DEFAULT_PLOT_CONFIG = {
    # Quantiles to display (max ~4 looks best)
    "qranks": ["qr_100", "qr_75", "qr_50", "qr_25"],
    "allow_missing_qranks": False,

    # Heatmap filters for H2/H3:
    #  "AUTO" picks a common value from DAILY (preferring prefixes); otherwise supply explicit lists.
    "H2_targets": "AUTO",
    "H2_bets": "AUTO",
    "H3_targets": "AUTO",
    "H3_bets": "AUTO",

    # Smoothing windows (trading days) for H1/H2/H3 temporal line pages
    "roll_h1_lines": 30,
    "roll_h2_lines": 30,
    "roll_h3_lines": 1,  # 1 => "expanding" style in code below

    # Rolling windows for temporal panels
    "roll_nrinstr": 1,
    "roll_ppd": 1,
    "roll_trades": 1,
    "roll_pnl": 1,
    "roll_size_notional": 1,
    "roll_sharpe": 60,  # used for optional rolling Sharpe on temporal plots

    # Temporal plots (per target/signal/bet)
    "variables_temporal_plot": ["pnl", "ppd", "nrTrades", "sizeNotional"],  # default set
    "variables_temporal_extras": [],  # e.g., ["hit_ratio", "long_ratio", "r2", "t_stat"]
    "arrayDim_temporal_plot": (2, 2),  # rows, cols per temporal page

    # Bar plots (SUMMARY ONLY)
    "bar_page_vars": ["signal", "bet_size_col"],   # facets
    "bar_x_vars": ["target"],                      # x-axis grouping
    "bar_metrics": [
        "pnl", "ppd", "sharpe", "hit_ratio", "long_ratio",
        "sizeNotional", "r2", "t_stat", "n_trades", "market_corr"
    ],
    "aspect_ratio_barplots": 16 / 9,  # widescreen layout for bar pages

    # Outliers (compact)
    # (Directory will be filled in from pipeline result; only behavior lives here.)
    "outlier_metrics_for_tables": ["pnl", "ppd", "sizeNotional", "nrInstr", "n_trades"],
    "outlier_top_k": 3,
    "outlier_tables_per_page": 3,  # compact 2-3 per page works well

    # Styles (secondary is dotted)
    "style_first": "-",     # solid
    "style_second": ":",    # dotted
    "quantile_colors": {"qr_100": "red", "qr_75": "green", "qr_50": "blue",  "qr_25": "black"},

    # CCF (cross-correlation vs SPY) plots using per-ticker dumps in per_ticker_dir
    # MAX_LAG = ccf_max_lag; we use lags in [-MAX_LAG, ..., +MAX_LAG]
    "ccf_enable": True,
    "ccf_max_lag": 5,

    # Titles / layout
    "meta_text": None,      # printed top-right on every page; if None, plotting will auto-fill
}


if __name__ == '__main__':
    # -----------------------------------------------------------------
    # 1) Build centralized configs for runner + plotting
    # -----------------------------------------------------------------
    runner_cfg = dict(DEFAULT_RUNNER_CONFIG)
    plot_cfg = dict(DEFAULT_PLOT_CONFIG)

    # ---- Optional JSON overrides for runner (same behavior as before) ----
    cfg_path = _env_or_none("FP_CONFIG")
    if cfg_path:
        runner_cfg.update(_load_json(cfg_path))

    # ---- Env overrides (interval, I/O) ----
    env_start = _env_or_none("FP_INTERVAL_START")
    env_end = _env_or_none("FP_INTERVAL_END")
    if env_start is not None:
        runner_cfg["interval_start"] = env_start
    if env_end is not None:
        runner_cfg["interval_end"] = env_end

    env_features_dir = _env_or_none("FP_FEATURES_DIR")
    if env_features_dir is not None:
        runner_cfg["features_input_dir"] = env_features_dir

    env_output_root = _env_or_none("FP_OUTPUT_ROOT")
    if env_output_root is not None:
        runner_cfg["output_root"] = env_output_root

    # ---- Keep plotting interval in sync with runner ----
    plot_cfg["interval_start"] = runner_cfg.get("interval_start")
    plot_cfg["interval_end"] = runner_cfg.get("interval_end")

    # ---- Keep CCF settings in sync between runner & plotting ----
    plot_cfg["ccf_enable"] = runner_cfg.get("ccf_enable", True)
    plot_cfg["ccf_max_lag"] = runner_cfg.get("ccf_max_lag", 5)

    # -----------------------------------------------------------------
    # 2) Run pipeline with centralized config
    # -----------------------------------------------------------------
    result = run_pipeline(runner_cfg)

    daily_dir = result.get('daily_dir')
    summary_path = result.get('summary_path')
    summary_dir = result.get('summary_dir')
    outliers_dir = result.get('outliers_dir')
    market_dist_dir = result.get('market_dist_dir') or result.get('per_ticker_dir')

    # -----------------------------------------------------------------
    # 3) Build a combined stats_df for backward compatibility
    # -----------------------------------------------------------------
    if isinstance(result, dict) and daily_dir:
        # Gather all daily frames
        daily_paths = sorted(glob.glob(os.path.join(daily_dir, 'stats_*.pkl')))
        daily_frames = [read_pickle_compat(p) for p in daily_paths]
        stats_daily = (
            pd.concat(daily_frames, ignore_index=True)
            if daily_frames else pd.DataFrame(columns=DEFAULT_COLS)
        )

        # Read summary (optional)
        if summary_path and os.path.exists(summary_path):
            stats_summary = read_pickle_compat(summary_path)
        else:
            stats_summary = pd.DataFrame(columns=DEFAULT_COLS)

        # Combined DataFrame for convenience/backwards-compat
        parts = [df for df in (stats_daily, stats_summary) if not df.empty]
        stats_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=DEFAULT_COLS)
    else:
        # Very defensive fallback
        stats_df = pd.DataFrame(columns=DEFAULT_COLS)

    print(f"\n📦 Loaded stats_df with shape: {stats_df.shape}")
    print("📄 Columns:", stats_df.columns.tolist())
    print("\n🔍 Preview of stats_df:")
    print(stats_df.head(10))

    # --- Backwards-compatible outputs in DAILY_SUMMARIES ---
    compat_dir = "./output/DAILY_SUMMARIES"
    os.makedirs(compat_dir, exist_ok=True)

    # Save pickle
    compat_pkl = os.path.join(compat_dir, "stats_tensor.pkl")
    stats_df.to_pickle(compat_pkl)

    # Also save CSV
    compat_csv = os.path.join(compat_dir, "stats_tensor.csv")
    stats_df.to_csv(compat_csv, index=False)

    # Demonstrate reloading the pickle with the standard loader (kept as-is)
    with open(compat_pkl, "rb") as f:
        obj = pkl.load(f)
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(compat_csv, index=False)

    # -----------------------------------------------------------------
    # 4) Wire pipeline outputs into plotting config and generate PDF
    # -----------------------------------------------------------------
    output_root = runner_cfg["output_root"]
    plot_cfg["daily_dir"] = daily_dir
    plot_cfg["summary_dir"] = summary_dir
    # Provide per-ticker/MDS directory (plotting can choose to use or ignore)
    plot_cfg["per_ticker_dir"] = market_dist_dir
    plot_cfg["outliers_dir"] = outliers_dir
    plot_cfg["output_pdf"] = os.path.join(output_root, "Quantile_Combined_Report.pdf")

    # Let plot config auto-fill a nice meta_text if user didn't set one
    # (the plotting module will use the actual data window too).
    generate_quantile_report(plot_cfg)

    end = time.perf_counter()
    print(f"\n⏱️ Total time (pipeline + report): {end - start:.3f} seconds")
