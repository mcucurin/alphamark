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

# ---- Runner / pipeline config (moved from runner.DEFAULT_CONFIG) ----
DEFAULT_RUNNER_CONFIG = {
    # ========== I/O Configuration ==========
    # Input directories.  In the simple case all three point to the same folder
    # containing combined daily PKL files (one per day, each with signal + target +
    # betsize columns).  Set them to different paths when each category lives in a
    # separate file tree; the pipeline merges on (date, ticker) automatically.
    #
    # Each entry is {"dir": <path>, "glob": <pattern>}.
    # "glob" can be None to fall back to "*.pkl".
    "signals_input":  {"dir": "input/DAILY_FEATURES_PKL", "glob": "features_*.pkl"},
    "targets_input":  {"dir": "input/DAILY_FEATURES_PKL", "glob": "features_*.pkl"},
    "betsizes_input": {"dir": "input/DAILY_FEATURES_PKL", "glob": "features_*.pkl"},

    # Root directory for all pipeline outputs
    "output_root": "output",

    # ========== Column Discovery ==========
    # Each column category is identified by prefix OR regex.
    # When regex is set (not None), it takes precedence over prefix.
    #   prefix:  simple startswith match  (e.g., "pret_" matches "pret_signal1")
    #   regex:   full regex search        (e.g., r"pret_signal.*")
    "signal_prefix": "pret_",       "signal_regex": None,
    "target_prefix": "fret_",       "target_regex": None,
    "bet_prefix":    "betsize_",    "bet_regex":    None,

    # ========== Market Proxy (SPY) ==========
    "spy_ticker":      "SPY",       # ticker used as market proxy
    "spy_col_base":    "spy",       # internal column naming: f"{spy_col_base}__{target}"
    "spy_single_name": "spy_ret",   # fallback single-column name

    # ========== Quantile Configuration ==========
    "quantiles": [1.0, 0.75, 0.5, 0.25],
    "type_quantile": "cumulative",  # "cumulative" (top-K) or "quantEach" (bucket)

    # ========== Pipeline Stage Toggles ==========
    "do_daily": True,
    "do_summary": True,
    "do_outliers": True,

    # ========== Summary Statistics Extras ==========
    "add_spearman": False,          # Spearman rank corr (expensive)
    "add_dcor": False,              # Distance correlation (expensive)
    "spearman_sample_cap_per_key": 10000,

    # ========== CCF (Cross-Correlation vs Market Proxy) ==========
    # When enabled, computes per-ticker cross-correlation between signal/PnL
    # and SPY at lags in [-ccf_max_lag, +ccf_max_lag].  Expensive.
    "ccf_enable": False,
    "ccf_max_lag": 5,
    # If True AND ccf_enable=True, dump per-ticker CCF detail PKLs to MDS_STATS/
    "ccf_dump_per_ticker": False,

    # ========== Outlier Detection ==========
    "outlier_metrics": ["pnl", "ppd", "sizeNotional", "n_trades"],

    # ========== Daily Processing Behavior ==========
    "empty_day_policy": "carry",          # "carry" | "close" | "skip"
    "report_empty_trades_as_nan": True,

    # ========== Parallelism Configuration ==========
    # Number of parallel jobs for I/O operations (loading feature files)
    "n_jobs_io": 1,
    
    # Number of parallel jobs for daily statistics computation
    "n_jobs_daily": 3,
    
    # Number of parallel jobs for summary statistics computation
    "n_jobs_summary": 3,

    # ========== Reproducibility ==========
    # Random seed for any random operations (e.g., sampling for Spearman correlation)
    "random_state": 123,

    # ========== Date Range Filter (Inclusive) ==========
    # Start date for filtering input data (inclusive). Accepts multiple formats:
    #   "2021-01-01" (ISO format)
    #   "01/01/2021" (US format)
    #   "20210101" (compact format)
    # This date range is applied during pipeline processing AND plotting
    "interval_start": "2019-01-01",
    
    # End date for filtering input data (inclusive). Same format options as interval_start
    # This date range is applied during pipeline processing AND plotting
    "interval_end": "2020-01-01"
}

# ---- Plotting / report config (moved from plot_quantile_bars.CONFIG) ----
DEFAULT_PLOT_CONFIG = {
    # ========== Quantile Display Configuration ==========
    # List of quantile ranks to display in plots (maximum ~4 quantiles recommended for readability)
    # Quantile ranks correspond to the quantiles defined in DEFAULT_RUNNER_CONFIG
    # Example: ["qr_100", "qr_75", "qr_50", "qr_25"] displays all 4 quantile levels
    "qranks": ["qr_100", "qr_75", "qr_50", "qr_25"],
    
    # If True, allow plotting even if some requested qranks are missing from data
    # If False, missing qranks will cause warnings and be ignored
    "allow_missing_qranks": False,

    # ========== Heatmap Filter Configuration (H2/H3) ==========
    # Target columns to use for H2 heatmap (per-quantile PnL daily cross-section correlation)
    # Set to "AUTO" to automatically pick common values from DAILY data (preferring prefixes)
    # Or provide explicit list: ["fret_1_MR", "fret_5_MR"] to use specific targets
    "H2_targets": ["fret_1_MR"],
    
    # Bet size columns to use for H2 heatmap
    # Set to "AUTO" for automatic selection, or provide explicit list
    "H2_bets": ["betsize_cap250k"],
    
    # Target columns to use for H3 heatmap (per-quantile time-series correlation of summed daily PnL)
    # Set to "AUTO" for automatic selection, or provide explicit list
    "H3_targets": ["fret_1_MR"],
    
    # Bet size columns to use for H3 heatmap
    # Set to "AUTO" for automatic selection, or provide explicit list
    "H3_bets": ["betsize_cap250k"],

    # ========== Temporal Line Smoothing Windows ==========
    # Rolling window (in trading days) for smoothing H1 temporal line plots
    # H1 shows average daily cross-section correlation across alphas
    "roll_h1_lines": 30,
    
    # Rolling window (in trading days) for smoothing H2 temporal line plots
    # H2 shows per-quantile PnL daily cross-section correlation
    "roll_h2_lines": 30,
    
    # Rolling window (in trading days) for smoothing H3 temporal line plots
    # H3 shows per-quantile time-series correlation of summed daily PnL vectors
    # Note: Setting to 1 creates an "expanding" style (cumulative from start)
    "roll_h3_lines": 1,

    # ========== Rolling Windows for Temporal Panels ==========
    # Rolling window (in trading days) for number of instruments metric
    # Set to 1 for no smoothing (raw daily values)
    "roll_nrinstr": 1,
    
    # Rolling window (in trading days) for profit per dollar (PPD) metric
    "roll_ppd": 1,
    
    # Rolling window (in trading days) for number of trades metric
    "roll_trades": 1,
    
    # Rolling window (in trading days) for profit and loss (PnL) metric
    "roll_pnl": 1,
    
    # Rolling window (in trading days) for size notional metric
    "roll_size_notional": 1,
    
    # Rolling window (in trading days) for Sharpe ratio metric
    # Used for optional rolling Sharpe display on temporal plots
    "roll_sharpe": 60,

    # ========== Temporal Plot Configuration ==========
    # List of metrics to display in temporal plots (per target/signal/bet combination)
    # Available metrics: "pnl", "ppd", "nrTrades", "sizeNotional", "sharpe", etc.
    # Add or remove metrics from this list to customize temporal plot pages
    "variables_temporal_plot": ["pnl", "ppd", "nrTrades", "sizeNotional"],
    
    # Grid dimensions for temporal plot pages: (rows, columns) per page
    # Example: (2, 2) creates a 2x2 grid with 4 subplots per page
    "arrayDim_temporal_plot": (2, 2),

    # ========== Bar Plot Configuration (SUMMARY Data Only) ==========
    # Faceting variables for bar plots (creates separate pages/facets for each combination)
    # Example: ["signal", "bet_size_col"] creates one page per signal/bet combination
    "bar_page_vars": ["signal", "bet_size_col"],
    
    # X-axis grouping variable for bar plots
    # Example: ["target"] groups bars by target column
    "bar_x_vars": ["target"],
    
    # List of metrics to display in bar plots
    # Available metrics: "pnl", "ppd", "sharpe", "hit_ratio", "long_ratio", 
    #                    "sizeNotional", "r2", "t_stat", "n_trades", "market_corr"
    "bar_metrics": [
        "pnl", "ppd", "sharpe", "hit_ratio", "long_ratio",
        "sizeNotional", "r2", "t_stat", "n_trades", "market_corr"
    ],
    
    # Aspect ratio (width/height) for bar plot pages
    # 16/9 creates widescreen layout suitable for presentations
    "aspect_ratio_barplots": 16 / 9,

    # ========== Outlier Table Configuration ==========
    # List of metrics to analyze for outliers in compact table format
    # These tables show top/bottom K extreme values per metric
    "outlier_metrics_for_tables": ["pnl", "ppd", "sizeNotional", "n_trades"],
    
    # Number of top and bottom outliers to display per metric (K)
    # Example: outlier_top_k=3 shows top 3 and bottom 3 outliers
    "outlier_top_k": 3,
    
    # Number of outlier tables to display per PDF page
    # Recommended: 2-3 tables per page for compact, readable layout
    "outlier_tables_per_page": 2,

    # ========== Plot Styling ==========
    # Line style for primary/primary quantile lines (matplotlib format)
    # "-" = solid line, "--" = dashed, "-." = dash-dot, ":" = dotted
    "style_first": "-",
    
    # Line style for secondary quantile lines
    "style_second": ":",
    
    # Color mapping for quantile ranks (matplotlib color names or hex codes)
    # Maps quantile rank labels to colors for consistent visualization
    "quantile_colors": {"qr_100": "red", "qr_75": "green", "qr_50": "blue", "qr_25": "black"},

    # ========== Layout and Metadata ==========
    # Custom text to display in top-right corner of every PDF page
    # If None, plotting module will auto-generate text showing date window and number of days
    "meta_text": None
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
        plot_cfg.update(_load_json(cfg_path))

    # ---- Optional JSON overrides for plotting only ----
    plot_cfg_path = _env_or_none("FP_PLOT_CONFIG")
    if plot_cfg_path:
        plot_cfg.update(_load_json(plot_cfg_path))

    # ---- Env overrides (interval, I/O) ----
    env_start = _env_or_none("FP_INTERVAL_START")
    env_end = _env_or_none("FP_INTERVAL_END")
    if env_start is not None:
        runner_cfg["interval_start"] = env_start
    if env_end is not None:
        runner_cfg["interval_end"] = env_end

    env_features_dir = _env_or_none("FP_FEATURES_DIR")
    if env_features_dir is not None:
        # Convenience: set all three input dirs to the same path
        for k in ("signals_input", "targets_input", "betsizes_input"):
            runner_cfg[k] = {"dir": env_features_dir, "glob": runner_cfg[k].get("glob")}

    # Per-category env overrides (take precedence over FP_FEATURES_DIR)
    for env_key, cfg_key in [
        ("FP_SIGNALS_DIR", "signals_input"),
        ("FP_TARGETS_DIR", "targets_input"),
        ("FP_BETSIZES_DIR", "betsizes_input"),
    ]:
        v = _env_or_none(env_key)
        if v is not None:
            runner_cfg[cfg_key] = {"dir": v, "glob": runner_cfg[cfg_key].get("glob")}

    env_output_root = _env_or_none("FP_OUTPUT_ROOT")
    if env_output_root is not None:
        runner_cfg["output_root"] = env_output_root

    # ---- Optional env override for temporal plot grid (rows, cols per page) ----
    env_temp_grid = _env_or_none("FP_TEMPORAL_GRID")
    parsed_grid = _parse_int_tuple(env_temp_grid)
    if parsed_grid:
        plot_cfg["arrayDim_temporal_plot"] = parsed_grid

    # ---- Optional env overrides for H2/H3 filters (targets/bets) ----
    env_h2_targets = _parse_list(_env_or_none("FP_H2_TARGETS"))
    env_h2_bets = _parse_list(_env_or_none("FP_H2_BETS"))
    env_h3_targets = _parse_list(_env_or_none("FP_H3_TARGETS"))
    env_h3_bets = _parse_list(_env_or_none("FP_H3_BETS"))
    if env_h2_targets is not None:
        plot_cfg["H2_targets"] = env_h2_targets
    if env_h2_bets is not None:
        plot_cfg["H2_bets"] = env_h2_bets
    if env_h3_targets is not None:
        plot_cfg["H3_targets"] = env_h3_targets
    if env_h3_bets is not None:
        plot_cfg["H3_bets"] = env_h3_bets

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
    print(f"\nTotal time (pipeline + report): {end - start:.3f} seconds")