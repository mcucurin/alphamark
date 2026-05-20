# main.py
from pipeline.runner import run_pipeline
from plotting.plot_quantile_bars import generate_quantile_report

import argparse
import pickle as pkl
import pandas as pd
import os, glob, json, re
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
    # Each entry: {"dir": "/path/to/pkl_files", "glob": "pattern_*.pkl"}
    # All three can point to the same directory if signals, targets, and bet sizes
    # are stored together in a single set of daily PKL files.
    "signals_input":  {"dir": "input/DAILY_FEATURES_PKL", "glob": "features_*.pkl"},
    "targets_input":  {"dir": "input/DAILY_FEATURES_PKL", "glob": "features_*.pkl"},
    "betsizes_input": {"dir": "input/DAILY_FEATURES_PKL", "glob": "features_*.pkl"},

    # Root directory where DAILY_STATS/, SUMMARY_STATS/, OUTLIERS/ are written.
    "output_root": "output",

    # ========== Column Discovery ==========
    # For each column type, provide either a regex pattern OR an explicit list of
    # column names — the list takes precedence when non-empty.
    #   signal_regex: matches columns that represent alpha signals (e.g. ^pret_ for
    #                 predicted returns).
    #   target_regex: matches columns that represent forward returns / targets.
    #   bet_regex:    matches columns that represent position / bet sizes.
    # Regex tip: use ^(col_a|col_b)$ to match an exact set of names via regex.
    "signal_regex": "^pret_",   "signal_list": None,
    "target_regex": "^fret_",   "target_list": None,
    "bet_regex":    "^betsize_","bet_list":    None,

    # ========== Market Proxy ==========
    # Ticker of the market proxy instrument present in the daily PKL files (e.g. SPY).
    # Used to compute market_corr (Spearman ρ between daily portfolio PnL and this
    # proxy return) shown in bar plots, and as the benchmark in CCF analysis.
    # Set to None or "" to skip both market correlation and CCF.
    "spy_ticker": "SPY",

    # ========== Quantile Portfolios ==========
    # List of quantile thresholds (fractions). Each value q builds a portfolio from
    # the top-q fraction of instruments ranked by |signal| each day.
    #   type_quantile options:
    #     "cumulative" — top-K cumulative portfolios (each q is a separate portfolio)
    #     "quantEach"  — exclusive bands (each q is a mutually exclusive bucket)
    "quantiles": [1.0, 0.75, 0.5, 0.25],
    "type_quantile": "cumulative",

    # ========== Correlation Diagnostics (slow) ==========
    # Cross-sectional Spearman ρ and distance correlation between signal and target
    # across instruments, computed per (signal, target, quantile) combination.
    # Results appear in bar plots only. Both are computationally expensive.
    #   spearman_sample_cap_per_key: max rows sampled per key when computing
    #   Spearman ρ — lower values run faster at the cost of precision.
    "add_spearman": False,
    "add_dcor":     False,
    "spearman_sample_cap_per_key": 10000,

    # ========== CCF vs Market Proxy (slow) ==========
    # Cross-correlation of daily portfolio PnL against the market proxy return,
    # computed at lags 0..ccf_max_lag (trading days).
    # Requires spy_ticker to be set. Results appear as a dedicated PDF section.
    #   ccf_dump_per_ticker: also saves each ticker's CCF series to disk for
    #   inspection outside the report.
    "ccf_enable": False,
    "ccf_max_lag": 5,
    "ccf_dump_per_ticker": False,

    # ========== Outlier Detection ==========
    # Metrics to flag as outliers using global z-scores across the date range.
    # Available: pnl, ppd, size_notional, nr_trades, long_ratio, hit_ratio
    "outlier_metrics": ["pnl", "ppd", "size_notional", "nr_trades", "long_ratio"],

    # ========== Parallelism ==========
    # n_jobs_io:      threads used when loading PKL files from disk
    # n_jobs_daily:   workers for per-day stat computation
    # n_jobs_summary: workers for summary stat computation across the full period
    "n_jobs_io": 1,
    "n_jobs_daily": 3,
    "n_jobs_summary": 3,

    # ========== Reproducibility ==========
    # Seed for Spearman sampling — set to any integer for deterministic results.
    "random_state": 123,

    # ========== Date Range (inclusive) ==========
    "interval_start": "2000-01-01",
    "interval_end":   "2021-12-31",
}

# ---- Plotting / report config ----
DEFAULT_PLOT_CONFIG = {
    # ========== Quantile Display ==========
    # Which quantile portfolios to include in all plots. Must match the quantile
    # thresholds in DEFAULT_RUNNER_CONFIG["quantiles"] (formatted as qr_<pct>).
    "qranks": ["qr_100", "qr_75", "qr_50", "qr_25"],

    # ========== Heatmap Target / Bet Filters (H2 and H3 pages) ==========
    # "AUTO" selects the most common targets/bets automatically.
    # Provide an explicit list to pin specific values, e.g. ["fret_1d", "fret_5d"].
    "H2_targets": "AUTO",
    "H2_bets":    "AUTO",
    "H3_targets": "AUTO",
    "H3_bets":    "AUTO",

    # ========== Heatmap Line Smoothing (rolling mean, days) ==========
    # Applied to the time-series lines overlaid on H1/H2/H3 heatmap pages.
    # Set to 1 to disable smoothing.
    "roll_h1_lines": 30,
    "roll_h2_lines": 30,
    "roll_h3_lines": 1,

    # ========== Temporal Panel Rolling Windows (days) ==========
    # Rolling mean applied to each metric before plotting on temporal pages.
    # Set to 1 for raw daily values (no smoothing).
    # sharpe_ratio: minimum 21 days (one trading month) — values below this are
    #               silently raised because std(ddof=1) on fewer observations is
    #               statistically meaningless for Sharpe.
    # pnl and ppd are always plotted as cumulative sums — no rolling option.
    "roll_nrinstr":       1,
    "roll_trades":        1,
    "roll_size_notional": 1,
    "roll_sharpe":        21,
    "roll_hit_ratio":     1,

    # ========== Temporal Plot Layout ==========
    # variables_temporal_plot: metrics shown as time-series panels. Each variable
    #   fills one cell of the grid left-to-right, top-to-bottom.
    #   Available: pnl, ppd, nr_trades, size_notional, sharpe_ratio, hit_ratio, nr_instr
    # arrayDim_temporal_plot: (rows, cols) grid dimensions per page.
    "variables_temporal_plot": ["pnl", "ppd", "nr_trades", "size_notional"],
    "arrayDim_temporal_plot":  (2, 2),

    # ========== Bar Plot Configuration (summary stats only) ==========
    # bar_page_vars: one PDF page is generated per unique combination of these
    #   dimensions. E.g. ["target", "bet_size_col"] → one page per target×bet pair.
    # bar_x_vars: dimension placed on the x-axis within each bar chart.
    #   A variable cannot appear in both bar_page_vars and bar_x_vars.
    # bar_metrics: metrics rendered as bar chart panels, left-to-right order.
    #   spearman / dcor only appear when add_spearman / add_dcor are enabled.
    # aspect_ratio_barplots: width/height ratio for each bar panel (16/9 ≈ 1.778).
    "bar_page_vars":        ["target", "bet_size_col"],
    "bar_x_vars":           ["signal"],
    "bar_metrics": [
        "pnl", "ppd", "sharpe_ratio", "hit_ratio", "long_ratio",
        "size_notional", "r2", "t_stat", "nr_trades", "market_corr"
    ],
    "aspect_ratio_barplots": 16 / 9,

    # ========== Outlier Tables ==========
    # outlier_metrics_for_tables: subset of outlier_metrics to show in PDF tables.
    # outlier_top_k: number of extreme high and low rows shown per metric per table.
    # outlier_tables_per_page: tables stacked per PDF page; partial pages scale down
    #   proportionally so each table keeps the same height across all pages.
    "outlier_metrics_for_tables": ["pnl", "ppd", "size_notional", "nr_trades"],
    "outlier_top_k":           3,
    "outlier_tables_per_page": 2,

    # ========== Line Style ==========
    # Matplotlib linestyle for temporal plot lines: "-" solid, "--" dashed,
    # "-." dash-dot, ":" dotted.
    "line_style":  "-",

    # ========== Quantile Color Palette ==========
    # One color per quantile portfolio. Keys must match qranks above.
    # Any valid matplotlib color string is accepted.
    "quantile_colors": {
        "qr_100": "#2166AC",   # steel blue
        "qr_75":  "#4DAC26",   # muted green
        "qr_50":  "#D6604D",   # muted coral
        "qr_25":  "#9970AB",   # muted purple
    },

    # ========== Report Footer ==========
    # Custom text shown bottom-right of every PDF page.
    # None = auto-generated from the date window (interval_start – interval_end).
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

    # ---- Keep plotting interval in sync with runner ----
    plot_cfg["interval_start"] = runner_cfg.get("interval_start")
    plot_cfg["interval_end"]   = runner_cfg.get("interval_end")

    # ---- Keep CCF settings in sync ----
    plot_cfg["ccf_enable"]  = runner_cfg.get("ccf_enable", False)
    plot_cfg["ccf_max_lag"] = runner_cfg.get("ccf_max_lag", 5)

    output_root = runner_cfg["output_root"]

    # ---- If explicit list provided, convert to exact-match regex ----
    for _lk, _rk in [("signal_list","signal_regex"),("target_list","target_regex"),("bet_list","bet_regex")]:
        _lst = runner_cfg.get(_lk)
        if _lst:
            runner_cfg[_rk] = "^(" + "|".join(re.escape(c) for c in _lst) + ")$"

    # ---- Strip correlation metrics from bar_metrics when not computed ----
    _bm = plot_cfg.get("bar_metrics", [])
    if not runner_cfg.get("add_spearman"): _bm = [m for m in _bm if m != "spearman"]
    if not runner_cfg.get("add_dcor"):     _bm = [m for m in _bm if m != "dcor"]
    plot_cfg["bar_metrics"] = _bm



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