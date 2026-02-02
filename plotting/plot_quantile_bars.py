# plotting/plot_quantile_bars.py
"""
Quantile Report PDF Generator (fixed H1 temporal coverage, restored distributions, compact outliers, CCF pages)

What this module does (when called via generate_quantile_report(config)):
- Loads precomputed DAILY stats (required) and SUMMARY stats (optional) from:
    <config['daily_dir']>/stats_YYYYMMDD.pkl
    <config['summary_dir']>/summary_stats_YYYYMMDD_YYYYMMDD.pkl
- Optionally loads per-ticker correlation / CCF dumps produced by the runner (if configured):
    <config['per_ticker_dir']>/mds_alpha_raw_spy_corr_*.pkl
    <config['per_ticker_dir']>/mds_alpha_pnl_spy_corr_*.pkl
    <config['per_ticker_dir']>/mds_alpha_raw_spy_ccf_*.pkl
    <config['per_ticker_dir']>/mds_alpha_pnl_spy_ccf_*.pkl
- Optionally loads outlier PKLs from:
    <config['outliers_dir']>/outliers_*.pkl
- Builds a multi-page PDF at: config['output_pdf']

Pages
1) Bar plots by quantile for selected metrics (from SUMMARY ONLY; skipped if no summary PKL).
2) H1: average daily cross-section correlation across alphas (Spearman), using base stat:
       alpha_sum > alpha_strength > pnl (no quantile/target/bet filter; uses all rows).
   H1 temporal lines (pairwise Spearman by day, optionally smoothed). No legend — line end labels.
3) H2: per-quantile PnL daily cross-section correlation (all targets, all bet sizes) — Spearman.
   H2 temporal lines for the same filter. No legend — line end labels.
4) H3: per-quantile time-series correlation of summed daily PnL vectors (Spearman, all targets / bets).
   H3 temporal lines (rolling/expanding time corr). No legend — line end labels.
5) Temporal pages per (target, signal, bet): configurable metric grid (default: pnl, ppd, n_trades, sizeNotional)
   with rolling/cumulative options and optional rolling Sharpe.
6) Distributions / CCF:
   - If CCF PKLs exist:
       - CCF summary page (mean/median/std vs lag) for RAW & PnL.
       - Histogram grids of CCF per lag (RAW vs SPY, PnL vs SPY).
     (MAX_LAG = config['ccf_max_lag'], default 5; lags in [-MAX_LAG, …, +MAX_LAG].)
   - Else:
       - Histograms (RAW↔SPY corr, PNL↔SPY corr) with mean/median/std annotations (if files exist).
7) Outlier tables (compact): top/bottom K for selected metrics (default: pnl, ppd, sizeNotional, nrInstr, n_trades).

Expected DAILY/SUMMARY columns:
  date (YYYY-MM-DD), signal, target, qrank (e.g., qr_100), bet_size_col,
  stat_type ('pnl','ppd','n_trades','nrInstr','sizeNotional','sharpe','market_corr',...),
  value (float)

Note
- All user-facing config comes from main.py and is passed in via `generate_quantile_report(config)`.
"""

import os
import glob
import warnings
import time
import pickle as pkl
from itertools import product, combinations
from contextlib import contextmanager
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore", category=UserWarning)

# --- Matplotlib theme (clean) ---
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "axes.grid": False,
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 18,
})
PAGE_SIZE = (14, 8.5)  # Standardize all PDF pages

# --- Layout guardrails ---
BAR_TITLE_Y     = 0.985
BAR_XLABEL_Y    = 0.962
BAR_AX_TOP      = 0.955
HEATMAP_AX_TOP  = 0.90
TEMPORAL_AX_TOP = 0.90

# Global meta text (set by generate_quantile_report)
META_TEXT = None

# Backward-compatible stat_type aliases
STAT_ALIASES = {
    "spy_corr": "market_corr",
    "mkt_corr": "market_corr",
    "nrTrades": "n_trades",
    "nr_trades": "n_trades",
    "ntrades": "n_trades",
}


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def _canonical_stat(stat: str) -> str:
    """Normalize stat_type tokens for plotting."""
    return STAT_ALIASES.get(stat, stat)


def _apply_stat_aliases(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Rename legacy stat_type labels (e.g., spy_corr -> market_corr)."""
    if df is None or not isinstance(df, pd.DataFrame):
        return df
    if "stat_type" not in df.columns:
        return df
    out = df.copy()
    out["stat_type"] = (
        out["stat_type"]
        .astype("string")
        .map(lambda x: STAT_ALIASES.get(x, x))
        .astype("category")
    )
    return out


def _normalize_metric_list(metrics) -> list[str]:
    """Deduplicate/alias a user-provided metric list."""
    out: list[str] = []
    for m in metrics or []:
        if m is None:
            continue
        name = _canonical_stat(str(m))
        if name not in out:
            out.append(name)
    return out


def _metric_label(metric: str) -> str:
    """Human-readable metric label; only PNL/PPD are fully uppercased."""
    name = _canonical_stat(str(metric))
    if name in ("pnl", "ppd"):
        return name.upper()
    # Split camelCase/snake_case into words, then title-case
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", name)
    spaced = spaced.replace("_", " ")
    return spaced.title()


def savefig_white(pdf, fig):
    """Save with white background; print META_TEXT if set (supplied via main.py)."""
    fig.set_size_inches(*PAGE_SIZE, forward=True)
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    if META_TEXT:
        fig.text(
            0.99,
            0.99,
            str(META_TEXT),
            ha="right",
            va="top",
            fontsize=10,
            color="0.35",
            weight="normal",
        )
    for ax in fig.get_axes():
        ax.set_facecolor("white")
    pdf.savefig(fig, facecolor="white", edgecolor="white")
    plt.close(fig)


def read_pickle_compat(path: str):
    """Unpickle objects across NumPy 1.x/2.x by remapping numpy._core -> numpy.core."""
    class NPCompatUnpickler(pkl.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core")
            return super().find_class(module, name)

    with open(path, "rb") as f:
        return NPCompatUnpickler(f).load()


def _parse_date_range(interval_start: str | None, interval_end: str | None):
    """
    Parse date range strings into normalized pd.Timestamps.
    
    Args:
        interval_start: Optional start date string (accepts formats like "2021-01-01", "01/01/2021", "20210101")
        interval_end: Optional end date string (accepts formats like "2021-01-01", "01/01/2021", "20210101")
    
    Returns:
        Tuple of (start_dt, end_dt) as pd.Timestamp objects, or (None, None) if parsing fails
    """
    def _parse_date(date_str):
        """Parse date string into normalized pd.Timestamp."""
        if date_str is None:
            return None
        try:
            dt = pd.to_datetime(date_str, errors="coerce")
            if pd.isna(dt):
                return None
            return pd.Timestamp(dt).normalize()
        except Exception:
            return None
    
    start_dt = _parse_date(interval_start)
    end_dt = _parse_date(interval_end)
    
    if start_dt is not None and end_dt is not None and end_dt < start_dt:
        # Swap if user reversed them
        start_dt, end_dt = end_dt, start_dt
    
    return start_dt, end_dt


def _filter_by_date_range(df: pd.DataFrame, start_dt: pd.Timestamp | None, end_dt: pd.Timestamp | None, data_type: str = "data"):
    """
    Filter DataFrame by date range.
    
    Args:
        df: DataFrame with a 'date' column
        start_dt: Optional start date (inclusive)
        end_dt: Optional end date (inclusive)
        data_type: Label for logging (e.g., "daily data", "summary data")
    
    Returns:
        Filtered DataFrame
    """
    if start_dt is not None:
        df = df[df["date"] >= start_dt]
        print(f"[INFO] Filtered {data_type}: start >= {start_dt:%Y-%m-%d}")
    
    if end_dt is not None:
        df = df[df["date"] <= end_dt]
        print(f"[INFO] Filtered {data_type}: end <= {end_dt:%Y-%m-%d}")
    
    return df


def _load_data(daily_dir: str, summary_dir: str, interval_start: str | None = None, interval_end: str | None = None):
    """
    Load DAILY and SUMMARY PKLs from the provided directories.
    
    Args:
        daily_dir: Directory containing daily stats PKL files
        summary_dir: Directory containing summary stats PKL files
        interval_start: Optional start date for filtering (inclusive). Accepts formats like "2021-01-01", "01/01/2021", "20210101"
        interval_end: Optional end date for filtering (inclusive). Accepts formats like "2021-01-01", "01/01/2021", "20210101"
    
    Returns:
        Tuple of (stats_daily, stats_summary, dmin, dmax, ndays) where dates are filtered if provided
    """
    if not os.path.isdir(daily_dir):
        raise FileNotFoundError(f"Expected DAILY stats dir: {daily_dir}")

    daily_paths = sorted(glob.glob(os.path.join(daily_dir, "stats_*.pkl")))
    if not daily_paths:
        raise FileNotFoundError(f"No daily files found in '{daily_dir}'.")

    daily_frames = []
    for p in daily_paths:
        try:
            df = read_pickle_compat(p)
        except Exception:
            continue
        if (
            isinstance(df, pd.DataFrame)
            and not df.empty
            and {"date", "value"}.issubset(df.columns)
        ):
            daily_frames.append(df)

    if not daily_frames:
        raise FileNotFoundError("All daily PKLs were empty or malformed.")

    stats_daily = pd.concat(daily_frames, ignore_index=True)

    # Parse + categories
    stats_daily["date"] = pd.to_datetime(stats_daily["date"], errors="coerce")
    stats_daily = stats_daily.dropna(subset=["date"])
    
    # Apply date range filter if provided
    if interval_start is not None or interval_end is not None:
        start_dt, end_dt = _parse_date_range(interval_start, interval_end)
        stats_daily = _filter_by_date_range(stats_daily, start_dt, end_dt, "daily data")
    
    for col in ("signal", "target", "bet_size_col", "qrank", "stat_type"):
        if col not in stats_daily.columns:
            stats_daily[col] = pd.NA
        stats_daily[col] = stats_daily[col].astype("string").astype("category")
    stats_daily["value"] = pd.to_numeric(stats_daily["value"], errors="coerce")

    dmin = stats_daily["date"].min()
    dmax = stats_daily["date"].max()
    ndays = int(stats_daily["date"].nunique())

    stats_summary = pd.DataFrame()
    if os.path.isdir(summary_dir):
        pkl_paths = sorted(glob.glob(os.path.join(summary_dir, "summary_stats_*.pkl")))
        if pkl_paths:
            try:
                stats_summary = read_pickle_compat(pkl_paths[-1])
                if isinstance(stats_summary, pd.DataFrame) and not stats_summary.empty:
                    if "date" in stats_summary.columns:
                        stats_summary["date"] = pd.to_datetime(
                            stats_summary["date"], errors="coerce"
                        )
                    
                    # Apply date range filter to summary if provided
                    if interval_start is not None or interval_end is not None:
                        start_dt, end_dt = _parse_date_range(interval_start, interval_end)
                        stats_summary = _filter_by_date_range(stats_summary, start_dt, end_dt, "summary data")
                    
                    if "value" in stats_summary.columns:
                        stats_summary["value"] = pd.to_numeric(
                            stats_summary["value"], errors="coerce"
                        )
                    for col in (
                        "signal",
                        "target",
                        "bet_size_col",
                        "qrank",
                        "stat_type",
                    ):
                        if col not in stats_summary.columns:
                            stats_summary[col] = pd.NA
                        stats_summary[col] = (
                            stats_summary[col].astype("string").astype("category")
                        )
                print(
                    f"[INFO] Loaded summary PKL: {os.path.basename(pkl_paths[-1])}  "
                    f"shape={stats_summary.shape}"
                )
            except Exception as e:
                print(f"[WARN] Failed to read summary PKL ({pkl_paths[-1]}): {e}")
        else:
            print(
                f"[WARN] No summary PKL found in {summary_dir} (bar plots will be skipped)."
            )
    else:
        print(
            f"[WARN] Summary directory not found: {summary_dir} (bar plots will be skipped)."
        )

    print(
        f"[INFO] DAILY date window: {dmin:%Y-%m-%d} → {dmax:%Y-%m-%d}  ({ndays} days)"
    )
    return stats_daily, stats_summary, dmin, dmax, ndays


# -------------------------------------------------
# Helpers for labels/quantiles/colors
# -------------------------------------------------
def _sorted_qranks(series):
    vals = [str(q) for q in pd.Series(series).dropna().unique()]
    try:
        return sorted(
            vals,
            key=lambda x: float(x.split("_")[1]) if "_" in x else float(x),
        )
    except Exception:
        return sorted(vals)


def _ensure_quantile_colors(labels, base_map):
    cmap = mpl.colormaps.get_cmap("tab20")
    out = dict(base_map or {})
    i = 0
    for lab in labels:
        if lab not in out:
            out[lab] = cmap(i % cmap.N)
            i += 1
    return out


def _plot_date_axis(ax):
    """Format date axis with intelligent formatting based on date range."""
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.margins(x=0.02, y=0.08)
    
    # Try to get date information from the axis
    try:
        from matplotlib.dates import num2date, DateFormatter
        
        # Get the actual tick locations
        locs = ax.get_xticks()
        if len(locs) < 2:
            ax.tick_params(axis="both", labelsize=11)
            return
        
        # Convert numeric positions to dates if possible
        # Filter out invalid dates (matplotlib uses large numbers for dates)
        dates = []
        for x in locs:
            try:
                dt = num2date(x)
                if dt.year >= 1900 and dt.year <= 2100:  # Reasonable date range
                    dates.append(dt)
            except (ValueError, OverflowError, OSError):
                continue
        
        if len(dates) >= 2:
            # Calculate approximate interval between consecutive dates
            intervals = []
            for i in range(1, len(dates)):
                diff = (dates[i] - dates[i-1]).days
                if diff > 0:
                    intervals.append(diff)
            
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                
                # Determine format based on interval
                if avg_interval >= 25:  # Monthly or longer intervals
                    # Use shorter format: "Jan 2021"
                    formatter = DateFormatter("%b %Y")
                    ax.xaxis.set_major_formatter(formatter)
                    ax.tick_params(axis="x", labelsize=9, rotation=45)
                    ax.tick_params(axis="y", labelsize=11)
                elif avg_interval >= 5:  # Weekly intervals
                    formatter = DateFormatter("%m/%d")
                    ax.xaxis.set_major_formatter(formatter)
                    ax.tick_params(axis="x", labelsize=10, rotation=45)
                    ax.tick_params(axis="y", labelsize=11)
                else:  # Daily intervals
                    formatter = DateFormatter("%m/%d")
                    ax.xaxis.set_major_formatter(formatter)
                    ax.tick_params(axis="x", labelsize=10, rotation=45)
                    ax.tick_params(axis="y", labelsize=11)
            else:
                # Fallback: use default formatting with rotation
                ax.tick_params(axis="x", labelsize=10, rotation=45)
                ax.tick_params(axis="y", labelsize=11)
        else:
            # Fallback: use default formatting with rotation
            ax.tick_params(axis="x", labelsize=10, rotation=45)
            ax.tick_params(axis="y", labelsize=11)
    except Exception:
        # Fallback if date formatting fails
        ax.tick_params(axis="x", labelsize=10, rotation=45)
        ax.tick_params(axis="y", labelsize=11)
    
    # Set horizontal alignment on labels after formatting (works for all cases)
    try:
        labels = ax.get_xticklabels()
        if labels:
            for label in labels:
                label.set_ha("right")
    except Exception:
        pass  # Ignore if labels can't be accessed


def _ellipsis(s, n):
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _heatmap_figure_size(k, widen=1.18, extra_height=0.0):
    s = max(12, min(30, 0.7 * k + 10))
    return (s * (widen + 0.04), s + extra_height + 0.5)


@contextmanager
def std_err():
    with np.errstate(invalid="ignore", divide="ignore"):
        yield


def _set_title_fit(
    fig,
    ax,
    text,
    base_size=14,
    min_size=8,
    pad=10,
    loc="center",
    allow_wrap=True,
    max_lines=2,
):
    text = " ".join(str(text).split())
    size = int(base_size)
    while size >= min_size:
        t = ax.set_title(text, fontsize=size, weight="bold", pad=pad, loc=loc)
        t.set_ha("center")
        t.set_x(0.5)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_bb = ax.get_window_extent(renderer=renderer)
        t_bb = t.get_window_extent(renderer=renderer)
        if (
            t_bb.width <= 0.98 * ax_bb.width
            and t_bb.x0 >= ax_bb.x0
            and t_bb.x1 <= ax_bb.x1
        ):
            return t
        size -= 1

    if allow_wrap:
        words = text.split()
        lines = []
        cur = ""
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_bb = ax.get_window_extent(renderer=renderer)
        for w in words:
            trial = (cur + " " + w).strip()
            t = ax.set_title(trial, fontsize=min_size, weight="bold", pad=pad, loc=loc)
            t.set_ha("center")
            t.set_x(0.5)
            fig.canvas.draw()
            if (
                t.get_window_extent(renderer=fig.canvas.get_renderer()).width
                <= 0.98 * ax_bb.width
            ):
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        t = ax.set_title(
            "\n".join(lines[:max_lines]),
            fontsize=min_size,
            weight="bold",
            pad=pad,
            loc=loc,
        )
        t.set_ha("center")
        t.set_x(0.5)
        fig.canvas.draw()
        return t

    t = ax.set_title(text, fontsize=min_size, weight="bold", pad=pad, loc=loc)
    t.set_ha("center")
    t.set_x(0.5)
    fig.canvas.draw()
    return t


def _centered_heatmap_axes(k):
    figsize = _heatmap_figure_size(k)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        nrows=1,
        ncols=2,
        figure=fig,
        left=0.10,
        right=0.90,
        bottom=0.10,
        top=HEATMAP_AX_TOP,
        width_ratios=[20, 1],
        wspace=0.15,
    )
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    return fig, ax, cax


def _plot_matrix_heatmap(
    fig,
    ax,
    cax,
    M,
    labels,
    title,
    vmin=-1,
    vmax=1,
    annotate_lower=True,
    fmt=".2f",
):
    if M is None or labels is None or len(labels) == 0:
        ax.axis("off")
        _set_title_fit(
            fig,
            ax,
            title,
            base_size=13,
            pad=8,
            loc="center",
            allow_wrap=True,
            max_lines=3,
        )
        if cax is not None:
            cax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    k = len(labels)
    fs_labels = 9 if k <= 18 else (7 if k <= 30 else 6)
    fs_cells = 8 if k <= 18 else (6 if k <= 30 else 5)

    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap="coolwarm", aspect="equal")
    _set_title_fit(
        fig, ax, title, base_size=13, pad=10, loc="center", allow_wrap=True, max_lines=2
    )
    ax.set_xticks(range(k))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fs_labels)
    ax.set_yticks(range(k))
    ax.set_yticklabels(labels, fontsize=fs_labels)
    ax.set_xlim(-0.5, k - 0.5)
    ax.set_ylim(k - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if annotate_lower:
        for i in range(k):
            for j in range(i + 1):
                val = M[i, j]
                if np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        format(val, fmt),
                        ha="center",
                        va="center",
                        fontsize=fs_cells,
                        color=("white" if abs(val) >= 0.5 else "black"),
                    )

    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=9)


def _minp(window, floor=3):
    if window is None:
        return 1
    w = int(max(1, window))
    return min(w, max(1, w // 5, floor))


def _roll_mean(s: pd.Series, window: int):
    mp = _minp(window, floor=3)
    return s.rolling(window, min_periods=mp).mean()


def _rolling_sharpe(s: pd.Series, window: int):
    """Rolling Sharpe using daily series; annualized with sqrt(252)."""
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)

    w = max(1, int(window))
    mp = _minp(w, floor=5)

    def _sharpe(x):
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        if not np.isfinite(mu) or not np.isfinite(sd) or sd <= 0:
            return np.nan
        return mu / sd * np.sqrt(252.0)

    return s.rolling(w, min_periods=mp).apply(_sharpe, raw=False)


def _as_list(x):
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple, set, pd.Series, np.ndarray)):
        return [str(v) for v in x]
    raise ValueError("Expected string or list-like.")


def _resolve_fixed(name, desired, series_values, prefer_prefix=None, top_k=1):
    vals = pd.Series(series_values).dropna().astype(str)
    if vals.empty:
        raise ValueError(f"No available values to resolve {name}.")
    if isinstance(desired, str) and desired.upper() == "AUTO":
        if prefer_prefix:
            vsub = vals[vals.str.startswith(prefer_prefix)]
            if not vsub.empty:
                top = vsub.value_counts().index.tolist()[:top_k]
                return top
        return vals.value_counts().index.tolist()[:top_k]
    out = _as_list(desired)
    missing = [v for v in out if v not in set(vals)]
    if missing:
        avail = ", ".join(sorted(set(vals)))
        raise ValueError(f"{name} contains unknown token(s): {missing}. Available: {avail}")
    return out


def _exclude_all_rows(df: pd.DataFrame) -> pd.DataFrame:
    m = (
        df["target"].ne("__ALL__")
        & df["bet_size_col"].ne("__ALL__")
        & df["qrank"].ne("__ALL__")
    )
    return df[m].copy()


# -------- Spearman helpers --------
def _spearman_corr_df(X: pd.DataFrame, min_periods: int = 2) -> pd.DataFrame:
    R = X.rank(axis=0, method="average", na_option="keep")
    return R.corr(method="pearson", min_periods=min_periods)


def _spearman_corr_pair(
    x: pd.Series, y: pd.Series, min_periods: int = 2
) -> float:
    m = x.notna() & y.notna()
    if m.sum() < min_periods:
        return np.nan
    xr = x[m].rank()
    yr = y[m].rank()
    if xr.nunique() <= 1 or yr.nunique() <= 1:
        return np.nan
    return float(np.corrcoef(xr.values, yr.values)[0, 1])

# -------------------------------------------------
# Heatmap builders (SPEARMAN)
# -------------------------------------------------
def _build_daily_cross_section(df_day, alphas, stat_type, qfilter=None,
                               targets=None, bets=None):
    d = df_day[df_day['stat_type'] == stat_type]
    if qfilter:   d = d[d['qrank'].isin(qfilter)]
    if targets:   d = d[d['target'].isin(targets)]
    if bets:      d = d[d['bet_size_col'].isin(bets)]
    if d.empty: return None

    piv = d.pivot_table(index=['target', 'bet_size_col', 'qrank'],
                        columns='signal', values='value', aggfunc='sum', observed=True)
    cols_present = [a for a in alphas if a in piv.columns]
    if not cols_present:
        return None
    piv = piv[cols_present].dropna(how='all')
    if piv.shape[0] < 2:  # need >=2 samples to compute corr
        return None
    X = pd.to_numeric(piv.stack(), errors='coerce').unstack().astype(float)
    return X


def _avg_mats_ignore_nan(mats):
    if not mats: return None
    k = mats[0].shape[0]
    sumM = np.zeros((k, k), float); cntM = np.zeros((k, k), int)
    for M in mats:
        if M is None or M.shape != (k, k): continue
        m = np.isfinite(M)
        sumM[m] += M[m]; cntM[m] += 1
    with std_err():
        H = sumM / np.where(cntM == 0, np.nan, cntM)
    H[cntM == 0] = np.nan
    for i in range(k):
        if not np.isfinite(H[i, i]): H[i, i] = 1.0
    return H


def compute_heatmap_daily_avg(stats_df, alphas, stat_type, min_pairs=2,
                              qfilter=None, targets=None, bets=None):
    alphas = list(alphas)
    if len(alphas) < 2: return None, alphas, 0

    # Special-case: alpha_sum comes as one value per (signal, qrank, day).
    # Cross-section per-day has only one sample, so instead build a time-series
    # Spearman corr across days using alpha_sum series per signal.
    if stat_type == "alpha_sum":
        df = stats_df[stats_df['stat_type'] == stat_type].copy()
        if qfilter: df = df[df['qrank'].isin(qfilter)]
        if targets: df = df[df['target'].isin(targets)]
        if bets:    df = df[df['bet_size_col'].isin(bets)]
        if df.empty: return None, alphas, 0
        wide = df.pivot_table(index='date', columns='signal', values='value', aggfunc='sum', observed=True)
        wide = wide.reindex(columns=alphas).sort_index()
        if wide.shape[0] < min_pairs or wide.shape[1] < 2:
            return None, list(wide.columns), int(wide.shape[0])
        C = _spearman_corr_df(wide, min_periods=min_pairs).reindex(index=alphas, columns=alphas)
        return C.to_numpy(dtype=float), list(C.columns), int(wide.shape[0])

    mats = []
    for _, df_day in stats_df.groupby('date', sort=True, observed=True):
        X = _build_daily_cross_section(df_day, alphas, stat_type,
                                       qfilter=qfilter, targets=targets, bets=bets)
        if X is None: continue
        C = _spearman_corr_df(X, min_periods=min_pairs).reindex(index=alphas, columns=alphas)
        M = C.to_numpy(dtype=float)
        if np.isfinite(M).sum() == 0:
            continue
        mats.append(M)
    if not mats:
        # Fallback: if filters reduce per-day samples to <2, try time-series corr across days
        df_filt = stats_df[stats_df['stat_type'] == stat_type].copy()
        if qfilter: df_filt = df_filt[df_filt['qrank'].isin(qfilter)]
        if targets: df_filt = df_filt[df_filt['target'].isin(targets)]
        if bets:    df_filt = df_filt[df_filt['bet_size_col'].isin(bets)]
        if df_filt.empty:
            return None, alphas, 0
        wide = (df_filt.groupby(['date', 'signal'], observed=True)['value']
                .sum()
                .unstack('signal')
                .reindex(columns=alphas)
                .sort_index())
        wide = wide.dropna(axis=1, how='all')
        if wide.shape[0] < max(2, min_pairs) or wide.shape[1] < 2:
            return None, list(wide.columns), int(wide.shape[0])
        C = _spearman_corr_df(wide, min_periods=min_pairs).reindex(index=alphas, columns=alphas)
        return C.to_numpy(dtype=float), list(C.columns), int(wide.shape[0])
    H = _avg_mats_ignore_nan(mats)
    return H, alphas, len(mats)


def compute_timeseries_heatmap(stats_df, alphas, stat_type, min_days=5, agg='sum',
                               qfilter=None, targets=None, bets=None):
    df = stats_df[stats_df['stat_type'] == stat_type].copy()
    if qfilter: df = df[df['qrank'].isin(qfilter)]
    if targets: df = df[df['target'].isin(targets)]
    if bets:    df = df[df['bet_size_col'].isin(bets)]
    if df.empty: return None, alphas, 0

    gb = df.groupby(['date', 'signal'], observed=True)['value']
    daily = (gb.sum() if agg == 'sum' else gb.mean()).unstack('signal')
    daily = daily.reindex(columns=alphas).dropna(axis=1, how='all').sort_index()

    if daily.shape[1] < 2 or daily.shape[0] < min_days:
        return None, list(daily.columns), int(daily.shape[0])

    C = _spearman_corr_df(daily, min_periods=min_days)
    return C.values, C.columns.tolist(), int(daily.shape[0])


# -------------------------------------------------
# Temporal correlation lines (SPEARMAN)
# -------------------------------------------------
def compute_daily_pair_corr_series(stats_df, alphas, stat_type, min_pairs=2,
                                   qfilter=None, targets=None, bets=None):
    """
    Returns dict: "A|B" -> Series(date -> corr)
    """
    alphas = [a for a in alphas]
    pairs = list(combinations(alphas, 2))
    dates = sorted(stats_df['date'].dropna().unique())
    out = {f"{a}|{b}": pd.Series(index=pd.DatetimeIndex(dates, name='date', dtype='datetime64[ns]'),
                                 dtype='float64') for a, b in pairs}

    for dt, df_day in stats_df.groupby('date', sort=True, observed=True):
        X = _build_daily_cross_section(df_day, alphas, stat_type,
                                       qfilter=qfilter, targets=targets, bets=bets)
        if X is None:
            continue
        cols = set(X.columns)
        for a, b in pairs:
            if (a not in cols) or (b not in cols):
                continue
            xa = pd.to_numeric(X[a], errors='coerce')
            xb = pd.to_numeric(X[b], errors='coerce')
            out[f"{a}|{b}"].loc[dt] = _spearman_corr_pair(xa, xb, min_periods=min_pairs)
    return out


def _rolling_spearman_pair(a: pd.Series, b: pd.Series, window=None, min_periods=2):
    s1 = a.copy(); s2 = b.copy()
    idx = s1.index.union(s2.index); s1 = s1.reindex(idx); s2 = s2.reindex(idx)
    out = pd.Series(index=idx, dtype='float64')

    if (window is None) or int(window) <= 1:
        # expanding
        for i in range(len(idx)):
            x = s1.iloc[:i + 1]; y = s2.iloc[:i + 1]
            m = x.notna() & y.notna()
            if m.sum() >= min_periods:
                xr = x[m].rank(); yr = y[m].rank()
                out.iloc[i] = np.corrcoef(xr, yr)[0, 1] if (xr.nunique() > 1 and yr.nunique() > 1) else np.nan
            else:
                out.iloc[i] = np.nan
        return out

    w = int(window)
    for i in range(len(idx)):
        start = max(0, i - w + 1)
        x = s1.iloc[start:i + 1]; y = s2.iloc[start:i + 1]
        m = x.notna() & y.notna()
        if m.sum() >= min_periods:
            xr = x[m].rank(); yr = y[m].rank()
            out.iloc[i] = np.corrcoef(xr, yr)[0, 1] if (xr.nunique() > 1 and yr.nunique() > 1) else np.nan
        else:
            out.iloc[i] = np.nan
    return out


def compute_pairwise_rolling_time_corr(stats_df, alphas, stat_type, window=1, min_periods=None,
                                       qfilter=None, targets=None, bets=None, agg='mean'):
    df = stats_df[stats_df['stat_type'] == stat_type].copy()
    if qfilter: df = df[df['qrank'].isin(qfilter)]
    if targets: df = df[df['target'].isin(targets)]
    if bets:    df = df[df['bet_size_col'].isin(bets)]
    if df.empty: return {}

    gb = df.groupby(['date', 'signal'], observed=True)['value']
    daily = (gb.sum() if agg == 'sum' else gb.mean()).unstack('signal')
    daily = daily.reindex(columns=alphas).dropna(axis=1, how='all').sort_index()
    if daily.shape[1] < 2: return {}

    if (window is None) or int(window) <= 1:
        mp = 2 if min_periods is None else int(min_periods)
        cols = [c for c in daily.columns if daily[c].notna().sum() >= mp]
        pairs = list(combinations(cols, 2))
        out = {}
        for a, b in pairs:
            out[f"{a}|{b}"] = _rolling_spearman_pair(daily[a], daily[b], window=None, min_periods=mp)
        return out

    win = int(window)
    # Use min_periods=1 so values appear from the first day (was _minp(win, floor=3))
    mp = 1 if min_periods is None else int(min_periods)
    cols = [c for c in daily.columns if daily[c].notna().sum() >= mp]
    pairs = list(combinations(cols, 2))
    out = {}
    for a, b in pairs:
        out[f"{a}|{b}"] = _rolling_spearman_pair(daily[a], daily[b], window=win, min_periods=mp)
    return out


def _label_last_points(ax, series_map, cmap=None, fontsize=9):
    """
    Label each line at its last valid point; no legend used.
    series_map: dict[name] -> pd.Series(index=date, values=corr)
    """
    if cmap is None:
        cmap = mpl.colormaps.get_cmap('tab20')
    for i, (name, s) in enumerate(series_map.items()):
        s = s.sort_index()
        color = cmap(i % cmap.N)
        ax.plot(s.index, s.values, lw=1.8, alpha=0.95, color=color)
        v = s.values
        if np.isfinite(v).any():
            all_vals = v[np.isfinite(v)]
        finite_idx = np.where(np.isfinite(s.values))[0]
        if finite_idx.size:
            j = finite_idx[-1]
            ax.text(s.index[j], s.values[j], f"  {name}", color=color, fontsize=fontsize, va='center')


# ---- Temporal plot helpers ----
def plot_cross_section_corr_lines(pdf, stats_df, alphas, stat_type, title_prefix,
                                  smooth_window=1, height=6.0,
                                  qfilter=None, targets=None, bets=None):
    corr_map = compute_daily_pair_corr_series(
        stats_df, alphas, stat_type, min_pairs=2,
        qfilter=qfilter, targets=targets, bets=bets
    )
    filt_str = " | ".join(filter(None, [
        f"qr={','.join(qfilter)}" if qfilter else "",
        f"tgt={','.join(targets)}" if targets else "",
        f"bet={','.join(bets)}" if bets else ""
    ]))
    title_text = (f"{title_prefix} — smoothed {int(smooth_window)}D mean"
                  if (smooth_window is not None and int(smooth_window) > 1)
                  else f"{title_prefix} — daily (no smoothing)")
    if filt_str: title_text += f" — {filt_str}"

    if not corr_map:
        fig, ax = plt.subplots(figsize=(14, height))
        _set_title_fit(fig, ax, title_text + " — no data", base_size=14, pad=8, loc='center')
        fig.tight_layout(rect=[0.02, 0.06, 0.98, TEMPORAL_AX_TOP]); savefig_white(pdf, fig); return

    coverage = [(k, v.notna().sum()) for k, v in corr_map.items()]
    coverage.sort(key=lambda x: x[1], reverse=True)
    chosen = [k for k, cnt in coverage[:8] if cnt > 0]

    dates_all = sorted(stats_df['date'].dropna().unique())
    if len(dates_all) == 0 or not chosen:
        fig, ax = plt.subplots(figsize=(14, height))
        _set_title_fit(fig, ax, title_text + " — insufficient", base_size=14, pad=8, loc='center')
        fig.tight_layout(rect=[0.02, 0.06, 0.98, TEMPORAL_AX_TOP]); savefig_white(pdf, fig); return

    fig, ax = plt.subplots(figsize=(14, height))
    _set_title_fit(fig, ax, title_text, base_size=14, pad=10, loc='center')

    cmap = mpl.colormaps.get_cmap('tab20')
    x_min, x_max = pd.to_datetime(dates_all[0]), pd.to_datetime(dates_all[-1])
    all_vals = []

    for i, key in enumerate(chosen):
        s = corr_map[key].copy().sort_index()
        if (smooth_window is not None) and int(smooth_window) > 1:
            win = int(smooth_window)
            # Use min_periods=1 so values appear from the first day
            s = s.rolling(win, min_periods=1).mean()
        color = cmap(i % cmap.N)
        ax.plot(s.index, s.values, lw=1.8, alpha=0.95, color=color)
        v = s.values
        if np.isfinite(v).any():
            all_vals.append(v[np.isfinite(v)])
        finite_idx = np.where(np.isfinite(s.values))[0]
        if finite_idx.size:
            j = finite_idx[-1]
            ax.text(s.index[j], s.values[j], f"  {key}", color=color, fontsize=9, va='center')

    if all_vals:
        vals = np.concatenate(all_vals)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        span = max(vmax - vmin, 1e-6); pad = 0.08 * span
        y0 = max(-1.05, vmin - pad); y1 = min(1.05, vmax + pad)
        if (y1 - y0) < 0.2:
            mid = 0.5 * (y0 + y1); y0, y1 = mid - 0.1, mid + 0.1
        ax.set_ylim(y0, y1)
    else:
        ax.set_ylim(-1.05, 1.05)

    ax.set_ylabel("Spearman corr")
    ax.set_xlim(x_min, x_max)
    ax.margins(x=0.03)
    _plot_date_axis(ax)

    fig.tight_layout(rect=[0.02, 0.06, 0.98, TEMPORAL_AX_TOP])
    savefig_white(pdf, fig)


def plot_pairwise_timecorr_lines(pdf, stats_df, alphas, stat_type, title_prefix,
                                 window=1, height=6.0, qfilter=None, targets=None, bets=None, agg='mean'):
    corr_map = compute_pairwise_rolling_time_corr(
        stats_df, alphas, stat_type, window=window, qfilter=qfilter,
        targets=targets, bets=bets, agg=agg
    )
    filt_str = " | ".join(filter(None, [
        f"qr={','.join(qfilter)}" if qfilter else "",
        f"tgt={','.join(targets)}" if targets else "",
        f"bet={','.join(bets)}" if bets else ""
    ]))
    title_text = (f"{title_prefix} — Rolling Spearman {int(window)}D"
                  if (window is not None and int(window) > 1)
                  else f"{title_prefix} — Expanding Spearman")
    if filt_str: title_text += f" — {filt_str}"

    if not corr_map:
        fig, ax = plt.subplots(figsize=(14, height))
        _set_title_fit(fig, ax, title_text + " — no data", base_size=14, pad=8, loc='center')
        fig.tight_layout(rect=[0.02, 0.06, 0.98, TEMPORAL_AX_TOP]); savefig_white(pdf, fig); return

    coverage = [(k, v.notna().sum()) for k, v in corr_map.items()]
    coverage.sort(key=lambda x: x[1], reverse=True)
    chosen = [k for k, cnt in coverage[:8] if cnt > 0]

    dates_all = sorted(stats_df['date'].dropna().unique())
    if len(dates_all) == 0 or not chosen:
        fig, ax = plt.subplots(figsize=(14, height))
        _set_title_fit(fig, ax, title_text + " — insufficient", base_size=14, pad=8, loc='center')
        fig.tight_layout(rect=[0.02, 0.06, 0.98, TEMPORAL_AX_TOP]); savefig_white(pdf, fig); return

    fig, ax = plt.subplots(figsize=(14, height))
    _set_title_fit(fig, ax, title_text, base_size=14, pad=10, loc='center')

    cmap = mpl.colormaps.get_cmap('tab20')
    x_min, x_max = pd.to_datetime(dates_all[0]), pd.to_datetime(dates_all[-1])
    all_vals = []

    for i, key in enumerate(chosen):
        s = corr_map[key].copy().sort_index()
        color = cmap(i % cmap.N)
        ax.plot(s.index, s.values, lw=1.8, alpha=0.95, color=color)
        v = s.values
        if np.isfinite(v).any():
            all_vals.append(v[np.isfinite(v)])
        finite_idx = np.where(np.isfinite(s.values))[0]
        if finite_idx.size:
            j = finite_idx[-1]
            ax.text(s.index[j], s.values[j], f"  {key}", color=color, fontsize=9, va='center')

    if all_vals:
        vals = np.concatenate(all_vals)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        span = max(vmax - vmin, 1e-6); pad = 0.08 * span
        y0 = max(-1.05, vmin - pad); y1 = min(1.05, vmax + pad)
        if (y1 - y0) < 0.2:
            mid = 0.5 * (y0 + y1); y0, y1 = mid - 0.1, mid + 0.1
        ax.set_ylim(y0, y1)
    else:
        ax.set_ylim(-1.05, 1.05)

    ax.set_ylabel("Spearman corr (time)")
    ax.set_xlim(x_min, x_max)
    ax.margins(x=0.03)
    _plot_date_axis(ax)

    fig.tight_layout(rect=[0.02, 0.06, 0.98, TEMPORAL_AX_TOP])
    savefig_white(pdf, fig)


# -------------------------------------------------
# Alpha autodetection (for heatmaps/lines)
# -------------------------------------------------
def _autodetect_alphas(df, max_k=16):
    df = df[df['signal'].notna()]
    base = df[df['stat_type'] == 'pnl']
    day_counts = (base.groupby('signal', observed=True)['date'].nunique()).sort_values(ascending=False)
    if day_counts.empty:
        day_counts = (df.groupby('signal', observed=True)['date'].nunique()).sort_values(ascending=False)
    candidates = day_counts[day_counts >= 5].index.tolist() or day_counts.index.tolist()
    return list(map(str, candidates[:max_k]))


# -------------------------------------------------
# Outlier tables helpers (compact)
# -------------------------------------------------
def _find_latest_outliers_pkl(root: str):
    if not os.path.isdir(root): return None
    cand = sorted(glob.glob(os.path.join(root, "outliers_*.pkl")))
    return cand[-1] if cand else None


def _metric_table_rows(odf, metric, top_k, have_z, have_rule):
    sub = odf[odf['stat_type'] == metric].copy()
    if sub.empty: return None, None
    # Order by z if present; otherwise by absolute value
    if 'z' in sub.columns:
        sub['z'] = pd.to_numeric(sub['z'], errors='coerce')
        sub['abs_z'] = sub['z'].abs()
        sub = sub.sort_values('z')
        lows = sub.head(top_k).copy()           # most negative z
        highs = sub.tail(top_k).iloc[::-1].copy()  # most positive z
    else:
        sub = sub.sort_values('value')
        lows = sub.head(top_k).copy()
        highs = sub.tail(top_k).iloc[::-1].copy()
    labels_tbl = ["Type", "Date", "Signal", "Bet", "Target", "Q", "Value"] + (["z"] if have_z else [])

    def row(r, kind):
        base = [
            r['date'].strftime('%Y-%m-%d') if pd.notna(r['date']) else "NaT",
            _ellipsis(r['signal'], 18), _ellipsis(r['bet_size_col'], 16),
            _ellipsis(r['target'], 16), str(r['qrank']), f"{r['value']:.6g}",
        ]
        base = [kind] + base
        if have_z:    base.append("" if pd.isna(r.get('z')) else f"{r.get('z'):.2f}")
        return base

    rows = [row(r, "High") for _, r in highs.iterrows()] + [row(r, "Low") for _, r in lows.iterrows()]
    return labels_tbl, rows


def _draw_table_in_axis(ax, title, col_labels, rows, fontsize=9):
    ax.axis('off'); ax.set_title(title, fontsize=13, weight='bold', loc='left', pad=6)
    base_w = {"Type": 0.08, "Date": 0.11, "Signal": 0.18, "Bet": 0.14,
              "Target": 0.14, "Q": 0.06, "Value": 0.11, "z": 0.06, "Rule": 0.12}
    colWidths = [base_w.get(lbl, 0.10) for lbl in col_labels]
    s = sum(colWidths)
    if s > 0.98: colWidths = [w * 0.98 / s for w in colWidths]
    tb = ax.table(cellText=rows, colLabels=col_labels, colWidths=colWidths,
                  loc='upper left', cellLoc='left', bbox=[0.0, 0.0, 1.0, 0.92])
    tb.auto_set_font_size(False); tb.set_fontsize(fontsize); tb.scale(1.0, 1.10)
    header_color = (0.9, 0.9, 0.92); even = (0.98, 0.98, 0.985); odd = (1.0, 1.0, 1.0)
    for (r, c), cell in tb.get_celld().items():
        if r == 0:
            cell.set_text_props(weight='bold', ha='left')
            cell.set_facecolor(header_color); cell.set_edgecolor('0.75')
        else:
            cell.set_edgecolor('0.85')
            cell.set_facecolor(even if r % 2 == 0 else odd)


def append_outlier_pages(outliers_pkl_path: str, pdf,
                         metrics=None, top_k: int = 3, tables_per_page: int = 3):
    try:
        if outliers_pkl_path is None or not os.path.isfile(outliers_pkl_path):
            raise FileNotFoundError("Outliers PKL not found.")
        odf = read_pickle_compat(outliers_pkl_path)
    except Exception as e:
        fig, ax = plt.subplots(figsize=(14, 4)); ax.axis('off')
        fig.suptitle("Outlier Tables", fontsize=18, weight='bold')
        ax.text(0.5, 0.5, f"No outlier tables appended:\n{e}", ha='center', va='center', fontsize=12)
        savefig_white(pdf, fig); return

    if odf is None or len(odf) == 0:
        fig, ax = plt.subplots(figsize=(14, 4)); ax.axis('off')
        fig.suptitle("Outlier Tables", fontsize=18, weight='bold')
        ax.text(0.5, 0.5, "No outliers found.", ha='center', va='center', fontsize=12)
        savefig_white(pdf, fig); return

    odf = odf.copy()
    odf = _apply_stat_aliases(odf)
    odf['date'] = pd.to_datetime(odf['date'], errors='coerce')
    odf = odf.dropna(subset=['date', 'value'])
    have_rule = False  # hide rule column from output
    have_z    = 'z' in odf.columns

    if metrics is None:
        metrics = sorted(odf['stat_type'].unique().tolist())
    else:
        metrics = [m for m in metrics if m in odf['stat_type'].unique()]
        if not metrics:
            fig, ax = plt.subplots(figsize=(14, 4)); ax.axis('off')
            fig.suptitle("Outlier Tables", fontsize=18, weight='bold')
            ax.text(0.5, 0.5, "Selected outlier metrics not present in file.", fontsize=12)
            savefig_white(pdf, fig); return

    # Drop __ALL__ targets to avoid duplicate rows
    odf = odf[odf.get('target', '') != "__ALL__"]

    tables = []
    for m in metrics:
        col_labels, rows = _metric_table_rows(odf, m, top_k=top_k, have_z=have_z, have_rule=have_rule)
        if col_labels is None or not rows: continue
        tables.append((m, col_labels, rows))
    if not tables: return

    per_page = max(1, int(tables_per_page))
    for page_idx in range(0, len(tables), per_page):
        chunk = tables[page_idx:page_idx + per_page]
        fig = plt.figure(figsize=(14, 8.5))
        fig.suptitle("Outlier Tables", fontsize=18, weight='bold', y=0.985)
        gs = GridSpec(nrows=len(chunk), ncols=1, figure=fig, left=0.03, right=0.97, top=0.90, bottom=0.06, hspace=0.35)
        for row_idx, (metric_name, col_labels, rows) in enumerate(chunk):
            ax = fig.add_subplot(gs[row_idx, 0])
            _draw_table_in_axis(ax, title=f"{metric_name} — Top Highs & Lows",
                                col_labels=col_labels, rows=rows, fontsize=9)
        savefig_white(pdf, fig)


# =========================
# Quantile report builder
# =========================
def _series(df, stat):
    s = df[df['stat_type'] == stat][['date', 'value']].set_index('date')['value'].astype(float)
    return s.sort_index()


def _title_token(base: str, window: int, cumulative: bool = False) -> str:
    if cumulative:
        base = f"cumulative {base}"
        if window and int(window) > 1:
            return f"Rolling-mean {base} ({int(window)}D)"
        return base.capitalize()
    else:
        return f"Rolling-mean {base} ({int(window)}D)" if (window and int(window) > 1) else base


def _distrib_page(pdf, df, title, bins=40):
    """Histogram for correlation distributions with mean/median/std annotated."""
    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(14, 4)); ax.axis('off')
        ax.set_title(title); ax.text(0.5, 0.5, "No data", ha='center', va='center')
        savefig_white(pdf, fig); return

    # Try to find correlation column
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if 'corr' in df.columns:
        x = pd.to_numeric(df['corr'], errors='coerce')
    elif num_cols:
        x = pd.to_numeric(df[num_cols[0]], errors='coerce')
    else:
        fig, ax = plt.subplots(figsize=(14, 4)); ax.axis('off')
        ax.set_title(title); ax.text(0.5, 0.5, "No numeric column found", ha='center', va='center')
        savefig_white(pdf, fig); return

    x = x[np.isfinite(x)]
    if x.empty:
        fig, ax = plt.subplots(figsize=(14, 4)); ax.axis('off')
        ax.set_title(title); ax.text(0.5, 0.5, "All NaN", ha='center', va='center')
        savefig_white(pdf, fig); return

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.hist(x.values, bins=bins, edgecolor='white', alpha=0.9)
    m = float(np.nanmean(x)); med = float(np.nanmedian(x)); sd = float(np.nanstd(x))
    ax.axvline(m,   linestyle='-',  linewidth=2)
    ax.axvline(med, linestyle=':',  linewidth=2)
    ax.axvline(m + sd, linestyle='--', linewidth=1)
    ax.axvline(m - sd, linestyle='--', linewidth=1)
    ax.set_xlabel("Correlation"); ax.set_ylabel("Count")
    ax.set_title(f"{title}\nmean={m:.3f}  median={med:.3f}  std={sd:.3f}")
    savefig_white(pdf, fig)


def _metric_series_for_temporal(metric: str, df: pd.DataFrame, roll_windows: dict,
                                roll_sharpe: int):
    """Return (title, series) for a temporal metric, handling cumulative/rolling logic."""
    name = _canonical_stat(str(metric))
    if df is None or df.empty:
        return None, None

    if name == "pnl":
        s = _series(df, "pnl")
        if s.empty:
            return None, None
        y = _roll_mean(s.cumsum(), roll_windows.get("pnl", 1))
        title = _title_token(_metric_label("pnl"), roll_windows.get("pnl", 1), cumulative=True)
        return title, y

    if name == "ppd":
        s = _series(df, "ppd")
        if s.empty:
            return None, None
        s = s * 10000.0  # basis points
        y = _roll_mean(s.cumsum(), roll_windows.get("ppd", 1))
        title = _title_token("PPD (bps)", roll_windows.get("ppd", 1), cumulative=True)
        return title, y

    if name == "nrInstr":
        s = _series(df, "nrInstr")
        if s.empty:
            return None, None
        window = roll_windows.get("nrInstr", 1)
        y = _roll_mean(s, window)
        title = _title_token("nrInstr", window, cumulative=False)
        return title, y

    if name == "n_trades":
        s = _series(df, "n_trades")
        if s.empty:
            return None, None
        window = roll_windows.get("n_trades", 1)
        y = _roll_mean(s, window)
        title = _title_token("n_trades", window, cumulative=False)
        return title, y

    if name == "sizeNotional":
        s = _series(df, "sizeNotional")
        if s.empty:
            return None, None
        window = roll_windows.get("sizeNotional", 1)
        y = _roll_mean(s, window)
        title = _title_token("Size Notional", window, cumulative=False)
        return title, y

    if name == "sharpe":
        pnl_series = _series(df, "pnl")
        if pnl_series.empty:
            return None, None
        window = max(1, int(roll_sharpe))
        y = _rolling_sharpe(pnl_series, window)
        return f"Rolling Sharpe ({window}D)", y

    s = _series(df, name)
    if s.empty:
        return None, None
    window = roll_windows.get(name, roll_windows.get("__default__", 1))
    y = _roll_mean(s, window) if window and int(window) > 1 else s
    title = _title_token(name, window, cumulative=False)
    return title, y


def _plot_temporal_grid(pdf,
                        df: pd.DataFrame,
                        qranks: list[str],
                        quantile_colors: dict,
                        metrics: list[str],
                        roll_windows: dict,
                        roll_sharpe: int,
                        grid_shape: tuple[int, int],
                        title_prefix: str,
                        style: str = "-"):
    """Plot a grid of temporal metrics for one (target, signal, bet) combination."""
    rows, cols = grid_shape
    per_page = max(1, rows * cols)
    metrics = [m for m in metrics if m]  # defensive
    if not metrics:
        return

    legend_handles = [
        Line2D([0], [0], color=quantile_colors.get(q, "gray"), linestyle=style, label=str(q))
        for q in qranks
    ]

    n_pages = int(np.ceil(len(metrics) / per_page))

    for page_idx, start in enumerate(range(0, len(metrics), per_page), start=1):
        chunk = metrics[start:start + per_page]
        fig, axs = plt.subplots(
            rows, cols, figsize=PAGE_SIZE
        )
        axs = np.atleast_1d(axs).ravel()

        for ax in axs[len(chunk):]:
            ax.axis("off")

        any_plotted = False
        for ax, metric in zip(axs, chunk):
            metric_any = False
            metric_title = None
            for q in qranks:
                sq = df[df["qrank"] == q]
                title, series = _metric_series_for_temporal(metric, sq, roll_windows, roll_sharpe)
                if series is None or series.empty:
                    continue
                ax.plot(
                    series.index,
                    series.values,
                    color=quantile_colors.get(q, "gray"),
                    linestyle=style,
                    linewidth=1.6,
                    label=str(q),
                )
                metric_any = True
                metric_title = title if title else metric

            if metric_any:
                any_plotted = True
                ax.set_ylabel(metric_title or str(metric))
                _plot_date_axis(ax)
            else:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    f"No data for {metric}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="0.35",
                )

        fig.suptitle(
            f"{title_prefix} — Temporal metrics",
            fontsize=16,
            weight="bold",
            y=0.97,
        )

        # Legend just below the title
        if any_plotted and legend_handles:
            fig.legend(
                handles=legend_handles,
                title="Quantile",
                fontsize=9,
                frameon=True,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.90),
                ncol=len(legend_handles),
            )

        fig.tight_layout(rect=[0.04, 0.07, 0.96, 0.88])
        savefig_white(pdf, fig)


# =====================================================
# CCF helpers: per-ticker RAW/PnL vs SPY cross-corr
# =====================================================
def _load_latest_ccf_pkl(root: str, pattern):
    """Return latest non-empty DataFrame for given CCF glob pattern(s) or None."""
    if root is None or not os.path.isdir(root):
        return None
    patterns = pattern if isinstance(pattern, (list, tuple)) else [pattern]
    paths: list[str] = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(root, pat)))
    paths = sorted(paths)
    if not paths:
        return None
    path = paths[-1]
    try:
        df = read_pickle_compat(path)
    except Exception as e:
        print(f"[WARN] Failed to read CCF PKL {path}: {e}")
        return None
    if not isinstance(df, pd.DataFrame) or df.empty:
        print(f"[WARN] CCF PKL {path} is empty or not a DataFrame.")
        return None
    print(f"[INFO] Loaded CCF PKL: {os.path.basename(path)}  shape={df.shape}")
    return df


def _ccf_prepare_df(df: pd.DataFrame, max_lag: int | None):
    """Normalize CCF DF to have integer 'lag' and float 'corr', filter by [-max_lag, +max_lag]."""
    if df is None or df.empty:
        return None
    df = df.copy()

    if 'lag' not in df.columns:
        print("[WARN] CCF DF missing 'lag' column; skipping.")
        return None
    if 'corr' not in df.columns:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            df = df.rename(columns={num_cols[0]: 'corr'})
        else:
            print("[WARN] CCF DF missing numeric correlation column; skipping.")
            return None

    df['lag'] = pd.to_numeric(df['lag'], errors='coerce')
    df['corr'] = pd.to_numeric(df['corr'], errors='coerce')
    df = df[np.isfinite(df['lag']) & np.isfinite(df['corr'])]
    if df.empty:
        return None

    if max_lag is not None:
        max_lag = int(max_lag)
        df = df[(df['lag'] >= -max_lag) & (df['lag'] <= max_lag)]
    if df.empty:
        return None

    df['lag'] = df['lag'].astype(int)
    return df


def _ccf_bar_summary_page(pdf, df_raw_ccf: pd.DataFrame | None,
                          df_pnl_ccf: pd.DataFrame | None,
                          max_lag: int = 5):
    """CCF summary: mean/median/std vs lag for RAW and PnL."""
    df_raw_ccf = _ccf_prepare_df(df_raw_ccf, max_lag)
    df_pnl_ccf = _ccf_prepare_df(df_pnl_ccf, max_lag)

    if df_raw_ccf is None and df_pnl_ccf is None:
        return

    n_panels = 1 if (df_raw_ccf is None or df_pnl_ccf is None) else 2
    fig, axs = plt.subplots(1, n_panels, figsize=(14, 4.8))
    axs = np.atleast_1d(axs)

    def _plot_one(ax, df, title):
        stats = (df.groupby('lag')['corr']
                 .agg(['mean', 'median', 'std'])
                 .sort_index())
        if stats.empty:
            ax.axis('off')
            ax.set_title(title + " — no data", fontsize=11)
            return
        lags = stats.index.values
        ax.errorbar(lags, stats['mean'].values, yerr=stats['std'].values,
                    fmt='-o', linewidth=1.8, capsize=3, label='mean ± std')
        ax.plot(lags, stats['median'].values,
                linestyle=':', marker='x', linewidth=1.5, label='median')
        ax.axhline(0.0, linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Correlation")
        ax.set_xticks(lags)
        ax.grid(True, linestyle=":", alpha=0.35, axis='y')
        ax.set_title(title, fontsize=11)
        ax.legend(loc='best', fontsize=8, frameon=True)

    idx = 0
    if df_raw_ccf is not None:
        _plot_one(axs[idx], df_raw_ccf, "RAW alpha vs SPY — CCF summary")
        idx += 1
    if df_pnl_ccf is not None:
        _plot_one(axs[idx], df_pnl_ccf, "PnL vs SPY — CCF summary")

    fig.suptitle(
        "Cross-correlation vs SPY (per-ticker; aggregated across stocks)",
        fontsize=16, weight='bold', y=0.96
    )
    fig.tight_layout(rect=[0.03, 0.05, 0.97, 0.90])
    savefig_white(pdf, fig)


def _ccf_hist_pages(pdf, df_ccf: pd.DataFrame, title_prefix: str,
                    max_lag: int = 5, bins: int = 40):
    """Histogram grids over CCF distributions per lag, across stocks."""
    df_ccf = _ccf_prepare_df(df_ccf, max_lag)
    if df_ccf is None or df_ccf.empty:
        fig, ax = plt.subplots(figsize=(14, 4)); ax.axis('off')
        ax.set_title(title_prefix); ax.text(0.5, 0.5, "No CCF data", ha='center', va='center')
        savefig_white(pdf, fig); return

    lags = sorted(df_ccf['lag'].unique())
    if not lags:
        fig, ax = plt.subplots(figsize=(14, 4)); ax.axis('off')
        ax.set_title(title_prefix); ax.text(0.5, 0.5, "No lag values", ha='center', va='center')
        savefig_white(pdf, fig); return

    rows, cols = 3, 4
    per_page = rows * cols

    for page_start in range(0, len(lags), per_page):
        page_lags = lags[page_start:page_start + per_page]
        fig, axs = plt.subplots(rows, cols, figsize=(14, 8.5))
        axs = np.atleast_1d(axs).ravel()

        for ax in axs:
            ax.axis('off')

        for ax, lag in zip(axs, page_lags):
            sub = df_ccf[df_ccf['lag'] == lag]
            x = pd.to_numeric(sub['corr'], errors='coerce')
            x = x[np.isfinite(x)]
            if x.empty:
                continue
            ax.hist(x.values, bins=bins, edgecolor='white', alpha=0.9)
            m = float(np.nanmean(x)); med = float(np.nanmedian(x)); sd = float(np.nanstd(x))
            n = int(x.size)
            ax.axvline(m,   linestyle='-',  linewidth=1.2)
            ax.axvline(med, linestyle=':',  linewidth=1.2)
            ax.axvline(m + sd, linestyle='--', linewidth=0.9)
            ax.axvline(m - sd, linestyle='--', linewidth=0.9)
            ax.set_title(f"lag={lag}  n={n}\nμ={m:.3f}  med={med:.3f}  σ={sd:.3f}",
                         fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, linestyle=":", alpha=0.25, axis='y')

        fig.suptitle(title_prefix, fontsize=16, weight='bold', y=0.98)
        fig.tight_layout(rect=[0.03, 0.04, 0.97, 0.94])
        savefig_white(pdf, fig)


def append_ccf_pages(per_ticker_dir: str, pdf, max_lag: int = 5) -> bool:
    """
    Build CCF pages if per-ticker CCF PKLs exist.

    Files:
      per_ticker_alpha_raw_spy_ccf_*.pkl
      per_ticker_alpha_pnl_spy_ccf_*.pkl

    Returns True if any CCF page was generated, else False.
    """
    if per_ticker_dir is None or not os.path.isdir(per_ticker_dir):
        return False

    df_raw_ccf = _load_latest_ccf_pkl(
        per_ticker_dir,
        ["mds_alpha_raw_spy_ccf_*.pkl", "per_ticker_alpha_raw_spy_ccf_*.pkl"],
    )
    df_pnl_ccf = _load_latest_ccf_pkl(
        per_ticker_dir,
        ["mds_alpha_pnl_spy_ccf_*.pkl", "per_ticker_alpha_pnl_spy_ccf_*.pkl"],
    )

    if df_raw_ccf is None and df_pnl_ccf is None:
        print("[INFO] No CCF PKLs found; skipping CCF pages.")
        return False

    # Metric (1): spy corr bar/line summary using CCF (RAW & PnL)
    _ccf_bar_summary_page(pdf, df_raw_ccf, df_pnl_ccf, max_lag=max_lag)

    # Metrics (2) & (3): distributions of CCF across stocks per lag
    if df_raw_ccf is not None:
        _ccf_hist_pages(pdf, df_raw_ccf,
                        "CCF distributions — RAW alpha vs SPY (per lag)",
                        max_lag=max_lag)
    if df_pnl_ccf is not None:
        _ccf_hist_pages(pdf, df_pnl_ccf,
                        "CCF distributions — PnL vs SPY (per lag)",
                        max_lag=max_lag)

    return True


def generate_quantile_report(config: dict):
    """
    Entry point used by main.py.

    Required keys in `config` (all configured in main.py):
      - daily_dir, summary_dir, per_ticker_dir, outliers_dir, output_pdf
      - qranks, allow_missing_qranks
      - roll_* windows, bar_* settings, outlier_* settings, styles, quantile_colors
      - interval_start, interval_end (for informational/meta only; data is already trimmed upstream)
    """
    global META_TEXT

    daily_dir        = config["daily_dir"]
    summary_dir      = config["summary_dir"]
    per_ticker_dir   = config["per_ticker_dir"]
    outliers_dir     = config["outliers_dir"]
    output_pdf       = config["output_pdf"]

    qranks_requested = [str(q) for q in config.get("qranks", [])][:4]
    allow_missing_q  = bool(config.get("allow_missing_qranks", False))

    roll_h1_lines    = int(config.get("roll_h1_lines", 60))
    roll_h2_lines    = int(config.get("roll_h2_lines", 60))
    roll_h3_lines    = int(config.get("roll_h3_lines", 1))
    roll_nrinstr     = int(config.get("roll_nrinstr", 1))
    roll_ppd         = int(config.get("roll_ppd", 1))
    roll_trades      = int(config.get("roll_trades", 1))
    roll_pnl         = int(config.get("roll_pnl", 1))
    roll_size_notnl  = int(config.get("roll_size_notional", 1))
    roll_sharpe      = int(config.get("roll_sharpe", 60))

    temporal_vars = _normalize_metric_list(config.get("variables_temporal_plot", [])) \
        or ["pnl", "ppd", "n_trades", "sizeNotional", "sharpe"]

    array_dim_cfg = config.get("arrayDim_temporal_plot", (2, 2))
    try:
        temp_rows, temp_cols = int(array_dim_cfg[0]), int(array_dim_cfg[1])
    except Exception:
        temp_rows, temp_cols = 2, 2
    temp_rows = max(1, temp_rows)
    temp_cols = max(1, temp_cols)

    bar_page_vars    = list(config.get("bar_page_vars", []))
    bar_x_vars       = list(config.get("bar_x_vars", []))
    bar_metrics      = _normalize_metric_list(config.get("bar_metrics", []))
    aspect_ratio_barplots = float(config.get("aspect_ratio_barplots", 16 / 9))

    outlier_metrics_for_tables = _normalize_metric_list(config.get("outlier_metrics_for_tables", []))
    outlier_top_k              = int(config.get("outlier_top_k", 3))
    outlier_tables_per_page    = int(config.get("outlier_tables_per_page", 3))

    style_first      = config.get("style_first", "-")
    style_second     = config.get("style_second", ":")
    quantile_colors_cfg = dict(config.get("quantile_colors", {}))

    ccf_enable       = bool(config.get("ccf_enable", True))
    ccf_max_lag      = int(config.get("ccf_max_lag", 5))

    # Load data (with optional date range filtering)
    interval_start = config.get("interval_start")
    interval_end = config.get("interval_end")
    stats_daily, stats_summary, interval_min, interval_max, interval_ndays = _load_data(
        daily_dir, summary_dir, interval_start=interval_start, interval_end=interval_end
    )
    stats_daily = _apply_stat_aliases(stats_daily)
    stats_summary = _apply_stat_aliases(stats_summary)

    # H2/H3: allow user-specified targets/bets (optional; defaults to AUTO/ALL)
    def _safe_resolve(name, desired, series_vals, prefer_prefix=None):
        try:
            return _resolve_fixed(name, desired, series_vals, prefer_prefix=prefer_prefix, top_k=1)
        except Exception:
            return None

    h23_targets = _safe_resolve("H2_targets", config.get("H2_targets", "AUTO"),
                                stats_daily['target'] if 'target' in stats_daily else [])
    h23_bets = _safe_resolve("H2_bets", config.get("H2_bets", "AUTO"),
                             stats_daily['bet_size_col'] if 'bet_size_col' in stats_daily else [],
                             prefer_prefix="betsize")
    h3_targets = _safe_resolve("H3_targets", config.get("H3_targets", "AUTO"),
                               stats_daily['target'] if 'target' in stats_daily else [])
    h3_bets = _safe_resolve("H3_bets", config.get("H3_bets", "AUTO"),
                            stats_daily['bet_size_col'] if 'bet_size_col' in stats_daily else [],
                            prefer_prefix="betsize")
    # If resolution fails or finds nothing, fall back to ALL (None)
    h23_targets = h23_targets or None
    h23_bets = h23_bets or None
    h3_targets = h3_targets or None
    h3_bets = h3_bets or None

    # Meta text (if caller didn't provide, auto-fill using actual data window)
    if config.get("meta_text"):
        META_TEXT = str(config["meta_text"])
    else:
        META_TEXT = f"Window: {interval_min:%Y-%m-%d} → {interval_max:%Y-%m-%d}  |  Days: {interval_ndays}"

    # ---------- SUMMARY ONLY for bar plots ----------
    bars_source = stats_summary if (isinstance(stats_summary, pd.DataFrame) and not stats_summary.empty) else None
    print("[INFO] Bar plots source:", "SUMMARY" if bars_source is not None else "NONE (skipped)")

    # -------------------------------------------------
    # QRanks & colors
    # -------------------------------------------------
    qr_source = (bars_source if bars_source is not None else stats_daily)
    qranks_all = _sorted_qranks(qr_source['qrank'])
    if not qranks_requested:
        qranks = qranks_all
    else:
        if not allow_missing_q:
            missing = [q for q in qranks_requested if q not in qranks_all]
            if missing:
                print(f"[WARN] These qranks not found and will be ignored: {missing}")
        qranks = [q for q in qranks_requested if (allow_missing_q or q in qranks_all)] or qranks_all

    quantile_colors = _ensure_quantile_colors(qranks, quantile_colors_cfg)
    bar_width = 0.18

    # -------------------------------------------------
    # Build datasets
    # -------------------------------------------------
    stats_daily_plot = _exclude_all_rows(stats_daily)

    if bars_source is not None:
        stats_summary_plot = _exclude_all_rows(bars_source)
        plot_signals = sorted(stats_summary_plot['signal'].dropna().unique())
        plot_targets = sorted(stats_summary_plot['target'].dropna().unique())
        plot_bets    = sorted(stats_summary_plot['bet_size_col'].dropna().unique())
    else:
        stats_summary_plot = pd.DataFrame()
        plot_signals = plot_targets = plot_bets = []

    # -------------------------------------------------
    # Alpha autodetection & H1 base stat
    # -------------------------------------------------
    alphas = _autodetect_alphas(stats_daily_plot, max_k=16)
    if len(alphas) < 2:
        print("[WARN] Not enough signals to build heatmaps/lines (need ≥2). Heatmap pages will be skipped.")

    h1_base_stat = 'alpha_sum'  # force Heatmap 1 to use alpha value (sum)
    print(f"[INFO] Heatmap 1 base stat (forced): {h1_base_stat} (time-series Spearman across days)")

    do_temporal = (len(alphas) <= 6)
    if not do_temporal:
        print(f"[INFO] {len(alphas)} alphas detected (>6). Skipping temporal line plots for heatmaps.")

    # =========================
    # ------- BUILD PDF -------
    # =========================
    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        print(f"[INFO] Requested QR: {qranks_requested}")
        print(f"[INFO] Window printed on each page: {META_TEXT}")

        # ---------- Bar Plots (SUMMARY only) ----------
        if bars_source is not None and bar_metrics:
            stats_summary_plot = _exclude_all_rows(bars_source)

            if all(stats_summary_plot[c].notna().any() for c in bar_page_vars) and bar_page_vars:
                page_iter = list(product(*[sorted(stats_summary_plot[col].dropna().unique()) for col in bar_page_vars]))
            else:
                page_iter = []

            for page_vals in page_iter:
                subset = stats_summary_plot.copy()
                title_bits = []
                for var, val in zip(bar_page_vars, page_vals):
                    subset = subset[subset[var] == val]; title_bits.append(f"{var}: {val}")
                if subset.empty:
                    continue

                if bar_x_vars:
                    subset = subset.copy()
                    subset['x_key'] = subset[bar_x_vars].astype(str).agg('|'.join, axis=1)
                    x_levels = sorted(subset['x_key'].unique().tolist())
                else:
                    subset['x_key'] = "ALL"; x_levels = ["ALL"]

                # Arrange metrics in a 2-column grid to improve readability and keep a single legend
                n_cols = 2
                n_rows = int(np.ceil(len(bar_metrics) / n_cols))
                fig_height = max(PAGE_SIZE[1], 3.2 * n_rows + 1.2)
                fig_height = max(PAGE_SIZE[1], 3.6 * n_rows + 2.2)
                fig, axs = plt.subplots(n_rows, n_cols, figsize=(PAGE_SIZE[0], fig_height))
                axs = np.atleast_1d(axs).ravel()
                main_title = "Bar Plots | " + " ".join(title_bits)
                fig.suptitle(main_title, fontsize=16, weight='bold', y=0.97)
                xlabel_descr = " | ".join(bar_x_vars) if bar_x_vars else "ALL"
                fig.text(0.5, 0.92, f"X-axis: {xlabel_descr}", ha="center", va="top", fontsize=10, color="0.35", weight='bold')

                legend_handles = []
                legend_labels = []
                for i, metric in enumerate(bar_metrics):
                    ax = axs[i]
                    metric_display = _metric_label(metric)
                    data = subset[subset['stat_type'] == metric].copy()
                    if data.empty:
                        ax.set_title(f"{metric_display}: no data in summary PKL", fontsize=11)
                        ax.axis('off')
                        continue

                    unit_suffix = ""
                    if metric.lower() == 'ppd':
                        data['value'] = data['value'] * 10000  # bps
                        unit_suffix = " (bps)"
                    elif metric == 'sizeNotional':
                        data['value'] = data['value'] / 1e6    # $M
                        unit_suffix = " ($M)"

                    if 'date' in data.columns:
                        data = data.sort_values('date')

                    if data['qrank'].notna().any():
                        # ---- Per-quantile grouped bars ----
                        keys = ['x_key', 'qrank']
                        data_dedup = data.drop_duplicates(subset=keys, keep='last')

                        try:
                            pivot = data_dedup.pivot(
                                index='x_key',
                                columns='qrank',
                                values='value',
                            )
                        except ValueError:
                            # If there are multiple rows per (x_key, qrank), aggregate first
                            data_dedup = (
                                data_dedup.groupby(keys, as_index=False)['value'].mean()
                            )
                            pivot = data_dedup.pivot(
                                index='x_key',
                                columns='qrank',
                                values='value',
                            )

                        use_q = [q for q in qranks if q in pivot.columns]
                        x = np.arange(len(x_levels))
                        plotted = False

                        if use_q:
                            # Center the bars around each x position
                            q_offsets = (
                                np.arange(-(len(use_q) - 1) / 2, (len(use_q) + 1) / 2)
                                * bar_width
                            )
                            for j, q in enumerate(use_q):
                                vals = (
                                    pd.to_numeric(pivot.get(q), errors='coerce')
                                    .reindex(x_levels)
                                    .astype(float)
                                )
                                if vals.notna().any():
                                    ax.bar(
                                        x + q_offsets[j],
                                        vals.fillna(0.0).values,
                                        width=bar_width,
                                        color=quantile_colors.get(q, 'gray'),
                                        label=q,
                                    )
                                    plotted = True
                                    if q not in legend_labels:
                                        legend_labels.append(q)
                                        legend_handles.append(
                                            plt.Rectangle((0, 0), 1, 1, color=quantile_colors.get(q, 'gray'))
                                        )

                        # drop per-axes titles; ylabel carries the metric name
                    else:
                        # ---- Single bar per x_key (no quantiles) ----
                        data_dedup = data.drop_duplicates(subset=['x_key'], keep='last')
                        vals = (
                            data_dedup.set_index('x_key')['value']
                            .reindex(x_levels)
                            .fillna(0.0)
                            .values
                        )
                        ax.bar(
                            np.arange(len(x_levels)),
                            vals,
                            width=bar_width,
                            color='gray',
                        )

                    # Reference line for ratios
                    if metric in ('long_ratio', 'hit_ratio'):
                        ymin, ymax = ax.get_ylim()
                        if ymin <= 0.5 <= ymax:
                            ax.axhline(
                                y=0.5,
                                color='red',
                                linestyle=':',
                                linewidth=1.5,
                                alpha=0.7,
                                zorder=0,
                            )

                    ax.set_ylabel(f"{metric_display}{unit_suffix}")
                    ax.set_xticks(np.arange(len(x_levels)))
                    ax.set_xticklabels(
                        [str(v) for v in x_levels],
                        rotation=10,
                        ha='center',
                        fontsize=7,
                    )
                    ax.grid(axis='y', linestyle=':', alpha=0.35)
                    ax.margins(y=0.2)
                    ax.tick_params(axis='y', labelsize=8)
                    ax.set_ylabel(f"{metric_display}{unit_suffix}", fontsize=9)

                # Hide any unused subplots
                for j in range(len(bar_metrics), len(axs)):
                    axs[j].axis('off')

                if legend_handles:
                    fig.legend(
                        legend_handles,
                        legend_labels,
                        title='Quantile (color)',
                        bbox_to_anchor=(0.5, 0.905),
                        loc='upper center',
                        ncol=len(legend_handles),
                        fontsize=9,
                        frameon=True,
                    )

                plt.tight_layout(rect=[0.05, 0.10, 0.95, 0.80], h_pad=2.8)
                savefig_white(pdf, fig)

        # ---------- Heatmap 1 (Spearman; raw alpha base) ----------
        stats_daily_nonall = _exclude_all_rows(stats_daily)
        # Use full daily stats (including __ALL__ target/bet) for alpha_sum
        stats_daily_h1 = stats_daily if h1_base_stat == "alpha_sum" else stats_daily_nonall
        if len(alphas) >= 2:
            H1, labels1, n_days1 = compute_heatmap_daily_avg(
                stats_daily_h1,
                alphas,
                stat_type=h1_base_stat,
                min_pairs=2,
                qfilter=None,
                targets=None,
                bets=None,
            )
            k1 = len(labels1) if labels1 else 0
            fig, ax, cax = _centered_heatmap_axes(k1)
            base_desc = (
                "Alpha Cross-Section Corr (Spearman)"
                if h1_base_stat in ("alpha_sum", "alpha_strength")
                else f"Cross-Section Corr (Spearman, {h1_base_stat})"
            )

            if H1 is None or n_days1 == 0:
                ax.axis("off")
                _set_title_fit(
                    fig,
                    ax,
                    f"Heatmap 1 — {base_desc}",
                    base_size=13,
                    pad=8,
                    loc="center",
                )
                ax.text(
                    0.5,
                    0.5,
                    "No sufficient daily cross-sections.",
                    ha="center",
                    va="center",
                )
                if cax is not None:
                    cax.axis("off")
            else:
                _plot_matrix_heatmap(
                    fig,
                    ax,
                    cax,
                    H1,
                    labels1,
                    f"Heatmap 1 — {base_desc} (avg over {n_days1} days)",
                    vmin=-1,
                    vmax=1,
                    annotate_lower=True,
                    fmt=".2f",
                )

            fig.tight_layout(rect=[0.02, 0.06, 0.98, HEATMAP_AX_TOP])
            savefig_white(pdf, fig)

            # H1 temporal (all targets, all bet sizes)
            if do_temporal:
                plot_cross_section_corr_lines(
                    pdf,
                    stats_daily_h1,
                    alphas,
                    stat_type=h1_base_stat,
                    title_prefix="[H1] Alpha vs Alpha",
                    smooth_window=roll_h1_lines,
                    height=6.0,
                    qfilter=None,
                    targets=None,
                    bets=None,
                )

        # ---------- Heatmaps 2 & 3 per-quantile (ALL targets/bets) ----------
        stats_daily_plot_nonall = stats_daily_nonall

        for q in qranks:
            q_masked_df = stats_daily_plot_nonall[
                stats_daily_plot_nonall["qrank"] == q
            ].copy()

            # Heatmap 2 — cross-section corr of P&L, filtered targets/bets
            # Heatmap 2 — time-series Spearman corr of daily PnL (filtered targets/bets)
            H2, labels2, n_days2 = compute_timeseries_heatmap(
                q_masked_df,
                alphas,
                stat_type="pnl",
                min_days=2,
                agg="sum",
                qfilter=[q],
                targets=h23_targets,
                bets=h23_bets,
            )
            h2_ts = True
            k2 = len(labels2) if labels2 else 0
            fig, ax, cax = _centered_heatmap_axes(k2)
            tdesc2 = f"targets={','.join(h23_targets) if h23_targets else 'ALL'}"
            bdesc2 = f"bets={','.join(h23_bets) if h23_bets else 'ALL'}"
            h2_title = (
                f"Heatmap 2 — Time-series Spearman corr of daily PnL "
                f"({tdesc2}, {bdesc2}) [{q}]"
            )

            if H2 is None or n_days2 == 0:
                ax.axis("off")
                _set_title_fit(
                    fig,
                    ax,
                    h2_title,
                    base_size=13,
                    pad=8,
                    loc="center",
                )
                ax.text(
                    0.5,
                    0.5,
                    "No sufficient daily cross-sections for 'pnl'.\n"
                    "Tip: ensure ≥2 rows/day across (signal, target, bet).",
                    ha="center",
                    va="center",
                )
                if cax is not None:
                    cax.axis("off")
            else:
                _plot_matrix_heatmap(
                    fig,
                    ax,
                    cax,
                    H2,
                    labels2,
                    f"{h2_title} (avg over {n_days2} days)",
                    vmin=-1,
                    vmax=1,
                    annotate_lower=True,
                    fmt=".2f",
                )

            fig.tight_layout(rect=[0.02, 0.06, 0.98, HEATMAP_AX_TOP])
            savefig_white(pdf, fig)

            # H2 temporal (all targets/bets)
            if do_temporal and len(alphas) >= 2 and (H2 is not None):
                plot_pairwise_timecorr_lines(
                    pdf,
                    q_masked_df,
                    alphas,
                    stat_type="pnl",
                    title_prefix=f"[H2 | {q}] Alpha vs Alpha — Time corr (P&L, {tdesc2}, {bdesc2})",
                    window=roll_h2_lines,
                    height=6.0,
                    qfilter=[q],
                    targets=h23_targets,
                    bets=h23_bets,
                    agg="sum",
                )

            # Heatmap 3 — time-series corr of summed daily P&L vectors, all targets/bets
            C3, labels3, n_days3 = compute_timeseries_heatmap(
                q_masked_df,
                alphas,
                stat_type="pnl",
                min_days=5,
                agg="sum",
                qfilter=[q],
                targets=h3_targets,
                bets=h3_bets,
            )
            k3 = len(labels3) if labels3 else 0
            fig, ax, cax = _centered_heatmap_axes(k3)
            tdesc3 = f"targets={','.join(h3_targets) if h3_targets else 'ALL'}"
            bdesc3 = f"bets={','.join(h3_bets) if h3_bets else 'ALL'}"
            h3_title = (
                f"Heatmap 3 — Time-series Spearman corr of daily PnL vectors "
                f"({tdesc3}, {bdesc3}) [{q}]"
            )

            if C3 is None or n_days3 < 5:
                ax.axis("off")
                _set_title_fit(
                    fig,
                    ax,
                    h3_title,
                    base_size=13,
                    pad=8,
                    loc="center",
                )
                ax.text(
                    0.5,
                    0.5,
                    "Not enough days (need ≥5) for time-series correlations.",
                    ha="center",
                    va="center",
                )
                if cax is not None:
                    cax.axis("off")
            else:
                _plot_matrix_heatmap(
                    fig,
                    ax,
                    cax,
                    C3,
                    labels3,
                    f"{h3_title} (days={n_days3})",
                    vmin=-1,
                    vmax=1,
                    annotate_lower=True,
                    fmt=".2f",
                )

            fig.tight_layout(rect=[0.02, 0.06, 0.98, HEATMAP_AX_TOP])
            savefig_white(pdf, fig)

            # H3 temporal (all targets/bets)
            if do_temporal and len(alphas) >= 2 and (C3 is not None):
                plot_pairwise_timecorr_lines(
                    pdf,
                    q_masked_df,
                    alphas,
                    stat_type="pnl",
                    title_prefix=f"[H3 | {q}] Alpha vs Alpha — Time corr (P&L vectors, {tdesc3}, {bdesc3})",
                    window=roll_h3_lines,
                    height=6.0,
                    qfilter=[q],
                    targets=h3_targets,
                    bets=h3_bets,
                    agg="sum",
                )

        # ---------- Temporal pages per (target, signal, bet) ----------
        stats_daily_plot_local = stats_daily_plot_nonall

        roll_windows = {
            "pnl": roll_pnl,
            "ppd": roll_ppd,
            "n_trades": roll_trades,
            "sizeNotional": roll_size_notnl,
            "nrInstr": roll_nrinstr,
            "__default__": 1,
        }

        for target in sorted(stats_daily_plot_local["target"].dropna().unique()):
            for signal in sorted(stats_daily_plot_local["signal"].dropna().unique()):
                for bet_strategy in sorted(
                    stats_daily_plot_local["bet_size_col"].dropna().unique()
                ):
                    # Skip explicit combination requested by user
                    if target == "__ALL__":
                        continue
                    mask_base = (
                        (stats_daily_plot_local["target"] == target)
                        & (stats_daily_plot_local["signal"] == signal)
                        & (stats_daily_plot_local["bet_size_col"] == bet_strategy)
                    )
                    sub_all = stats_daily_plot_local[mask_base].copy()
                    if sub_all.empty:
                        continue
                    title_prefix = f"{target} | {signal} | {bet_strategy}"
                    _plot_temporal_grid(
                        pdf,
                        sub_all,
                        qranks,
                        quantile_colors,
                        temporal_vars,
                        roll_windows,
                        roll_sharpe,
                        (temp_rows, temp_cols),
                        title_prefix=title_prefix,
                        style=style_first,
                    )

        # ---------- CCF / correlation distribution pages ----------
        ccf_done = False
        if ccf_enable:
            ccf_done = append_ccf_pages(
                per_ticker_dir,
                pdf,
                max_lag=ccf_max_lag,
            )

        if (not ccf_done) and per_ticker_dir and os.path.isdir(per_ticker_dir):
            # Fallback: distributions of simple SPY correlations if available
            raw_corr_paths = sorted(
                glob.glob(
                    os.path.join(
                        per_ticker_dir, "mds_alpha_raw_spy_corr_*.pkl"
                    )
                )
            )
            pnl_corr_paths = sorted(
                glob.glob(
                    os.path.join(
                        per_ticker_dir, "mds_alpha_pnl_spy_corr_*.pkl"
                    )
                )
            )
            # Backward compat patterns
            if not raw_corr_paths:
                raw_corr_paths = sorted(
                    glob.glob(os.path.join(per_ticker_dir, "per_ticker_alpha_raw_spy_corr_*.pkl"))
                )
            if not pnl_corr_paths:
                pnl_corr_paths = sorted(
                    glob.glob(os.path.join(per_ticker_dir, "per_ticker_alpha_pnl_spy_corr_*.pkl"))
                )

            if raw_corr_paths:
                try:
                    df_raw = read_pickle_compat(raw_corr_paths[-1])
                    _distrib_page(
                        pdf,
                        df_raw,
                        "Per-ticker correlation: RAW alpha vs SPY",
                    )
                except Exception as e:
                    print(f"[WARN] Failed to append RAW corr distrib page: {e}")
            if pnl_corr_paths:
                try:
                    df_pnl = read_pickle_compat(pnl_corr_paths[-1])
                    _distrib_page(
                        pdf,
                        df_pnl,
                        "Per-ticker correlation: PnL vs SPY",
                    )
                except Exception as e:
                    print(f"[WARN] Failed to append PnL corr distrib page: {e}")

        # ---------- Outlier tables ----------
        outliers_pkl = _find_latest_outliers_pkl(outliers_dir)
        if outliers_pkl is not None:
            append_outlier_pages(
                outliers_pkl,
                pdf,
                metrics=outlier_metrics_for_tables,
                top_k=outlier_top_k,
                tables_per_page=outlier_tables_per_page,
            )
        else:
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.axis("off")
            fig.suptitle(
                "Outlier Tables",
                fontsize=18,
                weight="bold",
                y=0.985,
            )
            ax.text(
                0.5,
                0.5,
                "No outlier PKL found; skipping tables.",
                ha="center",
                va="center",
                fontsize=12,
            )
            savefig_white(pdf, fig)

        # ---------- Done ----------
        t1 = time.perf_counter()
        print(
            f"[INFO] Quantile report PDF written to: {output_pdf}"
        )
        print(
            f"[INFO] Total plotting time: {t1 - t0:.2f} seconds"
        )
