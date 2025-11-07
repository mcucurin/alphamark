# plotting/plot_quantile_bars.py
"""
Quantile Report PDF Generator

What this script does
- Loads precomputed DAILY stats (required) and SUMMARY stats (optional) from:
    output/DAILY_STATS/stats_YYYYMMDD.pkl
    output/SUMMARY_STATS/summary_stats_YYYYMMDD_YYYYMMDD.pkl
- Builds a multi-page PDF at: output/Quantile_Combined_Report.pdf

Pages included
1) Bar plots by quantile for selected metrics (from SUMMARY ONLY; if SUMMARY missing, bar pages are skipped).
2) Heatmap 1: average daily cross-section correlation across alphas (Spearman),
   using a base stat (alpha_sum/alpha_strength/pnl depending on availability).
3) Heatmap 2: per-quantile PnL daily cross-section correlation (fixed target & bet) — Spearman.
4) Heatmap 3: per-quantile PnL time-series correlation across days (daily sum by alpha) — Spearman.
5) Temporal pages per (target, signal, bet):
     • Cumulative P&L (left) smoothed by rolling mean if window>1,
       vs nrInstr (right) smoothed by rolling mean if window>1
     • Cumulative PPD (left) smoothed by rolling mean if window>1,
       vs Size Notional (right) smoothed by rolling mean if window>1
     • Number of Trades (smoothed if window>1)
6) Outlier tables (if available).

Input file expectations
Both DAILY and SUMMARY PKLs must contain:
  - date (YYYY-MM-DD), signal, target, qrank (e.g., qr_100), bet_size_col,
    stat_type (e.g., 'pnl','ppd','n_trades','nrInstr','sizeNotional','sharpe','spy_corr',...),
    value (float)
Heatmaps & temporal lines use DAILY stats; bar plots use SUMMARY ONLY (no daily fallback).
"""

import os, glob, warnings, time, pickle as pkl
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from itertools import product, combinations
from contextlib import contextmanager

start = time.perf_counter()
warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# User configuration
# =========================
# Which quantile labels to display.
QR = ['qr_100','qr_75','qr_50','qr_25']
ALLOW_MISSING_QRANKS = False

# Heatmap 2 & 3 filters.
# "AUTO" auto-picks one common target/bet (preferring 'fret_'/'betsize_'), or set explicit lists:
#   H2_TARGETS = ["fret_1D"], H2_BETS = ["betsize_equal"]
H2_TARGETS = "AUTO"
H2_BETS    = "AUTO"
H3_TARGETS = "AUTO"
H3_BETS    = "AUTO"

# Temporal correlation-line smoothing windows (for the *line charts* that accompany heatmaps only).
# Heatmap matrices themselves are not smoothed.
ROLL_H1_LINES = 60
ROLL_H2_LINES = 60
ROLL_H3_LINES = 1

# Temporal rolling windows for plot pages (in trading days).
# PnL & PPD: take the cumulative SUM first, then apply a rolling MEAN over that cumulative curve.
# nrInstr, trades, sizeNotional: rolling MEAN over the raw daily series.
# Set to 1 to show the raw (cum or daily) series with no “rolling” text.
ROLL_NRINSTR        = 1  # rolling mean of 'nrInstr' (daily)
ROLL_PPD            = 1  # rolling mean of cumulative sum of 'ppd'
ROLL_TRADES         = 1  # rolling mean of 'n_trades' (daily)
ROLL_NORMALIZE_PNL  = 1  # rolling mean of cumulative sum of 'pnl'
ROLL_SIZE_NOTIONAL  = 1  # rolling mean of 'sizeNotional' (daily)

# Bar plot setup
BAR_PAGE_VARS = ['signal', 'bet_size_col']  # facets (one page per combination)
BAR_X_VARS    = ['target']                  # x-axis grouping (we print "X-axis: target" under title)

# Metrics expected in bar plots (pulled from SUMMARY ONLY)
metrics_to_plot = [
    'pnl','ppd','sharpe','hit_ratio','long_ratio',
    'nrInstr','sizeNotional','r2','t_stat','n_trades','spy_corr'
]

# Outliers
OUTLIERS_DIR               = "output/OUTLIERS"
OUTLIER_METRICS_FOR_TABLES = ['pnl','ppd','sizeNotional','nrInstr','n_trades']
OUTLIER_TOP_K              = 3
OUTLIER_TABLES_PER_PAGE    = 3

# Styles & colors
STYLE_FIRST  = '-'    # primary line style
STYLE_SECOND = '--'   # secondary line style
quantile_colors = {'qr_100':'red','qr_75':'green','qr_50':'blue','qr_25':'black'}

# Matplotlib theme
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "axes.grid": False,
})

META_TEXT = None  # printed top-right on each page

# --- Page spacing knobs / layout guardrails ---
BAR_TITLE_Y     = 0.985  # suptitle vertical position (bar pages)
BAR_XLABEL_Y    = 0.962  # "X-axis: ..." label vertical position (bar pages)
BAR_AX_TOP      = 0.955  # top bound for tight_layout rect on bar pages
HEATMAP_AX_TOP  = 0.90   # top bound to keep heatmap titles away from META_TEXT
TEMPORAL_AX_TOP = 0.90   # top bound for heatmap temporal line charts (prevents title cutoff)

# -------------------------------------------------
# Utilities
# -------------------------------------------------
def savefig_white(pdf, fig):
    """Save with a white background; leave room at top for META_TEXT."""
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    if META_TEXT:
        fig.text(
            0.99, 0.99, META_TEXT,
            ha="right", va="top", fontsize=10, color="0.35", weight='normal',
        )
    for ax in fig.get_axes():
        ax.set_facecolor("white")
    pdf.savefig(fig, facecolor="white", edgecolor="white")
    plt.close(fig)

def read_pickle_compat(path: str):
    class NPCompatUnpickler(pkl.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core")
            return super().find_class(module, name)
    with open(path, "rb") as f:
        return NPCompatUnpickler(f).load()

def _load_data():
    daily_dir   = "output/DAILY_STATS"
    summary_dir = "output/SUMMARY_STATS"

    if not os.path.isdir(daily_dir):
        raise FileNotFoundError("Expected 'output/DAILY_STATS' with stats_*.pkl files.")
    daily_paths = sorted(glob.glob(os.path.join(daily_dir, "stats_*.pkl")))
    if not daily_paths:
        raise FileNotFoundError("No daily files found in 'output/DAILY_STATS'.")

    daily_frames = []
    for p in daily_paths:
        try:
            df = read_pickle_compat(p)
        except Exception:
            continue
        if isinstance(df, pd.DataFrame) and not df.empty and {'date','value'}.issubset(df.columns):
            daily_frames.append(df)
    if not daily_frames:
        raise FileNotFoundError("All daily PKLs were empty or malformed.")
    stats_daily = pd.concat(daily_frames, ignore_index=True)

    stats_daily['date'] = pd.to_datetime(stats_daily['date'], errors='coerce')
    stats_daily = stats_daily.dropna(subset=['date'])
    dmin = stats_daily['date'].min()
    dmax = stats_daily['date'].max()
    ndays = int(stats_daily['date'].nunique())

    stats_summary = pd.DataFrame()
    if os.path.isdir(summary_dir):
        pkl_paths = sorted(glob.glob(os.path.join(summary_dir, "summary_stats_*.pkl")))
        if pkl_paths:
            try:
                stats_summary = read_pickle_compat(pkl_paths[-1])
                print(f"[INFO] Loaded summary PKL: {os.path.basename(pkl_paths[-1])}  shape={stats_summary.shape}")
            except Exception as e:
                print(f"[WARN] Failed to read summary PKL ({pkl_paths[-1]}): {e}")
        else:
            print("[WARN] No summary PKL found in output/SUMMARY_STATS (bar plots will be skipped).")
    else:
        print("[WARN] output/SUMMARY_STATS directory not found (bar plots will be skipped).")

    for df in (stats_daily, stats_summary):
        if df is None or df.empty: continue
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if 'value' in df.columns:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
        for col in ['signal','target','bet_size_col','qrank','stat_type']:
            if col not in df.columns: df[col] = pd.NA
            df[col] = df[col].astype('string')

    print(f"[INFO] DAILY date window: {dmin:%Y-%m-%d} → {dmax:%Y-%m-%d}  ({ndays} days)")
    return stats_daily, stats_summary, dmin, dmax, ndays

STATS_DAILY, STATS_SUMMARY, INTERVAL_MIN, INTERVAL_MAX, INTERVAL_NDAYS = _load_data()
META_TEXT = f"Window: {INTERVAL_MIN:%Y-%m-%d} → {INTERVAL_MAX:%Y-%m-%d}  |  Days: {INTERVAL_NDAYS}"

# ---------- IMPORTANT: SUMMARY ONLY for bar plots (no DAILY fallback) ----------
if isinstance(STATS_SUMMARY, pd.DataFrame) and not STATS_SUMMARY.empty:
    BARS_SOURCE = STATS_SUMMARY
    print("[INFO] Bar plots source: SUMMARY")
else:
    BARS_SOURCE = None
    print("[INFO] No SUMMARY PKL -> Bar plots disabled (daily fallback is OFF)")

# -------------------------------------------------
# Helpers for labels/quantiles/colors
# -------------------------------------------------
def _sorted_qranks(series):
    vals = [q for q in series.dropna().unique()]
    try:
        return sorted(vals, key=lambda x: float(str(x).split('_')[1]) if '_' in str(x) else str(x))
    except Exception:
        return sorted(vals)

def _ensure_quantile_colors(labels, base_map):
    cmap = mpl.colormaps.get_cmap('tab20')
    out = dict(base_map)
    i = 0
    for lab in labels:
        if lab not in out:
            out[lab] = cmap(i % cmap.N); i += 1
    return out

# Bar pages use SUMMARY if present; otherwise derive from DAILY for QR labels
qranks_all = _sorted_qranks((BARS_SOURCE if BARS_SOURCE is not None else STATS_DAILY)['qrank'])

if isinstance(QR, (list, tuple)):
    requested = [str(q) for q in QR][:4]
else:
    requested = []
if not requested:
    qranks = qranks_all
else:
    if not ALLOW_MISSING_QRANKS:
        missing = [q for q in requested if q not in qranks_all]
        if missing:
            print(f"[WARN] These qranks not found in selected source and will be ignored: {missing}")
    qranks = [q for q in requested if (ALLOW_MISSING_QRANKS or q in qranks_all)]
    if not qranks:
        print("[WARN] Requested qranks absent; falling back to available.")
        qranks = qranks_all

quantile_colors = _ensure_quantile_colors(qranks, quantile_colors)
bar_width = 0.15

# -------------------------------------------------
# Plotting helpers
# -------------------------------------------------
def _plot_date_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

def _ellipsis(s, n):
    s = "" if s is None else str(s)
    return s if len(s)<=n else s[:n-1]+"…"

def _heatmap_figure_size(k, widen=1.18, extra_height=0.0):
    s = max(10, min(28, 0.6 * k + 8))
    return (s * widen, s + extra_height)

@contextmanager
def std_err():
    with np.errstate(invalid='ignore', divide='ignore'):
        yield

# ----- title helpers (NO TRUNCATION; optional wrapping) -----
def _compose_filters_str(qfilter=None, targets=None, bets=None, short_keys=False):
    parts = []
    if qfilter: parts.append(f"{'qr' if short_keys else 'qranks'}={','.join(map(str, qfilter if isinstance(qfilter,(list,tuple,set)) else [qfilter]))}")
    if targets: parts.append(f"{'tgt' if short_keys else 'target'}={','.join(map(str, targets if isinstance(targets,(list,tuple,set)) else [targets]))}")
    if bets:    parts.append(f"bet={','.join(map(str, bets if isinstance(bets,(list,tuple,set)) else [bets]))}")
    return " | ".join(parts)

def _measure_text_width(fig, text, fontsize):
    # Create a transient text to measure width in pixels
    t = fig.text(0, 0, text, fontsize=fontsize)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w = t.get_window_extent(renderer=renderer).width
    t.remove()
    return w

def _wrap_text_greedy(fig, ax, text, fontsize, max_lines=2, max_frac=0.98):
    """
    Greedily wrap `text` into <= max_lines so that each line width <= max_frac * axis width.
    Returns list of lines.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(renderer=renderer)
    max_w = max_frac * ax_bb.width

    words = str(text).split()
    lines = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if _measure_text_width(fig, trial, fontsize) <= max_w:
            cur = trial
        else:
            if cur: lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    # If we exceeded max_lines, we'll shrink font elsewhere; here just return lines
    return lines

def _set_title_fit(fig, ax, text, base_size=14, min_size=8, pad=10, loc='center', allow_wrap=False, max_lines=2):
    """
    Fit title without truncation. Strategy:
    1) Try single line, shrinking font down to min_size.
    2) If still too wide and allow_wrap=True, wrap into <= max_lines with greedy wrapping,
       and (if necessary) shrink font so each line fits within the axes box.
    Never inserts '…' and keeps the title centered, avoiding left/right bleed.
    """
    text = " ".join(str(text).split())
    # Step 1: single-line attempt
    size = int(base_size)
    while size >= min_size:
        t = ax.set_title(text, fontsize=size, weight='bold', pad=pad, loc=loc)
        t.set_ha('center'); t.set_x(0.5)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_bb = ax.get_window_extent(renderer=renderer)
        t_bb  = t.get_window_extent(renderer=renderer)
        if (t_bb.width <= 0.98 * ax_bb.width) and (t_bb.x0 >= ax_bb.x0) and (t_bb.x1 <= ax_bb.x1):
            return t  # fits on one line
        size -= 1

    # Step 2: wrapping if allowed
    if allow_wrap:
        for size in range(min_size, min_size-1, -1):  # lock at min_size for measurement stability
            lines = _wrap_text_greedy(fig, ax, text, size, max_lines=max_lines, max_frac=0.98)
            # If too many lines, try bumping max_lines to 3 as a last resort
            if len(lines) > max_lines:
                lines = _wrap_text_greedy(fig, ax, text, size, max_lines=3, max_frac=0.98)
            # Ensure each line fits; if not, we accept smaller implicit scaling via Matplotlib layout
            t = ax.set_title("\n".join(lines), fontsize=size, weight='bold', pad=pad, loc=loc)
            t.set_ha('center'); t.set_x(0.5)
            fig.canvas.draw()
            return t

    # Fallback: set at min_size single-line (may slightly overflow in extreme cases, but no truncation)
    t = ax.set_title(text, fontsize=min_size, weight='bold', pad=pad, loc=loc)
    t.set_ha('center'); t.set_x(0.5)
    fig.canvas.draw()
    return t

# ----- centered heatmap layout -----
def _centered_heatmap_axes(k):
    """
    Center the heatmap with symmetric margins and a dedicated colorbar axis.
    Returns (fig, ax, cax).
    """
    figsize = _heatmap_figure_size(k)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        nrows=1, ncols=2, figure=fig,
        left=0.10, right=0.90, bottom=0.10, top=HEATMAP_AX_TOP,  # symmetric to avoid left bleed
        width_ratios=[20, 1], wspace=0.15
    )
    ax  = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    return fig, ax, cax

def _plot_matrix_heatmap(fig, ax, cax, M, labels, title, vmin=-1, vmax=1, annotate_lower=True, fmt=".2f"):
    if M is None or labels is None or len(labels) == 0:
        ax.axis('off'); _set_title_fit(fig, ax, title, base_size=13, pad=8, loc='center', allow_wrap=True, max_lines=3)
        if cax is not None: cax.axis('off')
        ax.text(0.5,0.5,"No data",ha='center',va='center')
        return
    k = len(labels)
    fs_labels = 9 if k <= 18 else (7 if k <= 30 else 6)
    fs_cells  = 8 if k <= 18 else (6 if k <= 30 else 5)

    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap='coolwarm', aspect='equal')
    _set_title_fit(fig, ax, title, base_size=13, pad=10, loc='center', allow_wrap=True, max_lines=2)
    ax.set_xticks(range(k)); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=fs_labels)
    ax.set_yticks(range(k)); ax.set_yticklabels(labels, fontsize=fs_labels)
    ax.set_xlim(-0.5, k-0.5); ax.set_ylim(k-0.5, -0.5)
    ax.set_xticks(np.arange(-.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-.5, k, 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.8)
    for spine in ax.spines.values(): spine.set_visible(False)

    if annotate_lower:
        for i in range(k):
            for j in range(i+1):
                val = M[i, j]
                if np.isfinite(val):
                    ax.text(j, i, format(val, fmt),
                            ha='center', va='center',
                            fontsize=fs_cells,
                            color=('white' if abs(val) >= 0.5 else 'black'))
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=9)

def _minp(window, floor=3):
    if window is None:
        return 1
    w = int(max(1, window))
    return min(w, max(1, w//5, floor))

def _roll_mean(s: pd.Series, window: int):
    mp = _minp(window, floor=3)
    return s.rolling(window, min_periods=mp).mean()

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

def _ensure_min_cross_section(df_q, targ_sel, bet_sel, auto_targ, auto_bet, min_rows=2):
    t = list(targ_sel)
    b = list(bet_sel)
    if len(t)*len(b) >= min_rows:
        return t, b, None
    avail_t = df_q['target'].dropna().astype(str).value_counts().index.tolist()
    avail_b = df_q['bet_size_col'].dropna().astype(str).value_counts().index.tolist()
    msg = None
    if auto_targ and auto_bet:
        if len(avail_t) >= 2:
            t = avail_t[:2]; b = avail_b[:1] if avail_b else b
        elif len(avail_b) >= 2:
            t = avail_t[:1] if avail_t else t; b = avail_b[:2]
        if len(t)*len(b) < min_rows:
            msg = "[WARN] Cross-section: only one (target, bet) combination exists; cross-section corr unavailable."
        return t, b, msg
    if auto_targ and not auto_bet:
        if len(avail_t) >= 2: t = avail_t[:2]
        elif len(avail_t) >= 1: t = avail_t[:1]
        if len(t)*len(b) < min_rows:
            msg = "[WARN] Cross-section: insufficient rows with fixed bet(s); consider adding more targets."
        return t, b, msg
    if not auto_targ and auto_bet:
        if len(avail_b) >= 2: b = avail_b[:2]
        elif len(avail_b) >= 1: b = avail_b[:1]
        if len(t)*len(b) < min_rows:
            msg = "[WARN] Cross-section: insufficient rows with fixed target(s); consider adding more bets."
        return t, b, msg
    msg = "[INFO] Cross-section: explicit filters yield <2 rows; cannot compute cross-section correlation."
    return t, b, msg

def _exclude_all_rows(df: pd.DataFrame) -> pd.DataFrame:
    m = (
        df['target'].ne('__ALL__') &
        df['bet_size_col'].ne('__ALL__') &
        df['qrank'].ne('__ALL__')
    )
    return df[m].copy()

# -------- Spearman helpers (no constant-input warnings) --------
def _spearman_corr_df(X: pd.DataFrame, min_periods: int = 2) -> pd.DataFrame:
    R = X.rank(axis=0, method='average', na_option='keep')
    return R.corr(method='pearson', min_periods=min_periods)

def _spearman_corr_pair(x: pd.Series, y: pd.Series, min_periods: int = 2) -> float:
    m = x.notna() & y.notna()
    if m.sum() < min_periods:
        return np.nan
    xr = x[m].rank()
    yr = y[m].rank()
    if xr.nunique() <= 1 or yr.nunique() <= 1:
        return np.nan
    return float(np.corrcoef(xr.values, yr.values)[0, 1])

# -------- Barplot helpers for 0.5 reference tick alignment --------
def _ensure_half_tick_and_line(ax):
    ymin, ymax = ax.get_ylim()
    if ymin <= 0.5 <= ymax:
        ax.axhline(y=0.5, color='red', linestyle=':', linewidth=1.5, alpha=0.7, zorder=0)
        ticks = ax.get_yticks()
        if not np.any(np.isclose(ticks, 0.5)):
            ax.set_yticks(np.sort(np.append(ticks, 0.5)))

# -------------------------------------------------
# Build datasets
# -------------------------------------------------
stats_daily_plot = _exclude_all_rows(STATS_DAILY)

# For bar plots (SUMMARY only)
if BARS_SOURCE is not None:
    stats_summary_plot = _exclude_all_rows(BARS_SOURCE)
    plot_signals   = sorted(stats_summary_plot['signal'].dropna().unique())
    plot_targets   = sorted(stats_summary_plot['target'].dropna().unique())
    plot_bets      = sorted(stats_summary_plot['bet_size_col'].dropna().unique())
    PLOT_LEVELS    = {'signal': plot_signals, 'target': plot_targets, 'bet_size_col': plot_bets}
else:
    stats_summary_plot = pd.DataFrame()
    plot_signals = plot_targets = plot_bets = []
    PLOT_LEVELS  = {'signal': [], 'target': [], 'bet_size_col': []}

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
    piv = d.pivot_table(index=['target','bet_size_col','qrank'],
                        columns='signal', values='value', aggfunc='sum')
    cols_present = [a for a in alphas if a in piv.columns]
    if not cols_present:
        return None
    piv = piv[cols_present].dropna(how='all')
    if piv.shape[0] < 2:
        return None
    return pd.to_numeric(piv.stack(), errors='coerce').unstack().astype(float)

def _avg_mats_ignore_nan(mats):
    if not mats: return None
    k = mats[0].shape[0]
    sumM = np.zeros((k,k), float); cntM = np.zeros((k,k), int)
    for M in mats:
        if M is None or M.shape != (k,k): continue
        m = np.isfinite(M)
        sumM[m] += M[m]; cntM[m] += 1
    with std_err():
        H = sumM / np.where(cntM==0, np.nan, cntM)
    H[cntM==0] = np.nan
    for i in range(k):
        if not np.isfinite(H[i,i]): H[i,i] = 1.0
    return H

def compute_heatmap_daily_avg(stats_df, alphas, stat_type, min_pairs=2,
                              qfilter=None, targets=None, bets=None):
    alphas = list(alphas)
    if len(alphas) < 2: return None, alphas, 0
    mats = []
    for _, df_day in stats_df.groupby('date'):
        X = _build_daily_cross_section(df_day, alphas, stat_type,
                                       qfilter=qfilter, targets=targets, bets=bets)
        if X is None: continue
        C = _spearman_corr_df(X, min_periods=min_pairs)
        C = C.reindex(index=alphas, columns=alphas)
        mats.append(C.to_numpy(dtype=float))
    if not mats: return None, alphas, 0
    H = _avg_mats_ignore_nan(mats)
    return H, alphas, len(mats)

def compute_timeseries_heatmap(stats_df, alphas, stat_type, min_days=5, agg='sum',
                               qfilter=None, targets=None, bets=None):
    df = stats_df[stats_df['stat_type'] == stat_type].copy()
    if qfilter: df = df[df['qrank'].isin(qfilter)]
    if targets: df = df[df['target'].isin(targets)]
    if bets:    df = df[df['bet_size_col'].isin(bets)]
    if df.empty: return None, alphas, 0

    gb = df.groupby(['date','signal'])['value']
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
    alphas = [a for a in alphas]
    pairs = list(combinations(alphas, 2))
    dates = sorted(stats_df['date'].dropna().unique())
    out = {f"{a}|{b}": pd.Series(index=pd.DatetimeIndex(dates, name='date', dtype='datetime64[ns]'),
                                 dtype='float64') for a, b in pairs}

    for dt, df_day in stats_df.groupby('date'):
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
    s1 = a.copy()
    s2 = b.copy()
    idx = s1.index.union(s2.index)
    s1 = s1.reindex(idx)
    s2 = s2.reindex(idx)
    out = pd.Series(index=idx, dtype='float64')

    if (window is None) or int(window) <= 1:
        for i in range(len(idx)):
            x = s1.iloc[:i+1]
            y = s2.iloc[:i+1]
            m = x.notna() & y.notna()
            if m.sum() >= min_periods:
                xr = x[m].rank()
                yr = y[m].rank()
                if xr.nunique() > 1 and yr.nunique() > 1:
                    out.iloc[i] = np.corrcoef(xr, yr)[0,1]
                else:
                    out.iloc[i] = np.nan
            else:
                out.iloc[i] = np.nan
        return out

    w = int(window)
    for i in range(len(idx)):
        start = max(0, i - w + 1)
        x = s1.iloc[start:i+1]
        y = s2.iloc[start:i+1]
        m = x.notna() & y.notna()
        if m.sum() >= min_periods:
            xr = x[m].rank()
            yr = y[m].rank()
            if xr.nunique() > 1 and yr.nunique() > 1:
                out.iloc[i] = np.corrcoef(xr, yr)[0,1]
            else:
                out.iloc[i] = np.nan
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

    gb = df.groupby(['date','signal'])['value']
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
    mp = _minp(win, floor=3) if min_periods is None else int(min_periods)
    cols = [c for c in daily.columns if daily[c].notna().sum() >= mp]
    pairs = list(combinations(cols, 2))
    out = {}
    for a, b in pairs:
        out[f"{a}|{b}"] = _rolling_spearman_pair(daily[a], daily[b], window=win, min_periods=mp)
    return out

def plot_cross_section_corr_lines(pdf, stats_df, alphas, stat_type, title_prefix,
                                  smooth_window=1, height=6.0,
                                  qfilter=None, targets=None, bets=None):
    corr_map = compute_daily_pair_corr_series(
        stats_df, alphas, stat_type, min_pairs=2,
        qfilter=qfilter, targets=targets, bets=bets
    )
    filt_str = _compose_filters_str(qfilter=qfilter, targets=targets, bets=bets, short_keys=True)
    if (smooth_window is None) or int(smooth_window) <= 1:
        title_text = f"{title_prefix} — daily (no smoothing){(' — ' + filt_str) if filt_str else ''}"
    else:
        title_text = f"{title_prefix} — smoothed {int(smooth_window)}D mean{(' — ' + filt_str) if filt_str else ''}"

    if not corr_map:
        fig, ax = plt.subplots(figsize=(14, height))
        _set_title_fit(fig, ax, title_text + " — no data", base_size=14, pad=8, loc='center', allow_wrap=False)
        fig.tight_layout(rect=[0.02, 0.06, 0.98, TEMPORAL_AX_TOP])
        savefig_white(pdf, fig); return

    coverage = [(k, v.notna().sum()) for k, v in corr_map.items()]
    coverage.sort(key=lambda x: x[1], reverse=True)
    chosen = [k for k,cnt in coverage[:8] if cnt > 0]

    dates_all = sorted(stats_df['date'].dropna().unique())
    if len(dates_all) == 0 or not chosen:
        fig, ax = plt.subplots(figsize=(14, height))
        _set_title_fit(fig, ax, title_text + " — insufficient", base_size=14, pad=8, loc='center', allow_wrap=False)
        fig.tight_layout(rect=[0.02, 0.06, 0.98, TEMPORAL_AX_TOP])
        savefig_white(pdf, fig); return

    fig, ax = plt.subplots(figsize=(14, height))
    _set_title_fit(fig, ax, title_text, base_size=14, pad=10, loc='center', allow_wrap=False)

    cmap = mpl.colormaps.get_cmap('tab20')
    x_min, x_max = pd.to_datetime(dates_all[0]), pd.to_datetime(dates_all[-1])
    all_vals = []

    for i, key in enumerate(chosen):
        s = corr_map[key].copy().sort_index()
        if (smooth_window is not None) and int(smooth_window) > 1:
            win = int(smooth_window)
            mp = _minp(win, floor=3)
            s = s.rolling(win, min_periods=mp).mean()
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
            mid = 0.5*(y0 + y1); y0, y1 = mid + (-0.1), mid + 0.1
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
    filt_str = _compose_filters_str(qfilter=qfilter, targets=targets, bets=bets, short_keys=True)
    if (window is None) or int(window) <= 1:
        title_text = f"{title_prefix} — Expanding Spearman{(' — ' + filt_str) if filt_str else ''}"
    else:
        title_text = f"{title_prefix} — Rolling Spearman {int(window)}D{(' — ' + filt_str) if filt_str else ''}"

    if not corr_map:
        fig, ax = plt.subplots(figsize=(14, height))
        _set_title_fit(fig, ax, title_text + " — no data", base_size=14, pad=8, loc='center', allow_wrap=False)
        fig.tight_layout(rect=[0.02, 0.06, 0.98, TEMPORAL_AX_TOP])
        savefig_white(pdf, fig); return

    coverage = [(k, v.notna().sum()) for k, v in corr_map.items()]
    coverage.sort(key=lambda x: x[1], reverse=True)
    chosen = [k for k,cnt in coverage[:8] if cnt > 0]

    dates_all = sorted(stats_df['date'].dropna().unique())
    if len(dates_all) == 0 or not chosen:
        fig, ax = plt.subplots(figsize=(14, height))
        _set_title_fit(fig, ax, title_text + " — insufficient", base_size=14, pad=8, loc='center', allow_wrap=False)
        fig.tight_layout(rect=[0.02, 0.06, 0.98, TEMPORAL_AX_TOP])
        savefig_white(pdf, fig); return

    fig, ax = plt.subplots(figsize=(14, height))
    _set_title_fit(fig, ax, title_text, base_size=14, pad=10, loc='center', allow_wrap=False)

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
            mid = 0.5*(y0 + y1); y0, y1 = mid - 0.1, mid + 0.1
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
    base = df[df['stat_type']=='pnl']
    day_counts = (base.groupby('signal')['date'].nunique()).sort_values(ascending=False)
    if day_counts.empty:
        day_counts = (df.groupby('signal')['date'].nunique()).sort_values(ascending=False)
    candidates = day_counts[day_counts >= 5].index.tolist() or day_counts.index.tolist()
    return candidates[:max_k]

ALPHAS = _autodetect_alphas(stats_daily_plot, max_k=16)
if len(ALPHAS) < 2:
    print("[WARN] Not enough signals in DAILY to build heatmaps/lines (need ≥2). Heatmap pages will be skipped.")

available_stats_daily = set(stats_daily_plot['stat_type'].dropna().unique())
H1_BASE_STAT = 'alpha_sum' if 'alpha_sum' in available_stats_daily else ('alpha_strength' if 'alpha_strength' in available_stats_daily else 'pnl')
print(f"[INFO] Heatmap 1 base stat: {H1_BASE_STAT}")

DO_TEMPORAL = (len(ALPHAS) <= 6)
if not DO_TEMPORAL:
    print(f"[INFO] {len(ALPHAS)} alphas detected (>6). Skipping temporal line plots for heatmaps.")

# -------------------------------------------------
# Outlier tables
# -------------------------------------------------
def _find_latest_outliers_pkl(root=OUTLIERS_DIR):
    if not os.path.isdir(root): return None
    cand = sorted(glob.glob(os.path.join(root, "outliers_*.pkl")))
    return cand[-1] if cand else None

def _metric_table_rows(odf, metric, top_k, have_z, have_rule):
    sub = odf[odf['stat_type']==metric].copy()
    if sub.empty: return None, None
    sub = sub.sort_values('value')
    lows  = sub.head(top_k).copy()
    highs = sub.tail(top_k).iloc[::-1].copy()  # fixed typo
    labels_tbl = ["Type","Date","Signal","Bet","Target","Q","Value"] + (["z"] if have_z else []) + (["Rule"] if have_rule else [])
    def row(r, kind):
        base = [
            kind,
            r['date'].strftime('%Y-%m-%d') if pd.notna(r['date']) else "NaT",
            _ellipsis(r['signal'],18), _ellipsis(r['bet_size_col'],16),
            _ellipsis(r['target'],16), str(r['qrank']), f"{r['value']:.6g}",
        ]
        if have_z:    base.append("" if pd.isna(r.get('z')) else f"{r.get('z'):.2f}")
        if have_rule: base.append(_ellipsis(r.get('rule',''),18))
        return base
    rows = [row(r,"High") for _,r in highs.iterrows()] + [row(r,"Low") for _,r in lows.iterrows()]
    return labels_tbl, rows

def _draw_table_in_axis(ax, title, col_labels, rows, fontsize=9):
    ax.axis('off'); ax.set_title(title, fontsize=13, weight='bold', loc='left', pad=6)
    base_w = {"Type":0.08,"Date":0.11,"Signal":0.18,"Bet":0.14,"Target":0.14,"Q":0.06,"Value":0.11,"z":0.06,"Rule":0.12}
    colWidths = [base_w.get(lbl,0.10) for lbl in col_labels]
    s = sum(colWidths)
    if s>0.98: colWidths = [w*0.98/s for w in colWidths]
    tb = ax.table(cellText=rows, colLabels=col_labels, colWidths=colWidths,
                  loc='upper left', cellLoc='left', bbox=[0.0, 0.0, 1.0, 0.92])
    tb.auto_set_font_size(False); tb.set_fontsize(fontsize); tb.scale(1.0, 1.15)
    header_color=(0.9,0.9,0.92); even=(0.98,0.98,0.985); odd=(1.0,1.0,1.0)
    for (r,c), cell in tb.get_celld().items():
        if r==0: cell.set_text_props(weight='bold', ha='left'); cell.set_facecolor(header_color); cell.set_edgecolor('0.75')
        else:    cell.set_edgecolor('0.85'); cell.set_facecolor(even if r%2==0 else odd)

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
    odf['date'] = pd.to_datetime(odf['date'], errors='coerce')
    odf = odf.dropna(subset=['date', 'value'])
    have_rule = 'rule' in odf.columns
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

    date_min = odf['date'].min()
    date_max = odf['date'].max()
    subtitle = f"Window: {date_min:%Y-%m-%d} → {date_max:%Y-%m-%d} | Metrics: {', '.join(metrics)} | Top-K: {top_k}"

    tables = []
    for m in metrics:
        col_labels, rows = _metric_table_rows(odf, m, top_k=top_k, have_z=have_z, have_rule=have_rule)
        if col_labels is None or not rows: continue
        tables.append((m, col_labels, rows))
    if not tables: return

    per_page = max(1, int(tables_per_page))
    for page_idx in range(0, len(tables), per_page):
        chunk = tables[page_idx:page_idx+per_page]
        fig = plt.figure(figsize=(14, 8.5))
        fig.suptitle("Outlier Tables", fontsize=18, weight='bold', y=0.985)
        if page_idx == 0:
            fig.text(0.03, 0.955, subtitle, ha='left', va='top', fontsize=11, color='0.25')
        gs = GridSpec(nrows=len(chunk), ncols=1, figure=fig, left=0.03, right=0.97, top=0.90, bottom=0.06, hspace=0.35)
        for row_idx, (metric_name, col_labels, rows) in enumerate(chunk):
            ax = fig.add_subplot(gs[row_idx, 0])
            _draw_table_in_axis(ax, title=f"{metric_name} — Top {top_k} Highs & Lows",
                                col_labels=col_labels, rows=rows, fontsize=9)
        savefig_white(pdf, fig)

# =========================
# ------- BUILD PDF -------
# =========================
def _series(df, stat):
    s = df[df['stat_type']==stat][['date','value']].set_index('date')['value'].astype(float)
    s = s.sort_index()
    return s

def _title_token(base: str, window: int, cumulative: bool = False) -> str:
    if cumulative:
        base = f"cumulative {base}"
        Base = f"Cumulative {base.split(' ',1)[1].capitalize()}"
        if window and int(window) > 1:
            return f"Rolling-mean {base} ({int(window)}D)"
        else:
            return Base
    else:
        if window and int(window) > 1:
            return f"Rolling-mean {base} ({int(window)}D)"
        else:
            return base

os.makedirs("output", exist_ok=True)
with PdfPages("output/Quantile_Combined_Report.pdf") as pdf:

    print(f"[INFO] Requested QR: {QR}")
    print(f"[INFO] Window printed on each page: {META_TEXT}")

    # ---------- Bar Plots (SUMMARY only; no DAILY fallback) ----------
    if BARS_SOURCE is None:
        print("[INFO] Skipping Bar Plots: SUMMARY not found.")
    else:
        stats_summary_plot = _exclude_all_rows(BARS_SOURCE)
        if all(stats_summary_plot[level_col].notna().any() for level_col in BAR_PAGE_VARS):
            page_iter = list(product(*[sorted(stats_summary_plot[col].dropna().unique()) for col in BAR_PAGE_VARS]))
        else:
            page_iter = []

        for page_vals in page_iter:
            subset = stats_summary_plot.copy()
            title_bits = []
            for var, val in zip(BAR_PAGE_VARS, page_vals):
                subset = subset[subset[var]==val]; title_bits.append(f"{var}: {val}")
            if subset.empty:
                continue

            if BAR_X_VARS:
                subset = subset.copy()
                subset['x_key'] = subset[BAR_X_VARS].astype(str).agg('|'.join, axis=1)
                x_levels = sorted(subset['x_key'].unique().tolist())
            else:
                subset['x_key'] = "ALL"; x_levels = ["ALL"]

            fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(14, 2.7*len(metrics_to_plot)))
            axs = np.atleast_1d(axs)
            main_title = "Bar Plots | " + " | ".join(title_bits)

            fig.suptitle(main_title, fontsize=18, weight='bold', y=BAR_TITLE_Y)
            xlabel_descr = " | ".join(BAR_X_VARS) if BAR_X_VARS else "ALL"
            fig.text(0.5, BAR_XLABEL_Y, f"X-axis: {xlabel_descr}",
                     ha="center", va="top", fontsize=13, color="0.35", weight='bold')

            for i, metric in enumerate(metrics_to_plot):
                ax = axs[i]
                data = subset[subset['stat_type']==metric].copy()
                if data.empty:
                    ax.set_title(f"{metric}: no data in summary PKL"); ax.axis('off'); continue

                unit_suffix = ""
                if metric.lower() == 'ppd':
                    data['value'] = data['value'] * 10000  # bps
                    unit_suffix = " (bpts)"
                elif metric == 'sizeNotional':
                    data['value'] = data['value'] / 1e6    # $M
                    unit_suffix = " ($M)"

                if 'date' in data.columns:
                    data = data.sort_values('date')

                if data['qrank'].notna().any():
                    keys = ['x_key', 'qrank']
                    data_dedup = data.drop_duplicates(subset=keys, keep='last')
                    try:
                        pivot = data_dedup.pivot(index='x_key', columns='qrank', values='value')
                    except ValueError:
                        data_dedup = (data_dedup.groupby(keys, as_index=False).first())
                        pivot = data_dedup.pivot(index='x_key', columns='qrank', values='value')

                    use_q = [q for q in qranks if q in pivot.columns]
                    x = np.arange(len(x_levels))
                    plotted = False
                    if use_q:
                        q_offsets = np.arange(-(len(use_q)-1)/2, (len(use_q)+1)/2) * bar_width
                        for j, q in enumerate(use_q):
                            vals = pd.to_numeric(pivot.get(q), errors='coerce').reindex(x_levels).astype(float)
                            if vals.notna().any():
                                ax.bar(x + q_offsets[j], vals.fillna(0.0).values,
                                       width=bar_width, color=quantile_colors.get(q,'gray'), label=q)
                                plotted = True
                    if plotted:
                        ax.legend(title='Quantile (color)', bbox_to_anchor=(1.02, 0.98),
                                  loc='upper left', fontsize=9)
                else:
                    data_dedup = data.drop_duplicates(subset=['x_key'], keep='last')
                    vals = (data_dedup.set_index('x_key')['value']
                            .reindex(x_levels)
                            .fillna(0.0)
                            .values)
                    ax.bar(np.arange(len(x_levels)), vals, width=bar_width, color='gray')

                if metric in ('long_ratio','hit_ratio'):
                    _ensure_half_tick_and_line(ax)

                ax.set_ylabel(f"{metric}{unit_suffix}")
                ax.set_xticks(np.arange(len(x_levels)))
                ax.set_xticklabels([str(v) for v in x_levels], rotation=45, ha='right', fontsize=11)
                ax.grid(axis='y', linestyle='--', alpha=0.35)

            plt.tight_layout(rect=[0.02, 0.04, 0.98, BAR_AX_TOP])
            savefig_white(pdf, fig)

    # ---------- Heatmap 1 (Spearman) ----------
    if len(ALPHAS) >= 2:
        H1, labels1, n_days1 = compute_heatmap_daily_avg(
            stats_daily_plot, ALPHAS, stat_type=H1_BASE_STAT, min_pairs=2, qfilter=qranks
        )
        k1 = len(labels1) if labels1 else 0
        fig, ax, cax = _centered_heatmap_axes(k1)
        base_desc = ("Alpha Cross-Section Corr (Spearman)"
                     if H1_BASE_STAT in ('alpha_sum', 'alpha_strength')
                     else f"Cross-Section Corr (Spearman, {H1_BASE_STAT})")
        if H1 is None or n_days1 == 0:
            ax.axis('off')
            _set_title_fit(fig, ax, f"Heatmap 1 — {base_desc}", base_size=13, pad=8, loc='center', allow_wrap=True, max_lines=2)
            ax.text(0.5,0.5,"No sufficient daily cross-sections.\nTip: include ≥2 quantiles in QR.", ha='center', va='center')
            if cax is not None: cax.axis('off')
        else:
            _plot_matrix_heatmap(
                fig, ax, cax, H1, labels1,
                f"Heatmap 1 — {base_desc}",
                vmin=-1, vmax=1, annotate_lower=True, fmt=".2f"
            )
        fig.tight_layout(rect=[0.02, 0.06, 0.98, HEATMAP_AX_TOP])
        savefig_white(pdf, fig)

        if DO_TEMPORAL:
            plot_cross_section_corr_lines(
                pdf, stats_daily_plot, ALPHAS, stat_type=H1_BASE_STAT,
                title_prefix=f"[H1] Alpha vs Alpha",
                smooth_window=int(ROLL_H1_LINES),
                height=6.0, qfilter=qranks
            )

    # ---------- Heatmaps 2 & 3 per-quantile (Spearman) ----------
    H2_TARGETS_RES = _resolve_fixed("H2_TARGETS", H2_TARGETS, stats_daily_plot['target'], prefer_prefix="fret_", top_k=1)
    H2_BETS_RES    = _resolve_fixed("H2_BETS",    H2_BETS,    stats_daily_plot['bet_size_col'], prefer_prefix="betsize_", top_k=1)
    H3_TARGETS_RES = _resolve_fixed("H3_TARGETS", H3_TARGETS, stats_daily_plot['target'], prefer_prefix="fret_", top_k=1)
    H3_BETS_RES    = _resolve_fixed("H3_BETS",    H3_BETS,    stats_daily_plot['bet_size_col'], prefer_prefix="betsize_", top_k=1)

    H2_AUTO_TARGETS = isinstance(H2_TARGETS, str) and H2_TARGETS.upper() == "AUTO"
    H2_AUTO_BETS    = isinstance(H2_BETS,    str) and H2_BETS.upper()    == "AUTO"

    for q in qranks:
        q_masked_df = stats_daily_plot[stats_daily_plot['qrank'] == q].copy()

        # Heatmap 2 — cross-section corr of P&L, fixed (target, bet)
        df_q_pnl = q_masked_df[q_masked_df['stat_type']=='pnl']
        t2, b2, msg2 = _ensure_min_cross_section(
            df_q_pnl, H2_TARGETS_RES, H2_BETS_RES, H2_AUTO_TARGETS, H2_AUTO_BETS, min_rows=2
        )
        if msg2: print(msg2)
        H2, labels2, n_days2 = compute_heatmap_daily_avg(
            q_masked_df, ALPHAS, stat_type='pnl', min_pairs=2,
            qfilter=[q], targets=t2, bets=b2
        )
        k2 = len(labels2) if labels2 else 0
        fig, ax, cax = _centered_heatmap_axes(k2)
        tdesc2 = f"target={', '.join(t2)}"
        bdesc2 = f"bet={', '.join(b2)}"
        if H2 is None or n_days2 == 0:
            ax.axis('off')
            _set_title_fit(fig, ax, f"Heatmap 2 — PnL Cross-Section Corr (Spearman, daily avg) [{q}] [{tdesc2} | {bdesc2}]",
                           base_size=13, pad=8, loc='center', allow_wrap=True, max_lines=2)
            ax.text(0.5,0.5,"No sufficient daily cross-sections for 'pnl' with fixed filters.", ha='center', va='center')
            if cax is not None: cax.axis('off')
        else:
            _plot_matrix_heatmap(
                fig, ax, cax, H2, labels2,
                f"Heatmap 2 — Cross-section corr of daily PnLs (Spearman) [{q}] [{tdesc2} | {bdesc2}]",
                vmin=-1, vmax=1, annotate_lower=True, fmt=".2f"
            )
        fig.tight_layout(rect=[0.02, 0.06, 0.98, HEATMAP_AX_TOP])
        savefig_white(pdf, fig)

        if DO_TEMPORAL and len(ALPHAS) >= 2:
            plot_cross_section_corr_lines(
                pdf, q_masked_df, ALPHAS, stat_type='pnl',
                title_prefix=f"[H2 | {q}] PnL vs PnL",
                smooth_window=int(ROLL_H2_LINES),
                height=6.0, qfilter=[q], targets=t2, bets=b2
            )

        # Heatmap 3 — time-series corr of summed daily P&L across days, fixed (target, bet)
        C3, labels3, n_days3 = compute_timeseries_heatmap(
            q_masked_df, ALPHAS, stat_type='pnl', min_days=5, agg='sum',
            qfilter=[q], targets=H3_TARGETS_RES, bets=H3_BETS_RES
        )
        k3 = len(labels3) if labels3 else 0
        fig, ax, cax = _centered_heatmap_axes(k3)
        tdesc3 = f"target={', '.join(H3_TARGETS_RES)}"
        bdesc3 = f"bet={', '.join(H3_BETS_RES)}"
        if C3 is None or n_days3 < 5:
            ax.axis('off')
            _set_title_fit(fig, ax, f"Heatmap 3 — PnL Time-Series Corr (Spearman) [{q}] [{tdesc3} | {bdesc3}]",
                           base_size=13, pad=8, loc='center', allow_wrap=True, max_lines=2)
            ax.text(0.5,0.5,"Not enough days (need ≥5) for time-series correlations.", ha='center', va='center')
            if cax is not None: cax.axis('off')
        else:
            _plot_matrix_heatmap(
                fig, ax, cax, C3, labels3,
                f"Heatmap 3 — corr of daily summed P&L across days (Spearman) [{q}] [{tdesc3} | {bdesc3}]",
                vmin=-1, vmax=1, annotate_lower=True, fmt=".2f"
            )
        fig.tight_layout(rect=[0.02, 0.06, 0.98, HEATMAP_AX_TOP])
        savefig_white(pdf, fig)

        if DO_TEMPORAL and len(ALPHAS) >= 2:
            plot_pairwise_timecorr_lines(
                pdf, q_masked_df, ALPHAS, stat_type='pnl',
                title_prefix=f"[H3 | {q}] Alpha vs Alpha — Time correlation (P&L)",
                window=int(ROLL_H3_LINES), height=6.0,
                qfilter=[q], targets=H3_TARGETS_RES, bets=H3_BETS_RES, agg='sum'
            )

    # ---------- Temporal pages per (target, signal, bet) ----------
    for target in sorted(stats_daily_plot['target'].dropna().unique()):
        for signal in sorted(stats_daily_plot['signal'].dropna().unique()):
            for bet_strategy in sorted(stats_daily_plot['bet_size_col'].dropna().unique()):
                mask_base = (
                    (stats_daily_plot['target']==target) &
                    (stats_daily_plot['signal']==signal) &
                    (stats_daily_plot['bet_size_col']==bet_strategy)
                )
                sub_all = stats_daily_plot[mask_base].copy()
                if sub_all.empty:
                    continue

                # ---- PnL (cum) vs nrInstr (daily)
                left_title  = _title_token("P&L", ROLL_NORMALIZE_PNL, cumulative=True)
                right_title = _title_token("nrInstr", ROLL_NRINSTR, cumulative=False)

                fig, ax = plt.subplots(figsize=(14, 6.0))
                right_ax = ax.twinx(); right_ax.grid(False)
                fig.suptitle(
                    f"{target} | {signal} | {bet_strategy}\n"
                    f"{left_title} (left, solid) vs {right_title} (right, dashed)",
                    fontsize=16, weight='bold', y=0.96
                )

                anyL=anyR=False
                for q in qranks:
                    sq = sub_all[sub_all['qrank']==q]
                    if sq.empty: continue
                    color = quantile_colors.get(q,'gray')

                    pnl    = _series(sq, 'pnl')
                    nrin   = _series(sq, 'nrInstr')

                    cum_pnl = pnl.cumsum()
                    yL = _roll_mean(cum_pnl, ROLL_NORMALIZE_PNL)
                    ax.plot(yL.index, yL.values, color=color, linestyle=STYLE_FIRST, linewidth=1.8); anyL=True

                    yR = _roll_mean(nrin, ROLL_NRINSTR)
                    right_ax.plot(yR.index, yR.values, color=color, linestyle=STYLE_SECOND, linewidth=1.8); anyR=True

                if anyL: ax.set_ylabel(left_title)
                if anyR: right_ax.set_ylabel(right_title)
                _plot_date_axis(ax)
                styles=[]
                if anyL: styles.append((f"{left_title} (left, solid)", STYLE_FIRST))
                if anyR: styles.append((f"{right_title} (right, dashed)", STYLE_SECOND))
                if styles:
                    legL = [Line2D([0],[0], color='gray', lw=2, linestyle=ls, label=lab) for (lab,ls) in styles]
                    leg_styles = ax.legend(handles=legL, loc='upper left', fontsize=9, frameon=True,
                                          bbox_to_anchor=(0, 0.98))
                    ax.add_artist(leg_styles)
                if qranks:
                    handles = [Line2D([0],[0], color=quantile_colors.get(str(q), 'gray'), lw=2, label=str(q)) for q in qranks]
                    ax.legend(handles=handles, title="Quantiles", loc="upper right", fontsize=9, frameon=True,
                             bbox_to_anchor=(1, 0.98))
                plt.tight_layout(rect=[0, 0.03, 1, 0.90]); savefig_white(pdf, fig)

                # ---- PPD (cum) vs Size Notional (daily)
                left_title_base  = _title_token("PPD", ROLL_PPD, cumulative=True)
                right_title_base = _title_token("Size Notional", ROLL_SIZE_NOTIONAL, cumulative=False)
                left_title  = f"{left_title_base} (bpts)"
                right_title = f"{right_title_base} ($M)"

                fig, ax = plt.subplots(figsize=(14, 6.0))
                right_ax = ax.twinx(); right_ax.grid(False)
                fig.suptitle(
                    f"{target} | {signal} | {bet_strategy}\n"
                    f"{left_title} (left, solid) vs {right_title} (right, dashed)",
                    fontsize=16, weight='bold', y=0.96
                )

                anyL=anyR=False
                for q in qranks:
                    sq = sub_all[sub_all['qrank']==q]
                    if sq.empty: continue
                    color = quantile_colors.get(q,'gray')

                    pnl    = _series(sq, 'pnl')
                    notnl  = _series(sq, 'sizeNotional')

                    if not pnl.empty and not notnl.empty:
                        merged = pd.DataFrame({'pnl': pnl, 'notional': notnl}).fillna(0.0)
                        cum_pnl = merged['pnl'].cumsum()
                        cum_notnl = merged['notional'].cumsum()
                        cum_ppd = pd.Series(
                            np.where(cum_notnl > 0, cum_pnl / cum_notnl, np.nan),
                            index=cum_pnl.index
                        ).ffill()
                        cum_ppd_bpts = cum_ppd * 10000
                        yL = _roll_mean(cum_ppd_bpts, ROLL_PPD)
                        ax.plot(yL.index, yL.values, color=color, linestyle=STYLE_FIRST, linewidth=1.8, alpha=0.9)
                        anyL = True

                    if not notnl.empty:
                        notnl_millions = notnl / 1e6
                        yR = _roll_mean(notnl_millions, ROLL_SIZE_NOTIONAL)
                        right_ax.plot(yR.index, yR.values, color=color, linestyle=STYLE_SECOND, linewidth=1.8, alpha=0.9)
                        anyR = True

                if anyL: ax.set_ylabel(left_title)
                if anyR: right_ax.set_ylabel(right_title)
                _plot_date_axis(ax)
                styles=[]
                if anyL: styles.append((f"{left_title} (left, solid)", STYLE_FIRST))
                if anyR: styles.append((f"{right_title} (right, dashed)", STYLE_SECOND))
                if styles:
                    legL = [Line2D([0],[0], color='gray', lw=2, linestyle=ls, label=lab) for (lab,ls) in styles]
                    leg_styles = ax.legend(handles=legL, loc='upper left', fontsize=9, frameon=True,
                                          bbox_to_anchor=(0, 0.98))
                    ax.add_artist(leg_styles)
                if qranks:
                    handles = [Line2D([0],[0], color=quantile_colors.get(str(q), 'gray'), lw=2, label=str(q)) for q in qranks]
                    ax.legend(handles=handles, title="Quantiles", loc="upper right", fontsize=9, frameon=True,
                             bbox_to_anchor=(1, 0.98))
                plt.tight_layout(rect=[0, 0.03, 1, 0.90]); savefig_white(pdf, fig)

                # ---- Number of Trades (daily)
                title_trades = _title_token("Trades", ROLL_TRADES, cumulative=False)
                fig, ax = plt.subplots(figsize=(14, 5.5))
                fig.suptitle(f"{target} | {signal} | {bet_strategy}\n{title_trades}",
                             fontsize=16, weight='bold', y=0.96)
                anyPlot=False
                for q in qranks:
                    sq = sub_all[sub_all['qrank']==q]
                    if sq.empty: continue
                    color = quantile_colors.get(q,'gray')
                    trades = _series(sq, 'n_trades')
                    y = _roll_mean(trades, ROLL_TRADES)
                    ax.plot(y.index, y.values, color=color, linestyle=STYLE_FIRST, linewidth=1.8); anyPlot=True
                if anyPlot: ax.set_ylabel(title_trades)
                _plot_date_axis(ax)
                if qranks:
                    handles = [Line2D([0],[0], color=quantile_colors.get(str(q), 'gray'), lw=2, label=str(q)) for q in qranks]
                    ax.legend(handles=handles, title="Quantiles", loc="upper right", fontsize=9, frameon=True,
                             bbox_to_anchor=(1, 0.98))
                plt.tight_layout(rect=[0, 0.03, 1, 0.90]); savefig_white(pdf, fig)

    # ---------- Outlier tables ----------
    latest_outliers = _find_latest_outliers_pkl(OUTLIERS_DIR)
    append_outlier_pages(latest_outliers, pdf,
                         metrics=OUTLIER_METRICS_FOR_TABLES,
                         top_k=OUTLIER_TOP_K,
                         tables_per_page=OUTLIER_TABLES_PER_PAGE)

print("✅ Saved to output/Quantile_Combined_Report.pdf")
end = time.perf_counter()
print(f"Time taken: {end - start} seconds")
