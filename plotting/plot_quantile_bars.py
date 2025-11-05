# plotting/plot_quantile_bars.py
# Full report generator with ALL plots:
#  - Bar plots (from SUMMARY PKL; falls back to DAILY only if PKL missing)
#  - Heatmap 1: alpha-vs-alpha daily cross-section corr (avg over days)
#  - Heatmap 2: per-quantile PnL cross-section corr (fixed target/bet)
#  - Heatmap 3: per-quantile PnL time-series corr across days
#  - Temporal line plots for H1/H2/H3 (optional if few alphas)
#  - Bottom pages: PnL vs nrInstr, PPD vs Notional, PPT vs Trades
#  - Outlier tables
#
# Every page prints the overall window (earliest date, latest date, #days) at the top-right.

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from itertools import product, combinations
import pickle as pkl
import time

start = time.perf_counter()
warnings.filterwarnings("ignore", category=UserWarning)

# =============================
# Matplotlib: clean white pages
# =============================
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "axes.grid": False,
})

META_TEXT = None  # set after loading (top-right stamp on each page)

def savefig_white(pdf, fig):
    """Save current figure with white background and a top-right meta stamp."""
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    # stamp top-right
    if META_TEXT:
        fig.text(0.985, 0.985, META_TEXT, ha="right", va="top", fontsize=10, color="0.3")
    for ax in fig.get_axes():
        ax.set_facecolor("white")
    pdf.savefig(fig, facecolor="white", edgecolor="white")
    plt.close(fig)


# =========================
# ---- USER CONFIG --------
# =========================
# Which quantiles to show (labels must match what's written by summary, e.g., qr_100, qr_75, ...)
QR = ['qr_100','qr_75','qr_50','qr_25']
ALLOW_MISSING_QRANKS = False

# Fixed filters for Heatmap 2 and Heatmap 3 (we still use DAILY for heatmaps/time series)
H2_TARGETS = "AUTO"
H2_BETS    = "AUTO"
H3_TARGETS = "AUTO"
H3_BETS    = "AUTO"

# Rolling windows for temporal plots
ROLL_H1_LINES = 60
ROLL_H2_LINES = 60
ROLL_H3_LINES = 1

# Bottom pages rolling windows:
ROLL_PNL_NRINSTR  = 1
ROLL_PPD_NOTIONAL = 1
ROLL_PPT_TRADES   = 1
NORMALIZE_LEFT_WITH_ROLL = False

# What to show as bar pages (x-axis buckets are in BAR_X_VARS)
BAR_PAGE_VARS = ['signal', 'bet_size_col']
BAR_X_VARS    = ['target']

# Metrics expected in SUMMARY_STATS (bars will read from summary PKL)
metrics_to_plot = [
    'pnl','ppd','sharpe','hit_ratio','long_ratio',
    'nrInstr','sizeNotional','r2','t_stat','n_trades','ppt'
]

# Cumulative/paired sections for bottom pages (still use DAILY)
CUM_SECTIONS = ['pnl','ppd','sizeNotional','nrInstr','n_trades','ppt']

OUTLIERS_DIR               = "output/OUTLIERS"
OUTLIER_METRICS_FOR_TABLES = ['pnl','ppd','ppt','sizeNotional','nrInstr','n_trades']
OUTLIER_TOP_K              = 3
OUTLIER_TABLES_PER_PAGE    = 3

# Colors/styles
STYLE_FIRST  = '-'
STYLE_SECOND = '--'
quantile_colors = {'qr_100':'red','qr_75':'green','qr_50':'blue','qr_25':'black'}


# ===================================
# NumPy 1.x/2.x compatible unpickler
# ===================================
def read_pickle_compat(path: str):
    """Unpickle objects across NumPy 1.x/2.x by remapping numpy._core -> numpy.core."""
    class NPCompatUnpickler(pkl.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core")
            return super().find_class(module, name)
    with open(path, "rb") as f:
        return NPCompatUnpickler(f).load()


# =========================
# ----- LOAD & PREP -------
# =========================
def _load_data():
    """Load DAILY PKLs (required) and SUMMARY (PKL only).
       Returns: stats_daily, stats_summary, (dmin, dmax, ndays)
    """
    daily_dir   = "output/DAILY_STATS"
    summary_dir = "output/SUMMARY_STATS"

    # --- DAILY (required)
    if not os.path.isdir(daily_dir):
        raise FileNotFoundError("Expected 'output/DAILY_STATS' with stats_*.pkl files.")
    daily_paths = sorted(glob.glob(os.path.join(daily_dir, "stats_*.pkl")))
    if not daily_paths:
        raise FileNotFoundError("No daily files found in 'output/DAILY_STATS'.")

    daily_frames = []
    for p in daily_paths:
        try:
            df = read_pickle_compat(p)  # NumPy-2 safe
        except Exception:
            continue
        if isinstance(df, pd.DataFrame) and not df.empty and {'date','value'}.issubset(df.columns):
            daily_frames.append(df)
    if not daily_frames:
        raise FileNotFoundError("All daily PKLs were empty or malformed.")
    stats_daily = pd.concat(daily_frames, ignore_index=True)

    # date range from DAILY (not from summary's single last_date stamp)
    stats_daily['date'] = pd.to_datetime(stats_daily['date'], errors='coerce')
    stats_daily = stats_daily.dropna(subset=['date'])
    dmin = stats_daily['date'].min()
    dmax = stats_daily['date'].max()
    ndays = int(stats_daily['date'].nunique())

    # --- SUMMARY (PKL only)
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
            print("[WARN] No summary PKL found in output/SUMMARY_STATS (bar plots will fall back to DAILY).")
    else:
        print("[WARN] output/SUMMARY_STATS directory not found (bar plots will fall back to DAILY).")

    # Clean dtypes
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

# Which dataset feeds which plot:
BARS_SOURCE = STATS_SUMMARY if (isinstance(STATS_SUMMARY, pd.DataFrame) and not STATS_SUMMARY.empty) else STATS_DAILY
print(f"[INFO] Bar plots source: {'SUMMARY' if BARS_SOURCE is STATS_SUMMARY else 'DAILY'}")

# ---------- Discover levels ----------
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

signals_all   = sorted([s for s in BARS_SOURCE['signal'].dropna().unique()])
targets_all   = sorted([t for t in BARS_SOURCE['target'].dropna().unique()])
bet_sizes_all = sorted([b for b in BARS_SOURCE['bet_size_col'].dropna().unique()])
qranks_all    = _sorted_qranks(BARS_SOURCE['qrank'])

if isinstance(QR, (list, tuple)):
    selected = [str(q) for q in QR][:4]
else:
    selected = []
if not selected:
    qranks = qranks_all
else:
    if not ALLOW_MISSING_QRANKS:
        missing = [q for q in selected if q not in qranks_all]
        if missing:
            print(f"[WARN] These qranks not found in summary and will be ignored: {missing}")
    qranks = [q for q in selected if (ALLOW_MISSING_QRANKS or q in qranks_all)]
    if not qranks:
        print("[WARN] Requested qranks absent; falling back to available.")
        qranks = qranks_all

quantile_colors = _ensure_quantile_colors(qranks, quantile_colors)
bar_width = 0.15

# =========================
# ---- HELPERS ------------
# =========================
def _plot_date_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

def _ellipsis(s, n): s = "" if s is None else str(s); return s if len(s)<=n else s[:n-1]+"…"

def _heatmap_figure_size(k, widen=1.18, extra_height=0.0):
    s = max(10, min(28, 0.6 * k + 8))
    return (s * widen, s + extra_height)

def _plot_matrix_heatmap(fig, ax, M, labels, title, vmin=-1, vmax=1, annotate_lower=True, fmt=".2f"):
    if M is None or labels is None or len(labels) == 0:
        ax.axis('off'); ax.set_title(title, fontsize=13, weight='bold', wrap=True)
        ax.text(0.5,0.5,"No data",ha='center',va='center'); return
    k = len(labels)
    fs_labels = 9 if k <= 18 else (7 if k <= 30 else 6)
    fs_cells  = 8 if k <= 18 else (6 if k <= 30 else 5)

    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap='coolwarm', aspect='equal')
    ax.set_title(title, fontsize=14, weight='bold', pad=12, wrap=True)
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
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.4/10)
    cbar.ax.tick_params(labelsize=9)

def _minp(window, floor=3):
    if window is None:
        return 1
    w = int(max(1, window))
    return min(w, max(1, w//5, floor))

def _roll_mean(s: pd.Series, window: int):
    mp = _minp(window, floor=3)
    return s.rolling(window, min_periods=mp).mean()

def _add_quantile_legend(ax, qranks, qcolors, loc="upper right", title="Quantiles"):
    """
    Add a small legend mapping quantile labels to their colors.
    Keeps any existing legend(s) by returning a new one rather than replacing.
    """
    if not qranks:
        return None
    handles = [Line2D([0],[0], color=qcolors.get(str(q), 'gray'), lw=2, label=str(q)) for q in qranks]
    leg = ax.legend(handles=handles, title=title, loc=loc, fontsize=9, frameon=True)
    return leg

# ------------- Fixed filters helpers -------------
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

# flags for AUTO
H2_AUTO_TARGETS = isinstance(H2_TARGETS, str) and H2_TARGETS.upper() == "AUTO"
H2_AUTO_BETS    = isinstance(H2_BETS,    str) and H2_BETS.upper()    == "AUTO"
H3_AUTO_TARGETS = isinstance(H3_TARGETS, str) and H3_TARGETS.upper() == "AUTO"
H3_AUTO_BETS    = isinstance(H3_BETS,    str) and H3_BETS.upper()    == "AUTO"


# =========================
# ---- DATASETS for PLOTS -
# =========================
def _exclude_all_rows(df: pd.DataFrame) -> pd.DataFrame:
    m = (
        df['target'].ne('__ALL__') &
        df['bet_size_col'].ne('__ALL__') &
        df['qrank'].ne('__ALL__')
    )
    return df[m].copy()

# Summary-backed frame for BAR pages:
stats_summary_plot = _exclude_all_rows(BARS_SOURCE)

plot_signals   = sorted(stats_summary_plot['signal'].dropna().unique())
plot_targets   = sorted(stats_summary_plot['target'].dropna().unique())
plot_bets      = sorted(stats_summary_plot['bet_size_col'].dropna().unique())
PLOT_LEVELS    = {'signal': plot_signals, 'target': plot_targets, 'bet_size_col': plot_bets}

# DAILY-backed frame for heatmaps/time lines/bottom pages:
stats_daily_plot = _exclude_all_rows(STATS_DAILY)

# =========================
# ---- HEATMAP BUILDERS ---
# =========================
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
    with np.errstate(invalid='ignore', divide='ignore'):
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
        C = X.corr(method='pearson', min_periods=min_pairs)
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

    C = daily.corr(method='pearson', min_periods=min_days)
    return C.values, C.columns.tolist(), int(daily.shape[0])

# =========================
# -- TEMPORAL LINES -------
# =========================
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
            m = xa.notna() & xb.notna()
            if m.sum() >= min_pairs and xa[m].std(ddof=0) > 0 and xb[m].std(ddof=0) > 0:
                r = np.corrcoef(xa[m].values, xb[m].values)[0,1]
                out[f"{a}|{b}"].loc[dt] = float(r)
            else:
                out[f"{a}|{b}"].loc[dt] = np.nan
    return out

def plot_cross_section_corr_lines(pdf, stats_df, alphas, stat_type, title_prefix,
                                  smooth_window=60, height=6.0,
                                  qfilter=None, targets=None, bets=None):
    corr_map = compute_daily_pair_corr_series(
        stats_df, alphas, stat_type, min_pairs=2,
        qfilter=qfilter, targets=targets, bets=bets
    )
    if not corr_map:
        fig, ax = plt.subplots(figsize=(17, height))
        ax.axis('off'); ax.set_title(f"{title_prefix} — Daily Cross-Section Corr (no data)", fontsize=14, weight='bold')
        savefig_white(pdf, fig); return

    coverage = [(k, v.notna().sum()) for k, v in corr_map.items()]
    coverage.sort(key=lambda x: x[1], reverse=True)
    chosen = [k for k,cnt in coverage[:8] if cnt > 0]

    dates_all = sorted(stats_df['date'].dropna().unique())
    if len(dates_all) == 0 or not chosen:
        fig, ax = plt.subplots(figsize=(17, height))
        ax.axis('off'); ax.set_title(f"{title_prefix} — Daily Cross-Section Corr (insufficient)", fontsize=14, weight='bold')
        savefig_white(pdf, fig); return

    fig, ax = plt.subplots(figsize=(17.5, height))
    fil_str = []
    if qfilter: fil_str.append(f"qranks={', '.join(qfilter)}")
    if targets: fil_str.append(f"target={', '.join(targets)}")
    if bets:    fil_str.append(f"bet={', '.join(bets)}")
    suffix = (" | " + " | ".join(fil_str)) if fil_str else ""

    if (smooth_window is None) or int(smooth_window) <= 1:
        ax.set_title(f"{title_prefix} — daily (no smoothing){suffix}",
                     fontsize=15, weight='bold', pad=10)
        apply_smoothing = False
    else:
        ax.set_title(f"{title_prefix} — (smoothed {int(smooth_window)}D mean){suffix}",
                     fontsize=15, weight='bold', pad=10)
        apply_smoothing = True

    cmap = mpl.colormaps.get_cmap('tab20')
    x_min, x_max = pd.to_datetime(dates_all[0]), pd.to_datetime(dates_all[-1])
    all_vals = []

    for i, key in enumerate(chosen):
        s = corr_map[key].copy().sort_index()
        if apply_smoothing:
            win = int(smooth_window)
            mp = _minp(win, floor=3)
            sm = s.rolling(win, min_periods=mp).mean()
        else:
            sm = s
        color = cmap(i % cmap.N)
        ax.plot(sm.index, sm.values, lw=1.8, alpha=0.95, color=color)
        v = sm.values
        if np.isfinite(v).any():
            all_vals.append(v[np.isfinite(v)])
        finite_idx = np.where(np.isfinite(sm.values))[0]
        if finite_idx.size:
            j = finite_idx[-1]
            ax.text(sm.index[j], sm.values[j], f"  {key}", color=color, fontsize=9, va='center')

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

    ax.set_ylabel("Pearson corr")
    ax.set_xlim(x_min, x_max)
    ax.margins(x=0.03)
    _plot_date_axis(ax)
    savefig_white(pdf, fig)

def compute_pairwise_rolling_time_corr(stats_df, alphas, stat_type, window=60, min_periods=None,
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

    out = {}
    if (window is None) or int(window) <= 1:
        mp = 2
        cols = [c for c in daily.columns if daily[c].notna().sum() >= mp]
        pairs = list(combinations(cols, 2))
        for a, b in pairs:
            s = daily[a].expanding(min_periods=mp).corr(daily[b])
            out[f"{a}|{b}"] = s
        return out

    win = int(window)
    if min_periods is None:
        min_periods = _minp(win, floor=3)
    cols = [c for c in daily.columns if daily[c].notna().sum() >= min_periods]
    pairs = list(combinations(cols, 2))
    for a, b in pairs:
        s = daily[a].rolling(win, min_periods=min_periods).corr(daily[b])
        out[f"{a}|{b}"] = s
    return out

def plot_pairwise_timecorr_lines(pdf, stats_df, alphas, stat_type, title_prefix,
                                 window=60, height=6.0, qfilter=None, targets=None, bets=None, agg='mean'):
    corr_map = compute_pairwise_rolling_time_corr(
        stats_df, alphas, stat_type, window=window, qfilter=qfilter,
        targets=targets, bets=bets, agg=agg
    )
    if not corr_map:
        fig, ax = plt.subplots(figsize=(17, height))
        lbl = "Expanding corr" if (window is None or int(window) <= 1) else f"Rolling time corr ({int(window)}D)"
        ax.axis('off'); ax.set_title(f"{title_prefix} — {lbl} (no data)", fontsize=14, weight='bold')
        savefig_white(pdf, fig); return

    coverage = [(k, v.notna().sum()) for k, v in corr_map.items()]
    coverage.sort(key=lambda x: x[1], reverse=True)
    chosen = [k for k,cnt in coverage[:8] if cnt > 0]

    dates_all = sorted(stats_df['date'].dropna().unique())
    if len(dates_all) == 0 or not chosen:
        fig, ax = plt.subplots(figsize=(17, height))
        lbl = "Expanding corr" if (window is None or int(window) <= 1) else f"Rolling time corr ({int(window)}D)"
        ax.axis('off'); ax.set_title(f"{title_prefix} — {lbl} (insufficient)", fontsize=14, weight='bold')
        savefig_white(pdf, fig); return

    fig, ax = plt.subplots(figsize=(17.5, height))
    fil_str = []
    if qfilter: fil_str.append(f"qranks={', '.join(qfilter)}")
    if targets: fil_str.append(f"target={', '.join(targets)}")
    if bets:    fil_str.append(f"bet={', '.join(bets)}")
    suffix = (" | " + " | ".join(fil_str)) if fil_str else ""
    if (window is None) or int(window) <= 1:
        ax.set_title(f"{title_prefix} — Expanding corr{suffix}", fontsize=15, weight='bold', pad=10)
    else:
        ax.set_title(f"{title_prefix} — Rolling time corr ({int(window)}D){suffix}", fontsize=15, weight='bold', pad=10)

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

    ax.set_ylabel("Pearson corr (time)")
    ax.set_xlim(x_min, x_max)
    ax.margins(x=0.03)
    _plot_date_axis(ax)
    savefig_white(pdf, fig)


# =========================
# --- ALPHA AUTODETECT ----
# =========================
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


# =========================
# ---- OUTLIER TABLES -----
# =========================
def _find_latest_outliers_pkl(root=OUTLIERS_DIR):
    if not os.path.isdir(root): return None
    cand = sorted(glob.glob(os.path.join(root, "outliers_*.pkl")))
    return cand[-1] if cand else None

def _metric_table_rows(odf, metric, top_k, have_z, have_rule):
    sub = odf[odf['stat_type']==metric].copy()
    if sub.empty: return None, None
    sub = sub.sort_values('value')
    lows  = sub.head(top_k).copy()
    highs = sub.tail(top_k).iloc[::-1].copy()
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
os.makedirs("output", exist_ok=True)
with PdfPages("output/Quantile_Combined_Report.pdf") as pdf:

    print(f"[INFO] Requested QR: {QR}")
    print(f"[INFO] Using qranks in bars: {', '.join(qranks) or '(none)'}")
    print(f"[INFO] Bar metrics: {', '.join(metrics_to_plot)}")
    print(f"[INFO] Window printed on each page: {META_TEXT}")

    # ---------- Bar Plots from SUMMARY PKL (fallback to DAILY only if PKL missing) ----------
    if all(PLOT_LEVELS[v] for v in BAR_PAGE_VARS):
        page_iter = list(product(*[PLOT_LEVELS[v] for v in BAR_PAGE_VARS]))
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
        fig.suptitle("Bar Plots (SUMMARY PKL) | " + " | ".join(title_bits), fontsize=18, weight='bold')

        for i, metric in enumerate(metrics_to_plot):
            ax = axs[i]
            data = subset[subset['stat_type']==metric].copy()
            if data.empty:
                ax.set_title(f"{metric}: no data in summary PKL"); ax.axis('off'); continue

            # pivot by qrank over x_key; summary already aggregated, so we just display values
            if data['qrank'].notna().any():
                pivot = data.pivot_table(index='x_key', columns='qrank', values='value',
                                         aggfunc='mean', fill_value=np.nan)
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
                    ax.legend(title='Quantile (color)', bbox_to_anchor=(1.01,1), loc='upper left', fontsize=9)
            else:
                vals = (data.groupby('x_key')['value'].mean()).reindex(x_levels).fillna(0.0).values
                ax.bar(np.arange(len(x_levels)), vals, width=bar_width, color='gray')

            ax.set_ylabel(metric)
            ax.set_xticks(np.arange(len(x_levels)))
            ax.set_xticklabels([str(v) for v in x_levels], rotation=45, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.35)

        plt.tight_layout(rect=[0,0.03,1,0.97])
        savefig_white(pdf, fig)

    # ---------- HEATMAP 1 ----------
    if len(ALPHAS) >= 2:
        H1, labels1, n_days1 = compute_heatmap_daily_avg(
            stats_daily_plot, ALPHAS, stat_type=H1_BASE_STAT, min_pairs=2, qfilter=qranks
        )
        k1 = len(labels1) if labels1 else 0
        fig, ax = plt.subplots(1,1, figsize=_heatmap_figure_size(k1), constrained_layout=True)
        base_desc = "Alpha Cross-Section Corr" if H1_BASE_STAT in ('alpha_sum', 'alpha_strength') else f"Cross-Section Corr ({H1_BASE_STAT})"
        if H1 is None or n_days1 == 0:
            ax.axis('off'); ax.set_title(f"Heatmap 1 — {base_desc} (avg over days)", fontsize=13, weight='bold', wrap=True)
            ax.text(0.5,0.5,"No sufficient daily cross-sections.\nTip: include ≥2 quantiles in QR.", ha='center', va='center')
        else:
            _plot_matrix_heatmap(
                fig, ax, H1, labels1,
                f"Heatmap 1 — {base_desc}",
                vmin=-1, vmax=1, annotate_lower=True, fmt=".2f"
            )
        savefig_white(pdf, fig)

        if DO_TEMPORAL:
            plot_cross_section_corr_lines(
                pdf, stats_daily_plot, ALPHAS, stat_type=H1_BASE_STAT,
                title_prefix=f"[H1] Alpha vs Alpha — Daily Cross-Section Corr",
                smooth_window=int(ROLL_H1_LINES),
                height=6.0, qfilter=qranks
            )

    # ---------- Per-Quantile HEATMAPS 2 & 3 ----------
    # Resolve fixed filters on DAILY (prefer pret*/fret* and betsize_*)
    H2_TARGETS_RES = _resolve_fixed("H2_TARGETS", H2_TARGETS, stats_daily_plot['target'], prefer_prefix="fret_", top_k=1)
    H2_BETS_RES    = _resolve_fixed("H2_BETS",    H2_BETS,    stats_daily_plot['bet_size_col'], prefer_prefix="betsize_", top_k=1)
    H3_TARGETS_RES = _resolve_fixed("H3_TARGETS", H3_TARGETS, stats_daily_plot['target'], prefer_prefix="fret_", top_k=1)
    H3_BETS_RES    = _resolve_fixed("H3_BETS",    H3_BETS,    stats_daily_plot['bet_size_col'], prefer_prefix="betsize_", top_k=1)

    for q in qranks:
        q_masked_df = stats_daily_plot[stats_daily_plot['qrank'] == q].copy()

        # Heatmap 2 (PnL cross-section, fixed filters)
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
        fig, ax = plt.subplots(1,1, figsize=_heatmap_figure_size(k2), constrained_layout=True)
        tdesc2 = f"target={', '.join(t2)}"
        bdesc2 = f"bet={', '.join(b2)}"
        if H2 is None or n_days2 == 0:
            ax.axis('off'); ax.set_title(f"Heatmap 2 — PnL Cross-Section Corr (daily avg) [{q}] [{tdesc2} | {bdesc2}]",
                                         fontsize=13, weight='bold', wrap=True)
            ax.text(0.5,0.5,"No sufficient daily cross-sections for 'pnl' with fixed filters.", ha='center', va='center')
        else:
            _plot_matrix_heatmap(
                fig, ax, H2, labels2,
                f"Heatmap 2 — Cross-section corr of daily PnLs [{q}] [{tdesc2} | {bdesc2}]",
                vmin=-1, vmax=1, annotate_lower=True, fmt=".2f"
            )
        savefig_white(pdf, fig)

        if DO_TEMPORAL and len(ALPHAS) >= 2:
            plot_cross_section_corr_lines(
                pdf, q_masked_df, ALPHAS, stat_type='pnl',
                title_prefix=f"[H2 | {q}] PnL vs PnL — Daily Cross-Section Corr",
                smooth_window=int(ROLL_H2_LINES),
                height=6.0, qfilter=[q], targets=t2, bets=b2
            )

        # Heatmap 3 (PnL time-series corr across days, fixed filters)
        C3, labels3, n_days3 = compute_timeseries_heatmap(
            q_masked_df, ALPHAS, stat_type='pnl', min_days=5, agg='sum',
            qfilter=[q], targets=H3_TARGETS_RES, bets=H3_BETS_RES
        )
        k3 = len(labels3) if labels3 else 0
        fig, ax = plt.subplots(1,1, figsize=_heatmap_figure_size(k3), constrained_layout=True)
        tdesc3 = f"target={', '.join(H3_TARGETS_RES)}"
        bdesc3 = f"bet={', '.join(H3_BETS_RES)}"
        if C3 is None or n_days3 < 5:
            ax.axis('off'); ax.set_title(f"Heatmap 3 — PnL Time-Series Corr [{q}] [{tdesc3} | {bdesc3}]",
                                         fontsize=13, weight='bold', wrap=True)
            ax.text(0.5,0.5,"Not enough days (need ≥5) for time-series correlations.", ha='center', va='center')
        else:
            _plot_matrix_heatmap(
                fig, ax, C3, labels3,
                f"Heatmap 3 — corr of daily summed PnL across days [{q}] [{tdesc3} | {bdesc3}]",
                vmin=-1, vmax=1, annotate_lower=True, fmt=".2f"
            )
        savefig_white(pdf, fig)

        if DO_TEMPORAL and len(ALPHAS) >= 2:
            plot_pairwise_timecorr_lines(
                pdf, q_masked_df, ALPHAS, stat_type='pnl',
                title_prefix=f"[H3 | {q}] Alpha vs Alpha — Time correlation (P&L)",
                window=int(ROLL_H3_LINES), height=6.0,
                qfilter=[q], targets=H3_TARGETS_RES, bets=H3_BETS_RES, agg='sum'
            )

    # ---------- Bottom pages (NO __ALL__), overlay selected quantiles ----------
    def _merge_two_stats(subset, stat_a, stat_b, name_a, name_b):
        a = subset[subset['stat_type']==stat_a][['date','value']].rename(columns={'value':name_a})
        b = subset[subset['stat_type']==stat_b][['date','value']].rename(columns={'value':name_b})
        merged = pd.merge(a, b, on='date', how='outer').sort_values('date')
        for c in [name_a,name_b]:
            if c not in merged: merged[c] = 0.0
        merged[[name_a,name_b]] = merged[[name_a,name_b]].fillna(0.0)
        merged[f'cum_{name_a}'] = merged[name_a].cumsum()
        merged[f'cum_{name_b}'] = merged[name_b].cumsum()
        return merged

    rk_pnl  = None if int(ROLL_PNL_NRINSTR)  <= 1 else int(ROLL_PNL_NRINSTR)
    rk_ppd  = None if int(ROLL_PPD_NOTIONAL) <= 1 else int(ROLL_PPD_NOTIONAL)
    rk_ppt  = None if int(ROLL_PPT_TRADES)   <= 1 else int(ROLL_PPT_TRADES)

    for target in sorted(stats_daily_plot['target'].dropna().unique()):
        for signal in sorted(stats_daily_plot['signal'].dropna().unique()):
            for bet_strategy in sorted(stats_daily_plot['bet_size_col'].dropna().unique()):
                base = ((stats_daily_plot['target']==target) &
                        (stats_daily_plot['signal']==signal) &
                        (stats_daily_plot['bet_size_col']==bet_strategy))
                if not base.any(): continue

                def want(tok): return tok in CUM_SECTIONS

                # ----------------- PnL vs nrInstr -----------------
                if want('pnl') or want('nrInstr'):
                    fig, ax = plt.subplots(figsize=(14,5.8))
                    left_title  = "Cum P&L"
                    right_title = "Daily nrInstr" if rk_pnl is None else f"Rolling {rk_pnl}D avg nrInstr"
                    ttl = [t for t, ok in [
                        (f"{left_title} (left, solid)", want('pnl')),
                        (f"{right_title} (right, dashed)", want('nrInstr'))
                    ] if ok]
                    fig.suptitle(f"{target} | {signal} | {bet_strategy}\n" + " vs ".join(ttl),
                                 fontsize=16, weight='bold')
                    sub = stats_daily_plot[base & stats_daily_plot['stat_type'].isin(['pnl','nrInstr'])]
                    if qranks: sub = sub[sub['qrank'].isin(qranks)]
                    ax_r = ax.twinx(); ax_r.grid(False)
                    anyL=anyR=False
                    for q in qranks:
                        sq = sub[sub['qrank']==q].sort_values('date')
                        if sq.empty: continue
                        color = quantile_colors.get(q,'gray')
                        pnl  = sq[sq['stat_type']=='pnl'][['date','value']].set_index('date')['value'].astype(float)
                        nrin = sq[sq['stat_type']=='nrInstr'][['date','value']].set_index('date')['value'].astype(float)
                        if want('pnl') and not pnl.empty:
                            y = pnl.cumsum()
                            ax.plot(y.index, y.values, color=color, linestyle=STYLE_FIRST, linewidth=1.8); anyL=True
                        if want('nrInstr') and not nrin.empty:
                            y2 = nrin if rk_pnl is None else _roll_mean(nrin, rk_pnl)
                            ax_r.plot(y2.index, y2.values, color=color, linestyle=STYLE_SECOND, linewidth=1.8); anyR=True
                    if anyL: ax.set_ylabel(left_title)
                    if anyR: ax_r.set_ylabel(right_title)
                    _plot_date_axis(ax)
                    styles=[]
                    if anyL: styles.append((f"{left_title} (left, solid)", STYLE_FIRST))
                    if anyR: styles.append((f"{right_title} (right, dashed)", STYLE_SECOND))
                    if styles:
                        legL = [Line2D([0],[0], color='gray', lw=2, linestyle=ls, label=lab) for (lab,ls) in styles]
                        leg_styles = ax.legend(handles=legL, loc='upper left', fontsize=9, frameon=True)
                        ax.add_artist(leg_styles)  # keep style legend when adding the quantile legend
                    # Quantile legend (color map) on the right
                    if qranks:
                        _add_quantile_legend(ax, qranks, quantile_colors, loc="upper right", title="Quantiles")
                    plt.tight_layout(); savefig_white(pdf, fig)

                # ----------------- PPD vs Notional -----------------
                if want('ppd') or want('sizeNotional'):
                    fig, ax = plt.subplots(figsize=(14,5.8))
                    left_title = (
                        "Cum PPD"
                        if (not NORMALIZE_LEFT_WITH_ROLL or rk_ppd is None)
                        else f"Cum PPD (P&L / rolling {rk_ppd}D avg Notional)"
                    )
                    right_title = "Daily Notional" if rk_ppd is None else f"Rolling {rk_ppd}D avg Notional"
                    ttl = [t for t, ok in [
                        (f"{left_title} (left, solid)", want('ppd')),
                        (f"{right_title} (right, dashed)", want('sizeNotional'))
                    ] if ok]
                    fig.suptitle(f"{target} | {signal} | {bet_strategy}\n" + " vs ".join(ttl),
                                 fontsize=16, weight='bold')
                    sub = stats_daily_plot[base & stats_daily_plot['stat_type'].isin(['pnl','sizeNotional'])]
                    if qranks: sub = sub[sub['qrank'].isin(qranks)]
                    ax_r = ax.twinx(); ax_r.grid(False)
                    anyL=anyR=False
                    for q in qranks:
                        sq = sub[sub['qrank']==q]
                        if sq.empty: continue
                        color = quantile_colors.get(q,'gray')
                        m = _merge_two_stats(sq.sort_values('date'), 'pnl', 'sizeNotional', 'pnl', 'notional')
                        if want('ppd'):
                            if NORMALIZE_LEFT_WITH_ROLL and rk_ppd is not None:
                                rn = _roll_mean(m['notional'], rk_ppd)
                                contrib = np.where((rn > 0) & np.isfinite(rn), m['pnl']/rn, 0.0)
                                y = pd.Series(contrib, index=m['date']).cumsum()
                            else:
                                m['cum_ppd'] = np.where(m['cum_notional']>0, m['cum_pnl']/m['cum_notional'], np.nan)
                                y = m.set_index('date')['cum_ppd'].fillna(method='ffill')
                            ax.plot(y.index, y.values, color=color, linestyle=STYLE_FIRST, linewidth=1.8); anyL=True
                        if want('sizeNotional'):
                            rn = m['notional'] if rk_ppd is None else _roll_mean(m['notional'], rk_ppd)
                            ax_r.plot(m['date'], rn, color=color, linestyle=STYLE_SECOND, linewidth=1.8); anyR=True
                    if anyL: ax.set_ylabel(left_title)
                    if anyR: ax_r.set_ylabel(right_title)
                    _plot_date_axis(ax)
                    styles=[]
                    if anyL: styles.append((f"{left_title} (left, solid)", STYLE_FIRST))
                    if anyR: styles.append((f"{right_title} (right, dashed)", STYLE_SECOND))
                    if styles:
                        legL = [Line2D([0],[0], color='gray', lw=2, linestyle=ls, label=lab) for (lab,ls) in styles]
                        leg_styles = ax.legend(handles=legL, loc='upper left', fontsize=9, frameon=True)
                        ax.add_artist(leg_styles)
                    if qranks:
                        _add_quantile_legend(ax, qranks, quantile_colors, loc="upper right", title="Quantiles")
                    plt.tight_layout(); savefig_white(pdf, fig)

                # ----------------- PPT vs Trades -----------------
                if want('ppt') or want('n_trades'):
                    fig, ax = plt.subplots(figsize=(14,5.8))
                    left_title = (
                        "Cum PPT"
                        if (not NORMALIZE_LEFT_WITH_ROLL or rk_ppt is None)
                        else f"Cum PPT (P&L / rolling {rk_ppt}D avg Trades)"
                    )
                    right_title = "Daily Trades" if rk_ppt is None else f"Rolling {rk_ppt}D avg Trades"
                    ttl = [t for t, ok in [
                        (f"{left_title} (left, solid)", want('ppt')),
                        (f"{right_title} (right, dashed)", want('n_trades'))
                    ] if ok]
                    fig.suptitle(f"{target} | {signal} | {bet_strategy}\n" + " vs ".join(ttl),
                                 fontsize=16, weight='bold')
                    sub = stats_daily_plot[base & stats_daily_plot['stat_type'].isin(['pnl','n_trades'])]
                    if qranks: sub = sub[sub['qrank'].isin(qranks)]
                    ax_r = ax.twinx(); ax_r.grid(False)
                    anyL=anyR=False
                    for q in qranks:
                        sq = sub[sub['qrank']==q].sort_values('date')
                        if sq.empty: continue
                        color = quantile_colors.get(q,'gray')
                        trades = sq[sq['stat_type']=='n_trades'][['date','value']].set_index('date')['value'].astype(float)
                        pnl    = sq[sq['stat_type']=='pnl'][['date','value']].set_index('date')['value'].astype(float)
                        if want('ppt'):
                            if NORMALIZE_LEFT_WITH_ROLL and rk_ppt is not None:
                                rt = _roll_mean(trades, rk_ppt).reindex(pnl.index)
                                contrib = pd.Series(0.0, index=pnl.index)
                                mask = (rt > 0) & rt.notna() & pnl.notna()
                                contrib[mask] = pnl[mask] / rt[mask]
                                y = contrib.cumsum()
                            else:
                                cum_pnl = pnl.cumsum()
                                cum_tr  = trades.cumsum().reindex(cum_pnl.index).ffill()
                                ratio   = np.where((cum_tr > 0) & np.isfinite(cum_tr), cum_pnl / cum_tr, np.nan)
                                y = pd.Series(ratio, index=cum_pnl.index).ffill()
                            ax.plot(y.index, y.values, color=color, linestyle=STYLE_FIRST, linewidth=1.8); anyL=True
                        if want('n_trades') and not trades.empty:
                            y2 = trades if rk_ppt is None else _roll_mean(trades, rk_ppt)
                            ax_r.plot(y2.index, y2.values, color=color, linestyle=STYLE_SECOND, linewidth=1.8); anyR=True
                    if anyL: ax.set_ylabel(left_title)
                    if anyR: ax_r.set_ylabel(right_title)
                    _plot_date_axis(ax)
                    styles=[]
                    if anyL: styles.append((f"{left_title} (left, solid)", STYLE_FIRST))
                    if anyR: styles.append((f"{right_title} (right, dashed)", STYLE_SECOND))
                    if styles:
                        legL = [Line2D([0],[0], color='gray', lw=2, linestyle=ls, label=lab) for (lab,ls) in styles]
                        leg_styles = ax.legend(handles=legL, loc='upper left', fontsize=9, frameon=True)
                        ax.add_artist(leg_styles)
                    if qranks:
                        _add_quantile_legend(ax, qranks, quantile_colors, loc="upper right", title="Quantiles")
                    plt.tight_layout(); savefig_white(pdf, fig)

    # ---------- Outlier tables ----------
    latest_outliers = _find_latest_outliers_pkl(OUTLIERS_DIR)
    append_outlier_pages(latest_outliers, pdf,
                         metrics=OUTLIER_METRICS_FOR_TABLES,
                         top_k=OUTLIER_TOP_K,
                         tables_per_page=OUTLIER_TABLES_PER_PAGE)

print("✅ Saved to output/Quantile_Combined_Report.pdf")
end = time.perf_counter()
print(f"Time taken: {end - start} seconds")
