import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from itertools import product
from scipy.stats import spearmanr, ConstantInputWarning

# =========================
# ---- USER CONFIG --------
# =========================

# Bar page grouping (pages split by these)
BAR_PAGE_VARS = ['signal', 'bet_size_col']
BAR_X_VARS    = ['target']

metrics_to_plot = [
    'pnl', 'ppd', 'sharpe', 'hit_ratio', 'long_ratio',
    'nrInstr', 'sizeNotional', 'r2', 't_stat',
    'n_trades', 'ppt', 'spearman', 'dcor'
]

# Cumulative sections to draw (omit any to hide the section)
CUM_SECTIONS = ['pnl', 'ppd', 'sizeNotional', 'nrInstr', 'n_trades', 'ppt']

# ===== Heatmap inputs =====
PANEL_DAILY_DIR      = "output/PANEL_DAILY"   # requires sidecars panel_YYYYMMDD.pkl
ALPHA_PREFIX         = "pret_"                # used only for inventory text
ALPHA_COLS_EXPLICIT  = None                   # or e.g. ['pret_1_MR','pret_1_RR',...]
# IMPORTANT: Heatmap 2 needs ONE exact realized return column name present in sidecars
TARGET_COL_FOR_PNL   = "fret_1d"              # e.g. 'fret_1d'  (MUST exist in sidecars)
BET_COL_FOR_PNL      = None                   # optional bet-size column in sidecars; None => unit
HEATMAP_QRANKS       = ['qr_100', 'qr_75', 'qr_50', 'qr_25']  # for H3–H6 (time-corr)

# ===== Distribution plots (explicit lists; no auto-detection) =====
# Put EXACT column names you want to plot (must exist in sidecars),
# or leave [] to skip that distribution page.
DIST_FEATURE_LIST = []                        # e.g. ['pret_1_MR','pret_1_RR','pret_3_MR','pret_3_RR']
DIST_BET_LIST     = ['betsize_equal','betsize_cap200k','betsize_cap250k']
DIST_FRET_LIST    = ['fret_1_MR','fret_3_MR','fret_1_RR','fret_3_RR']

DIST_BINS       = 60
MAX_SAMPLES     = 2_000_000        # cap rows concatenated from sidecars (None => all)

# ===== Outlier tables (PKL) =====
OUTLIERS_DIR                 = "output/OUTLIERS"
OUTLIER_METRICS_FOR_TABLES   = ['pnl', 'ppd', 'sizeNotional', 'nrInstr', 'n_trades', 'ppt']
OUTLIER_TOP_K                = 3        # highs & lows per metric
OUTLIER_TABLES_PER_PAGE      = 3        # stacked vertically per page

# Styles / colors
STYLE_FIRST  = '-'    # solid
STYLE_SECOND = '--'   # dashed
quantile_colors = {'qr_100': 'red', 'qr_75': 'green', 'qr_50': 'blue', 'qr_25': 'black'}

# Silence scipy constant-input spam in correlation loops
warnings.filterwarnings("ignore", category=ConstantInputWarning)

# =========================
# ---- LOAD & PREP --------
# =========================

def _load_stats_df():
    daily_dir = "output/DAILY_STATS"
    summary_dir = "output/SUMMARY_STATS"

    if not os.path.isdir(daily_dir):
        raise FileNotFoundError("Expected 'output/DAILY_STATS' with per-day pickles (stats_*.pkl).")

    daily_paths = sorted(glob.glob(os.path.join(daily_dir, "stats_*.pkl")))
    if not daily_paths:
        raise FileNotFoundError("No daily files found in 'output/DAILY_STATS'.")

    daily_frames = [pd.read_pickle(p) for p in daily_paths]
    stats_daily = pd.concat(daily_frames, ignore_index=True)

    summary_paths = sorted(glob.glob(os.path.join(summary_dir, "summary_stats_*.pkl"))) if os.path.isdir(summary_dir) else []
    stats_summary = pd.read_pickle(summary_paths[-1]) if summary_paths else pd.DataFrame()

    parts = [df for df in (stats_daily, stats_summary) if not df.empty]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out['date'] = pd.to_datetime(out['date'], errors='coerce')
    out = out.dropna(subset=['date', 'value'])
    return out

stats_df = _load_stats_df()

targets   = stats_df['target'].dropna().unique().tolist()
signals   = stats_df['signal'].dropna().unique().tolist()
bet_sizes = stats_df['bet_size_col'].dropna().unique().tolist()

_qr_raw = [q for q in stats_df['qrank'].dropna().unique().tolist() if isinstance(q, str)]
try:
    qranks = sorted(_qr_raw, key=lambda x: float(x.split('_')[1]) if '_' in x else x)
except Exception:
    qranks = sorted(_qr_raw)

bar_width = 0.15
q_offsets = np.arange(-(len(qranks)-1)/2, (len(qranks)+1)/2) * bar_width

LEVELS = {
    'signal': signals,
    'target': targets,
    'bet_size_col': bet_sizes,
}

# ================
# Helper functions
# ================

def _plot_date_axis(ax):
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

def _merge_two_stats(subset, stat_a, stat_b, name_a, name_b):
    a_df = subset[subset['stat_type'] == stat_a][['date','value']].rename(columns={'value': name_a})
    b_df = subset[subset['stat_type'] == stat_b][['date','value']].rename(columns={'value': name_b})
    merged = pd.merge(a_df, b_df, on='date', how='outer').sort_values('date')
    for col in [name_a, name_b]:
        if col not in merged:
            merged[col] = 0.0
    merged[[name_a, name_b]] = merged[[name_a, name_b]].fillna(0.0)
    merged[f'cum_{name_a}'] = merged[name_a].cumsum()
    merged[f'cum_{name_b}'] = merged[name_b].cumsum()
    return merged

def _add_dual_legends(ax, qranks, color_map, style_descs):
    quant_handles = [Line2D([0],[0], color=color_map.get(q, 'gray'), lw=2, label=q) for q in qranks]
    leg_colors = ax.legend(handles=quant_handles, title='Quantile (color)', loc='upper left', fontsize=9)
    if style_descs:
        style_handles = [Line2D([0],[0], color='gray', lw=2, linestyle=ls, label=lab) for (lab, ls) in style_descs]
        ax.legend(handles=style_handles, title='Series (style / axis)', loc='upper right', fontsize=9, frameon=True)
        ax.add_artist(leg_colors)

def _infer_alphas_from_columns(cols, explicit=None, prefix="pret_"):
    if explicit:
        return [c for c in explicit if c in cols]
    return sorted([c for c in cols if str(c).startswith(prefix)])

def _spearman_corr_across_stocks(mat):
    """
    mat: (n_stocks × k) values for a single day (alphas or per-stock pnl).
    Returns k×k Spearman correlation across stocks (NaN if <3 overlapping obs).
    """
    k = mat.shape[1]
    M = np.full((k, k), np.nan, float)
    for i in range(k):
        xi = mat[:, i]
        for j in range(i, k):
            xj = mat[:, j]
            m = np.isfinite(xi) & np.isfinite(xj)
            if m.sum() >= 3:
                rho = spearmanr(xi[m], xj[m], nan_policy='omit').correlation
                if np.isfinite(rho):
                    M[i, j] = M[j, i] = float(rho)
    return M

def _plot_matrix_heatmap(fig, ax, M: np.ndarray, labels, title: str,
                         vmin=-1, vmax=1, cmap='coolwarm'):
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title, fontsize=13, weight='bold')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

def _placeholder(ax, title, msg):
    ax.axis('off')
    ax.set_title(title, fontsize=13, weight='bold')
    ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=11)

# =========================
# Heatmap computations
# =========================

def compute_heatmap1_alpha_space(panel_daily_dir, alpha_prefix, alpha_cols_explicit=None):
    day_files = sorted(glob.glob(os.path.join(panel_daily_dir, "panel_*.pkl")))
    if not day_files:
        return None, None, 0

    sum_M = None
    n_used = 0
    labels = None

    for p in day_files:
        d = pd.read_pickle(p)
        if d.empty:
            continue
        if labels is None:
            labels = _infer_alphas_from_columns(d.columns, alpha_cols_explicit, alpha_prefix)
            if len(labels) < 2:
                return None, labels, 0
        X = d[labels].apply(pd.to_numeric, errors='coerce').to_numpy()
        if X.shape[0] < 3:
            continue
        M = _spearman_corr_across_stocks(X)
        if not np.isfinite(M).any():
            continue
        sum_M = M if sum_M is None else (sum_M + M)
        n_used += 1

    if n_used == 0:
        return None, labels, 0
    H1 = sum_M / n_used
    return H1, labels, n_used

def compute_heatmap2_pnl_across_stocks_avg(panel_daily_dir, alpha_prefix, target_col, bet_col=None, alpha_cols_explicit=None):
    day_files = sorted(glob.glob(os.path.join(panel_daily_dir, "panel_*.pkl")))
    if not day_files:
        return None, None, 0

    sum_M = None
    n_used = 0
    labels = None

    for p in day_files:
        d = pd.read_pickle(p)
        if d.empty or (target_col not in d.columns):
            continue

        if labels is None:
            labels = _infer_alphas_from_columns(d.columns, alpha_cols_explicit, alpha_prefix)
            if len(labels) < 2:
                return None, labels, 0

        Y = pd.to_numeric(d[target_col], errors='coerce').to_numpy()
        if bet_col and (bet_col in d.columns):
            B = np.abs(pd.to_numeric(d[bet_col], errors='coerce').to_numpy())
        else:
            B = np.ones_like(Y, dtype=float)

        X = np.column_stack([
            np.sign(pd.to_numeric(d[col], errors='coerce').to_numpy()) * Y * B
            for col in labels
        ])
        if X.shape[0] < 3:
            continue
        M = _spearman_corr_across_stocks(X)
        if not np.isfinite(M).any():
            continue
        sum_M = M if sum_M is None else (sum_M + M)
        n_used += 1

    if n_used == 0:
        return None, labels, 0
    H2 = sum_M / n_used
    return H2, labels, n_used

def compute_timecorr_from_daily_pkls(stats_df, target: str, bet: str, qrank: str):
    df = stats_df[(stats_df['stat_type'] == 'pnl') &
                  (stats_df['target'] == target) &
                  (stats_df['bet_size_col'] == bet) &
                  (stats_df['qrank'] == qrank)].copy()
    if df.empty:
        return None, None
    mat = (df.pivot_table(index='date', columns='signal', values='value', aggfunc='sum')
             .sort_index())
    mat = mat.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    if mat.shape[1] < 2 or mat.shape[0] < 5:
        return None, None
    C = mat.corr(method='pearson', min_periods=5)
    return C.values, mat.columns.tolist()

# =========================
# Outlier Tables (PKL)
# =========================

def _find_latest_outliers_pkl(root=OUTLIERS_DIR):
    if not os.path.isdir(root):
        return None
    cand = sorted(glob.glob(os.path.join(root, "outliers_*.pkl")))
    return cand[-1] if cand else None

def _ellipsis(s: str, maxlen: int) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= maxlen else (s[:maxlen-1] + "…")

def _metric_table_rows(odf: pd.DataFrame, metric: str, top_k: int, have_z: bool, have_rule: bool):
    sub = odf[odf['stat_type'] == metric].copy()
    if sub.empty:
        return None, None

    sub = sub.sort_values('value')
    lows  = sub.head(top_k).copy()
    highs = sub.tail(top_k).iloc[::-1].copy()

    labels = ["Type", "Date", "Signal", "Bet", "Target", "Q", "Value"]
    if have_z:    labels.append("z")
    if have_rule: labels.append("Rule")

    def row_from_series(r, kind):
        return [
            kind,
            r['date'].strftime('%Y-%m-%d') if pd.notna(r['date']) else "NaT",
            _ellipsis(r['signal'], 18),
            _ellipsis(r['bet_size_col'], 16),
            _ellipsis(r['target'], 16),
            str(r['qrank']),
            f"{r['value']:.6g}",
        ] + (
            [f"{r.get('z'):.2f}"] if have_z and pd.notna(r.get('z')) else ([""] if have_z else [])
        ) + (
            [_ellipsis(r.get('rule', ''), 18)] if have_rule else []
        )

    rows = []
    for _, r in highs.iterrows():
        rows.append(row_from_series(r, "High"))
    for _, r in lows.iterrows():
        rows.append(row_from_series(r, "Low"))

    return labels, rows

def _draw_table_in_axis(ax, title: str, col_labels, rows, fontsize=9):
    ax.axis('off')
    ax.set_title(title, fontsize=13, weight='bold', loc='left', pad=6)

    base_colwidths = {
        "Type": 0.08, "Date": 0.11, "Signal": 0.18, "Bet": 0.14,
        "Target": 0.14, "Q": 0.06, "Value": 0.11, "z": 0.06, "Rule": 0.12
    }
    colWidths = [base_colwidths.get(lbl, 0.10) for lbl in col_labels]
    s = sum(colWidths)
    if s > 0.98:
        colWidths = [w * 0.98 / s for w in colWidths]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        colWidths=colWidths,
        loc='upper left',
        cellLoc='left',
        bbox=[0.0, 0.0, 1.0, 0.92]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, 1.15)

    header_color = (0.9, 0.9, 0.92)
    even_color   = (0.98, 0.98, 0.985)
    odd_color    = (1.0, 1.0, 1.0)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', ha='left')
            cell.set_facecolor(header_color)
            cell.set_edgecolor('0.75')
        else:
            cell.set_edgecolor('0.85')
            cell.set_facecolor(even_color if row % 2 == 0 else odd_color)

def append_outlier_pages(outliers_pkl_path: str, pdf,
                         metrics=None, top_k: int = 3, tables_per_page: int = 3):
    try:
        if outliers_pkl_path is None or not os.path.isfile(outliers_pkl_path):
            raise FileNotFoundError("Outliers PKL not found.")
        odf = pd.read_pickle(outliers_pkl_path)
    except Exception as e:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.axis('off')
        fig.suptitle("Outlier Tables", fontsize=18, weight='bold')
        ax.text(0.5, 0.5, f"No outlier tables appended:\n{e}", ha='center', va='center', fontsize=12)
        plt.tight_layout(); pdf.savefig(fig); plt.close()
        return

    if odf is None or len(odf) == 0:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.axis('off')
        fig.suptitle("Outlier Tables", fontsize=18, weight='bold')
        ax.text(0.5, 0.5, "No outliers found.", ha='center', va='center', fontsize=12)
        plt.tight_layout(); pdf.savefig(fig); plt.close()
        return

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
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.axis('off')
            fig.suptitle("Outlier Tables", fontsize=18, weight='bold')
            ax.text(0.5, 0.5, "Selected outlier metrics not present in file.", fontsize=12)
            plt.tight_layout(); pdf.savefig(fig); plt.close()
            return

    date_min = odf['date'].min()
    date_max = odf['date'].max()
    subtitle = f"Window: {date_min:%Y-%m-%d} → {date_max:%Y-%m-%d} | Metrics: {', '.join(metrics)} | Top-K: {top_k}"

    tables = []
    for m in metrics:
        labels, rows = _metric_table_rows(odf, m, top_k=top_k, have_z=have_z, have_rule=have_rule)
        if labels is None or not rows:
            continue
        tables.append((m, labels, rows))
    if not tables:
        return

    per_page = max(1, int(tables_per_page))
    for page_idx in range(0, len(tables), per_page):
        chunk = tables[page_idx:page_idx+per_page]

        fig = plt.figure(figsize=(14, 8.5))
        fig.suptitle("Outlier Tables", fontsize=18, weight='bold', y=0.985)
        if page_idx == 0:
            fig.text(0.03, 0.955, subtitle, ha='left', va='top', fontsize=11, color='0.25')

        gs = GridSpec(
            nrows=len(chunk), ncols=1, figure=fig,
            left=0.03, right=0.97, top=0.90, bottom=0.06, hspace=0.35
        )

        for row_idx, (metric_name, labels, rows) in enumerate(chunk):
            ax = fig.add_subplot(gs[row_idx, 0])
            _draw_table_in_axis(
                ax,
                title=f"{metric_name} — Top {top_k} Highs & Lows",
                col_labels=labels,
                rows=rows,
                fontsize=9
            )

        pdf.savefig(fig)
        plt.close(fig)

# =========================
# Distribution helpers (explicit lists only)
# =========================

def _gather_sidecars(panel_dir):
    return sorted(glob.glob(os.path.join(panel_dir, "panel_*.pkl")))

def _concat_columns(files, col_list, max_samples=None):
    """
    Concatenate exactly the requested columns across all sidecars.
    IMPORTANT: we always include ALL requested columns and fill missing with NaN,
    so nothing disappears even if absent in some files.
    """
    if not col_list:
        return pd.DataFrame()

    parts = []
    for f in files:
        try:
            d = pd.read_pickle(f)
        except Exception:
            continue

        # Build a frame with ALL requested columns; fill missing with NaN
        block = pd.DataFrame(index=np.arange(len(d)), columns=col_list, dtype=float)
        for c in col_list:
            if c in d.columns:
                block[c] = pd.to_numeric(d[c], errors='coerce')
        parts.append(block)

    if not parts:
        # Return a DF that still has the columns (so plotting shows "(no data)")
        return pd.DataFrame(columns=col_list)

    big = pd.concat(parts, axis=0, ignore_index=True)

    # Drop rows where *all requested columns* are NaN; keep rows if any requested col has data
    big = big.dropna(how='all', subset=col_list)

    if (max_samples is not None) and (len(big) > max_samples):
        big = big.sample(int(max_samples), random_state=42)

    return big

def _nice_grid(n):
    c = min(3, max(1, int(np.ceil(np.sqrt(n)))))
    r = int(np.ceil(n / c))
    return r, c

def _plot_hist_percent(ax, x: pd.Series, bins=60, title="", label_min_pct=3.0):
    """
    Histogram where bar heights are percentages, with 1–99% clipping for scale.
    Adds labels to bars with >= label_min_pct and mean/median guide lines.
    """
    x = pd.to_numeric(pd.Series(x), errors='coerce').dropna()
    if x.empty:
        ax.set_title(f"{title} (no data)", fontsize=11)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center', va='center', fontsize=10, color='0.4')
        return

    p1, p99 = np.nanpercentile(x, [1, 99])
    use_clip = np.isfinite(p1) and np.isfinite(p99) and p1 < p99
    x_clip = x[(x >= p1) & (x <= p99)] if use_clip else x

    edges = np.histogram_bin_edges(x_clip, bins=bins)
    counts, edges = np.histogram(x_clip, bins=edges)
    total = max(len(x_clip), 1)
    perc = counts * 100.0 / total
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    bars = ax.bar(centers, perc, width=widths, align='center', alpha=0.9, edgecolor='white', linewidth=0.4)

    for b, p in zip(bars, perc):
        if p >= label_min_pct:
            ax.text(b.get_x() + b.get_width()/2.0, p, f"{p:.0f}%", ha='center', va='bottom', fontsize=8)

    mu = np.nanmean(x_clip)
    med = np.nanmedian(x_clip)
    ax.axvline(mu,  linestyle='--', linewidth=1.2, alpha=0.9)
    ax.axvline(med, linestyle=':',  linewidth=1.2, alpha=0.9)
    ax.text(mu, ax.get_ylim()[1]*0.95, "mean", rotation=90, va='top', ha='right', fontsize=8, alpha=0.8)
    ax.text(med, ax.get_ylim()[1]*0.95, "median", rotation=90, va='top', ha='left',  fontsize=8, alpha=0.8)

    ax.set_title(title, fontsize=11)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    if use_clip:
        clipped_pct = (1.0 - len(x_clip) / len(x)) * 100.0
        footer = f"N={len(x):,}  (clipped {clipped_pct:.1f}% to 1–99 pct)"
    else:
        footer = f"N={len(x):,}"
    ax.text(0.99, -0.12, footer, transform=ax.transAxes, ha='right', va='top', fontsize=8, color='0.35')

def _render_distribution_block(pdf, title, files, requested_cols, bins, label_min_pct=3.0):
    if not requested_cols:
        fig, ax = plt.subplots(figsize=(12, 2.6))
        ax.axis('off')
        fig.suptitle(title, fontsize=14, weight='bold')
        ax.text(0.03, 0.5, "No columns requested.", fontsize=11)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)
        return

    df = _concat_columns(files, requested_cols, max_samples=MAX_SAMPLES)
    n = len(requested_cols)
    r, c = _nice_grid(n)
    fig, axes = plt.subplots(r, c, figsize=(5*c + 1.2, 3.6*r + 1.0))
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(title, fontsize=16, weight='bold', y=0.98)

    for ax, col in zip(axes, requested_cols):
        if (df is not None) and (col in df.columns):
            _plot_hist_percent(ax, df[col], bins=bins, title=col, label_min_pct=label_min_pct)
        else:
            _plot_hist_percent(ax, pd.Series(dtype=float), bins=bins, title=col, label_min_pct=label_min_pct)

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)

def add_distribution_pages(pdf):
    files = _gather_sidecars(PANEL_DAILY_DIR)
    if not files:
        fig, ax = plt.subplots(figsize=(12, 2.6))
        ax.axis('off')
        fig.suptitle("Distribution Plots (features, bets, targets)", fontsize=14, weight='bold')
        ax.text(0.03, 0.5, f"No sidecars found in {PANEL_DAILY_DIR}. Skipping distributions.", fontsize=11)
        plt.tight_layout(); pdf.savefig(fig); plt.close(fig)
        return

    # Features
    _render_distribution_block(
        pdf,
        "Feature Return Distributions (pret_*)",
        files,
        DIST_FEATURE_LIST,
        bins=DIST_BINS,
        label_min_pct=3.0
    )

    # Bet sizes
    _render_distribution_block(
        pdf,
        "Bet Size Distributions (betsize_*)",
        files,
        DIST_BET_LIST,
        bins=DIST_BINS,
        label_min_pct=4.0
    )

    # Realized returns / targets
    _render_distribution_block(
        pdf,
        "Realized Return Distributions (fret_*)",
        files,
        DIST_FRET_LIST,
        bins=DIST_BINS,
        label_min_pct=4.0
    )

# =========================
# ---- BUILD THE PDF -------
# =========================

os.makedirs("output", exist_ok=True)
with PdfPages("output/Quantile_Combined_Report.pdf") as pdf:

    # -------- Bar Plots --------
    if BAR_PAGE_VARS:
        page_levels = [LEVELS[var] for var in BAR_PAGE_VARS]
        page_iter = list(product(*page_levels))
    else:
        page_iter = [()]

    for page_vals in page_iter:
        subset = stats_df.copy()
        title_bits = []
        for var, val in zip(BAR_PAGE_VARS, page_vals):
            subset = subset[subset[var] == val]
            title_bits.append(f"{var}: {val}")

        if subset.empty:
            continue

        if BAR_X_VARS:
            subset['x_key'] = subset[BAR_X_VARS].astype(str).agg('|'.join, axis=1)
            x_levels = sorted(subset['x_key'].unique().tolist())
        else:
            subset['x_key'] = "ALL"
            x_levels = ["ALL"]

        fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(14, 2.7 * len(metrics_to_plot)))
        if len(metrics_to_plot) == 1:
            axs = [axs]
        fig.suptitle(" | ".join(["Bar Plots"] + title_bits), fontsize=18, weight='bold')

        for i, metric in enumerate(metrics_to_plot):
            ax = axs[i]
            data = subset[subset['stat_type'] == metric]
            if data.empty:
                ax.set_title(f"{metric}: no data")
                ax.axis('off')
                continue

            pivot = data.pivot_table(index='x_key', columns='qrank', values='value',
                                     aggfunc='mean', fill_value=0.0)
            x = np.arange(len(x_levels))

            use_q = [q for q in qranks if q in pivot.columns]
            q_offsets_local = np.arange(-(len(use_q)-1)/2, (len(use_q)+1)/2) * bar_width if use_q else []

            for j, qrank in enumerate(use_q):
                values = pivot[qrank].reindex(x_levels).fillna(0.0).values
                color = quantile_colors.get(qrank, 'gray')
                ax.bar(x + q_offsets_local[j], values, width=bar_width, color=color, label=qrank)

            ax.set_ylabel(metric)
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in x_levels], rotation=45, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            if i == 0 and use_q:
                ax.legend(title='Quantile (color)', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        pdf.savefig(fig)
        plt.close()

    # =========================
    # HEATMAPS (six total, after bar plots)
    # =========================

    # Heatmap 1 & 2 (side-by-side)
    H1, labels1, n_days1 = compute_heatmap1_alpha_space(
        PANEL_DAILY_DIR, ALPHA_PREFIX, ALPHA_COLS_EXPLICIT
    )
    H2, labels2, n_days2 = compute_heatmap2_pnl_across_stocks_avg(
        PANEL_DAILY_DIR, ALPHA_PREFIX, TARGET_COL_FOR_PNL, bet_col=BET_COL_FOR_PNL,
        alpha_cols_explicit=ALPHA_COLS_EXPLICIT
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    if H1 is None or labels1 is None or n_days1 == 0:
        _placeholder(
            axes[0],
            "Heatmap 1 — Alpha Space",
            f"Missing or insufficient sidecars in {PANEL_DAILY_DIR} "
            f"(need per-stock alpha columns like '{ALPHA_PREFIX}*')."
        )
    else:
        _plot_matrix_heatmap(fig, axes[0], H1, labels1,
            f"Heatmap 1 — Alpha Space (Spearman across stocks; avg over {n_days1} days)")
    if H2 is None or labels2 is None or n_days2 == 0:
        msg = (f"Need per-stock target '{TARGET_COL_FOR_PNL}' in sidecars."
               f"{'' if BET_COL_FOR_PNL is None else f' Optional bet: {BET_COL_FOR_PNL}.'}")
        _placeholder(axes[1], "Heatmap 2 — PnL Space", msg)
    else:
        _plot_matrix_heatmap(fig, axes[1], H2, labels2,
            f"Heatmap 2 — PnL Space (Spearman across stocks; avg over {n_days2} days)")
    plt.tight_layout()
    pdf.savefig(fig); plt.close(fig)

    # Heatmaps 3–6 (2×2 grid): time corr of summed PnL per quantile
    target_for_hmaps = targets[0] if targets else None
    bet_for_hmaps    = bet_sizes[0] if bet_sizes else None

    H_time = []
    L_time = []
    if (target_for_hmaps is not None) and (bet_for_hmaps is not None):
        for q in HEATMAP_QRANKS:
            C, labels = compute_timecorr_from_daily_pkls(stats_df, target_for_hmaps, bet_for_hmaps, q)
            H_time.append(C); L_time.append(labels)
    else:
        H_time = [None, None, None, None]; L_time = [None, None, None, None]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    titles = [f"Heatmap 3 — {HEATMAP_QRANKS[0]}",
              f"Heatmap 4 — {HEATMAP_QRANKS[1]}",
              f"Heatmap 5 — {HEATMAP_QRANKS[2]}",
              f"Heatmap 6 — {HEATMAP_QRANKS[3]}"]
    for ax, C, labels, ttl in zip(axes.flatten(), H_time, L_time, titles):
        if C is None or labels is None:
            _placeholder(ax, ttl + " (PnL Time Corr)",
                         "No data for chosen target/bet/qrank.\n"
                         f"target={target_for_hmaps} | bet={bet_for_hmaps}")
        else:
            _plot_matrix_heatmap(fig, ax, C, labels,
                                 f"{ttl} — PnL Time Corr | target={target_for_hmaps} | bet={bet_for_hmaps}",
                                 vmin=-1, vmax=1, cmap='coolwarm')
    plt.tight_layout()
    pdf.savefig(fig); plt.close(fig)

    # =========================
    # DISTRIBUTION PAGES (explicit lists; always render all requested)
    # =========================
    add_distribution_pages(pdf)

    # =========================
    # CUMULATIVE PAGES
    # =========================
    for target in targets:
        for signal in signals:
            for bet_strategy in bet_sizes:
                base_filter = (
                    (stats_df['target'] == target) &
                    (stats_df['signal'] == signal) &
                    (stats_df['bet_size_col'] == bet_strategy)
                )
                if not base_filter.any():
                    continue

                def want(token: str):
                    return token in CUM_SECTIONS

                # Pair 1: Cum PnL vs Daily nrInstr
                want_left = want('pnl'); want_right = want('nrInstr')
                if want_left or want_right:
                    fig, ax = plt.subplots(figsize=(14, 5.8))
                    title_bits = []
                    if want_left: title_bits.append('Cum P&L (left, solid)')
                    if want_right: title_bits.append('Daily nrInstr (right, dashed)')
                    fig.suptitle(f"{target} | {signal} | {bet_strategy}\n" + " vs ".join(title_bits),
                                 fontsize=16, weight='bold')
                    subset_pair1 = stats_df[base_filter & stats_df['stat_type'].isin(['pnl','nrInstr'])]
                    ax_right = ax.twinx(); ax_right.grid(False)

                    any_left=False; any_right=False
                    for qrank in qranks:
                        sub_q = subset_pair1[subset_pair1['qrank'] == qrank].sort_values('date')
                        if sub_q.empty: continue
                        color = quantile_colors.get(qrank, 'gray')
                        if want_left:
                            pnl = sub_q[sub_q['stat_type']=='pnl'][['date','value']]
                            if not pnl.empty:
                                ax.plot(pnl['date'], pnl['value'].cumsum(), color=color, linestyle=STYLE_FIRST, linewidth=1.8)
                                any_left=True
                        if want_right:
                            nrin = sub_q[sub_q['stat_type']=='nrInstr'][['date','value']]
                            if not nrin.empty:
                                ax_right.plot(nrin['date'], nrin['value'], color=color, linestyle=STYLE_SECOND, linewidth=1.8)
                                any_right=True

                    if any_left: ax.set_ylabel("Cumulative P&L")
                    if any_right: ax_right.set_ylabel("Daily nrInstr")
                    _plot_date_axis(ax)
                    style_descs = []
                    if any_left: style_descs.append(('Cum P&L (left, solid)', STYLE_FIRST))
                    if any_right: style_descs.append(('Daily nrInstr (right, dashed)', STYLE_SECOND))
                    _add_dual_legends(ax, qranks, quantile_colors, style_descs)
                    plt.tight_layout(); pdf.savefig(fig); plt.close()

                # Pair 2: Cum PPD vs Daily Notional
                want_left = want('ppd'); want_right = want('sizeNotional')
                if want_left or want_right:
                    fig, ax = plt.subplots(figsize=(14, 5.8))
                    title_bits = []
                    if want_left: title_bits.append('Cum PPD (left, solid)')
                    if want_right: title_bits.append('Daily Notional (right, dashed)')
                    fig.suptitle(f"{target} | {signal} | {bet_strategy}\n" + " vs ".join(title_bits),
                                 fontsize=16, weight='bold')
                    subset_pair2 = stats_df[base_filter & stats_df['stat_type'].isin(['pnl','sizeNotional'])]
                    ax_right = ax.twinx(); ax_right.grid(False)

                    any_left=False; any_right=False
                    for qrank in qranks:
                        sub_q = subset_pair2[subset_pair2['qrank'] == qrank]
                        if sub_q.empty: continue
                        color = quantile_colors.get(qrank, 'gray')
                        merged = _merge_two_stats(sub_q, 'pnl', 'sizeNotional', 'pnl', 'notional')
                        merged['cum_ppd'] = np.where(merged['cum_notional']>0, merged['cum_pnl']/merged['cum_notional'], np.nan)
                        if want_left:
                            ax.plot(merged['date'], merged['cum_ppd'], color=color, linestyle=STYLE_FIRST, linewidth=1.8)
                            any_left=True
                        if want_right:
                            ax_right.plot(merged['date'], merged['notional'], color=color, linestyle=STYLE_SECOND, linewidth=1.8)
                            any_right=True

                    if any_left: ax.set_ylabel("Cumulative PPD")
                    if any_right: ax_right.set_ylabel("Daily Notional")
                    _plot_date_axis(ax)
                    style_descs = []
                    if any_left: style_descs.append(('Cum PPD (left, solid)', STYLE_FIRST))
                    if any_right: style_descs.append(('Daily Notional (right, dashed)', STYLE_SECOND))
                    _add_dual_legends(ax, qranks, quantile_colors, style_descs)
                    plt.tight_layout(); pdf.savefig(fig); plt.close()

                # Pair 3: Cum PPT vs Daily Trades
                want_left = want('ppt'); want_right = want('n_trades')
                if want_left or want_right:
                    fig, ax = plt.subplots(figsize=(14, 5.8))
                    title_bits = []
                    if want_left: title_bits.append('Cum PPT (left, solid)')
                    if want_right: title_bits.append('Daily Trades (right, dashed)')
                    fig.suptitle(f"{target} | {signal} | {bet_strategy}\n" + " vs ".join(title_bits),
                                 fontsize=16, weight='bold')
                    subset_pair3 = stats_df[base_filter & stats_df['stat_type'].isin(['pnl','n_trades'])]
                    ax_right = ax.twinx(); ax_right.grid(False)

                    any_left=False; any_right=False
                    for qrank in qranks:
                        sub_q = subset_pair3[subset_pair3['qrank'] == qrank].sort_values('date')
                        if sub_q.empty: continue
                        color = quantile_colors.get(qrank, 'gray')
                        trades = sub_q[sub_q['stat_type']=='n_trades'][['date','value']].rename(columns={'value':'trades'})
                        pnl = sub_q[sub_q['stat_type']=='pnl'][['date','value']].rename(columns={'value':'pnl'})
                        merged = pd.merge(pnl, trades, on='date', how='outer').fillna(0.0).sort_values('date')
                        merged['cum_pnl'] = merged['pnl'].cumsum()
                        merged['cum_trades'] = merged['trades'].cumsum()
                        merged['cum_ppt'] = np.where(merged['cum_trades']>0, merged['cum_pnl']/merged['cum_trades'], np.nan)
                        if want_left:
                            ax.plot(merged['date'], merged['cum_ppt'], color=color, linestyle=STYLE_FIRST, linewidth=1.8)
                            any_left=True
                        if want_right:
                            ax_right.plot(merged['date'], merged['trades'], color=color, linestyle=STYLE_SECOND, linewidth=1.8)
                            any_right=True

                    if any_left: ax.set_ylabel("Cumulative PPT")
                    if any_right: ax_right.set_ylabel("Daily Trades")
                    _plot_date_axis(ax)
                    style_descs = []
                    if any_left: style_descs.append(('Cum PPT (left, solid)', STYLE_FIRST))
                    if any_right: style_descs.append(('Daily Trades (right, dashed)', STYLE_SECOND))
                    _add_dual_legends(ax, qranks, quantile_colors, style_descs)
                    plt.tight_layout(); pdf.savefig(fig); plt.close()

    # =========================
    # OUTLIER TABLE PAGES (last)
    # =========================
    latest_outliers = _find_latest_outliers_pkl(OUTLIERS_DIR)
    append_outlier_pages(
        latest_outliers,
        pdf,
        metrics=OUTLIER_METRICS_FOR_TABLES,
        top_k=OUTLIER_TOP_K,
        tables_per_page=OUTLIER_TABLES_PER_PAGE
    )
