# plotting/plot_quantile_bars.py
"""
Quantile Report PDF Generator (fixed H1 temporal coverage, restored distributions, compact outliers)

What this module does (when called via generate_quantile_report(config)):
- Loads precomputed DAILY stats (required) and SUMMARY stats (optional) from:
    <config['daily_dir']>/stats_YYYYMMDD.pkl
    <config['summary_dir']>/summary_stats_YYYYMMDD_YYYYMMDD.pkl
- Optionally loads per-ticker correlation dumps produced by the runner (if configured):
    <config['per_ticker_dir']>/per_ticker_alpha_raw_spy_corr_*.pkl
    <config['per_ticker_dir']>/per_ticker_alpha_pnl_spy_corr_*.pkl
- Optionally loads outlier PKLs from:
    <config['outliers_dir']>/outliers_*.pkl
- Builds a multi-page PDF at: config['output_pdf']

Pages
1) Bar plots by quantile for selected metrics (from SUMMARY ONLY; skipped if no summary PKL).
2) H1: average daily cross-section correlation across alphas (Spearman), using base stat:
       alpha_sum > alpha_strength > pnl (no quantile/target/bet filter; uses all rows).
   H1 temporal lines (pairwise Spearman by day, optionally smoothed). No legend — line end labels.
3) H2: per-quantile PnL daily cross-section correlation (fixed target & bet) — Spearman.
   H2 temporal lines for the same filter. No legend — line end labels.
4) H3: per-quantile time-series correlation of summed daily PnL vectors (Spearman).
   H3 temporal lines (rolling/expanding time corr). No legend — line end labels.
5) Temporal pages per (target, signal, bet): cumulative P&L vs nrInstr; cumulative PPD vs Size Notional; daily n_trades.
6) Distributions: histograms (RAW↔SPY corr, PNL↔SPY corr) with mean/median/std annotations (if files exist).
7) Outlier tables (compact): top/bottom K for selected metrics (default: pnl, ppd, sizeNotional, nrInstr, n_trades).

Expected DAILY/SUMMARY columns:
  date (YYYY-MM-DD), signal, target, qrank (e.g., qr_100), bet_size_col,
  stat_type ('pnl','ppd','n_trades','nrInstr','sizeNotional','sharpe','spy_corr',...),
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
})

# --- Layout guardrails ---
BAR_TITLE_Y     = 0.985
BAR_XLABEL_Y    = 0.962
BAR_AX_TOP      = 0.955
HEATMAP_AX_TOP  = 0.90
TEMPORAL_AX_TOP = 0.90

# Global meta text (set by generate_quantile_report)
META_TEXT = None


# -------------------------------------------------
# Utilities
# -------------------------------------------------
def savefig_white(pdf, fig):
    """Save with white background; print META_TEXT if set (supplied via main.py)."""
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


def _load_data(daily_dir: str, summary_dir: str):
    """Load DAILY and SUMMARY PKLs from the provided directories."""
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
    ax.set_axisbelow(True)
    ax.grid(True, linestyle=":", alpha=0.35)


def _ellipsis(s, n):
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _heatmap_figure_size(k, widen=1.18, extra_height=0.0):
    s = max(10, min(28, 0.6 * k + 8))
    return (s * widen, s + extra_height)


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
    mats = []
    for _, df_day in stats_df.groupby('date', sort=True, observed=True):
        X = _build_daily_cross_section(df_day, alphas, stat_type,
                                       qfilter=qfilter, targets=targets, bets=bets)
        if X is None: continue
        C = _spearman_corr_df(X, min_periods=min_pairs).reindex(index=alphas, columns=alphas)
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
    mp = _minp(win, floor=3) if min_periods is None else int(min_periods)
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
        finite_idx = np.where(np.isfinite(v))[0]
        if finite_idx.size:
            j = finite_idx[-1]
            ax.text(s.index[j], v[j], f"  {name}", color=color, fontsize=fontsize, va='center')


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
            win = int(smooth_window); mp = _minp(win, floor=3)
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
    sub = sub.sort_values('value')
    lows = sub.head(top_k).copy()
    highs = sub.tail(top_k).iloc[::-1].copy()
    labels_tbl = ["Type", "Date", "Signal", "Bet", "Target", "Q", "Value"] + (["z"] if have_z else []) + (["Rule"] if have_rule else [])

    def row(r, kind):
        base = [
            r['date'].strftime('%Y-%m-%d') if pd.notna(r['date']) else "NaT",
            _ellipsis(r['signal'], 18), _ellipsis(r['bet_size_col'], 16),
            _ellipsis(r['target'], 16), str(r['qrank']), f"{r['value']:.6g}",
        ]
        base = [kind] + base
        if have_z:    base.append("" if pd.isna(r.get('z')) else f"{r.get('z'):.2f}")
        if have_rule: base.append(_ellipsis(r.get('rule', ''), 18))
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
        fig.suptitle("Outlier Tables (compact)", fontsize=18, weight='bold', y=0.985)
        if page_idx == 0:
            subtitle = f"Metrics: {', '.join([m for m, _, _ in tables])} | Top-K: {top_k}"
            fig.text(0.03, 0.955, subtitle, ha='left', va='top', fontsize=11, color='0.25')
        gs = GridSpec(nrows=len(chunk), ncols=1, figure=fig, left=0.03, right=0.97, top=0.90, bottom=0.06, hspace=0.35)
        for row_idx, (metric_name, col_labels, rows) in enumerate(chunk):
            ax = fig.add_subplot(gs[row_idx, 0])
            _draw_table_in_axis(ax, title=f"{metric_name} — Top {top_k} Highs & Lows",
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


def generate_quantile_report(config: dict):
    """
    Entry point used by main.py.

    Required keys in `config` (all configured in main.py):
      - daily_dir, summary_dir, per_ticker_dir, outliers_dir, output_pdf
      - qranks, allow_missing_qranks
      - H2_targets, H2_bets, H3_targets, H3_bets
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

    H2_targets_cfg   = config.get("H2_targets", "AUTO")
    H2_bets_cfg      = config.get("H2_bets", "AUTO")
    H3_targets_cfg   = config.get("H3_targets", "AUTO")
    H3_bets_cfg      = config.get("H3_bets", "AUTO")

    roll_h1_lines    = int(config.get("roll_h1_lines", 60))
    roll_h2_lines    = int(config.get("roll_h2_lines", 60))
    roll_h3_lines    = int(config.get("roll_h3_lines", 1))
    roll_nrinstr     = int(config.get("roll_nrinstr", 1))
    roll_ppd         = int(config.get("roll_ppd", 1))
    roll_trades      = int(config.get("roll_trades", 1))
    roll_pnl         = int(config.get("roll_pnl", 1))
    roll_size_notnl  = int(config.get("roll_size_notional", 1))

    bar_page_vars    = list(config.get("bar_page_vars", []))
    bar_x_vars       = list(config.get("bar_x_vars", []))
    bar_metrics      = list(config.get("bar_metrics", []))

    outlier_metrics_for_tables = list(config.get("outlier_metrics_for_tables", []))
    outlier_top_k              = int(config.get("outlier_top_k", 3))
    outlier_tables_per_page    = int(config.get("outlier_tables_per_page", 3))

    style_first      = config.get("style_first", "-")
    style_second     = config.get("style_second", ":")
    quantile_colors_cfg = dict(config.get("quantile_colors", {}))

    # Load data
    stats_daily, stats_summary, interval_min, interval_max, interval_ndays = _load_data(daily_dir, summary_dir)

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
    bar_width = 0.15

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

    available_stats_daily = set(stats_daily_plot['stat_type'].dropna().unique())
    if 'alpha_sum' in available_stats_daily:
        h1_base_stat = 'alpha_sum'
    elif 'alpha_strength' in available_stats_daily:
        h1_base_stat = 'alpha_strength'
    else:
        h1_base_stat = 'pnl'
    print(f"[INFO] Heatmap 1 base stat: {h1_base_stat}")

    do_temporal = (len(alphas) <= 6)
    if not do_temporal:
        print(f"[INFO] {len(alphas)} alphas detected (>6). Skipping temporal line plots for heatmaps.")

    # =========================
    # ------- BUILD PDF -------
    # =========================
    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        print("[INFO] Requested QR: {qranks_requested}")
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

                fig, axs = plt.subplots(len(bar_metrics), 1, figsize=(14, 2.6 * len(bar_metrics)))
                axs = np.atleast_1d(axs)
                main_title = "Bar Plots | " + " ".join(title_bits)

                fig.suptitle(main_title, fontsize=18, weight='bold', y=BAR_TITLE_Y)
                xlabel_descr = " | ".join(bar_x_vars) if bar_x_vars else "ALL"
                fig.text(0.5, BAR_XLABEL_Y, f"X-axis: {xlabel_descr}",
                         ha="center", va="top", fontsize=13, color="0.35", weight='bold')

                for i, metric in enumerate(bar_metrics):
                    ax = axs[i]
                    data = subset[subset['stat_type'] == metric].copy()
                    if data.empty:
                        ax.set_title(f"{metric}: no data in summary PKL")
                        ax.axis('off')
                        continue

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
                            q_offsets = np.arange(-(len(use_q) - 1) / 2, (len(use_q) + 1) / 2) * bar_width
                            for j, q in enumerate(use_q):
                                vals = pd.to_numeric(pivot.get(q), errors='coerce').reindex(x_levels).astype(float)
                                if vals.notna().any():
                                    ax.bar(x + q_offsets[j],
                                           vals.fillna(0.0).values,
                                           width=bar_width,
                                           color=quantile_colors.get(q, 'gray'),
                                           label=q)
                                    plotted = True
                        if plotted:
                            ax.legend(title='Quantile (color)', bbox_to_anchor=(1.02, 0.98),
                                      loc='upper left', fontsize=9, frameon=True)
                    else:
                        data_dedup = data.drop_duplicates(subset=['x_key'], keep='last')
                        vals = (data_dedup.set_index('x_key')['value']
                                .reindex(x_levels).fillna(0.0).values)
                        ax.bar(np.arange(len(x_levels)), vals, width=bar_width, color='gray')

                    if metric in ('long_ratio', 'hit_ratio'):
                        ymin, ymax = ax.get_ylim()
                        if ymin <= 0.5 <= ymax:
                            ax.axhline(y=0.5, color='red', linestyle=':', linewidth=1.5, alpha=0.7, zorder=0)

                    ax.set_ylabel(f"{metric}{unit_suffix}")
                    ax.set_xticks(np.arange(len(x_levels)))
                    ax.set_xticklabels([str(v) for v in x_levels], rotation=45, ha='right', fontsize=11)
                    ax.grid(axis='y', linestyle=':', alpha=0.35)

                plt.tight_layout(rect=[0.02, 0.04, 0.98, BAR_AX_TOP])
                savefig_white(pdf, fig)

        # ---------- Heatmap 1 (Spearman; raw alpha base) ----------
        stats_daily_nonall = _exclude_all_rows(stats_daily)
        if len(alphas) >= 2:
            H1, labels1, n_days1 = compute_heatmap_daily_avg(
                stats_daily_nonall, alphas, stat_type=h1_base_stat,
                min_pairs=2, qfilter=None, targets=None, bets=None
            )
            k1 = len(labels1) if labels1 else 0
            fig, ax, cax = _centered_heatmap_axes(k1)
            base_desc = ("Alpha Cross-Section Corr (Spearman)"
                         if h1_base_stat in ('alpha_sum', 'alpha_strength')
                         else f"Cross-Section Corr (Spearman, {h1_base_stat})")
            if H1 is None or n_days1 == 0:
                ax.axis('off')
                _set_title_fit(fig, ax, f"Heatmap 1 — {base_desc}", base_size=13, pad=8, loc='center')
                ax.text(0.5, 0.5, "No sufficient daily cross-sections.", ha='center', va='center')
                if cax is not None: cax.axis('off')
            else:
                _plot_matrix_heatmap(fig, ax, cax, H1, labels1,
                                     f"Heatmap 1 — {base_desc} (avg over {n_days1} days)",
                                     vmin=-1, vmax=1, annotate_lower=True, fmt=".2f")
            fig.tight_layout(rect=[0.02, 0.06, 0.98, HEATMAP_AX_TOP])
            savefig_white(pdf, fig)

            # H1 temporal
            if do_temporal:
                plot_cross_section_corr_lines(
                    pdf,
                    stats_daily_nonall,
                    alphas,
                    stat_type=h1_base_stat,
                    title_prefix="[H1] Alpha vs Alpha",
                    smooth_window=roll_h1_lines,
                    height=6.0,
                    qfilter=None,
                    targets=None,
                    bets=None,
                )

        # ---------- Heatmaps 2 & 3 per-quantile ----------
        stats_daily_plot_nonall = stats_daily_nonall
        try:
            H2_TARGETS_RES = _resolve_fixed("H2_TARGETS", H2_targets_cfg,
                                            stats_daily_plot_nonall['target'],
                                            prefer_prefix="fret_", top_k=2)
        except Exception as e:
            print(f"[WARN] H2 target resolution failed: {e}")
            H2_TARGETS_RES = []
        try:
            H2_BETS_RES    = _resolve_fixed("H2_BETS", H2_bets_cfg,
                                            stats_daily_plot_nonall['bet_size_col'],
                                            prefer_prefix="betsize_", top_k=2)
        except Exception as e:
            print(f"[WARN] H2 bet resolution failed: {e}")
            H2_BETS_RES = []
        try:
            H3_TARGETS_RES = _resolve_fixed("H3_TARGETS", H3_targets_cfg,
                                            stats_daily_plot_nonall['target'],
                                            prefer_prefix="fret_", top_k=1)
        except Exception as e:
            print(f"[WARN] H3 target resolution failed: {e}")
            H3_TARGETS_RES = []
        try:
            H3_BETS_RES    = _resolve_fixed("H3_BETS", H3_bets_cfg,
                                            stats_daily_plot_nonall['bet_size_col'],
                                            prefer_prefix="betsize_", top_k=1)
        except Exception as e:
            print(f"[WARN] H3 bet resolution failed: {e}")
            H3_BETS_RES = []

        for q in qranks:
            q_masked_df = stats_daily_plot_nonall[stats_daily_plot_nonall['qrank'] == q].copy()

            # Heatmap 2 — cross-section corr of P&L, fixed (target, bet)
            H2, labels2, n_days2 = compute_heatmap_daily_avg(
                q_masked_df, alphas, stat_type='pnl', min_pairs=2,
                qfilter=[q] if q else None,
                targets=H2_TARGETS_RES if H2_TARGETS_RES else None,
                bets=H2_BETS_RES if H2_BETS_RES else None
            )
            k2 = len(labels2) if labels2 else 0
            fig, ax, cax = _centered_heatmap_axes(k2)
            tdesc2 = f"target={', '.join(H2_TARGETS_RES)}" if H2_TARGETS_RES else "target=AUTO"
            bdesc2 = f"bet={', '.join(H2_BETS_RES)}" if H2_BETS_RES else "bet=AUTO"
            if H2 is None or n_days2 == 0:
                ax.axis('off')
                _set_title_fit(
                    fig, ax,
                    f"Heatmap 2 — Cross-section corr of daily PnLs (Spearman) [{q}] [{tdesc2} | {bdesc2}]",
                    base_size=13, pad=8, loc='center'
                )
                ax.text(0.5, 0.5,
                        "No sufficient daily cross-sections for 'pnl' with fixed filters.\n"
                        "Tip: ensure ≥2 rows/day across (target, bet).",
                        ha='center', va='center')
                if cax is not None: cax.axis('off')
            else:
                _plot_matrix_heatmap(
                    fig, ax, cax, H2, labels2,
                    f"Heatmap 2 — Cross-section corr of daily PnLs (Spearman) [{q}] [{tdesc2} | {bdesc2}] "
                    f"(avg over {n_days2} days)",
                    vmin=-1, vmax=1, annotate_lower=True, fmt=".2f"
                )
            fig.tight_layout(rect=[0.02, 0.06, 0.98, HEATMAP_AX_TOP])
            savefig_white(pdf, fig)

            # H2 temporal
            if do_temporal and len(alphas) >= 2 and (H2 is not None):
                plot_cross_section_corr_lines(
                    pdf,
                    q_masked_df,
                    alphas,
                    stat_type='pnl',
                    title_prefix=f"[H2 | {q}] PnL vs PnL",
                    smooth_window=roll_h2_lines,
                    height=6.0,
                    qfilter=[q],
                    targets=H2_TARGETS_RES if H2_TARGETS_RES else None,
                    bets=H2_BETS_RES if H2_BETS_RES else None,
                )

            # Heatmap 3 — time-series corr of summed daily P&L vectors, fixed (target, bet)
            C3, labels3, n_days3 = compute_timeseries_heatmap(
                q_masked_df, alphas, stat_type='pnl', min_days=5, agg='sum',
                qfilter=[q], targets=H3_TARGETS_RES if H3_TARGETS_RES else None,
                bets=H3_BETS_RES if H3_BETS_RES else None
            )
            k3 = len(labels3) if labels3 else 0
            fig, ax, cax = _centered_heatmap_axes(k3)
            tdesc3 = f"target={', '.join(H3_TARGETS_RES)}" if H3_TARGETS_RES else "target=AUTO"
            bdesc3 = f"bet={', '.join(H3_BETS_RES)}" if H3_BETS_RES else "bet=AUTO"
            if C3 is None or n_days3 < 5:
                ax.axis('off')
                _set_title_fit(
                    fig, ax,
                    f"Heatmap 3 — Time-series corr of summed daily P&L (Spearman) [{q}] [{tdesc3} | {bdesc3}]",
                    base_size=13, pad=8, loc='center'
                )
                ax.text(0.5, 0.5, "Not enough days (need ≥5) for time-series correlations.",
                        ha='center', va='center')
                if cax is not None: cax.axis('off')
            else:
                _plot_matrix_heatmap(
                    fig, ax, cax, C3, labels3,
                    f"Heatmap 3 — corr of daily P&L vectors (sum across day, Spearman) "
                    f"[{q}] [{tdesc3} | {bdesc3}] (days={n_days3})",
                    vmin=-1, vmax=1, annotate_lower=True, fmt=".2f"
                )
            fig.tight_layout(rect=[0.02, 0.06, 0.98, HEATMAP_AX_TOP])
            savefig_white(pdf, fig)

            # H3 temporal
            if do_temporal and len(alphas) >= 2 and (C3 is not None):
                corr_map = compute_pairwise_rolling_time_corr(
                    q_masked_df, alphas, stat_type='pnl',
                    window=roll_h3_lines, qfilter=[q],
                    targets=H3_TARGETS_RES if H3_TARGETS_RES else None,
                    bets=H3_BETS_RES if H3_BETS_RES else None,
                    agg='sum'
                )
                title_text = (
                    f"[H3 | {q}] Alpha vs Alpha — Time corr (P&L vectors) — "
                    f"{'Rolling ' + str(int(roll_h3_lines)) + 'D' if int(roll_h3_lines) > 1 else 'Expanding'} "
                    f"— tgt={','.join(H3_TARGETS_RES) if H3_TARGETS_RES else 'AUTO'} "
                    f"| bet={','.join(H3_BETS_RES) if H3_BETS_RES else 'AUTO'}"
                )
                dates_all = sorted(q_masked_df['date'].dropna().unique())
                fig, ax = plt.subplots(figsize=(14, 6.0))
                _set_title_fit(fig, ax, title_text, base_size=14, pad=10, loc='center')

                coverage = [(k, v.notna().sum()) for k, v in corr_map.items()]
                coverage.sort(key=lambda x: x[1], reverse=True)
                chosen = [k for k, cnt in coverage[:8] if cnt > 0]
                series_map = {k: corr_map[k].copy().sort_index() for k in chosen}
                _label_last_points(ax, series_map)
                if dates_all:
                    ax.set_xlim(pd.to_datetime(dates_all[0]), pd.to_datetime(dates_all[-1]))
                ax.set_ylabel("Spearman corr (time)")
                _plot_date_axis(ax)
                fig.tight_layout(rect=[0.02, 0.06, 0.98, TEMPORAL_AX_TOP])
                savefig_white(pdf, fig)

        # ---------- Temporal pages per (target, signal, bet) ----------
        stats_daily_plot_local = stats_daily_plot_nonall
        for target in sorted(stats_daily_plot_local['target'].dropna().unique()):
            for signal in sorted(stats_daily_plot_local['signal'].dropna().unique()):
                for bet_strategy in sorted(stats_daily_plot_local['bet_size_col'].dropna().unique()):
                    mask_base = (
                        (stats_daily_plot_local['target'] == target) &
                        (stats_daily_plot_local['signal'] == signal) &
                        (stats_daily_plot_local['bet_size_col'] == bet_strategy)
                    )
                    sub_all = stats_daily_plot_local[mask_base].copy()
                    if sub_all.empty:
                        continue

                    # ---- PnL (cum) vs nrInstr (daily)
                    left_title  = _title_token("P&L", roll_pnl, cumulative=True)
                    right_title = _title_token("nrInstr", roll_nrinstr, cumulative=False)

                    fig, ax = plt.subplots(figsize=(14, 6.0))
                    right_ax = ax.twinx(); right_ax.grid(False)
                    fig.suptitle(
                        f"{target} | {signal} | {bet_strategy}\n"
                        f"{left_title} (left, solid) vs {right_title} (right, dotted)",
                        fontsize=16, weight='bold', y=0.96
                    )

                    anyL = anyR = False
                    for q in qranks:
                        sq = sub_all[sub_all['qrank'] == q]
                        if sq.empty: continue
                        color = quantile_colors.get(q, 'gray')

                        pnl  = _series(sq, 'pnl')
                        nrin = _series(sq, 'nrInstr')

                        if not pnl.empty:
                            cum_pnl = pnl.cumsum()
                            yL = _roll_mean(cum_pnl, roll_pnl)
                            ax.plot(yL.index, yL.values, color=color,
                                    linestyle=style_first, linewidth=1.8)
                            anyL = True

                        if not nrin.empty:
                            yR = _roll_mean(nrin, roll_nrinstr)
                            right_ax.plot(yR.index, yR.values, color=color,
                                          linestyle=style_second, linewidth=1.8)
                            anyR = True

                    if anyL:
                        ax.set_ylabel(left_title)
                    if anyR:
                        right_ax.set_ylabel(right_title)
                    _plot_date_axis(ax)

                    handles_styles = [
                        Line2D([0], [0], color='gray', lw=2, linestyle=style_first,
                               label=f"{left_title} (left)"),
                        Line2D([0], [0], color='gray', lw=2, linestyle=style_second,
                               label=f"{right_title} (right)")
                    ]
                    leg_styles = ax.legend(handles=handles_styles, loc='upper left',
                                           fontsize=9, frameon=True, bbox_to_anchor=(0, 0.98))
                    ax.add_artist(leg_styles)
                    if qranks:
                        handles_q = [
                            Line2D([0], [0],
                                   color=quantile_colors.get(str(q), 'gray'),
                                   lw=2, label=str(q))
                            for q in qranks
                        ]
                        ax.legend(handles=handles_q, title="Quantiles",
                                  loc="upper right", fontsize=9, frameon=True,
                                  bbox_to_anchor=(1, 0.98))
                    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
                    savefig_white(pdf, fig)

                    # ---- PPD (cum) vs Size Notional (daily)
                    left_title_base  = _title_token("PPD", roll_ppd, cumulative=True)
                    right_title_base = _title_token("Size Notional", roll_size_notnl, cumulative=False)
                    left_title_ppd   = f"{left_title_base} (bpts)"
                    right_title_sz   = f"{right_title_base} ($M)"

                    fig, ax = plt.subplots(figsize=(14, 6.0))
                    right_ax = ax.twinx()
                    right_ax.grid(False)
                    fig.suptitle(
                        f"{target} | {signal} | {bet_strategy}\n"
                        f"{left_title_ppd} (left, solid) vs {right_title_sz} (right, dotted)",
                        fontsize=16, weight='bold', y=0.96
                    )

                    anyL = anyR = False
                    for q in qranks:
                        sq = sub_all[sub_all['qrank'] == q]
                        if sq.empty:
                            continue
                        color = quantile_colors.get(q, 'gray')

                        pnl   = _series(sq, 'pnl')
                        notnl = _series(sq, 'sizeNotional')

                        if not pnl.empty and not notnl.empty:
                            merged = pd.DataFrame({'pnl': pnl, 'notional': notnl}).fillna(0.0)
                            cum_pnl = merged['pnl'].cumsum()
                            cum_notnl = merged['notional'].cumsum()
                            cum_ppd = pd.Series(
                                np.where(cum_notnl > 0, cum_pnl / cum_notnl, np.nan),
                                index=cum_pnl.index
                            ).ffill()
                            cum_ppd_bpts = cum_ppd * 10000
                            yL = _roll_mean(cum_ppd_bpts, roll_ppd)
                            ax.plot(yL.index, yL.values, color=color,
                                    linestyle=style_first, linewidth=1.8, alpha=0.9)
                            anyL = True

                        if not notnl.empty:
                            notnl_millions = notnl / 1e6
                            yR = _roll_mean(notnl_millions, roll_size_notnl)
                            right_ax.plot(yR.index, yR.values, color=color,
                                          linestyle=style_second, linewidth=1.8, alpha=0.9)
                            anyR = True

                    if anyL:
                        ax.set_ylabel(left_title_ppd)
                    if anyR:
                        right_ax.set_ylabel(right_title_sz)
                    _plot_date_axis(ax)

                    handles_styles = [
                        Line2D([0], [0], color='gray', lw=2, linestyle=style_first,
                               label=f"{left_title_ppd} (left)"),
                        Line2D([0], [0], color='gray', lw=2, linestyle=style_second,
                               label=f"{right_title_sz} (right)")
                    ]
                    leg_styles = ax.legend(handles=handles_styles, loc='upper left',
                                           fontsize=9, frameon=True, bbox_to_anchor=(0, 0.98))
                    ax.add_artist(leg_styles)
                    if qranks:
                        handles_q = [
                            Line2D([0], [0],
                                   color=quantile_colors.get(str(q), 'gray'),
                                   lw=2, label=str(q))
                            for q in qranks
                        ]
                        ax.legend(handles=handles_q, title="Quantiles",
                                  loc="upper right", fontsize=9, frameon=True,
                                  bbox_to_anchor=(1, 0.98))
                    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
                    savefig_white(pdf, fig)

                    # ---- Number of Trades (daily)
                    title_trades = _title_token("Trades", roll_trades, cumulative=False)
                    fig, ax = plt.subplots(figsize=(14, 5.5))
                    fig.suptitle(
                        f"{target} | {signal} | {bet_strategy}\n{title_trades}",
                        fontsize=16, weight='bold', y=0.96
                    )
                    anyPlot = False
                    for q in qranks:
                        sq = sub_all[sub_all['qrank'] == q]
                        if sq.empty:
                            continue
                        color = quantile_colors.get(q, 'gray')
                        trades = _series(sq, 'n_trades')
                        if trades.empty:
                            continue
                        y = _roll_mean(trades, roll_trades)
                        ax.plot(y.index, y.values, color=color,
                                linestyle=style_first, linewidth=1.8)
                        anyPlot = True
                    if anyPlot:
                        ax.set_ylabel(title_trades)
                    _plot_date_axis(ax)
                    if qranks:
                        handles_q = [
                            Line2D([0], [0],
                                   color=quantile_colors.get(str(q), 'gray'),
                                   lw=2, label=str(q))
                            for q in qranks
                        ]
                        ax.legend(handles=handles_q, title="Quantiles",
                                  loc="upper right", fontsize=9, frameon=True,
                                  bbox_to_anchor=(1, 0.98))
                    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
                    savefig_white(pdf, fig)

        # ---------- Distributions: per-ticker correlations (if available) ----------
        raw_pkls = sorted(glob.glob(os.path.join(per_ticker_dir, "per_ticker_alpha_raw_spy_corr_*.pkl")))
        pnl_pkls = sorted(glob.glob(os.path.join(per_ticker_dir, "per_ticker_alpha_pnl_spy_corr_*.pkl")))

        if raw_pkls:
            try:
                df_raw = read_pickle_compat(raw_pkls[-1])
            except Exception:
                df_raw = None
            _distrib_page(pdf, df_raw, "Distribution — Alpha RAW ↔ SPY correlation (per ticker)")

        if pnl_pkls:
            try:
                df_pnl = read_pickle_compat(pnl_pkls[-1])
            except Exception:
                df_pnl = None
            _distrib_page(pdf, df_pnl, "Distribution — Alpha PnL ↔ SPY correlation (per ticker)")

        # ---------- Outlier tables (compact) ----------
        latest_outliers = _find_latest_outliers_pkl(outliers_dir)
        append_outlier_pages(
            latest_outliers,
            pdf,
            metrics=outlier_metrics_for_tables,
            top_k=outlier_top_k,
            tables_per_page=outlier_tables_per_page,
        )

    t1 = time.perf_counter()
    print("✅ Saved quantile report to", output_pdf)
    print(f"Time taken (plotting only): {t1 - t0:.3f} seconds")


if __name__ == "__main__":
    raise SystemExit("This module is intended to be called via generate_quantile_report(config) from main.py.")

