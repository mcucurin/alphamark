# plotting/plot_quantile_bars.py
"""
Quantile Report PDF Generator

SPEC: Alpha_Mark__financial_analysis_benchmark_pipeline.pdf (Cucuringu, priority)
      AlphaMark_Guide.pdf (Patel & Li, secondary)

Metric display conventions:
  pnl         — raw value (dollars)
  ppd         — ×10000 → basis points  (ppd = pnl/sizeNotional exactly)
  sizeNotional— ÷1e6   → $M
  sharpe      — mean(PnL)/std(PnL,ddof=1)*√252  (rolling window on temporal pages)
  hit_ratio   — fraction instruments where sign(si)=sign(fi), fi≠0  (ref line 0.5)
  long_ratio  — fraction instruments where sign(si)=1               (ref line 0.5)
  nrInstr     — count si≠0, bet-independent
  n_trades    — Σ_T |U_t^(q)|
  market_corr — Spearman(daily PnL_t, SPY_t)
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

mpl.rcParams.update({
    # Background
    "figure.facecolor":      "#FAFAFA",
    "axes.facecolor":        "#FAFAFA",
    "savefig.facecolor":     "white",
    "savefig.edgecolor":     "white",
    # Spines — remove top/right for clean look
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.linewidth":        0.7,
    "axes.edgecolor":        "0.55",
    # Grid — subtle
    "axes.grid":             True,
    "grid.color":            "0.88",
    "grid.linewidth":        0.5,
    "grid.linestyle":        ":",
    "axes.axisbelow":        True,
    # Typography
    "font.family":           "sans-serif",
    "font.size":             10,
    "axes.titlesize":        11,
    "axes.titleweight":      "semibold",
    "axes.titlepad":         8,
    "axes.labelsize":        10,
    "axes.labelcolor":       "0.25",
    "xtick.labelsize":       8.5,
    "ytick.labelsize":       8.5,
    "xtick.color":           "0.40",
    "ytick.color":           "0.40",
    "figure.titlesize":      13,
    "figure.titleweight":    "semibold",
    # Legend
    "legend.fontsize":       8.5,
    "legend.framealpha":     0.92,
    "legend.edgecolor":      "0.82",
    "legend.borderpad":      0.5,
    "legend.labelspacing":   0.3,
    # Ticks
    "xtick.major.size":      3,
    "ytick.major.size":      3,
    "xtick.major.width":     0.6,
    "ytick.major.width":     0.6,
    "xtick.major.pad":       4,
    "ytick.major.pad":       4,
    # Lines
    "lines.solid_capstyle":  "round",
    "patch.linewidth":       0.5,
})
PAGE_SIZE       = (14, 8.5)
HEATMAP_AX_TOP  = 0.90
TEMPORAL_AX_TOP = 0.90
META_TEXT       = None

STAT_ALIASES = {
    "spy_corr":  "market_corr",
    "mkt_corr":  "market_corr",
    "nrTrades":  "n_trades",
    "nr_trades": "n_trades",
    "ntrades":   "n_trades",
}


# ─── helpers ──────────────────────────────────────────────────────────────────

def _canonical(stat: str) -> str:
    return STAT_ALIASES.get(stat, stat)


def _apply_aliases(df):
    if df is None or not isinstance(df, pd.DataFrame) or "stat_type" not in df.columns:
        return df
    out = df.copy()
    out["stat_type"] = out["stat_type"].astype("string").map(lambda x: STAT_ALIASES.get(x, x)).astype("category")
    return out


def _norm_metrics(metrics) -> list:
    out = []
    for m in metrics or []:
        if m is None:
            continue
        n = _canonical(str(m))
        if n not in out:
            out.append(n)
    return out


def _metric_label(metric: str) -> str:
    n = _canonical(str(metric))
    if n in ("pnl", "ppd"):
        return n.upper()
    return re.sub(r"(?<!^)(?=[A-Z])", " ", n).replace("_", " ").title()


def savefig_white(pdf, fig):
    fig.set_size_inches(*PAGE_SIZE, forward=True)
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)
    # Footer: metadata in small italic bottom-right
    if META_TEXT:
        fig.text(0.99, 0.004, str(META_TEXT), ha="right", va="bottom",
                 fontsize=7.5, color="0.55", style="italic",
                 transform=fig.transFigure)
    # Thin bottom separator line above footer
    fig.add_artist(mpl.lines.Line2D([0.03, 0.97], [0.018, 0.018],
                   transform=fig.transFigure, color="0.82", linewidth=0.5))
    for ax in fig.get_axes():
        ax.set_facecolor("#FAFAFA")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    pdf.savefig(fig, facecolor="white", edgecolor="white", dpi=150)
    plt.close(fig)


def read_pickle_compat(path: str):
    class _U(pkl.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core")
            return super().find_class(module, name)
    with open(path, "rb") as f:
        return _U(f).load()


def _parse_date_range(start, end):
    def _p(x):
        if x is None:
            return None
        try:
            dt = pd.to_datetime(x, errors="coerce")
            return None if pd.isna(dt) else pd.Timestamp(dt).normalize()
        except Exception:
            return None
    s, e = _p(start), _p(end)
    if s and e and e < s:
        s, e = e, s
    return s, e


def _filter_dates(df, s, e, label="data"):
    if s is not None:
        df = df[df["date"] >= s]
        print(f"[INFO] Filtered {label}: start >= {s:%Y-%m-%d}")
    if e is not None:
        df = df[df["date"] <= e]
        print(f"[INFO] Filtered {label}: end <= {e:%Y-%m-%d}")
    return df


def _load_data(daily_dir, summary_dir, interval_start=None, interval_end=None):
    if not os.path.isdir(daily_dir):
        raise FileNotFoundError(f"Expected DAILY stats dir: {daily_dir}")
    paths = sorted(glob.glob(os.path.join(daily_dir, "stats_*.pkl")))
    if not paths:
        raise FileNotFoundError(f"No daily files found in '{daily_dir}'.")

    frames = []
    for p in paths:
        try:
            df = read_pickle_compat(p)
        except Exception:
            continue
        if isinstance(df, pd.DataFrame) and not df.empty and {"date", "value"}.issubset(df.columns):
            frames.append(df)
    if not frames:
        raise FileNotFoundError("All daily PKLs were empty or malformed.")

    stats_daily = pd.concat(frames, ignore_index=True)
    stats_daily["date"] = pd.to_datetime(stats_daily["date"], errors="coerce")
    stats_daily = stats_daily.dropna(subset=["date"])

    if interval_start is not None or interval_end is not None:
        s, e = _parse_date_range(interval_start, interval_end)
        stats_daily = _filter_dates(stats_daily, s, e, "daily data")

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
        spaths = sorted(glob.glob(os.path.join(summary_dir, "summary_stats_*.pkl")))
        if spaths:
            try:
                stats_summary = read_pickle_compat(spaths[-1])
                if isinstance(stats_summary, pd.DataFrame) and not stats_summary.empty:
                    if "date" in stats_summary.columns:
                        stats_summary["date"] = pd.to_datetime(stats_summary["date"], errors="coerce")
                    if interval_start is not None or interval_end is not None:
                        s, e = _parse_date_range(interval_start, interval_end)
                        stats_summary = _filter_dates(stats_summary, s, e, "summary data")
                    if "value" in stats_summary.columns:
                        stats_summary["value"] = pd.to_numeric(stats_summary["value"], errors="coerce")
                    for col in ("signal", "target", "bet_size_col", "qrank", "stat_type"):
                        if col not in stats_summary.columns:
                            stats_summary[col] = pd.NA
                        stats_summary[col] = stats_summary[col].astype("string").astype("category")
                print(f"[INFO] Loaded summary PKL: {os.path.basename(spaths[-1])}  shape={stats_summary.shape}")
            except Exception as e2:
                print(f"[WARN] Failed to read summary PKL: {e2}")
        else:
            print(f"[WARN] No summary PKL found in {summary_dir} (bar plots skipped).")
    else:
        print(f"[WARN] Summary dir not found: {summary_dir} (bar plots skipped).")

    print(f"[INFO] DAILY window: {dmin:%Y-%m-%d} → {dmax:%Y-%m-%d}  ({ndays} days)")
    return stats_daily, stats_summary, dmin, dmax, ndays


def _sorted_qranks(series):
    vals = [str(q) for q in pd.Series(series).dropna().unique()]
    try:
        return sorted(vals, key=lambda x: float(x.split("_")[1]) if "_" in x else float(x))
    except Exception:
        return sorted(vals)


def _ensure_colors(labels, base):
    cmap = mpl.colormaps.get_cmap("tab20")
    out  = dict(base or {})
    i    = 0
    for lab in labels:
        if lab not in out:
            out[lab] = cmap(i % cmap.N)
            i += 1
    return out


def _plot_date_axis(ax):
    ax.set_axisbelow(True)
    # Grid already set globally; just ensure margins and spine cleanup
    ax.margins(x=0.01, y=0.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    try:
        from matplotlib.dates import num2date, DateFormatter
        locs  = ax.get_xticks()
        dates = []
        for x in locs:
            try:
                dt = num2date(x)
                if 1900 <= dt.year <= 2100:
                    dates.append(dt)
            except Exception:
                continue
        if len(dates) >= 2:
            intervals = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))
                         if (dates[i] - dates[i - 1]).days > 0]
            avg = sum(intervals) / len(intervals) if intervals else 0
            fmt = "%b %Y" if avg >= 25 else "%m/%d"
            ax.xaxis.set_major_formatter(DateFormatter(fmt))
            ax.tick_params(axis="x", labelsize=9 if avg >= 25 else 10, rotation=45)
        else:
            ax.tick_params(axis="x", labelsize=10, rotation=45)
        ax.tick_params(axis="y", labelsize=11)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    except Exception:
        ax.tick_params(axis="x", labelsize=10, rotation=45)
        ax.tick_params(axis="y", labelsize=11)


def _ellipsis(s, n):
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n - 1] + "…"


@contextmanager
def _no_err():
    with np.errstate(invalid="ignore", divide="ignore"):
        yield


def _set_title(fig, ax, text, base=14, mn=8, pad=10):
    text = " ".join(str(text).split())
    for size in range(int(base), mn - 1, -1):
        t = ax.set_title(text, fontsize=size, weight="bold", pad=pad, loc="center")
        t.set_ha("center"); t.set_x(0.5)
        fig.canvas.draw()
        r  = fig.canvas.get_renderer()
        ab = ax.get_window_extent(renderer=r)
        tb = t.get_window_extent(renderer=r)
        if tb.width <= 0.98 * ab.width:
            return t
    t = ax.set_title(text, fontsize=mn, weight="bold", pad=pad, loc="center")
    t.set_ha("center"); t.set_x(0.5)
    return t


def _heatmap_axes(k):
    s   = max(12, min(30, 0.7 * k + 10))
    fig = plt.figure(figsize=(s * 1.22, s + 0.5))
    gs  = GridSpec(1, 2, figure=fig, left=0.10, right=0.90, bottom=0.10,
                   top=HEATMAP_AX_TOP, width_ratios=[20, 1], wspace=0.15)
    return fig, fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])


def _draw_heatmap(fig, ax, cax, M, labels, title, vmin=-1, vmax=1, fmt=".2f"):
    if M is None or not labels:
        ax.axis("off"); _set_title(fig, ax, title)
        if cax: cax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return
    k    = len(labels)
    fs_l = 9 if k <= 18 else (7 if k <= 30 else 6)
    # Larger annotation font for small matrices (e.g. 5x5)
    fs_c = 11 if k <= 6 else (9 if k <= 12 else (7 if k <= 18 else (5 if k <= 30 else 4)))

    # Data-adaptive colour range: expand symmetrically around the data's actual range
    # but always keep 0 centred so the colormap is interpretable.
    # For correlation matrices almost all values will be near 1; zooming in makes
    # small differences visible (e.g. 0.95 vs 0.99 actually look different).
    fin_vals = M[np.isfinite(M)]
    d_min_data = vmin  # track actual data min for colorbar ticks
    d_max_data = vmax
    if fin_vals.size > 0:
        d_min_data = float(np.nanmin(fin_vals))
        d_max_data = float(np.nanmax(fin_vals))
        d_range    = d_max_data - d_min_data
        if d_range < 0.15:
            # Tightly clustered — pad the colour scale beyond the data range
            # so neighbouring cells look similar rather than dramatically different.
            # Pad = 4× data range below, tiny pad above (keep max anchor visible).
            pad  = max(0.06, d_range * 4.0)
            if d_min_data >= 0:
                vmin = max(0.0,  d_min_data - pad)
                vmax = min(1.0,  d_max_data + 0.005)
            else:
                vmin = max(-1.0, d_min_data - pad)
                vmax = min(1.0,  d_max_data + pad)
        else:
            abs_max = max(abs(d_min_data), abs(d_max_data))
            vmin = max(-1.0, -abs_max)
            vmax = min(1.0,   abs_max)
            d_min_data = vmin; d_max_data = vmax
    # else keep caller-supplied vmin/vmax

    # Colormap selection:
    _cmap = "RdYlBu_r"
    im   = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=_cmap, aspect="equal",
                     interpolation="nearest")
    _set_title(fig, ax, title, base=13, pad=10)
    ax.set_xticks(range(k)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fs_l)
    ax.set_yticks(range(k)); ax.set_yticklabels(labels, fontsize=fs_l)
    ax.set_xlim(-0.5, k - 0.5); ax.set_ylim(k - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")
    # Annotation colour: white text on dark cells, dark text on light cells.
    # Use the 70th percentile of the data as the threshold so most cells get
    # white text (readable on the darker upper end of the scale).
    mid = float(np.nanpercentile(M[np.isfinite(M)], 70)) if fin_vals.size > 0 else 0.5 * (vmin + vmax)
    for i in range(k):
        for j in range(k):
            v = M[i, j]
            if np.isfinite(v):
                txt_color = "white" if v >= mid else "0.15"
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        fontsize=fs_c, color=txt_color, fontweight="semibold")
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=8.5)
    # Colorbar ticks: show actual data range (not padded vmin/vmax)
    # so the reader sees meaningful values, not the internal padding.
    t_lo  = round(d_min_data, 2)
    t_hi  = round(d_max_data, 2)
    t_mid = round((t_lo + t_hi) / 2, 2)
    ticks = sorted({t_lo, t_mid, t_hi})
    cb.set_ticks(ticks)
    cb.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))


def _minp(w, floor=3):
    if w is None:
        return 1
    w = int(max(1, w))
    return min(w, max(1, w // 5, floor))


def _roll_mean(s, w):
    return s.rolling(w, min_periods=_minp(w, 3)).mean()


def _rolling_sharpe(s, w):
    """Rolling annualised Sharpe: mean(PnL)/std(PnL,ddof=1)*√252 — matches Cucuringu Eq.7."""
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    w  = max(1, int(w))
    mp = _minp(w, 5)
    def _sr(x):
        mu = np.nanmean(x); sd = np.nanstd(x, ddof=1)
        return (mu / sd * np.sqrt(252.0)) if (np.isfinite(mu) and np.isfinite(sd) and sd > 0) else np.nan
    return s.rolling(w, min_periods=mp).apply(_sr, raw=False)


def _exclude_all(df):
    m = df["target"].ne("__ALL__") & df["bet_size_col"].ne("__ALL__") & df["qrank"].ne("__ALL__")
    return df[m].copy()


def _spearman_df(X, min_p=2):
    return X.rank(axis=0, method="average", na_option="keep").corr(method="pearson", min_periods=min_p)


def _spearman_pair(x, y, min_p=2):
    m = x.notna() & y.notna()
    if m.sum() < min_p:
        return np.nan
    xr = x[m].rank(); yr = y[m].rank()
    if xr.nunique() <= 1 or yr.nunique() <= 1:
        return np.nan
    return float(np.corrcoef(xr.values, yr.values)[0, 1])


# ─── heatmap computation ───────────────────────────────────────────────────────

def _cross_section(df_day, alphas, stat, qf=None, tgts=None, bets=None):
    d = df_day[df_day["stat_type"] == stat]
    if qf:   d = d[d["qrank"].isin(qf)]
    if tgts: d = d[d["target"].isin(tgts)]
    if bets: d = d[d["bet_size_col"].isin(bets)]
    if d.empty:
        return None
    piv = d.pivot_table(index=["target", "bet_size_col", "qrank"], columns="signal",
                        values="value", aggfunc="sum", observed=True)
    cols = [a for a in alphas if a in piv.columns]
    if not cols:
        return None
    piv = piv[cols].dropna(how="all")
    if piv.shape[0] < 2:
        return None
    return pd.to_numeric(piv.stack(), errors="coerce").unstack().astype(float)


def _avg_mats(mats):
    if not mats:
        return None
    k    = mats[0].shape[0]
    sumM = np.zeros((k, k)); cntM = np.zeros((k, k), int)
    for M in mats:
        if M is None or M.shape != (k, k):
            continue
        m = np.isfinite(M)
        sumM[m] += M[m]; cntM[m] += 1
    with _no_err():
        H = sumM / np.where(cntM == 0, np.nan, cntM)
    H[cntM == 0] = np.nan
    for i in range(k):
        if not np.isfinite(H[i, i]):
            H[i, i] = 1.0
    return H


def compute_heatmap_daily_avg(stats_df, alphas, stat, min_pairs=2, qf=None, tgts=None, bets=None):
    alphas = list(alphas)
    if len(alphas) < 2:
        return None, alphas, 0
    if stat == "alpha_sum":
        df = stats_df[stats_df["stat_type"] == stat].copy()
        if qf:   df = df[df["qrank"].isin(qf)]
        if tgts: df = df[df["target"].isin(tgts)]
        if bets: df = df[df["bet_size_col"].isin(bets)]
        if df.empty:
            return None, alphas, 0
        wide = df.pivot_table(index="date", columns="signal", values="value",
                              aggfunc="sum", observed=True).reindex(columns=alphas).sort_index()
        if wide.shape[0] < min_pairs or wide.shape[1] < 2:
            return None, list(wide.columns), int(wide.shape[0])
        C = _spearman_df(wide, min_p=min_pairs).reindex(index=alphas, columns=alphas)
        return C.to_numpy(dtype=float), list(C.columns), int(wide.shape[0])
    mats = []
    for _, df_day in stats_df.groupby("date", sort=True, observed=True):
        X = _cross_section(df_day, alphas, stat, qf, tgts, bets)
        if X is None:
            continue
        C = _spearman_df(X, min_p=min_pairs).reindex(index=alphas, columns=alphas)
        M = C.to_numpy(dtype=float)
        if np.isfinite(M).sum() == 0:
            continue
        mats.append(M)
    if not mats:
        df2 = stats_df[stats_df["stat_type"] == stat].copy()
        if qf:   df2 = df2[df2["qrank"].isin(qf)]
        if tgts: df2 = df2[df2["target"].isin(tgts)]
        if bets: df2 = df2[df2["bet_size_col"].isin(bets)]
        if df2.empty:
            return None, alphas, 0
        wide = (df2.groupby(["date", "signal"], observed=True)["value"]
                .sum().unstack("signal").reindex(columns=alphas).sort_index().dropna(axis=1, how="all"))
        if wide.shape[0] < max(2, min_pairs) or wide.shape[1] < 2:
            return None, list(wide.columns), int(wide.shape[0])
        C = _spearman_df(wide, min_p=min_pairs).reindex(index=alphas, columns=alphas)
        return C.to_numpy(dtype=float), list(C.columns), int(wide.shape[0])
    return _avg_mats(mats), alphas, len(mats)


def compute_timeseries_heatmap(stats_df, alphas, stat, min_days=5, agg="sum",
                               qf=None, tgts=None, bets=None):
    df = stats_df[stats_df["stat_type"] == stat].copy()
    if qf:   df = df[df["qrank"].isin(qf)]
    if tgts: df = df[df["target"].isin(tgts)]
    if bets: df = df[df["bet_size_col"].isin(bets)]
    if df.empty:
        return None, alphas, 0
    gb    = df.groupby(["date", "signal"], observed=True)["value"]
    daily = (gb.sum() if agg == "sum" else gb.mean()).unstack("signal")
    daily = daily.reindex(columns=alphas).dropna(axis=1, how="all").sort_index()
    if daily.shape[1] < 2 or daily.shape[0] < min_days:
        return None, list(daily.columns), int(daily.shape[0])
    C = _spearman_df(daily, min_p=min_days)
    return C.values, C.columns.tolist(), int(daily.shape[0])


# ─── temporal correlation lines ───────────────────────────────────────────────

def _rolling_sp_pair(a, b, window=None, min_p=2):
    idx = a.index.union(b.index)
    a   = a.reindex(idx); b = b.reindex(idx)
    out = pd.Series(index=idx, dtype="float64")
    w   = None if (not window or int(window) <= 1) else int(window)
    for i in range(len(idx)):
        start = 0 if w is None else max(0, i - w + 1)
        xa = a.iloc[start:i + 1]; xb = b.iloc[start:i + 1]
        m  = xa.notna() & xb.notna()
        if m.sum() >= min_p:
            xr = xa[m].rank(); yr = xb[m].rank()
            out.iloc[i] = np.corrcoef(xr, yr)[0, 1] if (xr.nunique() > 1 and yr.nunique() > 1) else np.nan
        else:
            out.iloc[i] = np.nan
    return out


def compute_pairwise_rolling_corr(stats_df, alphas, stat, window=1, min_p=None,
                                  qf=None, tgts=None, bets=None, agg="mean"):
    df = stats_df[stats_df["stat_type"] == stat].copy()
    if qf:   df = df[df["qrank"].isin(qf)]
    if tgts: df = df[df["target"].isin(tgts)]
    if bets: df = df[df["bet_size_col"].isin(bets)]
    if df.empty:
        return {}
    gb    = df.groupby(["date", "signal"], observed=True)["value"]
    daily = (gb.sum() if agg == "sum" else gb.mean()).unstack("signal")
    daily = daily.reindex(columns=alphas).dropna(axis=1, how="all").sort_index()
    if daily.shape[1] < 2:
        return {}
    mp   = 2 if min_p is None else int(min_p)
    cols = [c for c in daily.columns if daily[c].notna().sum() >= mp]
    return {f"{a}|{b}": _rolling_sp_pair(daily[a], daily[b],
                                          window=None if (not window or int(window) <= 1) else int(window),
                                          min_p=mp)
            for a, b in combinations(cols, 2)}


def _plot_corr_lines(pdf, corr_map, stats_df, title, ylabel="Spearman corr", height=6.0):
    if not corr_map:
        fig, ax = plt.subplots(figsize=(14, height))
        _set_title(fig, ax, title + " — no data")
        fig.tight_layout(rect=[0.02, 0.06, 0.98, TEMPORAL_AX_TOP])
        savefig_white(pdf, fig)
        return
    coverage = sorted(((k, v.notna().sum()) for k, v in corr_map.items()),
                      key=lambda x: x[1], reverse=True)
    chosen   = [k for k, cnt in coverage[:8] if cnt > 0]
    dates    = sorted(stats_df["date"].dropna().unique())
    if not dates or not chosen:
        fig, ax = plt.subplots(figsize=(14, height))
        _set_title(fig, ax, title + " — insufficient data")
        fig.tight_layout(rect=[0.02, 0.06, 0.98, TEMPORAL_AX_TOP])
        savefig_white(pdf, fig)
        return
    fig, ax = plt.subplots(figsize=(14, height))
    _set_title(fig, ax, title)
    cmap     = mpl.colormaps.get_cmap("tab20")
    all_vals = []
    for i, key in enumerate(chosen):
        s     = corr_map[key].copy().sort_index()
        color = cmap(i % cmap.N)
        ax.plot(s.index, s.values, lw=1.3, alpha=0.85, color=color)
        v = s.values[np.isfinite(s.values)]
        if v.size:
            all_vals.append(v)
        fi = np.where(np.isfinite(s.values))[0]
        if fi.size:
            ax.text(s.index[fi[-1]], s.values[fi[-1]], f"  {key}", color=color, fontsize=9, va="center")
    if all_vals:
        vals = np.concatenate(all_vals)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        span = max(vmax - vmin, 1e-6); pad_ = 0.08 * span
        y0 = max(-1.05, vmin - pad_); y1 = min(1.05, vmax + pad_)
        if y1 - y0 < 0.2:
            mid = 0.5 * (y0 + y1); y0, y1 = mid - 0.1, mid + 0.1
        ax.set_ylim(y0, y1)
    else:
        ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel(ylabel)
    ax.set_xlim(pd.to_datetime(dates[0]), pd.to_datetime(dates[-1]))
    ax.margins(x=0.03)
    _plot_date_axis(ax)
    fig.tight_layout(rect=[0.02, 0.06, 0.98, TEMPORAL_AX_TOP])
    savefig_white(pdf, fig)


def plot_cross_section_corr_lines(pdf, stats_df, alphas, stat, title_prefix,
                                  smooth=1, height=6.0, qf=None, tgts=None, bets=None):
    dates = sorted(stats_df["date"].dropna().unique())
    pairs = list(combinations(alphas, 2))
    corr_map = {f"{a}|{b}": pd.Series(index=pd.DatetimeIndex(dates, dtype="datetime64[ns]"), dtype="float64")
                for a, b in pairs}
    for dt, df_day in stats_df.groupby("date", sort=True, observed=True):
        X = _cross_section(df_day, alphas, stat, qf, tgts, bets)
        if X is None:
            continue
        for a, b in pairs:
            if a in X.columns and b in X.columns:
                corr_map[f"{a}|{b}"].loc[dt] = _spearman_pair(
                    pd.to_numeric(X[a], errors="coerce"),
                    pd.to_numeric(X[b], errors="coerce"))
    if smooth and int(smooth) > 1:
        corr_map = {k: s.rolling(int(smooth), min_periods=1).mean() for k, s in corr_map.items()}
    suffix = f" — smoothed {int(smooth)}D" if (smooth and int(smooth) > 1) else " — daily"
    _plot_corr_lines(pdf, corr_map, stats_df, title_prefix + suffix)


def plot_pairwise_timecorr_lines(pdf, stats_df, alphas, stat, title_prefix,
                                 window=1, height=6.0, qf=None, tgts=None, bets=None, agg="mean"):
    corr_map = compute_pairwise_rolling_corr(stats_df, alphas, stat, window=window,
                                             qf=qf, tgts=tgts, bets=bets, agg=agg)
    suffix   = (f" — Rolling Spearman {int(window)}D"
                if (window and int(window) > 1) else " — Expanding Spearman")
    _plot_corr_lines(pdf, corr_map, stats_df, title_prefix + suffix,
                     ylabel="Spearman corr (time)")


# ─── alpha autodetect ─────────────────────────────────────────────────────────

def _autodetect_alphas(df, max_k=16):
    df = df[df["signal"].notna()]
    base = df[df["stat_type"] == "pnl"]
    dc   = (base.groupby("signal", observed=True)["date"].nunique()).sort_values(ascending=False)
    if dc.empty:
        dc = df.groupby("signal", observed=True)["date"].nunique().sort_values(ascending=False)
    cands = dc[dc >= 5].index.tolist() or dc.index.tolist()
    return list(map(str, cands[:max_k]))


# ─── outlier tables ───────────────────────────────────────────────────────────

def _find_latest_outliers(root):
    if not os.path.isdir(root):
        return None
    cands = sorted(glob.glob(os.path.join(root, "outliers_*.pkl")))
    return cands[-1] if cands else None


def _metric_table_rows(odf, metric, top_k, have_z):
    sub = odf[odf["stat_type"] == metric].copy()
    if sub.empty:
        return None, None
    if have_z:
        sub["z"] = pd.to_numeric(sub["z"], errors="coerce")
        sub = sub.sort_values("z")
        lows  = sub.head(top_k).copy()
        highs = sub.tail(top_k).iloc[::-1].copy()
    else:
        sub   = sub.sort_values("value")
        lows  = sub.head(top_k).copy()
        highs = sub.tail(top_k).iloc[::-1].copy()
    labels = ["Type", "Date", "Signal", "Bet", "Target", "Q", "Value"] + (["z"] if have_z else [])
    def row(r, kind):
        base = [kind,
                r["date"].strftime("%Y-%m-%d") if pd.notna(r["date"]) else "NaT",
                _ellipsis(r["signal"], 18), _ellipsis(r["bet_size_col"], 16),
                _ellipsis(r["target"], 16), str(r["qrank"]), f"{r['value']:.6g}"]
        if have_z:
            base.append("" if pd.isna(r.get("z")) else f"{r.get('z'):.2f}")
        return base
    rows = [row(r, "High") for _, r in highs.iterrows()] + [row(r, "Low") for _, r in lows.iterrows()]
    return labels, rows


def _draw_table(ax, title, col_labels, rows, fontsize=9):
    ax.axis("off")
    ax.set_title(title, fontsize=13, weight="bold", loc="left", pad=6)
    bw  = {"Type": 0.08, "Date": 0.11, "Signal": 0.18, "Bet": 0.14,
           "Target": 0.14, "Q": 0.06, "Value": 0.11, "z": 0.06}
    cw  = [bw.get(l, 0.10) for l in col_labels]
    s   = sum(cw)
    if s > 0.98:
        cw = [w * 0.98 / s for w in cw]
    tb  = ax.table(cellText=rows, colLabels=col_labels, colWidths=cw,
                   loc="upper left", cellLoc="left", bbox=[0.0, 0.0, 1.0, 0.92])
    tb.auto_set_font_size(False); tb.set_fontsize(fontsize); tb.scale(1.0, 1.10)
    hc = (0.9, 0.9, 0.92); ev = (0.98, 0.98, 0.985); od = (1.0, 1.0, 1.0)
    for (r, c), cell in tb.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", ha="left")
            cell.set_facecolor(hc); cell.set_edgecolor("0.75")
        else:
            cell.set_edgecolor("0.85"); cell.set_facecolor(ev if r % 2 == 0 else od)


def append_outlier_pages(pkl_path, pdf, metrics=None, top_k=3, per_page=3):
    try:
        if not pkl_path or not os.path.isfile(pkl_path):
            raise FileNotFoundError("Outliers PKL not found.")
        odf = read_pickle_compat(pkl_path)
    except Exception as e:
        fig, ax = plt.subplots(figsize=(14, 4)); ax.axis("off")
        fig.suptitle("Outlier Tables", fontsize=18, weight="bold")
        ax.text(0.5, 0.5, f"No outlier tables:\n{e}", ha="center", va="center", fontsize=12)
        savefig_white(pdf, fig); return

    if odf is None or len(odf) == 0:
        fig, ax = plt.subplots(figsize=(14, 4)); ax.axis("off")
        fig.suptitle("Outlier Tables", fontsize=18, weight="bold")
        ax.text(0.5, 0.5, "No outliers found.", ha="center", va="center", fontsize=12)
        savefig_white(pdf, fig); return

    odf = _apply_aliases(odf.copy())
    odf["date"] = pd.to_datetime(odf["date"], errors="coerce")
    odf = odf.dropna(subset=["date", "value"])
    if "target" in odf.columns:
        odf = odf[odf["target"] != "__ALL__"]
    have_z = "z" in odf.columns

    avail = odf["stat_type"].dropna().unique().tolist()
    if metrics is None:
        metrics = sorted(avail)
    else:
        metrics = [m for m in metrics if m in avail]
    if not metrics:
        fig, ax = plt.subplots(figsize=(14, 4)); ax.axis("off")
        fig.suptitle("Outlier Tables", fontsize=18, weight="bold")
        ax.text(0.5, 0.5, "Selected metrics not found.", ha="center", va="center", fontsize=12)
        savefig_white(pdf, fig); return

    tables = []
    for m in metrics:
        col_labels, rows = _metric_table_rows(odf, m, top_k, have_z)
        if col_labels and rows:
            tables.append((m, col_labels, rows))
    if not tables:
        return

    pp = max(1, int(per_page))
    for start in range(0, len(tables), pp):
        chunk = tables[start:start + pp]
        fig   = plt.figure(figsize=(14, 8.5))
        fig.suptitle("Outlier Tables", fontsize=18, weight="bold", y=0.985)
        gs = GridSpec(len(chunk), 1, figure=fig, left=0.03, right=0.97,
                      top=0.90, bottom=0.06, hspace=0.35)
        for i, (mn, cl, rws) in enumerate(chunk):
            _draw_table(fig.add_subplot(gs[i, 0]),
                        f"{mn} — Top Highs & Lows", cl, rws)
        savefig_white(pdf, fig)


# ─── temporal metric series ───────────────────────────────────────────────────

def _series(df, stat):
    s = df[df["stat_type"] == stat][["date", "value"]].set_index("date")["value"].astype(float)
    return s.sort_index()


def _tok(base, w, cum=False):
    if cum:
        base = f"cumulative {base}"
        return f"Rolling-mean {base} ({int(w)}D)" if (w and int(w) > 1) else base[0].upper() + base[1:]
    return f"Rolling-mean {base} ({int(w)}D)" if (w and int(w) > 1) else base


def _metric_series(metric, df, roll_windows, roll_sharpe):
    name = _canonical(str(metric))
    if df is None or df.empty:
        return None, None

    if name == "pnl":
        s = _series(df, "pnl")
        if s.empty:
            return None, None
        return _tok("PNL", roll_windows.get("pnl", 1), cum=True), _roll_mean(s.cumsum(), roll_windows.get("pnl", 1))

    if name == "ppd":
        # Cumulative PPD in bps = (cumΣPnL / cumΣB_t) × 10000
        # sizeNotional = ΣB_t (Eq.11), same as PPD denominator, so this is exact
        pnl = _series(df, "pnl"); sn = _series(df, "sizeNotional")
        if pnl.empty or sn.empty:
            return None, None
        cum = pd.concat([pnl.rename("pnl"), sn.rename("sizeNotional")], axis=1).sort_index()
        if cum.empty:
            return None, None
        cum["cp"] = cum["pnl"].cumsum()
        cum["cs"] = cum["sizeNotional"].cumsum()
        cs_arr    = cum["cs"].to_numpy()
        # Suppress startup spike: require cumulative notional to reach at least
        # min(5% of max, value at day-20) before showing ratio.
        # For short intervals (<20 days), no suppression is applied.
        T         = len(cs_arr)
        cs_max    = np.nanmax(cs_arr) if cs_arr.size else 0.0
        if T >= 20 and cs_max > 0:
            warmup_val = cs_arr[min(19, T - 1)]               # notional at day 20
            cs_thresh  = min(0.05 * cs_max, float(warmup_val) if np.isfinite(warmup_val) else 0.0)
        else:
            cs_thresh = 0.0                                    # no suppression for short intervals
        valid = np.isfinite(cs_arr) & (cs_arr > max(cs_thresh, 1e-12))
        y = np.divide(cum["cp"], cum["cs"],
                      out=np.full(len(cum), np.nan, dtype=float),
                      where=valid)
        y = pd.Series(y, index=cum.index) * 10000.0
        return _tok("PPD (bps)", roll_windows.get("ppd", 1), cum=True), _roll_mean(y, roll_windows.get("ppd", 1))

    if name == "nrInstr":
        s = _series(df, "nrInstr")
        if s.empty:
            return None, None
        w = roll_windows.get("nrInstr", 1)
        return _tok("nrInstr", w), _roll_mean(s, w)

    if name == "n_trades":
        s = _series(df, "n_trades")
        if s.empty:
            return None, None
        w = roll_windows.get("n_trades", 1)
        return _tok("n_trades", w), _roll_mean(s, w)

    if name == "sizeNotional":
        s = _series(df, "sizeNotional")
        if s.empty:
            return None, None
        w = roll_windows.get("sizeNotional", 1)
        return _tok("Daily Notional ($M)", w), _roll_mean(s, w) / 1e6

    if name == "sharpe":
        pnl = _series(df, "pnl")
        if pnl.empty:
            return None, None
        w = max(1, int(roll_sharpe))
        return f"Rolling Sharpe ({w}D)", _rolling_sharpe(pnl, w)

    s = _series(df, name)
    if s.empty:
        return None, None
    w = roll_windows.get(name, roll_windows.get("__default__", 1))
    return _tok(name, w), (_roll_mean(s, w) if w and int(w) > 1 else s)


def _plot_temporal_grid(pdf, df, qranks, qcolors, metrics, roll_windows,
                        roll_sharpe, grid, title_prefix, style="-"):
    rows, cols = grid
    per_page   = max(1, rows * cols)
    metrics    = [m for m in metrics if m]
    if not metrics:
        return
    handles = [Line2D([0], [0], color=qcolors.get(q, "gray"), linestyle=style, label=q)
               for q in qranks]
    for start in range(0, len(metrics), per_page):
        chunk = metrics[start:start + per_page]
        fig, axs = plt.subplots(rows, cols, figsize=PAGE_SIZE)
        axs = np.atleast_1d(axs).ravel()
        for ax in axs[len(chunk):]:
            ax.axis("off")
        any_plot = False
        for ax, metric in zip(axs, chunk):
            m_any = False; m_title = None
            for q in qranks:
                sq = df[df["qrank"] == q]
                title, series = _metric_series(metric, sq, roll_windows, roll_sharpe)
                if series is None or series.empty:
                    continue
                ax.plot(series.index, series.values,
                        color=qcolors.get(q, "gray"), linestyle=style, linewidth=1.3,
                        alpha=0.85, label=q)
                m_any = True; m_title = title or metric
            if m_any:
                any_plot = True
                ax.set_ylabel(m_title or str(metric))
                _plot_date_axis(ax)
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center", fontsize=9, color="0.35")
        fig.suptitle(f"{title_prefix} — Temporal metrics", fontsize=13, weight="semibold", y=0.97)
        if any_plot and handles:
            leg = fig.legend(handles=handles, title="Quantile",
                             fontsize=8.5, frameon=True,
                             loc="upper center", bbox_to_anchor=(0.5, 0.925),
                             ncol=len(handles), handlelength=1.8, columnspacing=1.2)
            leg.get_title().set_fontsize(8.5)
            leg.get_title().set_fontweight("semibold")
        fig.tight_layout(rect=[0.04, 0.04, 0.96, 0.89])
        savefig_white(pdf, fig)


# ─── CCF pages ────────────────────────────────────────────────────────────────

def _load_ccf(root, patterns):
    if not root or not os.path.isdir(root):
        return None
    for pat in (patterns if isinstance(patterns, (list, tuple)) else [patterns]):
        paths = sorted(glob.glob(os.path.join(root, pat)))
        if paths:
            try:
                df = read_pickle_compat(paths[-1])
                if isinstance(df, pd.DataFrame) and not df.empty:
                    print(f"[INFO] Loaded CCF: {os.path.basename(paths[-1])}  shape={df.shape}")
                    return df
            except Exception as e:
                print(f"[WARN] CCF load failed: {e}")
    return None


def _prep_ccf(df, max_lag):
    if df is None or df.empty:
        return None
    df = df.copy()
    if "lag" not in df.columns:
        return None
    if "corr" not in df.columns:
        nc = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if nc:
            df = df.rename(columns={nc[0]: "corr"})
        else:
            return None
    df["lag"]  = pd.to_numeric(df["lag"],  errors="coerce")
    df["corr"] = pd.to_numeric(df["corr"], errors="coerce")
    df = df[np.isfinite(df["lag"]) & np.isfinite(df["corr"])]
    if max_lag is not None:
        df = df[(df["lag"] >= -int(max_lag)) & (df["lag"] <= int(max_lag))]
    if df.empty:
        return None
    df["lag"] = df["lag"].astype(int)
    return df


def _ccf_summary_page(pdf, df_raw, df_pnl, max_lag=5):
    df_raw = _prep_ccf(df_raw, max_lag); df_pnl = _prep_ccf(df_pnl, max_lag)
    if df_raw is None and df_pnl is None:
        return
    n = 1 if (df_raw is None or df_pnl is None) else 2
    fig, axs = plt.subplots(1, n, figsize=(14, 4.8))
    axs = np.atleast_1d(axs)
    def _one(ax, df, title):
        st = df.groupby("lag")["corr"].agg(["mean", "median", "std"]).sort_index()
        if st.empty:
            ax.axis("off"); ax.set_title(title + " — no data", fontsize=11); return
        lags = st.index.values
        ax.errorbar(lags, st["mean"].values, yerr=st["std"].values,
                    fmt="-o", lw=1.8, capsize=3, label="mean±std")
        ax.plot(lags, st["median"].values, linestyle=":", marker="x", lw=1.5, label="median")
        ax.axhline(0, linestyle="--", lw=0.8, alpha=0.7)
        ax.set_xlabel("Lag"); ax.set_ylabel("Correlation"); ax.set_xticks(lags)
        ax.grid(True, linestyle=":", alpha=0.35, axis="y")
        ax.set_title(title, fontsize=11); ax.legend(fontsize=8, frameon=True)
    i = 0
    if df_raw is not None: _one(axs[i], df_raw, "RAW alpha vs SPY — CCF summary"); i += 1
    if df_pnl is not None: _one(axs[i], df_pnl, "PnL vs SPY — CCF summary")
    fig.suptitle("Cross-correlation vs SPY (per-ticker, aggregated)", fontsize=16, weight="bold", y=0.96)
    fig.tight_layout(rect=[0.03, 0.05, 0.97, 0.90])
    savefig_white(pdf, fig)


def _ccf_hist_pages(pdf, df_ccf, title, max_lag=5, bins=40):
    df_ccf = _prep_ccf(df_ccf, max_lag)
    if df_ccf is None or df_ccf.empty:
        fig, ax = plt.subplots(figsize=(14, 4)); ax.axis("off")
        ax.set_title(title); ax.text(0.5, 0.5, "No CCF data", ha="center", va="center")
        savefig_white(pdf, fig); return
    lags = sorted(df_ccf["lag"].unique())
    rows, cols, pp = 3, 4, 12
    for start in range(0, len(lags), pp):
        page_lags = lags[start:start + pp]
        fig, axs  = plt.subplots(rows, cols, figsize=(14, 8.5))
        axs = np.atleast_1d(axs).ravel()
        for ax in axs:
            ax.axis("off")
        for ax, lag in zip(axs, page_lags):
            x = pd.to_numeric(df_ccf[df_ccf["lag"] == lag]["corr"], errors="coerce")
            x = x[np.isfinite(x)]
            if x.empty:
                continue
            ax.hist(x.values, bins=bins, edgecolor="white", alpha=0.9)
            m = float(np.nanmean(x)); med = float(np.nanmedian(x)); sd = float(np.nanstd(x))
            for v, ls in [(m, "-"), (med, ":"), (m + sd, "--"), (m - sd, "--")]:
                ax.axvline(v, linestyle=ls, linewidth=1.2)
            ax.set_title(f"lag={lag}  n={int(x.size)}\nμ={m:.3f}  med={med:.3f}  σ={sd:.3f}", fontsize=8)
            ax.tick_params(labelsize=7); ax.grid(True, linestyle=":", alpha=0.25, axis="y")
        fig.suptitle(title, fontsize=16, weight="bold", y=0.98)
        fig.tight_layout(rect=[0.03, 0.04, 0.97, 0.94])
        savefig_white(pdf, fig)


def append_ccf_pages(per_ticker_dir, pdf, max_lag=5):
    if not per_ticker_dir or not os.path.isdir(per_ticker_dir):
        return False
    df_raw = _load_ccf(per_ticker_dir, ["mds_alpha_raw_spy_ccf_*.pkl", "per_ticker_alpha_raw_spy_ccf_*.pkl"])
    df_pnl = _load_ccf(per_ticker_dir, ["mds_alpha_pnl_spy_ccf_*.pkl", "per_ticker_alpha_pnl_spy_ccf_*.pkl"])
    if df_raw is None and df_pnl is None:
        print("[INFO] No CCF PKLs found; skipping CCF pages.")
        return False
    _ccf_summary_page(pdf, df_raw, df_pnl, max_lag=max_lag)
    if df_raw is not None:
        _ccf_hist_pages(pdf, df_raw, "CCF distributions — RAW alpha vs SPY (per lag)", max_lag=max_lag)
    if df_pnl is not None:
        _ccf_hist_pages(pdf, df_pnl, "CCF distributions — PnL vs SPY (per lag)", max_lag=max_lag)
    return True


def _distrib_page(pdf, df, title, bins=40):
    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(14, 4)); ax.axis("off")
        ax.set_title(title); ax.text(0.5, 0.5, "No data", ha="center", va="center")
        savefig_white(pdf, fig); return
    nc = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    col = "corr" if "corr" in df.columns else (nc[0] if nc else None)
    if col is None:
        fig, ax = plt.subplots(figsize=(14, 4)); ax.axis("off")
        ax.set_title(title); ax.text(0.5, 0.5, "No numeric column", ha="center", va="center")
        savefig_white(pdf, fig); return
    x = pd.to_numeric(df[col], errors="coerce"); x = x[np.isfinite(x)]
    if x.empty:
        fig, ax = plt.subplots(figsize=(14, 4)); ax.axis("off")
        ax.set_title(title); ax.text(0.5, 0.5, "All NaN", ha="center", va="center")
        savefig_white(pdf, fig); return
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.hist(x.values, bins=bins, edgecolor="white", alpha=0.9)
    m = float(np.nanmean(x)); med = float(np.nanmedian(x)); sd = float(np.nanstd(x))
    ax.axvline(m, linestyle="-", lw=2); ax.axvline(med, linestyle=":", lw=2)
    ax.axvline(m + sd, linestyle="--", lw=1); ax.axvline(m - sd, linestyle="--", lw=1)
    ax.set_xlabel("Correlation"); ax.set_ylabel("Count")
    ax.set_title(f"{title}\nmean={m:.3f}  median={med:.3f}  std={sd:.3f}")
    savefig_white(pdf, fig)


def _safe_resolve(name, desired, series_vals, prefer_prefix=None):
    vals = pd.Series(series_vals).dropna().astype(str)
    if vals.empty:
        return None
    try:
        if isinstance(desired, str) and desired.upper() == "AUTO":
            if prefer_prefix:
                vsub = vals[vals.str.startswith(prefer_prefix)]
                if not vsub.empty:
                    return vsub.value_counts().index.tolist()[:1]
            return vals.value_counts().index.tolist()[:1]
        out     = [desired] if isinstance(desired, str) else list(desired)
        missing = [v for v in out if v not in set(vals)]
        if missing:
            return None
        return out
    except Exception:
        return None


# ─── main entry point ─────────────────────────────────────────────────────────

def generate_quantile_report(config: dict):
    global META_TEXT

    daily_dir      = config["daily_dir"]
    summary_dir    = config["summary_dir"]
    per_ticker_dir = config["per_ticker_dir"]
    outliers_dir   = config["outliers_dir"]
    output_pdf     = config["output_pdf"]

    qranks_req  = [str(q) for q in config.get("qranks", [])][:4]
    allow_miss  = bool(config.get("allow_missing_qranks", False))

    roll_h1     = int(config.get("roll_h1_lines",    30))
    roll_h2     = int(config.get("roll_h2_lines",    30))
    roll_h3     = int(config.get("roll_h3_lines",     1))
    roll_nr     = int(config.get("roll_nrinstr",      1))
    roll_ppd    = int(config.get("roll_ppd",          1))
    roll_tr     = int(config.get("roll_trades",       1))
    roll_pnl    = int(config.get("roll_pnl",          1))
    roll_sn     = int(config.get("roll_size_notional",1))
    roll_sr     = int(config.get("roll_sharpe",      60))

    temp_vars   = _norm_metrics(config.get("variables_temporal_plot", [])) or ["pnl", "ppd", "n_trades", "sizeNotional", "sharpe"]
    arr_dim     = config.get("arrayDim_temporal_plot", (2, 2))
    try:
        tr, tc  = int(arr_dim[0]), int(arr_dim[1])
    except Exception:
        tr, tc  = 2, 2
    tr = max(1, tr); tc = max(1, tc)

    bar_pv      = list(config.get("bar_page_vars", []))
    bar_xv      = list(config.get("bar_x_vars",   []))
    bar_metrics = _norm_metrics(config.get("bar_metrics", []))

    out_metrics = _norm_metrics(config.get("outlier_metrics_for_tables", []))
    out_topk    = int(config.get("outlier_top_k",          3))
    out_pp      = int(config.get("outlier_tables_per_page", 3))

    style1      = config.get("style_first", "-")
    # Default quantile color palette: muted, colorblind-safe, publication quality.
    # User can override via config["quantile_colors"] — but only if they've changed
    # from the old matplotlib defaults (red/green/blue/black). This prevents
    # main.py's hardcoded legacy colors from silently overriding the new palette.
    _DEFAULT_Q_COLORS = {
        "qr_100": "#2166AC",   # steel blue
        "qr_75":  "#4DAC26",   # muted green
        "qr_50":  "#D6604D",   # muted coral/red
        "qr_25":  "#9970AB",   # muted purple — more visible than grey on warm white
    }
    _LEGACY_Q_COLORS = {
        "qr_100": "red",  "qr_75": "green",
        "qr_50":  "blue", "qr_25": "black",
    }
    raw_colors = config.get("quantile_colors", {})
    # If user passed the old hardcoded defaults unchanged, ignore them so the
    # new professional palette applies. Only honour genuine user overrides.
    if raw_colors == _LEGACY_Q_COLORS or not raw_colors:
        qcolors_cfg = dict(_DEFAULT_Q_COLORS)
    else:
        qcolors_cfg = {**_DEFAULT_Q_COLORS, **raw_colors}
    ccf_enable  = bool(config.get("ccf_enable", True))
    ccf_max_lag = int(config.get("ccf_max_lag",  5))
    iv_start    = config.get("interval_start")
    iv_end      = config.get("interval_end")

    stats_daily, stats_summary, dmin, dmax, ndays = _load_data(
        daily_dir, summary_dir, interval_start=iv_start, interval_end=iv_end)
    stats_daily   = _apply_aliases(stats_daily)
    stats_summary = _apply_aliases(stats_summary)

    META_TEXT = config.get("meta_text") or \
        f"Window: {dmin:%Y-%m-%d} → {dmax:%Y-%m-%d}  |  Days: {ndays}"

    bars_src = stats_summary if (isinstance(stats_summary, pd.DataFrame) and not stats_summary.empty) else None
    print("[INFO] Bar plots source:", "SUMMARY" if bars_src is not None else "NONE (skipped)")

    qr_src     = bars_src if bars_src is not None else stats_daily
    qranks_all = _sorted_qranks(qr_src["qrank"])
    if not qranks_req:
        qranks = qranks_all
    else:
        if not allow_miss:
            missing = [q for q in qranks_req if q not in qranks_all]
            if missing:
                print(f"[WARN] qranks not found, ignored: {missing}")
        qranks = [q for q in qranks_req if (allow_miss or q in qranks_all)] or qranks_all

    qcolors  = _ensure_colors(qranks, qcolors_cfg)
    bar_w    = 0.18

    daily_nonall = _exclude_all(stats_daily)

    h23_tgts = _safe_resolve("H2_targets", config.get("H2_targets", "AUTO"),
                             stats_daily["target"] if "target" in stats_daily else []) or None
    h23_bets = _safe_resolve("H2_bets", config.get("H2_bets", "AUTO"),
                             stats_daily["bet_size_col"] if "bet_size_col" in stats_daily else [],
                             prefer_prefix="betsize") or None
    h3_tgts  = _safe_resolve("H3_targets", config.get("H3_targets", "AUTO"),
                             stats_daily["target"] if "target" in stats_daily else []) or None
    h3_bets  = _safe_resolve("H3_bets", config.get("H3_bets", "AUTO"),
                             stats_daily["bet_size_col"] if "bet_size_col" in stats_daily else [],
                             prefer_prefix="betsize") or None

    alphas     = _autodetect_alphas(daily_nonall, max_k=16)
    do_temp    = (len(alphas) <= 6)
    h1_stat    = "alpha_sum"
    print(f"[INFO] H1 stat: {h1_stat}  |  alphas detected: {len(alphas)}")

    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)

    with PdfPages(output_pdf) as pdf:

        # ── Bar plots (SUMMARY only) ──────────────────────────────────────────
        if bars_src is not None and bar_metrics:
            ssp = _exclude_all(bars_src)
            if bar_pv and all(ssp[c].notna().any() for c in bar_pv):
                page_iter = list(product(*[sorted(ssp[col].dropna().unique()) for col in bar_pv]))
            else:
                page_iter = []

            for page_vals in page_iter:
                subset = ssp.copy(); bits = []
                for var, val in zip(bar_pv, page_vals):
                    subset = subset[subset[var] == val]; bits.append(f"{var}: {val}")
                if subset.empty:
                    continue
                if bar_xv:
                    subset = subset.copy()
                    subset["x_key"] = subset[bar_xv].astype(str).agg("|".join, axis=1)
                    x_levels = sorted(subset["x_key"].unique())
                else:
                    subset["x_key"] = "ALL"; x_levels = ["ALL"]

                nc = 2; nr2 = int(np.ceil(len(bar_metrics) / nc))
                fh = max(PAGE_SIZE[1], 3.6 * nr2 + 2.2)
                fig, axs = plt.subplots(nr2, nc, figsize=(PAGE_SIZE[0], fh))
                axs = np.atleast_1d(axs).ravel()
                fig.suptitle("Summary Metrics  ·  " + "  ·  ".join(bits),
                             fontsize=12, weight="semibold", y=0.97, color="0.15")
                xl = " | ".join(bar_xv) if bar_xv else "ALL"
                fig.text(0.5, 0.925, f"grouped by  {xl}", ha="center", va="top",
                         fontsize=8.5, color="0.50", style="italic")

                leg_handles = []; leg_labels = []
                for i, metric in enumerate(bar_metrics):
                    ax   = axs[i]
                    disp = _metric_label(metric)
                    data = subset[subset["stat_type"] == metric].copy()
                    if data.empty:
                        ax.set_title(f"{disp}: no data", fontsize=11); ax.axis("off"); continue

                    unit = ""
                    if metric.lower() == "ppd":
                        data["value"] = data["value"] * 10000; unit = " (bps)"
                    elif metric == "sizeNotional":
                        data["value"] = data["value"] / 1e6;   unit = " ($M)"

                    if "date" in data.columns:
                        data = data.sort_values("date")

                    if data["qrank"].notna().any():
                        keys   = ["x_key", "qrank"]
                        dd     = data.drop_duplicates(subset=keys, keep="last")
                        try:
                            pivot = dd.pivot(index="x_key", columns="qrank", values="value")
                        except ValueError:
                            dd    = dd.groupby(keys, as_index=False)["value"].mean()
                            pivot = dd.pivot(index="x_key", columns="qrank", values="value")
                        use_q = [q for q in qranks if q in pivot.columns]
                        x     = np.arange(len(x_levels))
                        offs  = np.linspace(-(len(use_q) - 1) / 2, (len(use_q) - 1) / 2, len(use_q)) * bar_w
                        for j, q in enumerate(use_q):
                            vals = pd.to_numeric(pivot.get(q), errors="coerce").reindex(x_levels).astype(float)
                            if vals.notna().any():
                                ax.bar(x + offs[j], vals.fillna(0.0).values, width=bar_w,
                                       color=qcolors.get(q, "gray"), label=q)
                                if q not in leg_labels:
                                    leg_labels.append(q)
                                    leg_handles.append(plt.Rectangle((0, 0), 1, 1, color=qcolors.get(q, "gray")))
                    else:
                        dd   = data.drop_duplicates(subset=["x_key"], keep="last")
                        vals = dd.set_index("x_key")["value"].reindex(x_levels).fillna(0.0).values
                        ax.bar(np.arange(len(x_levels)), vals, width=bar_w, color="gray")

                    if metric in ("long_ratio", "hit_ratio"):
                        yl, yh = ax.get_ylim()
                        if yl <= 0.5 <= yh:
                            ax.axhline(y=0.5, color="red", linestyle=":", lw=1.5, alpha=0.7, zorder=0)

                    ax.set_ylabel(f"{disp}{unit}", fontsize=9, color="0.25")
                    ax.set_xticks(np.arange(len(x_levels)))
                    ax.set_xticklabels([str(v) for v in x_levels], rotation=15,
                                       ha="right", fontsize=7.5)
                    ax.margins(y=0.22)
                    ax.tick_params(axis="y", labelsize=8)
                    ax.tick_params(axis="x", pad=2)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.spines["left"].set_linewidth(0.5)
                    ax.spines["bottom"].set_linewidth(0.5)

                for j in range(len(bar_metrics), len(axs)):
                    axs[j].axis("off")
                if leg_handles:
                    leg = fig.legend(leg_handles, leg_labels, title="Quantile",
                                     bbox_to_anchor=(0.5, 0.908), loc="upper center",
                                     ncol=len(leg_handles), fontsize=8.5, frameon=True,
                                     handlelength=1.4, columnspacing=1.0)
                    leg.get_title().set_fontsize(8.5)
                    leg.get_title().set_fontweight("semibold")
                plt.tight_layout(rect=[0.04, 0.06, 0.96, 0.80], h_pad=3.0)
                savefig_white(pdf, fig)

        # ── H1 heatmap ────────────────────────────────────────────────────────
        h1_src = stats_daily if h1_stat == "alpha_sum" else daily_nonall
        if len(alphas) >= 2:
            H1, lbl1, nd1 = compute_heatmap_daily_avg(h1_src, alphas, stat=h1_stat, min_pairs=2)
            fig, ax, cax  = _heatmap_axes(len(lbl1) if lbl1 else 0)
            desc = "Alpha Time-Series Corr (Spearman, alpha_sum)"
            if H1 is None or nd1 == 0:
                ax.axis("off"); _set_title(fig, ax, f"Heatmap 1 — {desc}")
                ax.text(0.5, 0.5, "No sufficient data.", ha="center", va="center")
                if cax: cax.axis("off")
            else:
                _draw_heatmap(fig, ax, cax, H1, lbl1, f"Heatmap 1 — {desc} (avg {nd1} days)")
            fig.tight_layout(rect=[0.02, 0.06, 0.98, HEATMAP_AX_TOP])
            savefig_white(pdf, fig)
            if do_temp:
                plot_cross_section_corr_lines(pdf, h1_src, alphas, h1_stat,
                                              "[H1] Alpha vs Alpha", smooth=roll_h1)

        # ── H2 & H3 per quantile ──────────────────────────────────────────────
        for q in qranks:
            qdf = daily_nonall[daily_nonall["qrank"] == q].copy()
            td2 = f"targets={','.join(h23_tgts) if h23_tgts else 'ALL'}"
            bd2 = f"bets={','.join(h23_bets) if h23_bets else 'ALL'}"

            H2, lbl2, nd2 = compute_timeseries_heatmap(qdf, alphas, "pnl", min_days=2,
                                                        qf=[q], tgts=h23_tgts, bets=h23_bets)
            h2t = f"Heatmap 2 — Time-series Spearman of daily PnL ({td2}, {bd2}) [{q}]"
            fig, ax, cax = _heatmap_axes(len(lbl2) if lbl2 else 0)
            if H2 is None or nd2 == 0:
                ax.axis("off"); _set_title(fig, ax, h2t)
                ax.text(0.5, 0.5, "No sufficient data.", ha="center", va="center")
                if cax: cax.axis("off")
            else:
                _draw_heatmap(fig, ax, cax, H2, lbl2, f"{h2t} (avg {nd2} days)")
            fig.tight_layout(rect=[0.02, 0.06, 0.98, HEATMAP_AX_TOP])
            savefig_white(pdf, fig)
            if do_temp and len(alphas) >= 2 and H2 is not None:
                plot_pairwise_timecorr_lines(pdf, qdf, alphas, "pnl",
                                             f"[H2|{q}] Alpha vs Alpha — Time corr (PnL, {td2}, {bd2})",
                                             window=roll_h2, qf=[q], tgts=h23_tgts, bets=h23_bets)

            td3 = f"targets={','.join(h3_tgts) if h3_tgts else 'ALL'}"
            bd3 = f"bets={','.join(h3_bets) if h3_bets else 'ALL'}"
            C3, lbl3, nd3 = compute_timeseries_heatmap(qdf, alphas, "pnl", min_days=5,
                                                        qf=[q], tgts=h3_tgts, bets=h3_bets)
            h3t = f"Heatmap 3 — Time-series Spearman of daily PnL vectors ({td3}, {bd3}) [{q}]"
            fig, ax, cax = _heatmap_axes(len(lbl3) if lbl3 else 0)
            if C3 is None or nd3 < 5:
                ax.axis("off"); _set_title(fig, ax, h3t)
                ax.text(0.5, 0.5, "Not enough days (need ≥5).", ha="center", va="center")
                if cax: cax.axis("off")
            else:
                _draw_heatmap(fig, ax, cax, C3, lbl3, f"{h3t} (days={nd3})")
            fig.tight_layout(rect=[0.02, 0.06, 0.98, HEATMAP_AX_TOP])
            savefig_white(pdf, fig)
            if do_temp and len(alphas) >= 2 and C3 is not None:
                plot_pairwise_timecorr_lines(pdf, qdf, alphas, "pnl",
                                             f"[H3|{q}] Alpha vs Alpha — Time corr (PnL vectors, {td3}, {bd3})",
                                             window=roll_h3, qf=[q], tgts=h3_tgts, bets=h3_bets)

        # ── Temporal pages ────────────────────────────────────────────────────
        # Adaptive smoothing: apply a data-length-aware minimum smoothing window
        # for noisy daily series. The minimum is min(20, ndays//5) so it never
        # blanks out short intervals, and caps at ndays//3 to avoid over-smoothing.
        _ndays    = int(daily_nonall["date"].nunique()) if "date" in daily_nonall.columns else 252
        _smooth_n = max(1, min(20, _ndays // 5))    # min smoothing window
        _smooth_s = max(1, min(20, _ndays // 5))    # same for sizeNotional
        _smooth_i = max(1, min(5,  _ndays // 10))   # smaller for nrInstr
        roll_windows = {"pnl":          roll_pnl,
                        "ppd":          roll_ppd,
                        "n_trades":     max(roll_tr, _smooth_n),
                        "sizeNotional": max(roll_sn, _smooth_s),
                        "nrInstr":      max(roll_nr, _smooth_i),
                        "__default__":  1}
        for target in sorted(daily_nonall["target"].dropna().unique()):
            if target == "__ALL__":
                continue
            for signal in sorted(daily_nonall["signal"].dropna().unique()):
                for bet in sorted(daily_nonall["bet_size_col"].dropna().unique()):
                    sub = daily_nonall[
                        (daily_nonall["target"] == target) &
                        (daily_nonall["signal"] == signal) &
                        (daily_nonall["bet_size_col"] == bet)
                    ].copy()
                    if sub.empty:
                        continue
                    _plot_temporal_grid(pdf, sub, qranks, qcolors, temp_vars,
                                        roll_windows, roll_sr, (tr, tc),
                                        f"{target} | {signal} | {bet}", style=style1)

        # ── CCF / corr distribution pages ─────────────────────────────────────
        ccf_done = False
        if ccf_enable:
            ccf_done = append_ccf_pages(per_ticker_dir, pdf, max_lag=ccf_max_lag)

        if not ccf_done and per_ticker_dir and os.path.isdir(per_ticker_dir):
            for pat, title in [
                ("mds_alpha_raw_spy_corr_*.pkl",        "Per-ticker correlation: RAW alpha vs SPY"),
                ("per_ticker_alpha_raw_spy_corr_*.pkl", "Per-ticker correlation: RAW alpha vs SPY"),
                ("mds_alpha_pnl_spy_corr_*.pkl",        "Per-ticker correlation: PnL vs SPY"),
                ("per_ticker_alpha_pnl_spy_corr_*.pkl", "Per-ticker correlation: PnL vs SPY"),
            ]:
                paths = sorted(glob.glob(os.path.join(per_ticker_dir, pat)))
                if paths:
                    try:
                        _distrib_page(pdf, read_pickle_compat(paths[-1]), title)
                    except Exception as e:
                        print(f"[WARN] corr distrib page failed: {e}")

        # ── Outlier tables ────────────────────────────────────────────────────
        op = _find_latest_outliers(outliers_dir)
        if op:
            append_outlier_pages(op, pdf, metrics=out_metrics, top_k=out_topk, per_page=out_pp)
        else:
            fig, ax = plt.subplots(figsize=(14, 4)); ax.axis("off")
            fig.suptitle("Outlier Tables", fontsize=18, weight="bold", y=0.985)
            ax.text(0.5, 0.5, "No outlier PKL found; skipping.", ha="center", va="center", fontsize=12)
            savefig_white(pdf, fig)

        t1 = time.perf_counter()
        print(f"[INFO] PDF written: {output_pdf}")
        print(f"[INFO] Total time:  {t1 - t0:.2f}s")