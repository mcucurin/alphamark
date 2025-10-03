import os
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from itertools import product

# ────────────────────────────────────────────────────────────────
# CONFIG
# Choose any 1–3 x variables from: 'alpha' (= signal), 'target', 'bet_size'
# Order matters for page/grid sequencing.
# Examples:
#   x_variables = ['alpha']
#   x_variables = ['alpha', 'target']
#   x_variables = ['alpha', 'target', 'bet_size']
# ────────────────────────────────────────────────────────────────
x_variables = ['alpha']  # edit me

input_pkl = "output/DAILY_SUMMARIES/stats_tensor.pkl"
out_pdf   = "output/Quantile_Combined_Report.pdf"

# Metrics to bar-plot (each page will iterate over these)
metrics_to_plot = [
    'pnl', 'ppd', 'sharpe', 'hit_ratio', 'long_ratio',
    'nrInstr', 'sizeNotional', 'r2', 't_stat', 'n_trades', 'ppt'
]

# Colors for quantile ranks
quantile_colors = {
    'qr_100': 'red',
    'qr_75': 'green',
    'qr_50': 'blue',
    'qr_25': 'black'
}
bar_width = 0.15

# ────────────────────────────────────────────────────────────────
# LOAD DATA
# ────────────────────────────────────────────────────────────────
with open(input_pkl, "rb") as f:
    stats_df = pickle.load(f)

stats_df['date'] = pd.to_datetime(stats_df['date'], errors='coerce')
stats_df = stats_df.dropna(subset=['date', 'value'])

# Canonical uniques (stable order)
targets_all = list(pd.Index(stats_df['target'].dropna().unique()).sort_values())
signals_all = list(pd.Index(stats_df['signal'].dropna().unique()).sort_values())
bets_all    = list(pd.Index(stats_df['bet_size_col'].dropna().unique()).sort_values())
qranks_all  = sorted(stats_df['qrank'].dropna().unique(), key=lambda x: float(x.split('_')[1]))

# Mappings
NAME2COL = {
    'alpha': 'signal',
    'target': 'target',
    'bet_size': 'bet_size_col',
}
NAME2ALL = {
    'alpha': signals_all,
    'target': targets_all,
    'bet_size': bets_all,
}
PRETTY = {'alpha': 'Alpha (signal)', 'target': 'Target', 'bet_size': 'Bet size'}

# ────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────
def annotate_corner(fig, text):
    fig.text(0.995, 0.995, text, ha='right', va='top', fontsize=10, alpha=0.75)

def seq_qoffsets(nq):
    return np.arange(-(nq-1)/2, (nq+1)/2) * bar_width

def filter_df(df, **kwargs):
    mask = np.ones(len(df), dtype=bool)
    for k, v in kwargs.items():
        col = NAME2COL.get(k, k)  # allow passing raw columns too
        mask &= (df[col] == v)
    return df[mask].copy()

def collect_values_for_metric(df, metric, x_name, fixed_filters):
    x_col = NAME2COL[x_name]
    mask = (df['stat_type'] == metric)
    for k, v in fixed_filters.items():
        mask &= (df[NAME2COL[k]] == v)
    dsub = df[mask].copy()
    if dsub.empty:
        # return zeros for all categories
        x_vals = NAME2ALL[x_name]
        return x_vals, {q: np.zeros(len(x_vals)) for q in qranks_all}
    pivot = dsub.pivot_table(index=x_col, columns='qrank', values='value',
                             aggfunc='sum', fill_value=0.0)
    x_vals = NAME2ALL[x_name]
    out = {}
    for q in qranks_all:
        series = pivot[q] if q in pivot.columns else pd.Series(0.0, index=[])
        vec = pd.Series(series).reindex(x_vals).fillna(0.0).to_numpy()
        out[q] = vec
    return x_vals, out

def plot_metric_single_x(fig, ax, metric, x_name, fixed_filters, title=None):
    x_vals, values_by_q = collect_values_for_metric(stats_df, metric, x_name, fixed_filters)
    x = np.arange(len(x_vals))
    offsets = seq_qoffsets(len(qranks_all))
    for j, qrank in enumerate(qranks_all):
        color = quantile_colors.get(qrank, 'gray')
        ax.bar(x + offsets[j], values_by_q[qrank], width=bar_width, label=qrank, color=color)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x_vals], rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    if title:
        ax.set_title(title, fontsize=12)

def plot_small_multiple_for_metric_two(fig, axes, metric, x_pair, fixed_filters, grid_vals):
    """Two-variable small multiples (x_pair length == 2).
    Grid axes shape: (len(v1), len(v2)) in this order.
    Each tile shows bars (one per qrank) for that combo, with height=sum(value).
    """
    v1, v2 = x_pair
    col1, col2 = NAME2COL[v1], NAME2COL[v2]
    # prefilter
    mask = (stats_df['stat_type'] == metric)
    for k, v in fixed_filters.items():
        mask &= (stats_df[NAME2COL[k]] == v)
    dsub = stats_df[mask].copy()

    for i1, val1 in enumerate(grid_vals[v1]):
        for i2, val2 in enumerate(grid_vals[v2]):
            ax = axes[i1, i2]
            cell = dsub[(dsub[col1] == val1) & (dsub[col2] == val2)]
            heights = []
            for q in qranks_all:
                heights.append(cell[cell['qrank'] == q]['value'].sum() if not cell.empty else 0.0)
            x = np.arange(len(qranks_all))
            for j, qrank in enumerate(qranks_all):
                color = quantile_colors.get(qrank, 'gray')
                ax.bar(j, heights[j], width=0.8, color=color)
            ax.set_xticks(x)
            ax.set_xticklabels(qranks_all, rotation=45, ha='right', fontsize=8)
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            ax.set_title(f"{PRETTY[v1]}={val1} | {PRETTY[v2]}={val2}", fontsize=9)
    fig.supylabel(metric)

def tile_sequence_two(x_pair, grid_vals):
    """Return the tile sequence for 2D grid in nested loop order (v1 major, then v2)."""
    v1, v2 = x_pair
    seq = []
    for val1 in grid_vals[v1]:
        for val2 in grid_vals[v2]:
            seq.append({v1: val1, v2: val2})
    return seq

def plot_small_multiple_for_metric_three(fig, axes, metric, x_pair2, fixed_filters, grid_vals, title_suffix=""):
    """For 3+ x variables, we grid the first two and keep the rest fixed via fixed_filters.
    x_pair2 length == 2 (which two variables to form the grid).
    """
    plot_small_multiple_for_metric_two(fig, axes, metric, x_pair2, fixed_filters, grid_vals)
    if title_suffix:
        fig.suptitle(fig._suptitle.get_text() + " " + title_suffix, fontsize=16, weight='bold')

def tile_sequence_three(x_pair2, grid_vals, extra_fixed):
    """Sequence of tiles (first two x vars vary in nested order); returns dicts merging extra_fixed."""
    v1, v2 = x_pair2
    seq = []
    for val1 in grid_vals[v1]:
        for val2 in grid_vals[v2]:
            d = {**extra_fixed}
            d[v1] = val1
            d[v2] = val2
            seq.append(d)
    return seq

def make_cum_pnl_ppd_page(filters_dict, title_note, pdf):
    """Replicates 'Cumulative P&L + Cumulative PPD' for a fixed selection across
    variables and a single combination of x-variable values (and any fixed vars).
    Shows one line per qrank.
    """
    # We need pnl + sizeNotional over time for the selection
    subset = stats_df[
        (stats_df['stat_type'].isin(['pnl', 'sizeNotional']))
    ].copy()

    for k, v in filters_dict.items():
        subset = subset[subset[NAME2COL[k]] == v]

    if subset.empty:
        return

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()
    ax1.set_title(f"Cumulative P&L + Cumulative PPD  |  {title_note}", fontsize=16)

    for qrank in qranks_all:
        sub_q = subset[subset['qrank'] == qrank]
        pnl_data = sub_q[sub_q['stat_type'] == 'pnl'].sort_values('date')[['date','value']]
        notional_data = sub_q[sub_q['stat_type'] == 'sizeNotional'].sort_values('date')[['date','value']]
        if pnl_data.empty or notional_data.empty:
            continue

        merged = pd.merge(
            pnl_data.rename(columns={'value':'pnl'}),
            notional_data.rename(columns={'value':'notional'}),
            on='date', how='outer'
        ).sort_values('date')
        merged[['pnl','notional']] = merged[['pnl','notional']].fillna(0.0)
        merged['cum_pnl'] = merged['pnl'].cumsum()
        merged['cum_notional'] = merged['notional'].cumsum()
        merged['cum_ppd'] = np.where(merged['cum_notional'] > 0,
                                     merged['cum_pnl'] / merged['cum_notional'],
                                     np.nan)
        color = quantile_colors.get(qrank, 'gray')
        ax1.plot(merged['date'], merged['cum_pnl'], label=qrank, color=color, linewidth=1.5)
        ax2.plot(merged['date'], merged['cum_ppd'], color=color, linestyle='--', alpha=0.9)

    ax1.set_ylabel("Cumulative P&L")
    ax2.set_ylabel("Cumulative PPD (cum PnL / cum Notional)")
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()
    ax1.legend(title='Quantile Rank', loc='upper left')
    annotate_corner(fig, title_note)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def pretty_combo(d):
    return " | ".join(f"{PRETTY[k]}={d[k]}" for k in d)

# ────────────────────────────────────────────────────────────────
# PAGE ORCHESTRATION
# ────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

with PdfPages(out_pdf) as pdf:

    # Identify non-x (fixed) variables
    all_vars = ['alpha', 'target', 'bet_size']
    xvars = list(x_variables)
    assert 1 <= len(xvars) <= 3, "x_variables must be any subset of ['alpha','target','bet_size'] with length 1..3."
    fixed_vars = [v for v in all_vars if v not in xvars]

    # Prepare all combinations of fixed vars (cartesian)
    fixed_lists = [NAME2ALL[v] for v in fixed_vars]
    if len(fixed_vars) == 0:
        fixed_combos = [dict()]  # nothing fixed
    else:
        fixed_combos = []
        for tup in product(*fixed_lists):
            fixed_combos.append({fv: tup[i] for i, fv in enumerate(fixed_vars)})

    # ── CASE 1: single x variable -> big vertical stack of metrics; after-page cum per each x value
    if len(xvars) == 1:
        xvar = xvars[0]
        x_values = NAME2ALL[xvar]

        for fcombo in fixed_combos:
            # Bar page with all metrics stacked
            fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(14, 2.7 * len(metrics_to_plot)))
            fig.suptitle(
                f"{PRETTY[xvar]} on X   |   " +
                (" | ".join(f"{PRETTY[k]}={fcombo[k]}" for k in fcombo) if fcombo else "(no fixed vars)"),
                fontsize=18, weight='bold'
            )
            annotate_corner(fig, pretty_combo(fcombo) if fcombo else "")

            for i, metric in enumerate(metrics_to_plot):
                ax = axs[i] if isinstance(axs, np.ndarray) else axs
                plot_metric_single_x(fig, ax, metric, xvar, fixed_filters=fcombo,
                                     title=None if i else f"Bars colored by Quantile Rank")
                if i == 0:
                    # one legend at top
                    ax.legend(title='Quantile Rank', bbox_to_anchor=(1.01, 1), loc='upper left')

            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)

            # After-page cumulative sequence, one per x value in the same order as the x-axis
            for xv in x_values:
                sel = {**fcombo, xvar: xv}
                note = (pretty_combo(fcombo) + (" | " if fcombo else "") + f"{PRETTY[xvar]}={xv}") if fcombo else f"{PRETTY[xvar]}={xv}"
                make_cum_pnl_ppd_page(sel, note, pdf)

    # ── CASE 2: two x variables -> small multiples grid per metric; after-page cum per cell (same order)
    elif len(xvars) == 2:
        v1, v2 = xvars  # grid order: rows=v1, cols=v2
        v1_vals, v2_vals = NAME2ALL[v1], NAME2ALL[v2]
        grid_vals = {v1: v1_vals, v2: v2_vals}

        # Reasonable figure size scaling with grid
        fig_w = 3.6 * max(2, len(v2_vals))
        fig_h = 2.8 * max(2, len(v1_vals))

        for fcombo in fixed_combos:
            for metric in metrics_to_plot:
                fig, axes = plt.subplots(len(v1_vals), len(v2_vals), figsize=(fig_w, fig_h), squeeze=False)
                supt = (f"Small Multiples by {PRETTY[v1]}×{PRETTY[v2]} | " +
                        (pretty_combo(fcombo) if fcombo else "(no fixed vars)") +
                        f" | metric={metric}")
                fig.suptitle(supt, fontsize=16, weight='bold')
                annotate_corner(fig, pretty_combo(fcombo) if fcombo else "")

                plot_small_multiple_for_metric_two(fig, axes, metric, [v1, v2], fixed_filters=fcombo, grid_vals=grid_vals)
                # Legend for qrank
                handles = [plt.Rectangle((0,0),1,1, color=quantile_colors.get(q,'gray')) for q in qranks_all]
                fig.legend(handles, qranks_all, title='Quantile Rank',
                           loc='upper center', ncol=len(qranks_all), bbox_to_anchor=(0.5, 0.995))

                plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
                pdf.savefig(fig)
                plt.close(fig)

                # After-page cumulative sequence: per cell in nested order (v1 major, then v2)
                for cell in tile_sequence_two([v1, v2], grid_vals):
                    sel = {**fcombo, **cell}
                    note_parts = []
                    if fcombo:
                        note_parts.append(pretty_combo(fcombo))
                    note_parts.append(pretty_combo(cell))
                    note = " | ".join(note_parts)
                    make_cum_pnl_ppd_page(sel, note, pdf)

    # ── CASE 3: three x variables -> grid over first two; aggregate over the third (no pagination).
    # After each bar page, cumulative pages for each tile (grid) in same order, aggregated across v3.
    elif len(xvars) == 3:
        v1, v2, v3 = xvars  # grid over (v1 x v2), aggregate v3
        v1_vals, v2_vals = NAME2ALL[v1], NAME2ALL[v2]
        grid_vals = {v1: v1_vals, v2: v2_vals}

        fig_w = 3.6 * max(2, len(v2_vals))
        fig_h = 2.8 * max(2, len(v1_vals))

        for fcombo in fixed_combos:
            # NOTE: we DO NOT fix v3; we aggregate over all v3 values implicitly (aggfunc='sum' in pivots)
            for metric in metrics_to_plot:
                fig, axes = plt.subplots(len(v1_vals), len(v2_vals), figsize=(fig_w, fig_h), squeeze=False)
                supt = (f"Small Multiples by {PRETTY[v1]}×{PRETTY[v2]} | "
                        f"Aggregated over {PRETTY[v3]} (sum) " +
                        ("| " + pretty_combo(fcombo) if fcombo else "") +
                        f" | metric={metric}")
                fig.suptitle(supt, fontsize=16, weight='bold')
                annotate_corner(fig, f"Aggregated over {PRETTY[v3]} (sum)" +
                                (" | " + pretty_combo(fcombo) if fcombo else ""))

                # Draw grid: reuse the 2-var small-multiple routine by passing only fixed_vars (no v3 filter).
                plot_small_multiple_for_metric_two(
                    fig, axes, metric, [v1, v2],
                    fixed_filters=fcombo,   # v1 & v2 will vary per tile; v3 is NOT fixed => aggregated
                    grid_vals=grid_vals
                )

                # Legend for qrank
                handles = [plt.Rectangle((0,0),1,1, color=quantile_colors.get(q,'gray')) for q in qranks_all]
                fig.legend(handles, qranks_all, title='Quantile Rank',
                           loc='upper center', ncol=len(qranks_all), bbox_to_anchor=(0.5, 0.995))

                plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
                pdf.savefig(fig)
                plt.close(fig)

                # After-page cumulative sequence for each tile (v1 major, then v2), aggregated over v3
                for cell in tile_sequence_two([v1, v2], grid_vals):
                    sel = {**fcombo, **cell}  # NOTE: no v3 in sel => aggregated across all v3
                    parts = []
                    if fcombo:
                        parts.append(pretty_combo(fcombo))
                    parts.append(f"{PRETTY[v1]}={cell[v1]} | {PRETTY[v2]}={cell[v2]}")
                    parts.append(f"(Aggregated over {PRETTY[v3]}: sum)")
                    note = " | ".join(parts)
                    make_cum_pnl_ppd_page(sel, note, pdf)

print(f"✅ Full PDF saved to {out_pdf}")