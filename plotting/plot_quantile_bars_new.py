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
#   x_variables = ['alpha', 'target', 'bet_size']  # 3-var aggregated over the 3rd
# ────────────────────────────────────────────────────────────────
x_variables = ['alpha']  # edit me

input_pkl = "output/DAILY_SUMMARIES/stats_tensor.pkl"
out_pdf   = "output/Quantile_Combined_Report.pdf"

# Metrics to bar-plot (each page will iterate over these)
metrics_to_plot = [
    'pnl', 'ppd', 'sharpe', 'hit_ratio', 'long_ratio',
    'nrInstr', 'sizeNotional', 'r2', 't_stat', 'n_trades', 'ppt'
]

# Colors for quantile ranks (fallback to gray if missing)
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
        col = NAME2COL.get(k, k)
        mask &= (df[col] == v)
    return df[mask].copy()

def collect_values_for_metric(df, metric, x_name, fixed_filters):
    x_col = NAME2COL[x_name]
    mask = (df['stat_type'] == metric)
    for k, v in fixed_filters.items():
        mask &= (df[NAME2COL[k]] == v)
    dsub = df[mask].copy()
    if dsub.empty:
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
    """Two-variable small multiples (x_pair length == 2). Grid shape: (len(v1), len(v2)).
    Each tile shows bars (one per qrank) for that combo, height=sum(value)."""
    v1, v2 = x_pair
    col1, col2 = NAME2COL[v1], NAME2COL[v2]
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
    v1, v2 = x_pair
    seq = []
    for val1 in grid_vals[v1]:
        for val2 in grid_vals[v2]:
            seq.append({v1: val1, v2: val2})
    return seq

def pretty_combo(d):
    return " | ".join(f"{PRETTY[k]}={d[k]}" for k in d)

# ────────────────────────────────────────────────────────────────
# CUMULATIVE PAGE MAKERS (4 types)
# Each draws one page per selected tile (per qrank lines).
# They accept a selection dict keyed by logical names: 'alpha','target','bet_size'.
# If selection omits some keys, those dimensions are aggregated by sum.
# ────────────────────────────────────────────────────────────────

def _subset_for_selection(required_stats, selection):
    """Return subset with required stat_types and (optional) equality filters for selection."""
    sub = stats_df[stats_df['stat_type'].isin(required_stats)].copy()
    for k, v in selection.items():
        sub = sub[sub[NAME2COL[k]] == v]
    return sub

def make_cum_pnl_ppd_page(selection, title_note, pdf):
    subset = _subset_for_selection(['pnl', 'sizeNotional'], selection)
    if subset.empty:
        return
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()
    ax1.set_title(f"Cumulative P&L + Cumulative PPD  |  {title_note}", fontsize=16)

    for qrank in qranks_all:
        sub_q = subset[subset['qrank'] == qrank]
        pnl_data = sub_q[sub_q['stat_type'] == 'pnl'].sort_values('date')[['date','value']]
        notional = sub_q[sub_q['stat_type'] == 'sizeNotional'].sort_values('date')[['date','value']]
        if pnl_data.empty or notional.empty:
            continue
        merged = pd.merge(
            pnl_data.rename(columns={'value':'pnl'}),
            notional.rename(columns={'value':'notional'}),
            on='date', how='outer'
        ).sort_values('date')
        merged[['pnl','notional']] = merged[['pnl','notional']].fillna(0.0)
        merged['cum_pnl'] = merged['pnl'].cumsum()
        merged['cum_notional'] = merged['notional'].cumsum()
        merged['cum_ppd'] = np.where(merged['cum_notional'] > 0,
                                     merged['cum_pnl'] / merged['cum_notional'], np.nan)
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

def make_cum_trades_ppt_page(selection, title_note, pdf):
    subset = _subset_for_selection(['pnl', 'n_trades'], selection)
    if subset.empty:
        return
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()
    ax1.set_title(f"Cumulative Trades + Cumulative PPT  |  {title_note}", fontsize=16)

    for qrank in qranks_all:
        sub_q = subset[subset['qrank'] == qrank]
        pnl = sub_q[sub_q['stat_type'] == 'pnl'].sort_values('date')[['date','value']]
        trades = sub_q[sub_q['stat_type'] == 'n_trades'].sort_values('date')[['date','value']]
        if pnl.empty or trades.empty:
            continue
        merged = pd.merge(
            pnl.rename(columns={'value':'pnl'}),
            trades.rename(columns={'value':'trades'}),
            on='date', how='outer'
        ).sort_values('date')
        merged[['pnl','trades']] = merged[['pnl','trades']].fillna(0.0)
        merged['cum_pnl'] = merged['pnl'].cumsum()
        merged['cum_trades'] = merged['trades'].cumsum()
        merged['cum_ppt'] = np.where(merged['cum_trades'] > 0,
                                     merged['cum_pnl'] / merged['cum_trades'], np.nan)
        color = quantile_colors.get(qrank, 'gray')
        ax1.plot(merged['date'], merged['cum_trades'], label=qrank, color=color, linewidth=1.5)
        ax2.plot(merged['date'], merged['cum_ppt'], color=color, linestyle='--', alpha=0.9)

    ax1.set_ylabel("Cumulative Trades")
    ax2.set_ylabel("Cumulative PPT (cum PnL / cum Trades)")
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()
    ax1.legend(title='Quantile Rank', loc='upper left')
    annotate_corner(fig, title_note)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def make_cum_size_notional_page(selection, title_note, pdf):
    subset = _subset_for_selection(['sizeNotional'], selection)
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title(f"Cumulative Size Notional  |  {title_note}", fontsize=16)

    for qrank in qranks_all:
        notional = subset[(subset['qrank'] == qrank) & (subset['stat_type'] == 'sizeNotional')] \
                    .sort_values('date')[['date','value']]
        if notional.empty:
            continue
        series = notional['value'].cumsum()
        ax.plot(notional['date'], series, label=qrank, color=quantile_colors.get(qrank, 'gray'), linewidth=1.5)

    ax.set_ylabel("Cumulative Size Notional")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()
    ax.legend(title='Quantile Rank', loc='upper left')
    annotate_corner(fig, title_note)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def make_cum_bet_total_page(selection, title_note, pdf):
    # bet_total = mean bet size (stat_type 'bet_size') * nrInstr (stat_type 'nrInstr')
    subset = _subset_for_selection(['bet_size', 'nrInstr'], selection)
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title(f"Cumulative Total Bet Size (mean bet × #instr)  |  {title_note}", fontsize=16)

    for qrank in qranks_all:
        bs = subset[(subset['qrank'] == qrank) & (subset['stat_type'] == 'bet_size')] \
                .sort_values('date')[['date','value']].rename(columns={'value':'bet_mean'})
        n  = subset[(subset['qrank'] == qrank) & (subset['stat_type'] == 'nrInstr')] \
                .sort_values('date')[['date','value']].rename(columns={'value':'n_instr'})
        if bs.empty or n.empty:
            continue
        merged = pd.merge(bs, n, on='date', how='outer').sort_values('date')
        merged[['bet_mean','n_instr']] = merged[['bet_mean','n_instr']].fillna(0.0)
        merged['bet_total'] = merged['bet_mean'] * merged['n_instr']
        merged['cum_bet_total'] = merged['bet_total'].cumsum()
        ax.plot(merged['date'], merged['cum_bet_total'],
                label=qrank, color=quantile_colors.get(qrank, 'gray'), linewidth=1.5)

    ax.set_ylabel("Cumulative Total Bet Size")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()
    ax.legend(title='Quantile Rank', loc='upper left')
    annotate_corner(fig, title_note)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

# ────────────────────────────────────────────────────────────────
# PAGE ORCHESTRATION
# ────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

with PdfPages(out_pdf) as pdf:

    all_vars = ['alpha', 'target', 'bet_size']
    xvars = list(x_variables)
    assert 1 <= len(xvars) <= 3, "x_variables must be any subset of ['alpha','target','bet_size'] with length 1..3."
    fixed_vars = [v for v in all_vars if v not in xvars]

    # All combinations of the non-x (fixed) variables → loop outermost
    fixed_lists = [NAME2ALL[v] for v in fixed_vars]
    if len(fixed_vars) == 0:
        fixed_combos = [dict()]
    else:
        fixed_combos = []
        for tup in product(*fixed_lists):
            fixed_combos.append({fv: tup[i] for i, fv in enumerate(fixed_vars)})

    # 1-VAR CASE: x only; pages per fixed combo; afterpage cumulative per x in order
    if len(xvars) == 1:
        xvar = xvars[0]
        x_values = NAME2ALL[xvar]

        for fcombo in fixed_combos:
            # Bar page with all metrics stacked vertically
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
                    ax.legend(title='Quantile Rank', bbox_to_anchor=(1.01, 1), loc='upper left')

            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            pdf.savefig(fig)
            plt.close(fig)

            # After-page cumulative sequence per x in order
            for xv in x_values:
                sel = {**fcombo, xvar: xv}
                note = (pretty_combo(fcombo) + (" | " if fcombo else "") + f"{PRETTY[xvar]}={xv}") if fcombo else f"{PRETTY[xvar]}={xv}"
                # 4 types:
                make_cum_pnl_ppd_page(sel, note, pdf)
                make_cum_trades_ppt_page(sel, note, pdf)
                make_cum_size_notional_page(sel, note, pdf)
                make_cum_bet_total_page(sel, note, pdf)

    # 2-VAR CASE: grid over xvars (rows=v1, cols=v2); paginate over remaining (if any)
    elif len(xvars) == 2:
        v1, v2 = xvars
        v1_vals, v2_vals = NAME2ALL[v1], NAME2ALL[v2]
        grid_vals = {v1: v1_vals, v2: v2_vals}
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
                handles = [plt.Rectangle((0,0),1,1, color=quantile_colors.get(q,'gray')) for q in qranks_all]
                fig.legend(handles, qranks_all, title='Quantile Rank',
                           loc='upper center', ncol=len(qranks_all), bbox_to_anchor=(0.5, 0.995))

                plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
                pdf.savefig(fig)
                plt.close(fig)

                # After-page cumulative sequence per cell (row-major)
                for cell in tile_sequence_two([v1, v2], grid_vals):
                    sel = {**fcombo, **cell}
                    parts = []
                    if fcombo:
                        parts.append(pretty_combo(fcombo))
                    parts.append(prety_combo := pretty_combo(cell))
                    note = " | ".join(parts) if parts else prety_combo
                    make_cum_pnl_ppd_page(sel, note, pdf)
                    make_cum_trades_ppt_page(sel, note, pdf)
                    make_cum_size_notional_page(sel, note, pdf)
                    make_cum_bet_total_page(sel, note, pdf)

    # 3-VAR CASE: grid over first two; AGGREGATE over the third (no pagination).
    # After each bar page, cumulative pages per tile (aggregated over 3rd var).
    elif len(xvars) == 3:
        v1, v2, v3 = xvars
        v1_vals, v2_vals = NAME2ALL[v1], NAME2ALL[v2]
        grid_vals = {v1: v1_vals, v2: v2_vals}
        fig_w = 3.6 * max(2, len(v2_vals))
        fig_h = 2.8 * max(2, len(v1_vals))

        for fcombo in fixed_combos:
            for metric in metrics_to_plot:
                fig, axes = plt.subplots(len(v1_vals), len(v2_vals), figsize=(fig_w, fig_h), squeeze=False)
                supt = (f"Small Multiples by {PRETTY[v1]}×{PRETTY[v2]} | "
                        f"Aggregated over {PRETTY[v3]} (sum) " +
                        ("| " + pretty_combo(fcombo) if fcombo else "") +
                        f" | metric={metric}")
                fig.suptitle(supt, fontsize=16, weight='bold')
                annotate_corner(fig, f"Aggregated over {PRETTY[v3]} (sum)" +
                                (" | " + pretty_combo(fcombo) if fcombo else ""))

                # v3 not fixed → aggregated in sums
                plot_small_multiple_for_metric_two(
                    fig, axes, metric, [v1, v2],
                    fixed_filters=fcombo, grid_vals=grid_vals
                )
                handles = [plt.Rectangle((0,0),1,1, color=quantile_colors.get(q,'gray')) for q in qranks_all]
                fig.legend(handles, qranks_all, title='Quantile Rank',
                           loc='upper center', ncol=len(qranks_all), bbox_to_anchor=(0.5, 0.995))

                plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
                pdf.savefig(fig)
                plt.close(fig)

                # After-page cumulative per tile (aggregated over v3)
                for cell in tile_sequence_two([v1, v2], grid_vals):
                    sel = {**fcombo, **cell}  # no v3 key => aggregated across all v3 values
                    parts = []
                    if fcombo:
                        parts.append(pretty_combo(fcombo))
                    parts.append(f"{PRETTY[v1]}={cell[v1]} | {PRETTY[v2]}={cell[v2]}")
                    parts.append(f"(Aggregated over {PRETTY[v3]}: sum)")
                    note = " | ".join(parts)
                    make_cum_pnl_ppd_page(sel, note, pdf)
                    make_cum_trades_ppt_page(sel, note, pdf)
                    make_cum_size_notional_page(sel, note, pdf)
                    make_cum_bet_total_page(sel, note, pdf)

print(f"✅ Full PDF saved to {out_pdf}")
