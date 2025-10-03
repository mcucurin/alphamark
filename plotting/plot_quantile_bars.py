import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
import pickle
import numpy as np
from itertools import product

# =========================
# ---- USER CONFIG --------
# =========================

# Bar plots: choose page vars (0–3) and x-axis vars (0–3) from:
#   'signal', 'target', 'bet_size_col'
BAR_PAGE_VARS = ['signal','bet_size_col']   # separate pages by these vars (0–3)
BAR_X_VARS    = ['target']                  # x-axis categories built from these vars (0–3)

# Define metrics for bar plots
metrics_to_plot = [
    'pnl', 'ppd', 'sharpe', 'hit_ratio', 'long_ratio',
    'nrInstr', 'sizeNotional', 'r2', 't_stat',
    'n_trades', 'ppt', 'spearman', 'dcor'
]

# Cumulative plots to show per (target, signal, bet_size_col) page.
# Allowed keys:
#   'pnl_ppd'          -> cum PnL (left) + cum PPD (right)
#   'sizeNotional'     -> cum notional
#   'trades_ppt'       -> cum trades (left) + cum PPT (right)
#   'avg_bet_per_name' -> cumulative average bet per instrument (cum_notional / cum_nrInstr)
CUM_SECTIONS = ['pnl_ppd', 'sizeNotional', 'trades_ppt', 'avg_bet_per_name']   # choose any subset/order

# =========================
# ---- LOAD & PREP --------
# =========================

with open("output/DAILY_SUMMARIES/stats_tensor.pkl", "rb") as f:
    stats_df = pickle.load(f)

stats_df['date'] = pd.to_datetime(stats_df['date'], errors='coerce')
stats_df = stats_df.dropna(subset=['date', 'value'])

# Validate config
_allowed = {'signal', 'target', 'bet_size_col'}
assert set(BAR_PAGE_VARS).issubset(_allowed), "BAR_PAGE_VARS must use allowed fields"
assert set(BAR_X_VARS).issubset(_allowed), "BAR_X_VARS must use allowed fields"
assert len(set(BAR_PAGE_VARS) & set(BAR_X_VARS)) == 0, "Page vars and x vars must be disjoint"

_allowed_cum = {'pnl_ppd', 'sizeNotional', 'trades_ppt', 'avg_bet_per_name'}
assert set(CUM_SECTIONS).issubset(_allowed_cum), f"CUM_SECTIONS must be within {_allowed_cum}"

# Precompute unique levels (keep originals for order)
targets   = stats_df['target'].dropna().unique().tolist()
signals   = stats_df['signal'].dropna().unique().tolist()
bet_sizes = stats_df['bet_size_col'].dropna().unique().tolist()

qranks = sorted(stats_df['qrank'].unique(), key=lambda x: float(x.split('_')[1]))
bar_width = 0.15
q_offsets = np.arange(-(len(qranks)-1)/2, (len(qranks)+1)/2) * bar_width

# Colors for quantile ranks
quantile_colors = {
    'qr_100': 'red',
    'qr_75': 'green',
    'qr_50': 'blue',
    'qr_25': 'black'
}

# Helper: ordered levels for each categorical field
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
    """Return merged df with columns: date, name_a, name_b, and cumsums for each."""
    a_df = (subset[subset['stat_type'] == stat_a]
            .sort_values('date')[['date','value']].rename(columns={'value': name_a}))
    b_df = (subset[subset['stat_type'] == stat_b]
            .sort_values('date')[['date','value']].rename(columns={'value': name_b}))
    if a_df.empty and b_df.empty:
        return pd.DataFrame(columns=['date', name_a, name_b, f'cum_{name_a}', f'cum_{name_b}'])
    merged = pd.merge(a_df, b_df, on='date', how='outer').sort_values('date')
    merged[[name_a, name_b]] = merged[[name_a, name_b]].fillna(0.0)
    merged[f'cum_{name_a}'] = merged[name_a].cumsum()
    merged[f'cum_{name_b}'] = merged[name_b].cumsum()
    return merged

def _cum_only(subset, stat_name, out_col):
    """Return df with date and cumulative of a single stat."""
    df = (subset[subset['stat_type'] == stat_name]
          .sort_values('date')[['date','value']].rename(columns={'value': out_col}))
    if df.empty:
        return pd.DataFrame(columns=['date', out_col, f'cum_{out_col}'])
    df[f'cum_{out_col}'] = df[out_col].cumsum()
    return df

# =========================
# ---- BUILD THE PDF -------
# =========================

os.makedirs("output", exist_ok=True)
with PdfPages("output/Quantile_Combined_Report.pdf") as pdf:

    # ==========================
    # -------- Bar Plots -------
    # ==========================
    if BAR_PAGE_VARS:
        page_levels = [LEVELS[var] for var in BAR_PAGE_VARS]
        page_iter = list(product(*page_levels))
    else:
        page_iter = [()]  # single page

    for page_vals in page_iter:
        subset = stats_df.copy()
        title_bits = []
        for var, val in zip(BAR_PAGE_VARS, page_vals):
            subset = subset[subset[var] == val]
            title_bits.append(f"{var}: {val}")

        # composite x-axis
        if BAR_X_VARS:
            if not all(v in subset.columns for v in BAR_X_VARS):
                continue
            subset = subset.copy()
            subset['x_key'] = subset[BAR_X_VARS].astype(str).agg('|'.join, axis=1)

            def tuple_levels(var_list):
                bases = [LEVELS[v] for v in var_list]
                all_tuples = list(product(*bases))
                present_tuples = set(map(tuple, subset[var_list].astype(str).values.tolist()))
                ordered = []
                for t in all_tuples:
                    if tuple(map(str, t)) in present_tuples:
                        ordered.append('|'.join(map(str, t)))
                return ordered

            x_levels = tuple_levels(BAR_X_VARS)
            if len(x_levels) == 0:
                continue
        else:
            subset = subset.copy()
            subset['x_key'] = "ALL"
            x_levels = ["ALL"]

        fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(14, 2.7 * len(metrics_to_plot)))
        if len(metrics_to_plot) == 1:
            axs = [axs]

        x_axis_label = ', '.join(BAR_X_VARS) if BAR_X_VARS else 'ALL'
        fig.suptitle(" | ".join(["Bar Plots"] + title_bits + [f"x: {x_axis_label}"]),
                     fontsize=18, weight='bold')

        for i, metric in enumerate(metrics_to_plot):
            ax = axs[i]
            data = subset[subset['stat_type'] == metric]
            if data.empty:
                ax.set_ylabel(metric, fontsize=10)
                ax.set_xticks([])
                ax.set_title("(no data)")
                continue

            pivot = data.pivot_table(index='x_key', columns='qrank',
                                     values='value', aggfunc='mean', fill_value=0.0)
            x = np.arange(len(x_levels))

            for j, qrank in enumerate(qranks):
                values = pivot[qrank].reindex(x_levels).values if qrank in pivot.columns else np.zeros(len(x_levels))
                color = quantile_colors.get(qrank, 'gray')
                ax.bar(x + q_offsets[j], values, width=bar_width, label=qrank, color=color)

            ax.set_ylabel(metric, fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in x_levels], rotation=45, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            if i == 0:
                ax.legend(title='Quantile Rank', bbox_to_anchor=(1.01, 1), loc='upper left')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        pdf.savefig(fig)
        plt.close()

    # ==============================================
    # ---- CUMULATIVE PLOTS ON ONE PAGE PER COMBO ---
    # ==============================================
    # For every (target, signal, bet_size_col), draw a page with selected CUM_SECTIONS
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

                nrows = len(CUM_SECTIONS)
                if nrows == 0:
                    continue

                fig_height = 4.5 * nrows
                fig, axes = plt.subplots(nrows, 1, figsize=(14, fig_height))
                if nrows == 1:
                    axes = [axes]

                fig.suptitle(f"Target: {target} | Alpha: {signal} | Bet Size: {bet_strategy}",
                             fontsize=16, weight='bold')

                for row_idx, section in enumerate(CUM_SECTIONS):
                    ax = axes[row_idx]

                    if section == 'pnl_ppd':
                        subset = stats_df[base_filter & stats_df['stat_type'].isin(['pnl','sizeNotional'])].copy()
                        ax_right = ax.twinx()
                        for qrank in qranks:
                            sub_q = subset[subset['qrank'] == qrank]
                            if sub_q.empty:
                                continue
                            merged = _merge_two_stats(sub_q, 'pnl', 'sizeNotional', 'pnl', 'notional')
                            if merged.empty:
                                continue
                            merged['cum_ppd'] = np.where(merged['cum_notional'] > 0,
                                                         merged['cum_pnl'] / merged['cum_notional'],
                                                         np.nan)
                            color = quantile_colors.get(qrank, 'gray')
                            ax.plot(merged['date'], merged['cum_pnl'], label=qrank, color=color, linewidth=1.5)
                            ax_right.plot(merged['date'], merged['cum_ppd'], color=color, linestyle='--', alpha=0.9)
                        ax.set_ylabel("Cumulative P&L")
                        ax_right.set_ylabel("Cumulative PPD (cum PnL / cum Notional)")
                        _plot_date_axis(ax)
                        if row_idx == 0:
                            ax.legend(title='Quantile Rank', loc='upper left')

                    elif section == 'sizeNotional':
                        subset = stats_df[base_filter & (stats_df['stat_type'] == 'sizeNotional')].copy()
                        for qrank in qranks:
                            notional_data = (subset[subset['qrank'] == qrank]
                                             .sort_values('date')[['date','value']])
                            if not notional_data.empty:
                                notional_data['cum_notional'] = notional_data['value'].cumsum()
                                color = quantile_colors.get(qrank, 'gray')
                                ax.plot(notional_data['date'], notional_data['cum_notional'],
                                        label=qrank, color=color, linewidth=1.5)
                        ax.set_ylabel("Cumulative Size Notional")
                        _plot_date_axis(ax)
                        if row_idx == 0 and 'pnl_ppd' not in CUM_SECTIONS:
                            ax.legend(title='Quantile Rank', loc='upper left')

                    elif section == 'trades_ppt':
                        subset = stats_df[base_filter & stats_df['stat_type'].isin(['pnl','n_trades'])].copy()
                        ax_right = ax.twinx()
                        for qrank in qranks:
                            sub_q = subset[subset['qrank'] == qrank]
                            if sub_q.empty:
                                continue
                            merged = _merge_two_stats(sub_q, 'pnl', 'n_trades', 'pnl', 'trades')
                            if merged.empty:
                                continue
                            merged['cum_ppt'] = np.where(merged['cum_trades'] > 0,
                                                         merged['cum_pnl'] / merged['cum_trades'],
                                                         np.nan)
                            color = quantile_colors.get(qrank, 'gray')
                            ax.plot(merged['date'], merged['cum_trades'], label=qrank, color=color, linewidth=1.5)
                            ax_right.plot(merged['date'], merged['cum_ppt'], color=color, linestyle='--', alpha=0.9)
                        ax.set_ylabel("Cumulative Trades")
                        ax_right.set_ylabel("Cumulative PPT (cum PnL / cum Trades)")
                        _plot_date_axis(ax)
                        if row_idx == 0 and ('pnl_ppd' not in CUM_SECTIONS and 'sizeNotional' not in CUM_SECTIONS):
                            ax.legend(title='Quantile Rank', loc='upper left')

                    elif section == 'avg_bet_per_name':
                        # Show cumulative *average bet per instrument* = cum_notional / cum_nrInstr
                        n_subset   = stats_df[base_filter & (stats_df['stat_type'] == 'nrInstr')].copy()
                        not_subset = stats_df[base_filter & (stats_df['stat_type'] == 'sizeNotional')].copy()

                        for qrank in qranks:
                            # daily breadth
                            n_q  = (n_subset[n_subset['qrank'] == qrank]
                                    .sort_values('date')[['date','value']]
                                    .rename(columns={'value':'n_instr'}))
                            # daily total notional (sum of bets)
                            not_q = (not_subset[not_subset['qrank'] == qrank]
                                     .sort_values('date')[['date','value']]
                                     .rename(columns={'value':'notional'}))

                            if not_q.empty or n_q.empty:
                                continue

                            merged = pd.merge(not_q, n_q, on='date', how='outer').sort_values('date')
                            merged[['notional','n_instr']] = merged[['notional','n_instr']].fillna(0.0)
                            merged['cum_notional'] = merged['notional'].cumsum()
                            merged['cum_n_instr']  = merged['n_instr'].cumsum()
                            merged['cum_avg_bet_per_name'] = np.where(
                                merged['cum_n_instr'] > 0,
                                merged['cum_notional'] / merged['cum_n_instr'],
                                np.nan
                            )

                            color = quantile_colors.get(qrank, 'gray')
                            ax.plot(merged['date'], merged['cum_avg_bet_per_name'],
                                    label=qrank, color=color, linewidth=1.5)

                        ax.set_ylabel("Cumulative Avg Bet per Name")
                        _plot_date_axis(ax)
                        if row_idx == 0 and all(sec not in CUM_SECTIONS for sec in ['pnl_ppd','sizeNotional','trades_ppt']):
                            ax.legend(title='Quantile Rank', loc='upper left')

                    else:
                        ax.text(0.5, 0.5, f"Unknown section: {section}",
                                transform=ax.transAxes, ha='center', va='center')

                    fig.autofmt_xdate()

                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                pdf.savefig(fig)
                plt.close()

print("✅ Full PDF saved to output/Quantile_Combined_Report.pdf")

