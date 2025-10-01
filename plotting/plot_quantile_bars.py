import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
import pickle
import numpy as np

# Load stats DataFrame
with open("output/DAILY_SUMMARIES/stats_tensor.pkl", "rb") as f:
    stats_df = pickle.load(f)

# Clean and prep
stats_df['date'] = pd.to_datetime(stats_df['date'], errors='coerce')
stats_df = stats_df.dropna(subset=['date', 'value'])

# Define config
metrics_to_plot = [
    'pnl', 'ppd', 'sharpe', 'hit_ratio', 'long_ratio',
    'nrInstr', 'sizeNotional', 'r2', 't_stat',
    'n_trades', 'ppt'   # <-- added
]
targets = stats_df['target'].unique()
signals = stats_df['signal'].unique()
bet_sizes = stats_df['bet_size_col'].unique()
qranks = sorted(stats_df['qrank'].unique(), key=lambda x: float(x.split('_')[1]))
bar_width = 0.15
q_offsets = np.arange(-(len(qranks)-1)/2, (len(qranks)+1)/2) * bar_width

# Color map for quantile ranks
quantile_colors = {
    'qr_100': 'red',
    'qr_75': 'green',
    'qr_50': 'blue',
    'qr_25': 'black'
}

# Create PDF
with PdfPages("output/Quantile_Combined_Report.pdf") as pdf:

    # ---- Bar Plots ----
    for target in targets:
        for bet_strategy in bet_sizes:
            subset = stats_df[stats_df['bet_size_col'] == bet_strategy]
            fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(14, 2.7 * len(metrics_to_plot)))
            fig.suptitle(f"Target: {target} | Bet Size: {bet_strategy}", fontsize=18, weight='bold')

            for i, metric in enumerate(metrics_to_plot):
                ax = axs[i]
                data = subset[(subset['target'] == target) & (subset['stat_type'] == metric)]
                pivot = data.pivot_table(index='signal', columns='qrank', values='value', fill_value=0)
                x = np.arange(len(signals))
                for j, qrank in enumerate(qranks):
                    values = pivot[qrank].reindex(signals).values if qrank in pivot.columns else np.zeros(len(signals))
                    color = quantile_colors.get(qrank, 'gray')
                    ax.bar(x + q_offsets[j], values, width=bar_width, label=qrank, color=color)
                ax.set_ylabel(metric, fontsize=10)
                ax.set_xticks(x)
                ax.set_xticklabels(signals, rotation=45, ha='right')
                ax.grid(axis='y', linestyle='--', alpha=0.4)
                if i == 0:
                    ax.legend(title='Quantile Rank', bbox_to_anchor=(1.01, 1), loc='upper left')

            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            pdf.savefig(fig)
            plt.close()

    # ---- Cumulative P&L + Cumulative PPD (ratio-of-cums) ----
    for target in targets:
        for signal in signals:
            for bet_strategy in bet_sizes:
                subset = stats_df[
                    (stats_df['target'] == target) &
                    (stats_df['signal'] == signal) &
                    (stats_df['bet_size_col'] == bet_strategy) &
                    (stats_df['stat_type'].isin(['pnl', 'sizeNotional']))
                ]
                if subset.empty:
                    continue

                fig, ax1 = plt.subplots(figsize=(14, 6))
                ax2 = ax1.twinx()
                ax1.set_title(f"Target: {target} | Alpha: {signal} | Bet Size: {bet_strategy}", fontsize=16)

                for qrank in qranks:
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
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()

    # ---- Cumulative Size Notional ----
    for target in targets:
        for signal in signals:
            for bet_strategy in bet_sizes:
                subset = stats_df[
                    (stats_df['target'] == target) &
                    (stats_df['signal'] == signal) &
                    (stats_df['bet_size_col'] == bet_strategy) &
                    (stats_df['stat_type'] == 'sizeNotional')
                ]
                if subset.empty:
                    continue

                fig, ax = plt.subplots(figsize=(14, 6))
                ax.set_title(f"Target: {target} | Alpha: {signal} | Bet Size: {bet_strategy}", fontsize=16)

                for qrank in qranks:
                    notional_data = subset[(subset['qrank'] == qrank)].sort_values('date')[['date','value']]
                    if notional_data.empty:
                        continue
                    color = quantile_colors.get(qrank, 'gray')
                    ax.plot(notional_data['date'], notional_data['value'].cumsum(), label=qrank, color=color, linewidth=1.5)

                ax.set_ylabel("Cumulative Size Notional")
                ax.grid(True, linestyle='--', alpha=0.4)
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                fig.autofmt_xdate()
                ax.legend(title='Quantile Rank', loc='upper left')
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()

    # ---- Cumulative Bet Size (proportional to quantile breadth) ----
    for target in targets:
        for signal in signals:
            for bet_strategy in bet_sizes:
                bs_subset = stats_df[
                    (stats_df['target'] == target) &
                    (stats_df['signal'] == signal) &
                    (stats_df['bet_size_col'] == bet_strategy) &
                    (stats_df['stat_type'] == 'bet_size')      # mean bet per instrument
                ]
                n_subset = stats_df[
                    (stats_df['target'] == target) &
                    (stats_df['signal'] == signal) &
                    (stats_df['bet_size_col'] == bet_strategy) &
                    (stats_df['stat_type'] == 'nrInstr')       # number of instruments
                ]

                if bs_subset.empty or n_subset.empty:
                    continue

                fig, ax = plt.subplots(figsize=(14, 6))
                ax.set_title(f"Target: {target} | Alpha: {signal} | Bet Size: {bet_strategy}", fontsize=16)

                for qrank in qranks:
                    bs_q = bs_subset[bs_subset['qrank'] == qrank][['date','value']].rename(columns={'value':'bet_mean'})
                    n_q  = n_subset[n_subset['qrank'] == qrank][['date','value']].rename(columns={'value':'n_instr'})

                    if bs_q.empty or n_q.empty:
                        continue

                    merged = (pd.merge(bs_q, n_q, on='date', how='outer')
                                .sort_values('date'))
                    merged[['bet_mean','n_instr']] = merged[['bet_mean','n_instr']].fillna(0.0)

                    # Total bet per quantile per day = mean bet * number of instruments
                    merged['bet_total'] = merged['bet_mean'] * merged['n_instr']
                    merged['cum_bet_total'] = merged['bet_total'].cumsum()

                    color = quantile_colors.get(qrank, 'gray')
                    ax.plot(merged['date'], merged['cum_bet_total'], label=qrank, color=color, linewidth=1.5)

                ax.set_ylabel("Cumulative Total Bet Size (mean bet × #instr)")
                ax.grid(True, linestyle='--', alpha=0.4)
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                fig.autofmt_xdate()
                ax.legend(title='Quantile Rank', loc='upper left')
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()

    # ---- Cumulative Trades + Cumulative PPT (ratio-of-cums) ----
    # (Uses pnl and n_trades; cum_ppt = cum_pnl / cum_trades)
    for target in targets:
        for signal in signals:
            for bet_strategy in bet_sizes:
                subset = stats_df[
                    (stats_df['target'] == target) &
                    (stats_df['signal'] == signal) &
                    (stats_df['bet_size_col'] == bet_strategy) &
                    (stats_df['stat_type'].isin(['pnl', 'n_trades']))
                ]
                if subset.empty:
                    continue

                fig, ax1 = plt.subplots(figsize=(14, 6))
                ax2 = ax1.twinx()
                ax1.set_title(f"Target: {target} | Alpha: {signal} | Bet Size: {bet_strategy}", fontsize=16)

                for qrank in qranks:
                    sub_q = subset[subset['qrank'] == qrank]
                    pnl_data = sub_q[sub_q['stat_type'] == 'pnl'].sort_values('date')[['date','value']]
                    trades_data = sub_q[sub_q['stat_type'] == 'n_trades'].sort_values('date')[['date','value']]

                    if pnl_data.empty or trades_data.empty:
                        continue

                    merged = pd.merge(
                        pnl_data.rename(columns={'value':'pnl'}),
                        trades_data.rename(columns={'value':'trades'}),
                        on='date', how='outer'
                    ).sort_values('date')
                    merged[['pnl','trades']] = merged[['pnl','trades']].fillna(0.0)

                    merged['cum_pnl'] = merged['pnl'].cumsum()
                    merged['cum_trades'] = merged['trades'].cumsum()
                    merged['cum_ppt'] = np.where(merged['cum_trades'] > 0,
                                                 merged['cum_pnl'] / merged['cum_trades'],
                                                 np.nan)

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
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()

print("✅ Full PDF saved to output/Quantile_Combined_Report.pdf")
