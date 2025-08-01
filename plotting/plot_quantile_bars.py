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
metrics_to_plot = ['pnl', 'ppd', 'sharpe', 'hit_ratio', 'long_ratio', 'nrInstr', 'sizeNotional', 'r2', 't_stat']
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

    # ---- Cumulative P&L + PPD Plots ----
    for target in targets:
        for signal in signals:
            for bet_strategy in bet_sizes:
                subset = stats_df[
                    (stats_df['target'] == target) &
                    (stats_df['signal'] == signal) &
                    (stats_df['bet_size_col'] == bet_strategy) &
                    (stats_df['stat_type'].isin(['pnl', 'ppd']))
                ]
                if subset.empty:
                    continue

                fig, ax1 = plt.subplots(figsize=(14, 6))
                ax2 = ax1.twinx()
                ax1.set_title(f"Target: {target} | Alpha: {signal} | Bet Size: {bet_strategy}", fontsize=16)

                for qrank in qranks:
                    pnl_data = subset[(subset['qrank'] == qrank) & (subset['stat_type'] == 'pnl')].sort_values('date')
                    ppd_data = subset[(subset['qrank'] == qrank) & (subset['stat_type'] == 'ppd')].sort_values('date')
                    if pnl_data.empty or ppd_data.empty:
                        continue
                    color = quantile_colors.get(qrank, 'gray')
                    ax1.plot(pnl_data['date'], pnl_data['value'].cumsum(), label=qrank, color=color, linewidth=1.5)
                    ax2.plot(ppd_data['date'], ppd_data['value'], color=color, linestyle='--', alpha=0.7)

                ax1.set_ylabel("Cumulative P&L")
                ax2.set_ylabel("PPD")
                ax1.grid(True, linestyle='--', alpha=0.4)
                ax1.xaxis.set_major_locator(mdates.YearLocator())
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                fig.autofmt_xdate()
                ax1.legend(title='Quantile Rank', loc='upper left')
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()

print("✅ Full PDF saved to output/Quantile_Combined_Report.pdf")
