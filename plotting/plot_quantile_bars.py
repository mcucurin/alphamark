import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import numpy as np

# Load the stats DataFrame
with open("output/DAILY_SUMMARIES/stats_tensor.pkl", "rb") as f:
    stats_df = pickle.load(f)

# Unique values
metrics_to_plot = ['pnl', 'ppd', 'sharpe', 'hit_ratio', 'long_ratio', 'nrInstr', 'sizeNotional']
targets = stats_df['target'].unique()
signals = stats_df['signal'].unique()
qranks = sorted(stats_df['qrank'].unique())  # ensure order

# Define bar width and offsets
bar_width = 0.15
q_offsets = np.arange(-(len(qranks)-1)/2, (len(qranks)+1)/2) * bar_width

# Create one PDF with one page per target
with PdfPages("output/Quantile_Barplots.pdf") as pdf:
    for target in targets:
        fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(14, 4 * len(metrics_to_plot)))
        fig.suptitle(f"Target: {target}", fontsize=16)

        for i, metric in enumerate(metrics_to_plot):
            ax = axs[i]
            data = stats_df[
                (stats_df['target'] == target) &
                (stats_df['stat_type'] == metric)
            ]
            # Pivot: index=signal, columns=qrank, values=value
            pivot = data.pivot_table(index='signal', columns='qrank', values='value', fill_value=0)
            x = np.arange(len(signals))  # one base position per signal
            for j, qrank in enumerate(qranks):
                # Use .reindex to ensure order matches signals
                values = pivot[qrank].reindex(signals).values if qrank in pivot.columns else np.zeros(len(signals))
                ax.bar(x + q_offsets[j], values, width=bar_width, label=qrank)
            ax.set_ylabel(metric)
            ax.set_xticks(x)
            ax.set_xticklabels(signals, rotation=45)
            ax.legend(title='Quantile')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close()

print("✅ Grouped bar plots saved to output/Quantile_Barplots.pdf")
