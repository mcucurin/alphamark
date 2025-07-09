import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle

# Load the stats DataFrame
with open("output/DAILY_SUMMARIES/stats_tensor.pkl", "rb") as f:
    stats_df = pickle.load(f)

# Get unique values
metrics_to_plot = ['pnl', 'ppd', 'sharpe', 'hit_ratio', 'long_ratio']
targets = stats_df['target'].unique()
signals = stats_df['signal'].unique()
qranks = stats_df['qrank'].unique()

# Create one PDF with one page per target
with PdfPages("output/Quantile_Barplots.pdf") as pdf:
    for target in targets:
        fig, axs = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 3.5 * len(metrics_to_plot)))
        fig.suptitle(f"Target: {target}", fontsize=16)

        for i, metric in enumerate(metrics_to_plot):
            ax = axs[i]

            data = stats_df[
                (stats_df['target'] == target) &
                (stats_df['stat_type'] == metric)
            ]

            for qrank in sorted(qranks):
                values = []
                labels = []
                for sig in signals:
                    val = data[
                        (data['signal'] == sig) & (data['qrank'] == qrank)
                    ]['value']
                    values.append(val.values[0] if not val.empty else 0)
                    labels.append(sig)

                ax.bar([f"{l}\n{qrank}" for l in labels], values, label=qrank, alpha=0.7)

            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title='Quantile')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close()
