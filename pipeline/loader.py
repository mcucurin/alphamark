import os
import pandas as pd

def load_daily_files(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith('.csv') or f.endswith('.csv.gz')])
    return [(f, pd.read_csv(os.path.join(folder, f))) for f in files]
