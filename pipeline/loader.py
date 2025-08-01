import os
import pandas as pd

def load_daily_files(root_folder):
    file_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for f in filenames:
            if f.endswith('.csv') or f.endswith('.csv.gz'):
                full_path = os.path.join(dirpath, f)
                file_paths.append(full_path)
    
    file_paths.sort()  # Optional: ensure consistent order
    return [(os.path.basename(f), pd.read_csv(f)) for f in file_paths]
