# pipeline/export_daily_features.py
import os
from pathlib import Path
import pandas as pd

from pipeline.features import generate_signals_and_targets

def save_daily_feature_files(
    daily_data_by_date,
    horizons=(1, 3),
    include_bet_caps=True,
    out_dir="data/features_daily",
    fmt="parquet",              # "parquet" or "csv"
    overwrite=True,
    csv_gzip=True,
    include_date_col=False,     # write the date column into each file (redundant but handy)
    parquet_compression="snappy"
):
    """
    For each day, write a file with columns:
        ['ticker'] + all pret_* (signals), fret_* (targets), betsize_* (bet sizes)
    Returns: list of output file paths.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    written_paths = []
    results = generate_signals_and_targets(
        daily_data_by_date,
        horizons=list(horizons),
        include_bet_caps=include_bet_caps,
    )

    for fname, df in results:
        date_str = str(fname)[:8]

        feat_cols = [c for c in df.columns if c.startswith(("pret_", "fret_", "betsize_"))]
        keep_cols = ["ticker"] + sorted(feat_cols)
        if include_date_col and "date" in df.columns:
            keep_cols = ["date"] + keep_cols

        out_df = df[keep_cols].copy()
        # optional: stable order by ticker for readability
        if "ticker" in out_df.columns:
            out_df = out_df.sort_values("ticker").reset_index(drop=True)

        # choose path
        if fmt.lower() == "parquet":
            out_path = Path(out_dir) / f"{date_str}.parquet"
            if out_path.exists() and not overwrite:
                written_paths.append(str(out_path))
                continue
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            # parquet (pyarrow recommended)
            out_df.to_parquet(tmp_path, index=False, compression=parquet_compression)
            os.replace(tmp_path, out_path)

        elif fmt.lower() == "csv":
            suffix = "csv.gz" if csv_gzip else "csv"
            out_path = Path(out_dir) / f"{date_str}.{suffix}"
            if out_path.exists() and not overwrite:
                written_paths.append(str(out_path))
                continue
            tmp_path = Path(str(out_path) + ".tmp")
            out_df.to_csv(
                tmp_path,
                index=False,
                compression=("gzip" if csv_gzip else None),
            )
            os.replace(tmp_path, out_path)

        else:
            raise ValueError("fmt must be 'parquet' or 'csv'.")

        written_paths.append(str(out_path))

    return written_paths

# Example usage:
# files = save_daily_feature_files(daily_data_by_date, horizons=(1,3), out_dir="out/daily_features", fmt="parquet")
# print(f"Wrote {len(files)} daily files")
