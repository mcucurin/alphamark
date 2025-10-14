# runner.py
import os
import pandas as pd
import numpy as np  # used when flattening dicts
from tempfile import NamedTemporaryFile
import shutil

from pipeline.loader import load_daily_files
from pipeline.features import generate_signals_and_targets
from pipeline.daily_stats import compute_daily_stats
from pipeline.summary_stats import compute_summary_stats_over_days

# Outliers (explicit metrics, no defaults) - PKL variant
from pipeline.outliers_stats import compute_outliers, save_outliers

# NEW: sidecars (for Heatmaps 1–2) and heatmaps bundle (1–6)
from pipeline.panel_sidecars import save_panel_sidecar
from pipeline.heatmap_stats import build_heatmaps_bundle, save_heatmaps_pkl

RAW_DATA_DIR = 'data/RAW_DATA'

# === output layout ===
OUTPUT_ROOT         = 'output'
DAILY_STATS_DIR     = os.path.join(OUTPUT_ROOT, 'DAILY_STATS')       # one file per day
SUMMARY_STATS_DIR   = os.path.join(OUTPUT_ROOT, 'SUMMARY_STATS')     # one file for summary
OUTLIERS_DIR        = os.path.join(OUTPUT_ROOT, 'OUTLIERS')          # outliers (PKL)
PANEL_DAILY_DIR     = os.path.join(OUTPUT_ROOT, 'PANEL_DAILY')       # per-stock sidecars (PKL)
HEATMAPS_DIR        = os.path.join(OUTPUT_ROOT, 'HEATMAPS')          # heatmaps bundle (PKL)
os.makedirs(DAILY_STATS_DIR, exist_ok=True)
os.makedirs(SUMMARY_STATS_DIR, exist_ok=True)
os.makedirs(OUTLIERS_DIR, exist_ok=True)
os.makedirs(PANEL_DAILY_DIR, exist_ok=True)
os.makedirs(HEATMAPS_DIR, exist_ok=True)

# --- Outlier config (explicit; change as needed) ---
OUTLIER_METRICS = ['pnl', 'ppd', 'sizeNotional', 'nrInstr', 'n_trades', 'ppt']
OUTLIER_Z_THRESH = 3.0
OUTLIER_TOP_K    = 5

# --- Heatmap config (sidecars + bundle) ---
ALPHA_PREFIX          = "pret_"     # infer alpha/“signal” columns from this prefix in sidecars
PNL_TARGET_COL        = "fret_1d"   # realized return column for PnL-space heatmap (if present)
PNL_BET_COL_FOR_HM2   = None        # e.g. 'betsize_equal' to weight pnl; None => unit weights
HM_QRANKS             = ['qr_100','qr_75','qr_50','qr_25']


def _atomic_pickle_dump(df: pd.DataFrame, path: str) -> None:
    """Write a dataframe to pickle atomically to avoid partial files."""
    with NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as tmp:
        tmp_name = tmp.name
        df.to_pickle(tmp_name)
    shutil.move(tmp_name, path)


def run_pipeline():
    # 1) Load & enrich
    daily_data = load_daily_files(RAW_DATA_DIR)
    enriched_data = generate_signals_and_targets(daily_data)

    # Keep raw day DataFrames (with a date column) for the summary pass
    raw_days_for_summary = []
    all_signal_cols = set()
    all_target_cols = set()
    all_bet_cols    = set()

    # Optional: tracking for an index of daily outputs
    per_day_index_rows = []

    # Accumulate daily stats frames in-memory for outliers/heatmaps 3–6
    daily_stats_frames = []

    # 2) Per-day loop -> compute & save daily stats + sidecars
    for fname, df in enriched_data:
        # infer per-file columns
        signal_cols = [col for col in df.columns if col.startswith('pret_')]
        target_cols = [col for col in df.columns if col.startswith('fret_')]

        bet_size_cols = ['betsize_equal', 'betsize_cap200k', 'betsize_cap250k']
        valid_bet_size_cols = [b for b in bet_size_cols if b in df.columns]
        if not valid_bet_size_cols:
            print(f"Skipping {fname}: no valid bet size columns")
            continue

        # date extraction used for filenames and summary union
        day_str = fname[:8]  # expects YYYYMMDD prefix in filename
        day_dt = pd.to_datetime(day_str, format='%Y%m%d', errors='coerce')

        # track for summary pass
        raw_days_for_summary.append(df.assign(date=day_dt))
        all_signal_cols.update(signal_cols)
        all_target_cols.update(target_cols)
        all_bet_cols.update(valid_bet_size_cols)

        # --- (A) SAVE PER-STOCK SIDECAR FOR THIS DAY (for Heatmaps 1–2) ---
        # Prefer PNL_TARGET_COL if present; helper will fallback to first 'fret_*' if None.
        sidecar_path = save_panel_sidecar(
            df,
            day_str=day_str,
            out_dir=PANEL_DAILY_DIR,
            alpha_prefix=ALPHA_PREFIX,
            target_col=PNL_TARGET_COL if PNL_TARGET_COL in df.columns else None,
            bet_cols_preferred=valid_bet_size_cols
        )
        if sidecar_path:
            print(f"🧩 Saved sidecar -> {sidecar_path}")

        # --- (B) compute DAILY stats for THIS day ---
        stats = compute_daily_stats(
            df,
            signal_cols=signal_cols,
            target_cols=target_cols,
            quantiles=[1.0, 0.75, 0.5, 0.25],
            bet_size_cols=valid_bet_size_cols
        )

        # flatten nested dict -> rows
        gen = (
            (day_str, s, t, q, stat_type, b, v)
            for stat_type, sig_dict in stats.items()
            for s, qd in sig_dict.items()
            for q, td in qd.items()
            for t, bd in td.items()
            for b, v in bd.items()
        )
        arr = np.array(list(gen), dtype=object)
        cols = ['date','signal','target','qrank','stat_type','bet_size_col','value']
        day_df = pd.DataFrame([dict(zip(cols, rec)) for rec in arr], columns=cols)

        # save this day's stats to its own file (PKL)
        out_path = os.path.join(DAILY_STATS_DIR, f'stats_{day_str}.pkl')
        _atomic_pickle_dump(day_df, out_path)
        per_day_index_rows.append({'date': day_str, 'path': out_path, 'n_rows': len(day_df)})
        print(f"📦 Saved daily stats for {day_str} -> {out_path} ({len(day_df)} rows)")

        # keep in memory for outlier calc & heatmaps 3–6
        daily_stats_frames.append(day_df)

    # 3) Summary over all days (single file)
    summary_path = None
    if raw_days_for_summary:
        big_df = pd.concat(raw_days_for_summary, ignore_index=True)

        # Sorted lists of columns available in the concatenated df
        sig_list = sorted([c for c in all_signal_cols if c in big_df.columns])
        tgt_list = sorted([c for c in all_target_cols if c in big_df.columns])
        bet_list = sorted([c for c in all_bet_cols    if c in big_df.columns])

        if sig_list and tgt_list and bet_list:
            summary = compute_summary_stats_over_days(
                big_df,
                date_col='date',
                signal_cols=sig_list,
                target_cols=tgt_list,
                quantiles=[1.0, 0.75, 0.5, 0.25],
                bet_size_cols=bet_list,
                type_quantile='cumulative',
                add_spearman=True,
                add_dcor=False
            )

            # flatten summary -> rows; tag with date range
            all_days = pd.to_datetime(big_df['date'], errors='coerce')
            first_date = pd.to_datetime(all_days.min())
            last_date  = pd.to_datetime(all_days.max())
            date_tag   = f"{first_date.strftime('%Y%m%d')}_{last_date.strftime('%Y%m%d')}" \
                         if pd.notna(first_date) and pd.notna(last_date) else "summary"

            gen_sum = (
                (last_date.strftime('%Y%m%d') if pd.notna(last_date) else None, s, t, q, stat_type, b, v)
                for stat_type, sig_dict in summary.items()
                for s, qd in sig_dict.items()
                for q, td in qd.items()
                for t, bd in td.items()
                for b, v in bd.items()
            )
            arr_sum = np.array(list(gen_sum), dtype=object)
            cols = ['date','signal','target','qrank','stat_type','bet_size_col','value']
            summary_df = pd.DataFrame([dict(zip(cols, rec)) for rec in arr_sum], columns=cols)

            summary_path = os.path.join(SUMMARY_STATS_DIR, f'summary_stats_{date_tag}.pkl')
            _atomic_pickle_dump(summary_df, summary_path)
            print(f"✅ Saved summary stats -> {summary_path} ({len(summary_df)} rows)")
        else:
            print("No valid columns for summary stats; skipping summary save.")
    else:
        print("No daily data collected; skipping summary computation.")

    # 4) Compute & save outliers over all daily stats (PKL)
    outliers_path = None
    if daily_stats_frames:
        stats_all = pd.concat(daily_stats_frames, ignore_index=True)

        # ensure proper dtype for date to build a tag
        dates = pd.to_datetime(stats_all['date'], errors='coerce')
        first_date = pd.to_datetime(dates.min())
        last_date  = pd.to_datetime(dates.max())
        date_tag   = f"{first_date.strftime('%Y%m%d')}_{last_date.strftime('%Y%m%d')}" \
                     if pd.notna(first_date) and pd.notna(last_date) else "all"

        odf = compute_outliers(
            stats_all,
            stats_list=OUTLIER_METRICS,   # explicit metrics
            z_thresh=OUTLIER_Z_THRESH,
            top_k=OUTLIER_TOP_K,
        )
        outliers_path = os.path.join(OUTLIERS_DIR, f'outliers_{date_tag}.pkl')
        save_outliers(odf, outliers_path)
        print(f"⚠️  Saved outliers -> {outliers_path} ({len(odf)} rows)")
    else:
        print("No daily stats frames accumulated; skipping outlier computation.")

    # 5) Compute & save heatmaps bundle (PKL) using sidecars (1–2) + daily PKLs (3–6)
    heatmaps_path = None
    if daily_stats_frames:
        stats_all = pd.concat(daily_stats_frames, ignore_index=True)
        dates_hm = pd.to_datetime(stats_all['date'], errors='coerce')
        first_hm = pd.to_datetime(dates_hm.min())
        last_hm  = pd.to_datetime(dates_hm.max())
        date_tag = (
            f"{first_hm.strftime('%Y%m%d')}_{last_hm.strftime('%Y%m%d')}"
            if pd.notna(first_hm) and pd.notna(last_hm) else "all"
        )

        bundle = build_heatmaps_bundle(
            stats_df=stats_all,
            qranks=HM_QRANKS,
            panel_daily_dir=PANEL_DAILY_DIR,
            alpha_prefix=ALPHA_PREFIX,
            pnl_target_col=PNL_TARGET_COL,
            pnl_bet_col=PNL_BET_COL_FOR_HM2,
        )
        heatmaps_path = os.path.join(HEATMAPS_DIR, f"heatmaps_bundle_{date_tag}.pkl")
        save_heatmaps_pkl(bundle, heatmaps_path)
        print(f"🗺️  Saved heatmaps bundle -> {heatmaps_path}")
    else:
        print("No daily stats frames accumulated; skipping heatmaps bundle.")

    # 6) Optional: write an index of daily stat files
    if per_day_index_rows:
        index_df = pd.DataFrame(per_day_index_rows).sort_values('date')
        index_path = os.path.join(DAILY_STATS_DIR, '_index.csv')
        with NamedTemporaryFile(dir=os.path.dirname(index_path), delete=False, mode='w', newline='') as tmp:
            tmp_name = tmp.name
            index_df.to_csv(tmp_name, index=False)
        shutil.move(tmp_name, index_path)
        print(f"🧭 Wrote daily index -> {index_path}")

    # Return paths for downstream use
    return {
        'daily_dir': DAILY_STATS_DIR,
        'summary_path': summary_path,
        'outliers_path': outliers_path,
        'panel_dir': PANEL_DAILY_DIR,
        'heatmaps_path': heatmaps_path,
    }
