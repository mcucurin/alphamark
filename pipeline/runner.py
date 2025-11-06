# pipeline/runner.py — consume daily feature PKLs -> produce stats/summary/outliers (all PKL)
import os
import re
import pickle
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd

from pipeline.daily_stats import compute_daily_stats
from pipeline.summary_stats import compute_summary_stats_over_days
from pipeline.outliers_stats import compute_outliers, save_outliers

# ===================== CONFIG =====================
FEATURES_INPUT_DIR  = "output/DAILY_FEATURES_PKL"   # where features_{YYYYMMDD}.pkl live
FEATURES_GLOB       = "features_*.pkl"

OUTPUT_ROOT         = "output"
DAILY_STATS_DIR     = os.path.join(OUTPUT_ROOT, "DAILY_STATS")
SUMMARY_STATS_DIR   = os.path.join(OUTPUT_ROOT, "SUMMARY_STATS")
OUTLIERS_DIR        = os.path.join(OUTPUT_ROOT, "OUTLIERS")

# Market proxy settings
SPY_TICKER          = "SPY"          # the ticker string inside 'ticker' column
SPY_COL_NAME        = "spy_ret"      # column we'll synthesize per day with SPY return

os.makedirs(DAILY_STATS_DIR, exist_ok=True)
os.makedirs(SUMMARY_STATS_DIR, exist_ok=True)
os.makedirs(OUTLIERS_DIR, exist_ok=True)

# Stats knobs
QUANTILES           = [1.0, 0.75, 0.5, 0.25]
TYPE_QUANTILE       = "cumulative"

# Outliers knobs (MUST be non-empty for compute_outliers)
# NOTE: PPT removed everywhere.
OUTLIER_METRICS     = ["pnl", "ppd", "sizeNotional", "nrInstr", "n_trades"]
OUTLIER_Z_THRESH    = 3.0

# ===================== UTILS =====================
def _atomic_pickle_dump(obj, path: str) -> None:
    """Atomic write of a pickle file (uses highest protocol)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as tmp:
        pickle.dump(obj, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_name = tmp.name
    os.replace(tmp_name, path)

def _extract_date_str(name: str) -> str | None:
    """Extract YYYYMMDD from a filename like 'features_20240103.pkl' or any name containing 8 digits."""
    m = re.search(r"(\d{8})", name)
    return m.group(1) if m else None

def _read_pickle_compat(path: Path):
    """
    Load a pickle across NumPy 1.x/2.x by remapping module paths:
    'numpy._core.*' -> 'numpy.core'
    """
    class NPCompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core")
            return super().find_class(module, name)
    with path.open("rb") as f:
        return NPCompatUnpickler(f).load()

# ===================== PIPELINE =====================
def run_pipeline():
    # 1) Discover and load feature PKLs (one per day)
    features_dir = Path(FEATURES_INPUT_DIR)
    files = sorted(features_dir.glob(FEATURES_GLOB), key=lambda p: p.name)
    if not files:
        print(f"[stop] No feature PKLs matching {features_dir / FEATURES_GLOB}")
        return {
            "daily_dir": DAILY_STATS_DIR,
            "summary_path": None,
            "outliers_path": None,
            "index_csv": None,
            "index_pkl": None,
        }

    items = []
    for p in files:
        day_str = _extract_date_str(p.name)
        if not day_str:
            print(f"[skip] cannot parse date from {p.name}")
            continue

        try:
            df = _read_pickle_compat(p)
        except Exception as e:
            print(f"[skip] failed to read {p}: {e}")
            continue

        # Must have ticker and at least one pret_/fret_/betsize_
        if "ticker" not in df.columns:
            print(f"[skip] {p.name}: missing 'ticker'")
            continue
        signal_cols = [c for c in df.columns if c.startswith("pret_")]
        target_cols = [c for c in df.columns if c.startswith("fret_")]
        bet_cols    = [c for c in df.columns if c.startswith("betsize_")]
        if not signal_cols or not target_cols or not bet_cols:
            print(f"[skip] {p.name}: missing features (pret_*, fret_*, betsize_*)")
            continue

        keep = ["ticker"] + sorted(set(signal_cols + target_cols + bet_cols))
        df_use = df[keep].copy()

        # Derive a single daily SPY return and broadcast to a column (SPY_COL_NAME)
        spy_val = None
        try:
            if (df_use['ticker'] == SPY_TICKER).any():
                # Prefer 'fret_1D' if present, else first available 'fret_*'
                if 'fret_1D' in target_cols:
                    tcol = 'fret_1D'
                else:
                    t_candidates = [c for c in target_cols if c.startswith('fret_')]
                    tcol = t_candidates[0] if t_candidates else None
                if tcol:
                    spy_row = df_use.loc[df_use['ticker'] == SPY_TICKER, tcol]
                    if not spy_row.empty:
                        spy_val = float(pd.to_numeric(spy_row, errors='coerce').dropna().mean())
        except Exception:
            spy_val = None

        if spy_val is not None:
            df_use[SPY_COL_NAME] = float(spy_val)
        else:
            # keep column absent if we couldn't compute SPY for the day
            pass

        day_dt = pd.to_datetime(day_str, format="%Y%m%d", errors="coerce")
        if pd.isna(day_dt):
            print(f"[skip] bad date {day_str} from {p.name}")
            continue

        items.append((day_dt, day_str, str(p), df_use))

    # sort chronologically
    items.sort(key=lambda x: x[0])
    if not items:
        print("[stop] No usable feature files after filtering.")
        return {
            "daily_dir": DAILY_STATS_DIR,
            "summary_path": None,
            "outliers_path": None,
            "index_csv": None,
            "index_pkl": None,
        }

    # 2) Per-day stats (PKL), and build inputs for summary/outliers
    daily_stats_frames = []
    per_day_index_rows = []
    raw_days_for_summary = []
    needed_sig, needed_tgt, needed_bet = set(), set(), set()
    spy_present_anywhere = False

    for day_dt, day_str, src_path, df in items:
        signal_cols    = [c for c in df.columns if c.startswith("pret_")]
        target_cols    = [c for c in df.columns if c.startswith("fret_")]
        bet_size_cols  = [c for c in df.columns if c.startswith("betsize_")]
        has_spy_col    = (SPY_COL_NAME in df.columns)

        needed_sig.update(signal_cols)
        needed_tgt.update(target_cols)
        needed_bet.update(bet_size_cols)
        spy_present_anywhere = spy_present_anywhere or has_spy_col

        # DAILY STATS (stateful carry lives inside compute_daily_stats)
        stats = compute_daily_stats(
            df,
            signal_cols=signal_cols,
            target_cols=target_cols,
            quantiles=QUANTILES,
            bet_size_cols=bet_size_cols,
            type_quantile=TYPE_QUANTILE,
            empty_day_policy="carry",
            report_empty_trades_as_nan=True,
        )

        # Flatten nested dict -> rows
        rows = []
        for stat_type, sig_dict in stats.items():
            for s, qd in sig_dict.items():
                for q, td in qd.items():
                    for t, bd in td.items():
                        for b, v in bd.items():
                            rows.append((day_str, s, t, q, stat_type, b, v))

        if not rows:
            print(f"[skip] {day_str}: no stats produced")
            continue

        day_df_stats = pd.DataFrame(
            rows,
            columns=["date", "signal", "target", "qrank", "stat_type", "bet_size_col", "value"],
        )
        out_path = os.path.join(DAILY_STATS_DIR, f"stats_{day_str}.pkl")
        _atomic_pickle_dump(day_df_stats, out_path)
        per_day_index_rows.append({"date": day_str, "path": out_path, "n_rows": len(day_df_stats)})
        daily_stats_frames.append(day_df_stats)
        print(f"📦 Saved daily stats PKL for {day_str} -> {out_path} ({len(day_df_stats)} rows)")

        # lean frame for summary (add 'date' and keep SPY_COL_NAME if present)
        keep_cols = ["date"] + signal_cols + target_cols + bet_size_cols + ([SPY_COL_NAME] if has_spy_col else [])
        raw_days_for_summary.append(
            df[signal_cols + target_cols + bet_size_cols + ([SPY_COL_NAME] if has_spy_col else [])]
              .assign(date=day_dt)[keep_cols]
        )

    # 3) Summary stats over all days (PKL)
    summary_path = None
    if raw_days_for_summary:
        big_df = pd.concat(raw_days_for_summary, ignore_index=True, copy=False)

        sig_list = sorted([c for c in needed_sig if c in big_df.columns])
        tgt_list = sorted([c for c in needed_tgt if c in big_df.columns])
        bet_list = sorted([c for c in needed_bet if c in big_df.columns])

        if sig_list and tgt_list and bet_list:
            summary = compute_summary_stats_over_days(
                big_df,
                date_col="date",
                signal_cols=sig_list,
                target_cols=tgt_list,
                bet_size_cols=bet_list,
                quantiles=QUANTILES,
                type_quantile=TYPE_QUANTILE,
                add_spearman=False,   # bounded per-row Spearman (signal vs target)
                add_dcor=False,
                spearman_sample_cap_per_key=10000,
                random_state=123,
                spy_col=SPY_COL_NAME if spy_present_anywhere else None,  # Spearman corr(PnL, SPY) per (s,q,t,b)
            )

            # flatten summary -> rows; tag with date range
            all_days  = pd.to_datetime(big_df["date"], errors="coerce")
            first_day = pd.to_datetime(all_days.min())
            last_day  = pd.to_datetime(all_days.max())
            date_tag  = (
                f"{first_day:%Y%m%d}_{last_day:%Y%m%d}"
                if pd.notna(first_day) and pd.notna(last_day) else "summary"
            )

            s_rows = []
            for stat_type, sig_dict in summary.items():
                for s, qd in sig_dict.items():
                    for q, td in qd.items():
                        for t, bd in td.items():
                            for b, v in bd.items():
                                s_rows.append((
                                    f"{last_day:%Y%m%d}" if pd.notna(last_day) else None,
                                    s, t, q, stat_type, b, v
                                ))

            if s_rows:
                summary_df = pd.DataFrame(
                    s_rows,
                    columns=["date", "signal", "target", "qrank", "stat_type", "bet_size_col", "value"],
                )
                summary_path = os.path.join(SUMMARY_STATS_DIR, f"summary_stats_{date_tag}.pkl")
                _atomic_pickle_dump(summary_df, summary_path)
                print(f"✅ Saved summary stats PKL -> {summary_path} ({len(summary_df)} rows)")
            else:
                print("[info] Summary produced no rows; not saving.")
        else:
            print("[info] No valid columns for summary stats; skipping summary save.")
    else:
        print("[info] No daily data collected; skipping summary computation.")

    # 4) Outliers across all daily-stat frames (PKL)
    outliers_path = None
    if daily_stats_frames:
        stats_all = pd.concat(daily_stats_frames, ignore_index=True, copy=False)
        dates = pd.to_datetime(stats_all["date"], errors="coerce")
        first_day = pd.to_datetime(dates.min())
        last_day  = pd.to_datetime(dates.max())
        date_tag  = (
            f"{first_day:%Y%m%d}_{last_day:%Y%m%d}"
            if pd.notna(first_day) and pd.notna(last_day) else "all"
        )

        odf = compute_outliers(
            stats_all,
            stats_list=OUTLIER_METRICS,      # NOTE: PPT removed from this list.
            z_thresh=OUTLIER_Z_THRESH,
        )
        outliers_path = os.path.join(OUTLIERS_DIR, f"outliers_{date_tag}.pkl")
        save_outliers(odf, outliers_path)
        print(f"⚠️  Saved outliers PKL -> {outliers_path} ({len(odf)} rows)")
    else:
        print("[info] No daily stats frames accumulated; skipping outlier computation.")

    # 5) Write index for daily stats (CSV + PKL)
    index_csv = None
    index_pkl = None
    if per_day_index_rows:
        index_df = pd.DataFrame(per_day_index_rows).sort_values("date")
        index_csv = os.path.join(DAILY_STATS_DIR, "_index.csv")
        with NamedTemporaryFile(dir=os.path.dirname(index_csv), delete=False, mode="w", newline="") as tmp:
            index_df.to_csv(tmp.name, index=False)
            tmp_name = tmp.name
        shutil.move(tmp_name, index_csv)

        index_pkl = os.path.join(DAILY_STATS_DIR, "_index.pkl")
        _atomic_pickle_dump(index_df, index_pkl)
        print(f"🧭 Wrote daily index -> {index_csv} and {index_pkl}")

    return {
        "daily_dir": DAILY_STATS_DIR,
        "summary_path": summary_path,
        "outliers_path": outliers_path,
        "index_csv": index_csv,
        "index_pkl": index_pkl,
    }

# ===================== ENTRY =====================
if __name__ == "__main__":
    run_pipeline()
