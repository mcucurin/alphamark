# pipeline/heatmap_stats.py
import os
import glob
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from tempfile import NamedTemporaryFile
import shutil
from scipy.stats import spearmanr


# -----------------------------
# Atomic pickle I/O
# -----------------------------
def _atomic_pickle_dump(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as tmp:
        tmp_name = tmp.name
        pd.to_pickle(obj, tmp_name)
    shutil.move(tmp_name, path)


def save_heatmaps_pkl(bundle: Dict[str, Any], path: str) -> None:
    _atomic_pickle_dump(bundle, path)


def load_heatmaps_pkl(path: str) -> Dict[str, Any]:
    return pd.read_pickle(path)


# -----------------------------
# Helpers
# -----------------------------
def _pick_default(series: pd.Series) -> Optional[str]:
    if series is None:
        return None
    s = series.dropna()
    if s.empty:
        return None
    return s.value_counts().idxmax()


def _infer_alphas_from_columns(cols: List[str], explicit: Optional[List[str]], prefix: str) -> List[str]:
    if explicit:
        return [c for c in explicit if c in cols]
    return sorted([c for c in cols if c.startswith(prefix)])


def _dedupe_identical_columns(df: pd.DataFrame, cols: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Drop columns among `cols` that are exactly identical (bitwise equal) for this day.
    Returns (kept_cols, dropped_pairs_info).
    """
    if len(cols) <= 1:
        return cols, []
    X = df[cols]
    # We'll detect duplicates by hashing each column's values (including NaNs pattern)
    hashes = {}
    equal_map = []
    kept = []
    for c in cols:
        # Use tuple of values for hashing (NaNs are fine in tuples)
        key = tuple(X[c].values.tolist())
        if key in hashes:
            equal_map.append((hashes[key], c))  # (kept_col, dup_col)
        else:
            hashes[key] = c
            kept.append(c)
    return kept, equal_map


def _spearman_corr_across_stocks(mat: np.ndarray) -> np.ndarray:
    k = mat.shape[1]
    M = np.full((k, k), np.nan, float)
    for i in range(k):
        xi = mat[:, i]
        for j in range(i, k):
            xj = mat[:, j]
            m = np.isfinite(xi) & np.isfinite(xj)
            if m.sum() >= 3:
                rho = spearmanr(xi[m], xj[m], nan_policy='omit').correlation
                if np.isfinite(rho):
                    M[i, j] = M[j, i] = float(rho)
    return M


def _first_sidecar_columns(panel_daily_dir: str) -> Optional[List[str]]:
    files = sorted(glob.glob(os.path.join(panel_daily_dir, "panel_*.pkl")))
    if not files:
        return None
    d = pd.read_pickle(files[0])
    return list(d.columns)


def _autodetect_target_from_sidecars(panel_daily_dir: str) -> Optional[str]:
    cols = _first_sidecar_columns(panel_daily_dir) or []
    # Prefer common names if present
    prefs = ["fret_1d", "fret_1_RR", "fret_1D", "fret"]
    for p in prefs:
        if p in cols:
            return p
    # fallback: any column starting with 'fret_'
    cands = [c for c in cols if c.startswith("fret_")]
    return cands[0] if cands else None


# -----------------------------
# Heatmap 1 & 2 from sidecars
# -----------------------------
def _compute_heatmap1_alpha_space(
    panel_daily_dir: str,
    alpha_prefix: str,
    alpha_cols_explicit: Optional[List[str]] = None,
) -> Tuple[Optional[np.ndarray], Optional[List[str]], int, dict]:
    """
    Alpha space: for each day, compute Spearman corr across stocks between alphas,
    then average over days. Deduplicate identical alpha columns per day.
    """
    day_files = sorted(glob.glob(os.path.join(panel_daily_dir, "panel_*.pkl")))
    if not day_files:
        return None, None, 0, {"reason": "no_sidecars"}

    sum_M = None
    n_used = 0
    labels_global = None
    notes = {"dedup_examples": [], "k_seen": []}

    for p in day_files:
        d = pd.read_pickle(p)
        if d.empty:
            continue

        # identify alpha cols for this day, then dedupe if identical
        cols_all = list(d.columns)
        labels_day = _infer_alphas_from_columns(cols_all, alpha_cols_explicit, alpha_prefix)
        if len(labels_day) < 2:
            continue

        labels_day, dup_pairs = _dedupe_identical_columns(d, labels_day)
        if dup_pairs:
            # record one example (file name and duplicates)
            notes["dedup_examples"].append({"file": os.path.basename(p), "duplicates": dup_pairs[:5]})

        notes["k_seen"].append(len(labels_day))
        if len(labels_day) < 2:
            continue

        # set global labels to the intersection across days to ensure consistent ordering
        if labels_global is None:
            labels_global = labels_day
        else:
            # keep only those that exist every day we've used so far
            labels_global = [c for c in labels_global if c in labels_day]

        if len(labels_global) < 2:
            # not enough common alphas across days
            continue

        X = d[labels_global].apply(pd.to_numeric, errors='coerce').to_numpy()
        if X.shape[0] < 3:
            continue

        M = _spearman_corr_across_stocks(X)
        if not np.isfinite(M).any():
            continue
        sum_M = (M if sum_M is None else sum_M + M)
        n_used += 1

    if n_used == 0 or labels_global is None or len(labels_global) < 2:
        return None, labels_global, 0, {"reason": "insufficient_data", **notes}

    H1 = sum_M / n_used
    return H1, labels_global, n_used, notes


def _compute_heatmap2_pnl_across_stocks_avg(
    panel_daily_dir: str,
    alpha_prefix: str,
    pnl_target_col: Optional[str],
    pnl_bet_col: Optional[str] = None,
    alpha_cols_explicit: Optional[List[str]] = None,
) -> Tuple[Optional[np.ndarray], Optional[List[str]], int, dict]:
    """
    PnL space: per day, build per-stock per-signal pnl = sign(alpha) * target * |bet| (or 1),
    compute Spearman corr across stocks, then average over days.
    Auto-detect target if not provided or missing.
    Deduplicate identical alpha columns per day.
    """
    day_files = sorted(glob.glob(os.path.join(panel_daily_dir, "panel_*.pkl")))
    if not day_files:
        return None, None, 0, {"reason": "no_sidecars"}

    # Auto-detect a valid target column if needed/missing
    if pnl_target_col is None:
        pnl_target_col = _autodetect_target_from_sidecars(panel_daily_dir)

    sum_M = None
    n_used = 0
    labels_global = None
    notes = {"target_used": pnl_target_col, "dedup_examples": [], "k_seen": []}

    for p in day_files:
        d = pd.read_pickle(p)
        if d.empty:
            continue
        if pnl_target_col not in d.columns:
            # try one-shot fallback on this day (any fret_*)
            fallback = next((c for c in d.columns if c.startswith("fret_")), None)
            if fallback is None:
                continue
            notes["target_used"] = fallback
            target_col = fallback
        else:
            target_col = pnl_target_col

        cols_all = list(d.columns)
        labels_day = _infer_alphas_from_columns(cols_all, alpha_cols_explicit, alpha_prefix)
        if len(labels_day) < 2:
            continue

        labels_day, dup_pairs = _dedupe_identical_columns(d, labels_day)
        if dup_pairs:
            notes["dedup_examples"].append({"file": os.path.basename(p), "duplicates": dup_pairs[:5]})

        notes["k_seen"].append(len(labels_day))
        if len(labels_day) < 2:
            continue

        if labels_global is None:
            labels_global = labels_day
        else:
            labels_global = [c for c in labels_global if c in labels_day]

        if len(labels_global) < 2:
            continue

        Y = pd.to_numeric(d[target_col], errors='coerce').to_numpy()
        if pnl_bet_col and (pnl_bet_col in d.columns):
            B = np.abs(pd.to_numeric(d[pnl_bet_col], errors='coerce').to_numpy())
        else:
            B = np.ones_like(Y, dtype=float)

        X = np.column_stack([
            np.sign(pd.to_numeric(d[col], errors='coerce').to_numpy()) * Y * B
            for col in labels_global
        ])

        if X.shape[0] < 3:
            continue

        M = _spearman_corr_across_stocks(X)
        if not np.isfinite(M).any():
            continue
        sum_M = (M if sum_M is None else sum_M + M)
        n_used += 1

    if n_used == 0 or labels_global is None or len(labels_global) < 2:
        return None, labels_global, 0, {"reason": "insufficient_data", **notes}

    H2 = sum_M / n_used
    return H2, labels_global, n_used, notes


# -----------------------------
# Heatmaps 3–6 from DAILY PKLs
# -----------------------------
def _time_corr_matrix_for(
    stats_df: pd.DataFrame, target: str, bet: str, qrank: str
) -> Optional[pd.DataFrame]:
    df = stats_df[
        (stats_df['stat_type'] == 'pnl') &
        (stats_df['target'] == target) &
        (stats_df['bet_size_col'] == bet) &
        (stats_df['qrank'] == qrank)
    ].copy()

    if df.empty:
        return None

    mat = (
        df.pivot_table(index='date', columns='signal', values='value', aggfunc='sum')
          .sort_index()
    )
    mat = mat.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')

    if mat.shape[1] < 2 or mat.shape[0] < 5:
        return None

    C = mat.corr(method='pearson', min_periods=5)
    return C


# -----------------------------
# Public: build full bundle (1–6)
# -----------------------------
def build_heatmaps_bundle(
    stats_df: pd.DataFrame,
    qranks: List[str],
    panel_daily_dir: Optional[str] = None,
    alpha_prefix: str = "pret_",
    pnl_target_col: Optional[str] = None,
    pnl_bet_col: Optional[str] = None,
    alpha_cols_explicit: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Returns a dictionary with:
      meta: info about choices and availability
      heatmap_1: dict or None -> {'matrix': np.ndarray, 'labels': [..], 'n_days': int, 'notes': {...}}
      heatmap_2: dict or None -> {'matrix': np.ndarray, 'labels': [..], 'n_days': int, 'notes': {...}}
      heatmaps_by_quantile: {qrank: pd.DataFrame or None}
    """
    bundle: Dict[str, Any] = {
        "meta": {
            "qranks": list(qranks),
            "panel_daily_dir": panel_daily_dir,
            "alpha_prefix": alpha_prefix,
            "pnl_target_requested": pnl_target_col,
            "pnl_bet_col": pnl_bet_col,
        },
        "heatmap_1": None,
        "heatmap_2": None,
        "heatmaps_by_quantile": {},
    }

    # Dates ready for quantile heatmaps
    sdf = stats_df.copy()
    sdf['date'] = pd.to_datetime(sdf['date'], errors='coerce')

    target_chosen = _pick_default(sdf['target'])
    bet_chosen    = _pick_default(sdf['bet_size_col'])
    bundle["meta"]["target_chosen"] = target_chosen
    bundle["meta"]["bet_chosen"]    = bet_chosen

    # ---- Heatmap 1 & 2 (require sidecars) ----
    have_sidecars = bool(panel_daily_dir) and os.path.isdir(panel_daily_dir) and \
                    len(glob.glob(os.path.join(panel_daily_dir, "panel_*.pkl"))) > 0

    if have_sidecars:
        # Heatmap 1
        H1, labels1, n1, notes1 = _compute_heatmap1_alpha_space(
            panel_daily_dir, alpha_prefix, alpha_cols_explicit
        )
        if H1 is not None and labels1 is not None and n1 > 0:
            bundle["heatmap_1"] = {"matrix": H1, "labels": labels1, "n_days": n1, "notes": notes1}
        else:
            bundle["meta"]["heatmap_1_unavailable_reason"] = f"Alpha-space unavailable: {notes1}"

        # Heatmap 2
        H2, labels2, n2, notes2 = _compute_heatmap2_pnl_across_stocks_avg(
            panel_daily_dir, alpha_prefix, pnl_target_col, pnl_bet_col, alpha_cols_explicit
        )
        if H2 is not None and labels2 is not None and n2 > 0:
            bundle["heatmap_2"] = {"matrix": H2, "labels": labels2, "n_days": n2, "notes": notes2}
        else:
            bundle["meta"]["heatmap_2_unavailable_reason"] = f"PnL-space unavailable: {notes2}"
    else:
        bundle["meta"]["heatmap_1_unavailable_reason"] = "No sidecars found."
        bundle["meta"]["heatmap_2_unavailable_reason"] = "No sidecars found."

    # ---- Heatmaps 3–6 (from daily PKLs only) ----
    if (target_chosen is None) or (bet_chosen is None):
        bundle["meta"]["quantile_heatmaps_unavailable_reason"] = (
            "Could not infer target or bet_size_col from stats_df."
        )
        return bundle

    for q in qranks:
        C = _time_corr_matrix_for(sdf, target=target_chosen, bet=bet_chosen, qrank=q)
        bundle["heatmaps_by_quantile"][q] = C  # may be None

    return bundle
