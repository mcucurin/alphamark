# pipeline/panel_sidecars.py
import os
import shutil
from tempfile import NamedTemporaryFile
from typing import List, Optional, Sequence
import pandas as pd


def _atomic_pickle_dump(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as tmp:
        tmp_name = tmp.name
        df.to_pickle(tmp_name)
    shutil.move(tmp_name, path)


def _pick_one_target(df: pd.DataFrame, preferred: str = "fret_1_RR") -> Optional[str]:
    # Prefer 'fret_1d' if present, else the first column that starts with 'fret_'
    if preferred in df.columns:
        return preferred
    cands = [c for c in df.columns if str(c).startswith("fret_")]
    return cands[0] if cands else None


def _pick_all_targets(
    df: pd.DataFrame,
    prefixes: Sequence[str]
) -> List[str]:
    """Return all target columns that start with any of the given prefixes, preserving df column order."""
    prefixes = tuple(prefixes) if prefixes else tuple()
    if not prefixes:
        return []
    return [c for c in df.columns if any(str(c).startswith(p) for p in prefixes)]


def extract_panel_sidecar(
    df: pd.DataFrame,
    alpha_prefix: str = "pret_",
    # Back-compat: if set, we’ll include that single target (if present).
    target_col: Optional[str] = None,
    # New: explicit list of targets to include (subset of df.columns). If provided, wins over target_col/prefix scan.
    target_cols: Optional[List[str]] = None,
    # New: collect *all* targets matching these prefixes when target_cols is None. Defaults to all 'fret_' targets.
    target_prefixes: Optional[Sequence[str]] = ("fret_",),
    bet_cols_preferred: Optional[List[str]] = None,
    id_candidates: Optional[Sequence[str]] = ("ticker", "permno", "ric", "secid"),
) -> pd.DataFrame:
    """
    Build a per-stock sidecar DataFrame with:
      - all alpha columns (prefix `alpha_prefix`)
      - one or more realized return columns (targets) – either:
          * explicit `target_cols`, OR
          * if `target_col` given, that single column (if present), OR
          * all columns starting with any of `target_prefixes` (default: 'fret_')
      - any bet-size columns that exist (from `bet_cols_preferred`)
      - an id column if present (first match from `id_candidates`)
    """
    cols: List[str] = []

    # alphas
    alpha_cols = [c for c in df.columns if str(c).startswith(alpha_prefix)]
    cols += alpha_cols

    # targets
    if target_cols:
        tcols = [c for c in target_cols if c in df.columns]
    elif target_col:
        tcols = [target_col] if target_col in df.columns else []
        if not tcols:
            picked = _pick_one_target(df)
            tcols = [picked] if picked else []
    else:
        tcols = _pick_all_targets(df, target_prefixes or ())

    if not tcols:
        return pd.DataFrame()  # nothing to save if no targets found
    cols += tcols

    # bets (only those that actually exist)
    if bet_cols_preferred:
        cols += [b for b in bet_cols_preferred if b in df.columns]

    # stable id if present
    if id_candidates:
        for cand in id_candidates:
            if cand in df.columns:
                cols.append(cand)
                break

    # dedupe column list while preserving order and validity
    seen = set()
    keep: List[str] = []
    for c in cols:
        if c not in seen and c in df.columns:
            keep.append(c)
            seen.add(c)

    panel = df[keep].copy()
    return panel


def save_panel_sidecar(
    df: pd.DataFrame,
    day_str: str,
    out_dir: str = "output/PANEL_DAILY",
    alpha_prefix: str = "pret_",
    # Back-compat single target (optional)
    target_col: Optional[str] = None,
    # New multi-target support
    target_cols: Optional[List[str]] = None,
    target_prefixes: Optional[Sequence[str]] = ("fret_",),
    bet_cols_preferred: Optional[List[str]] = None,
    id_candidates: Optional[Sequence[str]] = ("ticker", "permno", "ric", "secid"),
) -> Optional[str]:
    """
    Extract and save a per-stock sidecar for this day, including ALL targets by default
    (any column matching `target_prefixes`, e.g. all 'fret_*').

    Returns path or None if nothing saved.
    """
    panel = extract_panel_sidecar(
        df,
        alpha_prefix=alpha_prefix,
        target_col=target_col,
        target_cols=target_cols,
        target_prefixes=target_prefixes,
        bet_cols_preferred=bet_cols_preferred,
        id_candidates=id_candidates,
    )
    if panel.empty:
        return None
    path = os.path.join(out_dir, f"panel_{day_str}.pkl")
    _atomic_pickle_dump(panel, path)
    return path
