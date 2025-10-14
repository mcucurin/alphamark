# pipeline/panel_sidecars.py
import os
import pandas as pd
from typing import List, Optional
from tempfile import NamedTemporaryFile
import shutil


def _atomic_pickle_dump(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with NamedTemporaryFile(dir=os.path.dirname(path), delete=False) as tmp:
        tmp_name = tmp.name
        df.to_pickle(tmp_name)
    shutil.move(tmp_name, path)


def _pick_one_target(df: pd.DataFrame, preferred: str = "fret_1d") -> Optional[str]:
    # Prefer 'fret_1d' if present, else the first column that starts with 'fret_'
    if preferred in df.columns:
        return preferred
    cands = [c for c in df.columns if str(c).startswith("fret_")]
    return cands[0] if cands else None


def extract_panel_sidecar(
    df: pd.DataFrame,
    alpha_prefix: str = "pret_",
    target_col: Optional[str] = None,
    bet_cols_preferred: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Build a per-stock sidecar DataFrame with:
      - all alpha columns (prefix `alpha_prefix`)
      - realized return column (target_col)
      - any bet-size columns that exist (from bet_cols_preferred)
      - (optional) a ticker/id column if present
    """
    cols = []

    # alphas
    alpha_cols = [c for c in df.columns if str(c).startswith(alpha_prefix)]
    cols += alpha_cols

    # realized return
    tgt = target_col or _pick_one_target(df)
    if tgt is None or tgt not in df.columns:
        return pd.DataFrame()  # nothing to save
    cols.append(tgt)

    # bets (only those that actually exist)
    if bet_cols_preferred:
        cols += [b for b in bet_cols_preferred if b in df.columns]

    # stable id if present
    id_col = None
    for cand in ("ticker", "permno", "ric", "secid"):
        if cand in df.columns:
            id_col = cand
            cols.append(cand)
            break

    # dedupe column list while preserving order
    seen = set()
    keep = []
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
    target_col: Optional[str] = None,
    bet_cols_preferred: Optional[List[str]] = None
) -> Optional[str]:
    """
    Extract and save a per-stock sidecar for this day.
    Returns path or None if nothing saved.
    """
    panel = extract_panel_sidecar(
        df,
        alpha_prefix=alpha_prefix,
        target_col=target_col,
        bet_cols_preferred=bet_cols_preferred
    )
    if panel.empty:
        return None
    path = os.path.join(out_dir, f"panel_{day_str}.pkl")
    _atomic_pickle_dump(panel, path)
    return path
