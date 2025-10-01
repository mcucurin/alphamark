# stats.py
import numpy as np
from scipy.stats import spearmanr, linregress
from collections import defaultdict
from itertools import product

# ---- optional dcor ----
try:
    import dcor as _dcor
    def _distance_correlation(x, y):
        return _dcor.distance_correlation(x, y)
except Exception:
    def _distance_correlation(x, y):
        return np.nan


def create_5d_stats():
    """
    Access pattern:
      stats[stat_type][signal][qrank][target][bet_size] = value
    """
    return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))


# ========================= DAILY (nested loops, lean) =========================
def compute_daily_stats(
    df,
    signal_cols,
    target_cols,
    quantiles=[1.0, 0.75, 0.5, 0.25],
    bet_size_cols=['betsize_equal'],
    prev_state=None  # optional state to compute daily trades/ppt
):
    """
    Per-day cross-section stats (single date snapshot):
      - pnl (sum over names in bucket)
      - ppd  (pnl / notional)
      - sizeNotional (sum of bet sizes)    [code uses sum(b)]
      - nrInstr (names in bucket)
      - bet_size (mean bet for names in bucket)

    NEW (daily, optional if prev_state provided):
      - n_trades          (exact if an ID column is present; else proxy ≈ |ΔB_t| / mean_bet)
      - ppt               = pnl_t / n_trades_t  (if n_trades_t > 0)
      - turnover_notional = ∑ |Δ position_i| notional (exact with IDs; else |ΔB_t|)

    Quantile semantics (cumulative): for q in {1.0,0.75,0.5,0.25}
      mask = |sig| >= quantile_{day}(|sig|, 1 - q)
      -> qr_100 is the whole universe; qr_25 is the top 25% by |sig|.
    """
    stats = create_5d_stats()

    # Auto-detect an instrument ID column if present (used for exact trade counts)
    id_candidates = ['id', 'instrument', 'inst_id', 'sid', 'asset', 'symbol', 'ticker',
                     'permno', 'permno_', 'ric', 'secid']
    id_col = next((c for c in id_candidates if c in df.columns), None)

    needed = list(set(signal_cols + target_cols + bet_size_cols + ([id_col] if id_col else [])))
    df = df[needed].replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if df.empty:
        return stats

    # rolling state per (signal, qlabel, bet): Bt, mean_bet, and per-name positions if IDs exist
    if prev_state is None:
        prev_state = {}  # key=(signal, qlabel, bet) -> {'Bt': float, 'mean_bet': float, 'pos_map': {id: pos}}

    # minimal-change: cache trades/turnover per (signal, qlabel, bet) so all targets share it
    trades_cache = {}  # key=(signal, qlabel, bet) -> dict(n_trades, turnover, Bt_today, mean_bet_today, pos_map_today)

    for signal in signal_cols:
        if signal not in df.columns:
            continue

        # thresholds per q (computed once over df; expected df is a single day already)
        sig_abs = df[signal].abs().to_numpy(float)
        thresholds = {}
        for q in quantiles:
            thr = np.nanquantile(sig_abs, 1.0 - q)
            thresholds[f'qr_{int(q*100)}'] = thr

        for bet in bet_size_cols:
            if bet not in df.columns:
                continue

            for qlabel, thr in thresholds.items():
                # mask is target-agnostic
                mask = (df[signal].abs().to_numpy(float) >= thr)
                if not np.any(mask):
                    continue

                # Build a base subset that is target-agnostic (so trades don't depend on target NaNs)
                base_cols = [signal, bet] + ([id_col] if id_col else [])
                sub_base = df.loc[mask, base_cols].dropna(how="any")
                if sub_base.empty:
                    # still allow target-specific metrics to be computed (unlikely)
                    pass

                # Compute Bt/mean_bet from base subset (or fallback later if empty)
                if not sub_base.empty:
                    s_base = sub_base[signal].to_numpy(float)
                    b_base = sub_base[bet].to_numpy(float)
                    ids_base = sub_base[id_col].to_numpy() if id_col else None
                else:
                    s_base = np.array([], dtype=float)
                    b_base = np.array([], dtype=float)
                    ids_base = None

                # --- compute trades/turnover ONCE per (signal,q,bet); cache and update prev_state once ---
                key_sb = (signal, qlabel, bet)
                if key_sb not in trades_cache:
                    # derive today's notional and mean bet from base (or recompute later if empty)
                    Bt_today_base = float(b_base.sum()) if b_base.size else 0.0
                    mean_bet_today_base = float(np.nanmean(np.abs(b_base))) if b_base.size else 0.0

                    prev_entry = prev_state.get(key_sb, {})
                    prev_Bt = prev_entry.get('Bt', np.nan)
                    prev_mb = prev_entry.get('mean_bet', np.nan)
                    prev_map = prev_entry.get('pos_map', {}) if isinstance(prev_entry.get('pos_map', {}), dict) else {}

                    if id_col and ids_base is not None and ids_base.size:
                        # Exact per-instrument turnover/trades using positions (target exposure proxy)
                        pos_today = (np.sign(s_base) * b_base).astype(float)
                        pos_map_today = {inst: float(pos) for inst, pos in zip(ids_base, pos_today)}

                        day_turnover = 0.0
                        day_trades = 0
                        # entries/updates
                        for inst, pos in pos_map_today.items():
                            prev_pos = float(prev_map.get(inst, 0.0))
                            delta = abs(pos - prev_pos)
                            if delta > 0.0:
                                day_turnover += delta
                                day_trades += 1
                        # exits
                        if prev_map:
                            gone = set(prev_map.keys()) - set(pos_map_today.keys())
                            for inst in gone:
                                prev_pos = float(prev_map[inst])
                                delta = abs(0.0 - prev_pos)
                                if delta > 0.0:
                                    day_turnover += delta
                                    day_trades += 1

                        n_trades_today = float(day_trades)
                        trades_cache[key_sb] = {
                            'n_trades': n_trades_today,
                            'turnover': float(day_turnover),
                            'Bt_today': Bt_today_base,
                            'mean_bet_today': mean_bet_today_base,
                            'pos_map_today': pos_map_today
                        }
                        # update prev_state ONCE here
                        prev_state[key_sb] = {
                            'Bt': Bt_today_base,
                            'mean_bet': mean_bet_today_base,
                            'pos_map': pos_map_today
                        }
                    else:
                        # Fallback proxy: |ΔBt| / mean_bet
                        if np.isfinite(prev_Bt):
                            dBt = abs(Bt_today_base - prev_Bt)
                            denom = mean_bet_today_base if mean_bet_today_base > 0 else (prev_mb if np.isfinite(prev_mb) and prev_mb > 0 else 0.0)
                            n_trades_today = (dBt / denom) if denom > 0 else 0.0
                            turnover = float(dBt)
                        else:
                            n_trades_today = np.nan
                            turnover = np.nan

                        trades_cache[key_sb] = {
                            'n_trades': float(n_trades_today) if np.isfinite(n_trades_today) else np.nan,
                            'turnover': turnover,
                            'Bt_today': Bt_today_base,
                            'mean_bet_today': mean_bet_today_base,
                            'pos_map_today': {}
                        }
                        # update prev_state ONCE here
                        prev_state[key_sb] = {
                            'Bt': Bt_today_base,
                            'mean_bet': mean_bet_today_base,
                            'pos_map': {}
                        }

                # From here on, trades/turnover reused for ALL targets of this (signal,q,bet)
                n_trades_today_cached = trades_cache[key_sb]['n_trades']
                turnover_cached = trades_cache[key_sb]['turnover']

                # ---- per-target metrics (PnL, PPD, PPT) ----
                for target in target_cols:
                    if target not in df.columns:
                        continue

                    cols = [signal, target, bet]
                    if id_col:
                        cols.append(id_col)
                    sub = df.loc[mask, cols].dropna(how="any")
                    if sub.empty:
                        continue

                    s = sub[signal].to_numpy(float)
                    y = sub[target].to_numpy(float)
                    b = sub[bet].to_numpy(float)

                    pnl_vec = np.sign(s) * y * b
                    pnl = float(pnl_vec.sum())
                    Bt_today = float(b.sum())
                    mean_bet_today = float(np.nanmean(np.abs(b))) if b.size else 0.0

                    stats['pnl'][signal][qlabel][target][bet] = pnl
                    stats['ppd'][signal][qlabel][target][bet] = (pnl / Bt_today) if Bt_today > 0 else np.nan
                    stats['sizeNotional'][signal][qlabel][target][bet] = Bt_today
                    stats['nrInstr'][signal][qlabel][target][bet] = int(s.size)
                    stats['bet_size'][signal][qlabel][target][bet] = mean_bet_today

                    # broadcast cached trades/turnover to ALL targets
                    stats['n_trades'][signal][qlabel][target][bet] = n_trades_today_cached
                    stats['turnover_notional'][signal][qlabel][target][bet] = turnover_cached
                    # target-specific PPT (since PnL depends on target)
                    if np.isfinite(n_trades_today_cached) and n_trades_today_cached > 1e-12:
                        ppt_today = pnl / n_trades_today_cached
                    else:
                        ppt_today = np.nan
                    stats['ppt'][signal][qlabel][target][bet] = float(ppt_today) if np.isfinite(ppt_today) else np.nan

    return stats


# ===================== SUMMARY over multiple days (nested) ====================
def compute_summary_stats_over_days(
    df,
    date_col,
    signal_cols,
    target_cols,
    quantiles=[1.0, 0.75, 0.5, 0.25],
    bet_size_cols=['betsize_equal'],
    type_quantile='cumulative',   # 'cumulative' (>=thr) or 'quantEach' (exact bucket)
    add_spearman=True,
    add_dcor=False
):
    """
    Returns ONE summary value per (signal, qrank, target, bet):
      - sharpe     : time-series Sharpe of DAILY PPD (not rolling)
      - hit_ratio  : pooled fraction sign(sig)==sign(target), EXCLUDING target==0
      - long_ratio : pooled fraction sign(sig)>0 among traded names
      - r2, t_stat : from pooled linear regression y ~ s (scipy linregress over all pooled pairs)
      - spearman   : optional pooled Spearman(s, y)
      - dcor       : optional pooled distance correlation(s, y)
    """
    out = create_5d_stats()

    need = list(set([date_col] + signal_cols + target_cols + bet_size_cols))
    df = df[need].copy().replace([np.inf, -np.inf], np.nan)
    # keep raw date values (e.g., 20000103); grouping uses provided dtype
    df = df.dropna(subset=[date_col])
    if df.empty:
        return out

    # Accumulators
    daily_ppd = defaultdict(list)   # key=(signal, qlabel, target, bet) -> list of daily PPD
    pooled_pairs = defaultdict(lambda: {'s': [], 'y': []})  # for r2/t, spearman, dcor
    hit_num = defaultdict(int)
    hit_den = defaultdict(int)
    long_num = defaultdict(int)     # key=(signal, qlabel, bet)
    long_den = defaultdict(int)

    # Per-day loop
    for day, d in df.sort_values(date_col).groupby(date_col, sort=True):
        d = d.dropna(how='any')
        if d.empty:
            continue

        for signal in signal_cols:
            if signal not in d.columns:
                continue

            s_all = d[signal].to_numpy(float)
            sgn_all = np.sign(s_all)
            sabs_all = np.abs(s_all)

            # Build thresholds/buckets for this day and signal
            day_thresholds = {}
            for q in quantiles:
                day_thresholds[f'qr_{int(q*100)}'] = np.nanquantile(sabs_all, 1.0 - q)

            # For quantEach, precompute bin edges (equal-frequency) once per day/signal
            if type_quantile == 'quantEach':
                K = len(quantiles)
                probs = [i / K for i in range(K + 1)]
                edges = np.nanquantile(sabs_all, probs)

            for q in quantiles:
                qlabel = f'qr_{int(q*100)}'

                if type_quantile == 'cumulative':
                    mask = (sabs_all >= day_thresholds[qlabel])
                else:  # 'quantEach'
                    j = quantiles.index(q) + 1  # 1..K
                    lo = edges[j-1]
                    hi = edges[j]
                    mask = (sabs_all >= lo) & (sabs_all <= hi)

                if not np.any(mask):
                    continue

                s = s_all[mask]
                sgn = sgn_all[mask]
                n_names = int(s.size)

                for bet in bet_size_cols:
                    if bet not in d.columns:
                        continue
                    b = d.loc[mask, bet].to_numpy(float)

                    # --- accumulate long ratio counts per (signal, qlabel, bet) ---
                    long_num[(signal, qlabel, bet)] += int(np.sum(sgn > 0))
                    long_den[(signal, qlabel, bet)] += n_names

                    for target in target_cols:
                        if target not in d.columns:
                            continue
                        y = d.loc[mask, target].to_numpy(float)

                        # --- daily PPD for Sharpe (one per day) ---
                        pnl_day = float(np.nansum(np.sign(s) * y * b))
                        # Align PPD denominator with daily definition: Bt = sum_i b_i
                        notional_day = float(np.nansum(b))
                        key_full = (signal, qlabel, target, bet)
                        if notional_day > 0:
                            daily_ppd[key_full].append(pnl_day / notional_day)

                        # --- pooled pairs for r2/t, spearman, dcor ---
                        m = np.isfinite(s) & np.isfinite(y)
                        if m.any():
                            pooled_pairs[key_full]['s'].append(s[m])
                            pooled_pairs[key_full]['y'].append(y[m])

                        # --- hit ratio counts (exclude y==0) ---
                        y_sign = np.sign(y)
                        denom_mask = (y_sign != 0.0)
                        if denom_mask.any():
                            hit_num[key_full] += int(np.sum(np.sign(s)[denom_mask] == y_sign[denom_mask]))
                            hit_den[key_full] += int(np.sum(denom_mask))

    # Finalize one value per combo
    sqrt_252 = np.sqrt(252.0)

    # Sharpe on daily PPD
    for key_full, series in daily_ppd.items():
        arr = np.array(series, float)
        arr = arr[np.isfinite(arr)]
        if arr.size >= 2:
            mu = float(arr.mean())
            sd = float(arr.std(ddof=0))
            sharpe = mu / sd * sqrt_252 if sd > 0 else np.nan
        else:
            sharpe = np.nan
        signal, qlabel, target, bet = key_full
        out['sharpe'][signal][qlabel][target][bet] = float(sharpe) if np.isfinite(sharpe) else np.nan

    # r2, t_stat, spearman, dcor
    for key_full, bits in pooled_pairs.items():
        x = np.concatenate(bits['s']) if bits['s'] else np.array([])
        y = np.concatenate(bits['y']) if bits['y'] else np.array([])
        if x.size >= 3 and x.size == y.size:
            try:
                slope, intercept, r_val, p_val, stderr = linregress(x, y)
                r2 = float(r_val**2) if np.isfinite(r_val) else np.nan
                t_stat = float(slope / stderr) if (stderr is not None and stderr > 0) else np.nan
            except Exception:
                r2 = np.nan
                t_stat = np.nan
            if add_spearman:
                try:
                    sp = float(spearmanr(x, y, nan_policy='omit').correlation)
                except Exception:
                    sp = np.nan
                signal, qlabel, target, bet = key_full
                out['spearman'][signal][qlabel][target][bet] = sp if np.isfinite(sp) else np.nan
            if add_dcor:
                try:
                    dc = float(_distance_correlation(x, y))
                except Exception:
                    dc = np.nan
                signal, qlabel, target, bet = key_full
                out['dcor'][signal][qlabel][target][bet] = dc if np.isfinite(dc) else np.nan
        else:
            r2 = np.nan
            t_stat = np.nan

        signal, qlabel, target, bet = key_full
        out['r2'][signal][qlabel][target][bet] = r2 if np.isfinite(r2) else np.nan
        out['t_stat'][signal][qlabel][target][bet] = t_stat if np.isfinite(t_stat) else np.nan

    # hit ratio
    for key_full, hn in hit_num.items():
        hd = hit_den[key_full]
        signal, qlabel, target, bet = key_full
        out['hit_ratio'][signal][qlabel][target][bet] = (hn / hd) if hd > 0 else np.nan

    # finalize long_ratio (accumulated per (signal, qlabel, bet), broadcast to targets)
    for (signal, qlabel, bet), ln in long_num.items():
        ld = long_den[(signal, qlabel, bet)]
        val = (ln / ld) if ld > 0 else np.nan
        for target in target_cols:
            out['long_ratio'][signal][qlabel][target][bet] = val

    return out
