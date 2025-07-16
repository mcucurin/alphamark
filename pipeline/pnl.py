import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

def comp_r2_tval_pnl(
    fut_rets: pd.DataFrame,
    bet_size: pd.DataFrame,
    x_df: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """
    For each column, regress the *signed* PnL on the exposure x:
        PnL = fut_rets * bet_size * sign(x)
        model:  PnL ~ x
    Returns two pd.Series (indexed by column):
      - tvals:  the t‑statistic on the x coefficient
      - rsqs:   the R² of each regression
    """
    pnl = fut_rets * bet_size * np.sign(x_df)

    tvals = {}
    rsqs  = {}

    for col in pnl.columns:
        y = pnl[col]
        xi = x_df[col]

        # align non‑missing
        mask = y.notna() & xi.notna()
        if mask.sum() < 2:
            tvals[col] = np.nan
            rsqs[col]  = np.nan
            continue

        Y = y[mask]
        X = sm.add_constant(xi[mask])  # adds intercept

        model = sm.OLS(Y, X).fit()
        tvals[col] = model.tvalues[col]
        rsqs[col]  = model.rsquared

    return pd.Series(tvals), pd.Series(rsqs)

def comp_pnl_all_quantiles(
    qranks: pd.DataFrame,
    fut_rets: pd.DataFrame,
    bet_size: pd.DataFrame = None,
    xo: bool = False,
    add_corr: bool = True,
    add_long_ratio: bool = True,
    add_hit_ratio: bool = True,
    probs: np.ndarray = None,
    type_quantile: str = None,
    max_qrank: int = None,
    limit_counts_qr: dict = None,
    add_r2tval: bool = False,
    raw_alpha: pd.DataFrame = None,
    allow_r2_qranks: bool = False,
    spread_cost_in_out: float = None,
    add_stdev: bool = False,
) -> dict:
    """
    Compute PnL statistics across quantile portfolios.
    Returns: dict[label -> DataFrame(stats × alphas)]
    """
    if probs is not None:
        thresholds = np.arange(1, len(probs)+1)
    else:
        if max_qrank is None:
            max_qrank = int(np.nanmax(np.abs(qranks.values)))
        thresholds = np.arange(1, max_qrank+1)

    list_info = {}
    # standardize raw_alpha if needed
    if raw_alpha is not None:
        raw_std = (raw_alpha / raw_alpha.std(axis=0))

    for qr in thresholds:
        # mask out all |rank| < qr
        mask = qranks.abs() >= qr
        if type_quantile == 'quantEach':
            # also drop |rank| > qr
            mask &= (qranks.abs() <= qr)

        # optionally limit to first N rows
        if limit_counts_qr and qr in limit_counts_qr:
            N = min(limit_counts_qr[qr], mask.shape[0])
            mask = mask.iloc[:N]

        # build effective bet sizes and (optionally) raw_alpha
        if bet_size is not None:
            bs = bet_size.where(mask, other=0.0)
        else:
            bs = pd.DataFrame(1.0, index=qranks.index, columns=qranks.columns)

        if raw_alpha is not None:
            ra = raw_std.where(mask, other=0.0)
        else:
            ra = None

        # compute PNL matrix
        pnl_mat = np.sign(qranks) * fut_rets * bs
        if spread_cost_in_out is not None:
            pnl_mat = pnl_mat - spread_cost_in_out * bs

        stats = {}
        # total pnl
        pnl = pnl_mat.sum(axis=0)
        stats['pnl'] = pnl

        # pnl per dollar traded
        size_not = bs.sum(axis=0)
        stats['ppt'] = pnl / size_not

        # number of trades
        stats['nrTrades'] = mask.sum(axis=0)

        if add_corr:
            # Spearman corr per column
            corr = {
                col: spearmanr(qranks[col].where(mask[col]), fut_rets[col], nan_policy='omit').correlation
                for col in qranks.columns
            }
            stats['corr_SP'] = pd.Series(corr).round(3)

        if add_hit_ratio:
            hits = (pnl_mat > 0).sum(axis=0)
            misses = (pnl_mat < 0).sum(axis=0)
            stats['hitRatio'] = (hits / (hits + misses)).round(3)

        if add_long_ratio:
            longs = (qranks > 0).where(mask).sum(axis=0)
            shorts = (qranks < 0).where(mask).sum(axis=0)
            stats['longRatio'] = (longs / (longs + shorts)).round(3)

        if add_r2tval:
            tvals, rsqs = comp_r2_tval_pnl(fut_rets, bs, qranks if allow_r2_qranks else ra)
            stats['tval'] = pd.Series(tvals, index=qranks.columns)
            stats['rsq']  = pd.Series(rsqs, index=qranks.columns)

        if add_stdev:
            stdev = pnl_mat.std(axis=0)
            stats['stdev'] = stdev
            stats['sharpe'] = stats['ppt'] / stdev * np.sqrt(252)

        info_df = pd.DataFrame(stats)
        label = f"qr_{qr}"
        if limit_counts_qr and qr in limit_counts_qr:
            label += f"_{limit_counts_qr[qr]}"
        if type_quantile == 'SH':
            label += "_SH"

        list_info[label] = info_df

    return list_info

