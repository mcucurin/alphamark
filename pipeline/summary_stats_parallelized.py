# =============================
# summary_stats_fastest.py — fully parallel (by-day, all cores)
# =============================
from __future__ import annotations
import numpy as np
from collections import defaultdict
from typing import Sequence
from scipy.stats import spearmanr, linregress
from joblib import Parallel, delayed
import pandas as pd

# Optional distance correlation
try:
    import dcor as _dcor
    def _distance_correlation(x, y):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        n = min(x.size, y.size)
        if n < 3 or np.all(x == x[0]) or np.all(y == y[0]):
            return np.nan
        return float(_dcor.distance_correlation(x, y))
except Exception:
    def _distance_correlation(x, y):
        return np.nan


# ---- minimal store ----
def create_5d_stats():
    return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))


# ---- helpers ----
def _float_clean(a): 
    a = np.asarray(a, float)
    a[~np.isfinite(a)] = np.nan
    return a

def _qlabel(q): 
    return f"qr_{int(round(q*100))}"

def _topk_mask_desc(abs_vals, finite_mask, q):
    idx_fin = np.where(finite_mask)[0]
    if idx_fin.size == 0 or q <= 0: 
        return np.zeros_like(finite_mask, bool)
    if q >= 1: 
        m = np.zeros_like(finite_mask, bool); m[idx_fin]=True; return m
    k = int(np.ceil(q * idx_fin.size))
    order = np.argsort(-abs_vals[idx_fin], kind="mergesort")
    choose = idx_fin[order[:k]]
    m = np.zeros_like(finite_mask, bool)
    m[choose] = True
    return m


# ---- single-day computation ----
def _summarize_day(day_df, signal_cols, target_cols, bet_cols, quantiles):
    S = np.column_stack([_float_clean(day_df[c]) for c in signal_cols])
    Y = np.column_stack([_float_clean(day_df[c]) for c in target_cols])
    B = np.column_stack([np.abs(_float_clean(day_df[c])) for c in bet_cols])

    daily_ppd = defaultdict(list)
    pooled_pairs = defaultdict(lambda:{'s':[],'y':[]})
    hit_num = defaultdict(int)
    hit_den = defaultdict(int)
    long_num = defaultdict(int)
    long_den = defaultdict(int)
    daily_effp = defaultdict(list)

    for si, sname in enumerate(signal_cols):
        s = S[:, si]
        m_fin = np.isfinite(s)
        if not m_fin.any(): continue
        sgn = np.sign(s)
        abs_s = np.abs(s)

        for q in quantiles:
            ql = _qlabel(q)
            mask = _topk_mask_desc(abs_s, m_fin, q)
            if not mask.any(): continue

            s_q = s[mask]; sgn_q = sgn[mask]
            Y_q = Y[mask]; B_q = B[mask]
            Yf = np.isfinite(Y_q); Bf = np.isfinite(B_q)
            Yz = np.where(Yf, Y_q, 0); Bz = np.where(Bf, B_q, 0)

            pnl = (Yz.T @ (sgn_q[:,None]*Bz))
            notional = (Yf.astype(float).T @ Bz)
            ppd = np.divide(pnl, notional, out=np.full_like(pnl,np.nan), where=notional>0)
            denom_abs = (np.abs(Yz).T @ Bz)
            effP = np.divide(np.abs(pnl), denom_abs, out=np.full_like(pnl,np.nan), where=denom_abs>0)

            for ti, tname in enumerate(target_cols):
                for bi, bname in enumerate(bet_cols):
                    v = ppd[ti,bi]
                    if np.isfinite(v): daily_ppd[(sname,ql,tname,bname)].append(float(v))
                    e = effP[ti,bi]
                    if np.isfinite(e): daily_effp[(sname,ql,tname,bname)].append(float(e))

            y_sign = np.sign(Y_q)
            nonzero = (y_sign != 0) & Yf
            denom = (nonzero.astype(float).T @ Bf.astype(float))
            numer = (((np.sign(s_q)[:,None]==y_sign)&nonzero).astype(float).T @ Bf.astype(float))
            for ti, tname in enumerate(target_cols):
                for bi, bname in enumerate(bet_cols):
                    d=int(denom[ti,bi])
                    if d>0:
                        hit_den[(sname,ql,tname,bname)] += d
                        hit_num[(sname,ql,tname,bname)] += int(numer[ti,bi])

            ldv = np.sum(Bf,axis=0).astype(int)
            lnv = np.sum((sgn_q>0)[:,None]&Bf,axis=0).astype(int)
            for bi,bname in enumerate(bet_cols):
                if ldv[bi]>0:
                    long_den[(sname,ql,bname)] += int(ldv[bi])
                    long_num[(sname,ql,bname)] += int(lnv[bi])

            for bi,bname in enumerate(bet_cols):
                b_ok = Bf[:,bi]
                if not b_ok.any(): continue
                for ti,tname in enumerate(target_cols):
                    y_ok = Yf[:,ti]; m=b_ok&y_ok
                    if not m.any(): continue
                    xs=s_q[m].astype(float); ys=Y_q[m,ti].astype(float)
                    pooled_pairs[(sname,ql,tname,bname)]['s'].append(xs)
                    pooled_pairs[(sname,ql,tname,bname)]['y'].append(ys)
    return daily_ppd, pooled_pairs, hit_num, hit_den, long_num, long_den, daily_effp


def _merge(dest, src):
    dpp, pp, hn, hd, ln, ld, effp = src
    for k,v in dpp.items(): dest[0][k].extend(v)
    for k,d in pp.items():
        dest[1][k]['s'].extend(d['s']); dest[1][k]['y'].extend(d['y'])
    for k,v in hn.items(): dest[2][k]+=v
    for k,v in hd.items(): dest[3][k]+=v
    for k,v in ln.items(): dest[4][k]+=v
    for k,v in ld.items(): dest[5][k]+=v
    for k,v in effp.items(): dest[6][k].extend(v)


# ---- main driver ----
def compute_summary_stats_over_days(df, date_col, signal_cols, target_cols,
                                    quantiles=(1.0,0.75,0.5,0.25),
                                    bet_size_cols=('betsize_equal',)):
    out = create_5d_stats()
    signal_cols=list(signal_cols); target_cols=list(target_cols); bet_size_cols=list(bet_size_cols)
    df=df.copy(); df[date_col]=pd.to_datetime(df[date_col],errors='coerce')
    df=df.dropna(subset=[date_col])
    if df.empty: return out
    for c in signal_cols+target_cols+list(bet_size_cols):
        df[c]=_float_clean(df[c])

    grouped = list(df.sort_values(date_col).groupby(date_col))
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(_summarize_day)(gdf, signal_cols, target_cols, bet_size_cols, quantiles)
        for _, gdf in grouped
    )

    acc = (defaultdict(list),
           defaultdict(lambda:{'s':[],'y':[]}),
           defaultdict(int),defaultdict(int),
           defaultdict(int),defaultdict(int),
           defaultdict(list))
    for part in results: _merge(acc,part)

    daily_ppd, pooled_pairs, hit_num, hit_den, long_num, long_den, daily_effp = acc
    sqrt_252=np.sqrt(252.0)

    for k,v in daily_ppd.items():
        arr=np.asarray(v,float); arr=arr[np.isfinite(arr)]
        sharpe=np.nan
        if arr.size>=2:
            mu,sd=arr.mean(),arr.std(ddof=0)
            sharpe=mu/sd*sqrt_252 if sd>0 else np.nan
        s,ql,t,b=k; out['sharpe'][s][ql][t][b]=sharpe

    for k,v in daily_effp.items():
        arr=np.asarray(v,float); arr=arr[np.isfinite(arr)]
        val=arr.mean() if arr.size>0 else np.nan
        s,ql,t,b=k; out['efficiencyP'][s][ql][t][b]=val

    for k,v in pooled_pairs.items():
        x=np.concatenate([np.asarray(a,float).ravel() for a in v['s']]) if v['s'] else np.array([])
        y=np.concatenate([np.asarray(a,float).ravel() for a in v['y']]) if v['y'] else np.array([])
        n=min(x.size,y.size)
        if n>=3:
            x,y=x[:n],y[:n]
            slope,intercept,r,p,stderr=linregress(x,y)
            r2=r**2 if np.isfinite(r) else np.nan
            tstat=slope/stderr if stderr>0 else np.nan
            sp=spearmanr(x,y,nan_policy='omit').correlation
            dc=_distance_correlation(x,y)
        else: r2=tstat=sp=dc=np.nan
        s,ql,t,b=k
        out['r2'][s][ql][t][b]=r2; out['t_stat'][s][ql][t][b]=tstat
        out['spearman'][s][ql][t][b]=sp; out['dcor'][s][ql][t][b]=dc

    for k,hn in hit_num.items():
        hd=hit_den[k]; s,ql,t,b=k
        out['hit_ratio'][s][ql][t][b]=(hn/hd) if hd>0 else np.nan

    for (s,ql,b),ln in long_num.items():
        ld=long_den[(s,ql,b)]; val=(ln/ld) if ld>0 else np.nan
        for t in target_cols:
            out['long_ratio'][s][ql][t][b]=val
    return out