"""Paired inference: exact conditional sign-flip Wilcoxon (ties/zeros handled exactly),
Hodges-Lehmann, test-inversion CI, rank-biserial, Holm, AR(1) heuristic. (Fixes audit P1-D.)

Zero policy: zero differences are dropped (Wilcoxon 1945 convention) at every tested shift.
Tie policy: midranks; the null distribution is enumerated exactly over all 2^n' sign
vectors using the observed midranks, so p-values are exact conditional on the ties.
"""
import numpy as np
from scipy.stats import rankdata


def _signrank_null_counts(ranks2):
    """ranks2: integer (2*midrank) array. Returns dict sum->count over all sign vectors (exact DP)."""
    total = int(np.sum(ranks2))
    counts = np.zeros(total + 1, dtype=object)
    counts[0] = 1
    for r in ranks2:
        r = int(r)
        new = counts.copy()
        new[r:] = new[r:] + counts[:total + 1 - r]
        counts = new
    return counts


def signrank_exact(d):
    """Two-sided exact conditional p for H0: distribution of d symmetric about 0."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    n = len(d)
    if n == 0:
        return dict(p=1.0, T_plus=0.0, n_nonzero=0, rank_biserial=np.nan)
    ranks = rankdata(np.abs(d))                 # midranks
    r2 = np.round(ranks * 2).astype(int)
    t_obs2 = int(r2[d > 0].sum())
    counts = _signrank_null_counts(r2)
    total2 = int(r2.sum())
    probs = np.array([c for c in counts], dtype=float) / float(2 ** n)
    vals = np.arange(total2 + 1)
    center = total2 / 2.0
    dev = abs(t_obs2 - center)
    p = float(probs[np.abs(vals - center) >= dev - 1e-9].sum())
    t_plus = t_obs2 / 2.0
    t_minus = total2 / 2.0 - t_plus
    return dict(p=min(1.0, p), T_plus=t_plus, n_nonzero=n, rank_biserial=(t_plus - t_minus) / (t_plus + t_minus))


def hodges_lehmann(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2.0))


def signrank_ci(d, conf=0.95):
    """CI = closure of shifts delta not rejected by signrank_exact(d - delta) at 1-conf.
    p only changes at Walsh averages, so it is evaluated between consecutive distinct ones."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return dict(lo=np.nan, hi=np.nan, note="no data")
    i, j = np.triu_indices(len(d))
    W = np.unique((d[i] + d[j]) / 2.0)
    alpha = 1 - conf
    if len(W) == 1:
        return dict(lo=float(W[0]), hi=float(W[0]), note="all Walsh averages identical")
    pts = np.r_[W[0] - 1.0, (W[:-1] + W[1:]) / 2.0, W[-1] + 1.0]
    acc = np.array([signrank_exact(d - x)["p"] > alpha for x in pts])
    if not acc.any():
        return dict(lo=np.nan, hi=np.nan, note="no shift accepted")
    first, last = np.flatnonzero(acc)[[0, -1]]
    lo = -np.inf if first == 0 else float(W[first - 1])
    hi = np.inf if last == len(pts) - 1 else float(W[last])
    return dict(lo=lo, hi=hi, note="exact conditional sign-flip inversion; zeros dropped per shift; midranks")


def paired_test(d, conf=0.95):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    t = signrank_exact(d)
    ci = signrank_ci(d, conf)
    return dict(n=len(d), n_nonzero=t["n_nonzero"], hl=hodges_lehmann(d), median=float(np.median(d)) if len(d) else np.nan,
                ci_lo=ci["lo"], ci_hi=ci["hi"], p=t["p"], rank_biserial=t["rank_biserial"], n_pos=int((d > 0).sum()),
                n_neg=int((d < 0).sum()), n_zero=int((d == 0).sum()))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def ar1_neff(x):
    """Model-dependent UPPER-BOUND heuristic, never a count of independent engines."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3 or np.std(x) == 0:
        return n, np.nan, None, "undefined: n<3 or zero variance"
    xc = x - x.mean()
    rho = float(np.sum(xc[1:] * xc[:-1]) / np.sum(xc * xc))
    if not (0 < rho < 1):
        return n, rho, None, "not applied: rho1 not in (0,1)"
    return n, rho, n * (1 - rho) / (1 + rho), "AR(1) upper-bound heuristic (mixed 1/2 s sampling)"


def nonoverlap_blocks(t, block=300.0):
    t = np.asarray(t, float)
    t = t[np.isfinite(t)]
    return 0 if len(t) == 0 else int(len(np.unique(np.floor((t - t.min()) / block))))
