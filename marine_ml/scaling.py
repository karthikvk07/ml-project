"""Robust residual centering/scaling and anomaly-score aggregation."""
import numpy as np

MAD_K, IQR_K = 1.4826, 1.349


def robust_center_scale(r, floor=1e-6):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return np.nan, np.nan, "empty", 0
    mu = float(np.median(r))
    mad = MAD_K * float(np.median(np.abs(r - mu)))
    if mad > floor:
        return mu, mad, "mad", len(r)
    q75, q25 = np.quantile(r, [0.75, 0.25])
    iqr = float((q75 - q25) / IQR_K)
    if iqr > floor:
        return mu, iqr, "iqr", len(r)
    return mu, floor, "floor", len(r)


def params_matrix(R, mask, floor=1e-6):
    """Column-wise robust params over rows where mask (n,) & finite."""
    mu = np.full(R.shape[1], np.nan); sg = np.full(R.shape[1], np.nan)
    how, n = [], []
    for j in range(R.shape[1]):
        m, s, h, c = robust_center_scale(R[mask, j], floor)
        mu[j], sg[j] = m, s
        how.append(h); n.append(c)
    return mu, sg, how, n


def rms_score(Z):
    Z = np.asarray(Z, float)
    n = np.isfinite(Z).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.sqrt(np.nansum(Z ** 2, axis=1) / np.where(n > 0, n, np.nan))
    return s, n


def top5_mean_abs(Z):
    A = np.abs(np.asarray(Z, float))
    A = np.where(np.isfinite(A), A, -np.inf)
    top = -np.sort(-A, axis=1)[:, :5]
    top = np.where(np.isfinite(top), top, np.nan)
    n = np.isfinite(top).sum(axis=1)
    out = np.where(n > 0, np.nansum(np.where(np.isfinite(top), top, 0.0), axis=1) / np.maximum(n, 1), np.nan)
    return out
