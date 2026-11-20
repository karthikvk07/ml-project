"""Residual scoring, calibration offsets, thresholds and elapsed-time alarm persistence.
Shared by the reference threshold simulation, the scenario evaluation and the dashboard."""
import numpy as np
from .scaling import rms_score, top5_mean_abs


def calibration_offset(R, mu, t_rel, support, mode, plan, cfg):
    """Per-channel bias b (residual units). Uses only rows inside the calibration window."""
    if mode == "uncal" or (mode == "stability" and plan["fallback"]):
        return np.zeros(R.shape[1]), None
    if mode == "fixed500":
        lo, hi = cfg["calibration"]["fixed_window"]
    elif mode == "stability":
        hi = plan["cal_end"]
        lo = hi - cfg["calibration"]["window_s"]
    else:
        raise ValueError(mode)
    w = (t_rel >= lo) & (t_rel <= hi) & support
    C = R[w] - mu
    with np.errstate(all="ignore"):
        b = np.nanmedian(C, axis=0) if w.any() else np.full(R.shape[1], np.nan)
    b = np.where(np.isfinite(b), b, 0.0)
    return b, (float(lo), float(hi), int(w.sum()))


def standardize(R, mu, sigma, b, support):
    Z = (R - mu - b) / sigma
    Z[~support] = np.nan
    return Z


def scores(Z):
    rms, nch = rms_score(Z)
    return dict(rms=rms, top5=top5_mean_abs(Z), n_channels=nch)


def alarm_stream(t, score, thr, start, persist=10.0, gap=10.0):
    """Active when score > thr has persisted >= `persist` elapsed s (reset on gap > `gap` s or non-exceedance)."""
    t = np.asarray(t, float)
    active = np.zeros(len(t), bool)
    exceed = np.zeros(len(t), bool)
    streak, prev = None, None
    for i in range(len(t)):
        if t[i] < start:
            continue
        if prev is not None and t[i] - prev > gap:
            streak = None
        prev = t[i]
        ex = bool(np.isfinite(score[i]) and score[i] > thr)
        exceed[i] = ex
        if ex:
            if streak is None:
                streak = t[i]
            active[i] = t[i] - streak >= persist
        else:
            streak = None
    return active, exceed


def episodes(t, active, gap=10.0):
    """List of (start, end) for contiguous active rows."""
    eps, s, last = [], None, None
    for ti, a in zip(t, active):
        if a:
            if s is not None and ti - last > gap:
                eps.append((s, last)); s = None
            if s is None:
                s = ti
            last = ti
        elif s is not None:
            eps.append((s, last)); s = None
    if s is not None:
        eps.append((s, last))
    return eps


def time_weighted_fraction(t, mask, sel, cap_dt=10.0):
    t = np.asarray(t, float)
    dt = np.r_[np.diff(t), 0.0]
    dt = np.clip(dt, 0, cap_dt)
    tot = dt[sel].sum()
    return float(dt[sel & mask].sum() / tot) if tot > 0 else np.nan
