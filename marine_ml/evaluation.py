"""Run-level evaluation metrics (labels used post hoc only)."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from .scoring import time_weighted_fraction


def window_auc(t, score, center, W):
    sel = (t >= center - W) & (t <= center + W) & np.isfinite(score)
    y = t[sel] >= center
    if y.all() or (~y).all():
        return np.nan, np.nan, int(sel.sum())
    return float(roc_auc_score(y, score[sel])), float(average_precision_score(y, score[sel])), int(sel.sum())


def eligible(onset, eval_start, t_end, W):
    return onset is not None and (onset - eval_start >= 3 * W) and (t_end - onset >= W)


def placebo_center(eval_start, onset, W):
    return (eval_start + onset - W) / 2.0


def run_level(t, score, active, episodes, onset, eval_start, labels, W):
    """All run-level metrics for one score stream / alarm stream / window size."""
    t = np.asarray(t, float)
    t_end = float(t[-1])
    out = dict(W=W, eval_start=eval_start, onset=onset)
    ev = t >= eval_start
    fin = np.isfinite(score)
    out["frac_scored_eval"] = float(fin[ev].mean()) if ev.any() else np.nan
    if onset is None:
        out.update(eligible=False)
        out["alarm_time_frac"] = time_weighted_fraction(t, active, ev)
        out["first_alarm_after_eval_s"] = (episodes[0][0] - eval_start) if episodes else np.nan
        out["n_episodes"] = len(episodes)
        return out
    el = eligible(onset, eval_start, t_end, W)
    out["eligible"] = el
    if el:
        c = placebo_center(eval_start, onset, W)
        out["auc_true"], out["pr_true"], out["n_true"] = window_auc(t, score, onset, W)
        out["auc_plac"], out["pr_plac"], out["n_plac"] = window_auc(t, score, c, W)
        out["dAUC"] = out["auc_true"] - out["auc_plac"]
        out["dPR"] = out["pr_true"] - out["pr_plac"]
        out["placebo_center"] = c
        pre_p = (t >= c - W) & (t < c) & fin
        post_p = (t >= c) & (t <= c + W) & fin
        pre_t = (t >= onset - W) & (t < onset) & fin
        post_t = (t >= onset) & (t <= onset + W) & fin
        out["placebo_window_median_score"] = float(np.median(score[pre_p | post_p])) if (pre_p | post_p).any() else np.nan
        out["jump_true"] = float(np.median(score[post_t]) - np.median(score[pre_t])) if pre_t.any() and post_t.any() else np.nan
        out["jump_plac"] = float(np.median(score[post_p]) - np.median(score[pre_p])) if pre_p.any() and post_p.any() else np.nan
    pre = ev & (t < onset)
    out["fa_time_frac_pre"] = time_weighted_fraction(t, active, pre)
    pre_eps = [e for e in episodes if eval_start <= e[0] < onset]
    hours = max((onset - eval_start) / 3600.0, 1e-9)
    out["fa_episodes_pre"] = len(pre_eps)
    out["fa_per_hour"] = len(pre_eps) / hours
    out["alarm_active_at_onset"] = any(e[0] < onset <= e[1] for e in episodes)
    post_eps = [e for e in episodes if e[1] >= onset]
    first = None
    for e in post_eps:
        first = max(e[0], onset)
        break
    out["detected"] = first is not None
    out["detection_delay_s"] = (first - onset) if first is not None else np.nan
    out["detected_within_W"] = first is not None and (first - onset) <= W
    out["post_alarm_time_frac"] = time_weighted_fraction(t, active, t >= onset)
    lab = np.asarray(labels)[ev & fin]
    if lab.min() != lab.max():
        out["fullrun_auc"] = float(roc_auc_score(lab, score[ev & fin]))
        out["fullrun_pr"] = float(average_precision_score(lab, score[ev & fin]))
    return out
