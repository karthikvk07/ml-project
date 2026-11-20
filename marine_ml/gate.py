"""Model-free stability gate (decoupled load SD + one-sided MAD L2 thermal index).

Pure-NumPy port of the Stage 1 numba implementation (scripts/gateb_stage1_verified.py),
with identical window, coverage, SD formula, exact all-pairs Theil-Sen slopes and lock rule.
Verified against the Stage 1 scenario lock table in tests/test_gate.py.
"""
from functools import lru_cache
import json
import numpy as np
import pandas as pd
from .config import load_config, path

LOAD = ["Engine Speed", "Water Brake Weight"]
THERM = ["Engine room Temp.", "Charge Air IC Cooling Water Temp. In", "LO Cooling Water Temp. In",
         "Cooling Water Temp. Engine In", "LO Temp. Engine In", "Exh.Gas Temp. Turbine In"]
GATE_VARS = LOAD + THERM
METRICS = THERM + ["abs_slope_" + c for c in THERM]


@lru_cache(maxsize=512)
def _pairs(n):
    return np.triu_indices(n, 1)


def _theil_sen_abs(t, y):
    ok = np.isfinite(y)
    if ok.sum() < 30:
        return np.nan
    t, y = t[ok], y[ok]
    i, j = _pairs(len(t))
    dt = t[j] - t[i]
    keep = dt != 0
    if not keep.any():
        return np.nan
    return abs(float(np.median((y[j][keep] - y[i][keep]) / dt[keep])) * 60.0)


def window_features(t, X, b, cfg_gate):
    """14 features for the trailing window ending at row b: 2 load SDs, 6 thermal SDs, 6 |slopes|."""
    out = np.full(14, np.nan)
    a = int(np.searchsorted(t, t[b] - cfg_gate["window_s"], side="left"))
    n = b - a + 1
    if n < cfg_gate["min_obs"] or t[b] - t[a] < cfg_gate["min_span_s"]:
        return out
    W = X[a:b + 1]
    for j in range(8):
        v = W[:, j]
        f = np.isfinite(v)
        c = int(f.sum())
        if c < cfg_gate["min_obs"] or c / n < cfg_gate["min_nonmissing_frac"]:
            return np.full(14, np.nan)
        sm, sm2 = v[f].sum(), (v[f] * v[f]).sum()
        out[j] = np.sqrt(max(0.0, (sm2 - sm * sm / c) / (c - 1)))
    tw = t[a:b + 1]
    for j in range(6):
        out[8 + j] = _theil_sen_abs(tw, W[:, 2 + j])
    return out


def all_window_features(t, X, endpoints):
    g = load_config()["gate"]
    return np.vstack([window_features(t, X, int(b), g) for b in endpoints]) if len(endpoints) else np.zeros((0, 14))


def deployed_params():
    g = json.loads(path(load_config()["gate"]["deployed_params"]).read_text(encoding="utf-8"))
    order = g["thermal_metric_order"]
    if order != METRICS:
        raise ValueError("Gate metric order mismatch")
    return dict(load=np.array([g["load_sd_q95"][c] for c in LOAD]),
                median=np.array([g["thermal_median"][m] for m in METRICS]),
                scale=np.array([g["thermal_robust_scale"][m] for m in METRICS]),
                thermal_q95=float(g["thermal_l2_q95"]))


def fold_params(k):
    """Stage 1 train_k gate parameters for reference fold k (0-based)."""
    cfg = load_config()["gate"]
    th = pd.read_csv(path(cfg["fold_thermal_params"]))
    th = th[th.Fold == k + 1].set_index("thermal_metric").loc[METRICS]
    v = pd.read_csv(path(cfg["fold_validation"])).set_index("Fold").loc[k + 1]
    return dict(load=np.array([v["Load_Speed_SD_Q95"], v["Load_Brake_SD_Q95"]]),
                median=th["median"].to_numpy(float), scale=th["robust_scale"].to_numpy(float),
                thermal_q95=float(v["Thermal_L2_Q95"]))


def pass_mask(F, par):
    finite = np.isfinite(F).all(axis=1)
    score = np.sqrt(np.sum(np.maximum(0.0, (F[:, 2:] - par["median"]) / par["scale"]) ** 2, axis=1) / 12.0)
    lp = np.all(F[:, :2] <= par["load"], axis=1) & finite
    tp = (score <= par["thermal_q95"]) & finite
    return lp, tp, lp & tp, score


def find_lock(t_rel, passed, persist=60.0, gap=10.0):
    streak, prev = None, None
    for ti, ok in zip(t_rel, passed):
        if prev is not None and ti - prev > gap:
            streak = None
        prev = ti
        if ok:
            if streak is None:
                streak = ti
            if ti - streak >= persist:
                return float(ti)
        else:
            streak = None
    return None


def run_gate(t_rel, df_or_X, par=None, horizon=1200.0):
    """Gate over rows with t_rel <= horizon. Returns dict(lock_time, t, passed, load_pass, thermal_pass, score)."""
    g = load_config()["gate"]
    par = par or deployed_params()
    X = df_or_X[GATE_VARS].to_numpy(float) if hasattr(df_or_X, "columns") else np.asarray(df_or_X, float)
    t_rel = np.asarray(t_rel, float)
    ends = np.flatnonzero(t_rel <= horizon)
    F = all_window_features(t_rel, X, ends)
    lp, tp, comb, score = pass_mask(F, par) if len(F) else (np.array([], bool),) * 3 + (np.array([]),)
    lock = find_lock(t_rel[ends], comb, g["lock_persist_s"], g["gap_reset_s"])
    return dict(lock_time=lock, t=t_rel[ends], passed=comb, load_pass=lp, thermal_pass=tp, score=score, features=F)


def calibration_plan(lock_time, cap, cfg=None):
    cfg = cfg or load_config()
    locked = lock_time is not None and lock_time <= cap
    cal_end = lock_time if locked else float(cap)
    return dict(locked=locked, fallback=not locked, cal_end=cal_end,
                eval_start=max(cfg["calibration"]["min_eval_start_s"], cal_end))
