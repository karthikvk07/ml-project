"""Reference-only matched threshold simulation (no scenario data).

For each held-out reference fold k: start a simulated deployment at
t_w = max(fold start, physical session start + 3000 s) (primary; 'unshifted' = fold start as
sensitivity), run the fold-k Stage 1 gate on rows t >= t_w only (no prior history, mimicking a
fresh run), derive cal_end/eval_start exactly as for scenarios, apply each calibration mode,
standardize the fold's OOF residuals with fold-k scaling (nested = primary; legacy = sensitivity),
and pool in-support evaluation scores across folds. Thresholds = Q99 (primary) / Q95.
Stability-mode folds that do not lock contribute nothing (they route to the uncal threshold).
"""
import numpy as np
import pandas as pd
from .config import load_config
from .gate import GATE_VARS, run_gate, fold_params, calibration_plan
from .scoring import calibration_offset, standardize, scores
from .stats import ar1_neff, nonoverlap_blocks


def warm_start(k, variant):
    f = load_config()["reference_folds"]
    start = f["start_times_s"][k]
    if variant == "unshifted":
        return float(start)
    sess = max(s for s in f["session_starts_s"] if s <= start)
    return float(max(start, sess + f["warmup_s"]))


def simulate(ref, tr_out, gate_cache=None):
    cfg = load_config()
    t, fid, R, insup = tr_out["time"], tr_out["fold"], tr_out["resid"], tr_out["in_support"]
    gate_cache = {} if gate_cache is None else gate_cache
    pooled, sim_rows = {}, []
    for k in range(5):
        for variant in ("warm", "unshifted"):
            tw = warm_start(k, variant)
            rows = np.flatnonzero((fid == k) & (t >= tw))
            tau = t[rows] - tw
            key = (k, variant)
            if key not in gate_cache:
                gate_cache[key] = run_gate(tau, ref.iloc[rows][GATE_VARS], par=fold_params(k), horizon=1200.0)["lock_time"]
            lock = gate_cache[key]
            for cap in cfg["gate"]["caps_s"]:
                plan = calibration_plan(lock, cap, cfg)
                ev = tau >= plan["eval_start"]
                sup = insup[rows]
                for scaling in ("nested", "legacy"):
                    mu, sg = tr_out[scaling][k][0], tr_out[scaling][k][1]
                    for mode in cfg["calibration"]["modes"]:
                        if mode == "stability" and plan["fallback"]:
                            sim_rows.append(dict(fold=k + 1, start=variant, t_w=tw, cap=cap, scaling=scaling, mode=mode,
                                                 lock=lock, fallback=True, n_eval=0, cal_window=""))
                            continue
                        b, win = calibration_offset(R[rows], mu, tau, sup, mode, plan, cfg)
                        Z = standardize(R[rows], mu, sg, b, sup)
                        s = scores(Z)
                        sel = ev & np.isfinite(s["rms"])
                        for st in ("rms", "top5"):
                            pooled.setdefault((variant, cap, scaling, mode, st), []).append((s[st][sel], t[rows][sel]))
                        sim_rows.append(dict(fold=k + 1, start=variant, t_w=tw, cap=cap, scaling=scaling, mode=mode,
                                             lock=lock, fallback=plan["fallback"], eval_start_rel=plan["eval_start"],
                                             n_eval=int(sel.sum()), cal_window=str(win)))
    thr = []
    q1, q2 = cfg["thresholds"]["primary_quantile"], cfg["thresholds"]["sensitivity_quantile"]
    for (variant, cap, scaling, mode, st), parts in pooled.items():
        v = np.concatenate([p[0] for p in parts])
        neff = [ar1_neff(p[0])[2] for p in parts]
        thr.append(dict(start=variant, cap=cap, scaling=scaling, mode=mode, score=st,
                        Q99=float(np.quantile(v, q1)) if len(v) else np.nan,
                        Q95=float(np.quantile(v, q2)) if len(v) else np.nan,
                        n_folds=len(parts), N_raw=len(v),
                        N_eff_AR1_upper=float(np.nansum([x for x in neff if x is not None])) if any(x is not None for x in neff) else np.nan,
                        nonoverlap_300s_blocks=int(sum(nonoverlap_blocks(p[1]) for p in parts))))
    return pd.DataFrame(thr), pd.DataFrame(sim_rows)
