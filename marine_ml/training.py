"""Reference-only training engine (no scenario data is read here).

Per model:
  * outer 5-fold OOF (strict 300 s purge, inner blocked 4-fold alpha search per target)
  * NESTED cross-fit scaling (primary; fixes audit P0-B): for held-out fold k, residuals used to
    estimate mu_k, sigma_k come from models trained without fold k AND without the residual fold j
  * legacy fold-excluded scaling (sensitivity, original protocol definition)
  * full-OOF scaling (used for scenario/deployment scoring)
  * forward-chaining sensitivity (sigma_FC / sigma_OOF on matched support)
  * deployment refit on all complete-case reference rows, alpha by purged 5-fold blocked CV
Support hulls use the COMMON complete-X purged training rows (fixes audit P1-C).
"""
import time
import numpy as np
from .config import load_config
from .data import design_matrix
from .folds import (fold_ids, check_fold_times, outer_split, nested_split, inner_blocked_splits,
                    forward_chain_splits, deployment_alpha_splits)
from .models import fit_multi, MODEL_INPUTS
from .scaling import params_matrix
from .support import Hull2D


def reference_arrays(ref, targets):
    t = ref["Time"].to_numpy(float)
    check_fold_times(t)
    fid = fold_ids(len(t))
    X5 = design_matrix(ref, "all5")
    xok = np.isfinite(X5).all(axis=1)
    L = design_matrix(ref, "load2")
    Y = ref[targets].to_numpy(float)
    return t, fid, X5, xok, L, Y


def model_inputs(model_id, X5, xok):
    X = X5[:, :2].copy() if MODEL_INPUTS[model_id] == "load2" else X5.copy()
    X[~xok] = np.nan            # identical listwise X policy for every model
    return X


def common_hulls(t, fid, xok, L):
    purge = load_config()["purge_s"]
    hulls, insup = [], np.zeros(len(t), bool)
    for k in range(5):
        tr, va = outer_split(t, fid, k, purge)
        h = Hull2D(L[tr[xok[tr]]])
        hulls.append(h)
        insup[va] = h.contains(L[va]) & xok[va]
    return hulls, insup


def train_model(model_id, ref, targets, n_jobs=1, log=print):
    cfg = load_config()
    purge, floor = cfg["purge_s"], cfg["scale_floor"]
    t, fid, X5, xok, L, Y = reference_arrays(ref, targets)
    X = model_inputs(model_id, X5, xok)
    inner = lambda rows: inner_blocked_splits(rows, t, cfg["inner_cv_blocks"], purge)
    hulls, insup = common_hulls(t, fid, xok, L)
    T = len(targets)
    out = dict(model_id=model_id, targets=list(targets), fit_log=[], fold_models=[])

    # ---- outer OOF
    oof = np.full(Y.shape, np.nan)
    for k in range(5):
        t0 = time.perf_counter()
        tr, va = outer_split(t, fid, k, purge)
        m = fit_multi(model_id, X, Y, t, tr, inner, n_jobs)
        oof[va] = m.predict(X[va])
        out["fold_models"].append(m)
        out["fit_log"].append(dict(model=model_id, stage="outer", fold=k + 1, n_train_rows=int(xok[tr].sum()),
                                   n_val=len(va), n_val_in_support=int(insup[va].sum()),
                                   n_purged=int((fid != k).sum() - len(tr)), seconds=round(time.perf_counter() - t0, 2),
                                   alphas=";".join("nan" if np.isnan(a) else f"{a:g}" for a in m.alphas)))
        log(f"  {model_id} outer fold {k+1} done ({out['fit_log'][-1]['seconds']} s)")
    R = Y - oof

    # ---- nested cross-fit scaling (primary) and legacy fold-excluded (sensitivity)
    nested, legacy = {}, {}
    for k in range(5):
        t0 = time.perf_counter()
        pooled = []
        for j in range(5):
            if j == k:
                continue
            tr, va = nested_split(t, fid, k, j, purge)
            m = fit_multi(model_id, X, Y, t, tr, inner, n_jobs)
            r = Y[va] - m.predict(X[va])
            sup = Hull2D(L[tr[xok[tr]]]).contains(L[va]) & xok[va]
            pooled.append(r[sup])
        Rn = np.vstack(pooled)
        nested[k] = params_matrix(Rn, np.ones(len(Rn), bool), floor)
        legacy[k] = params_matrix(R, (fid != k) & insup, floor)
        out["fit_log"].append(dict(model=model_id, stage="nested", fold=k + 1, n_train_rows=None, n_val=len(Rn),
                                   n_val_in_support=len(Rn), n_purged=None,
                                   seconds=round(time.perf_counter() - t0, 2), alphas=""))
        log(f"  {model_id} nested scaling for fold {k+1} done ({out['fit_log'][-1]['seconds']} s)")
    full = params_matrix(R, insup, floor)

    # ---- forward chaining sensitivity (matched folds/support; fold 5 degenerate by construction)
    fc_rows = []
    for k, tr, va in forward_chain_splits(t, fid, purge):
        m = fit_multi(model_id, X, Y, t, tr, inner, n_jobs)
        r_fc = Y[va] - m.predict(X[va])
        sup = Hull2D(L[tr[xok[tr]]]).contains(L[va]) & insup[va]
        for j, name in enumerate(targets):
            s_fc = params_matrix(r_fc[:, [j]], sup, floor)[1][0]
            s_oof = params_matrix(R[va][:, [j]], sup, floor)[1][0]
            fc_rows.append(dict(model=model_id, target=name, val_fold=k + 1, n_matched=int(sup.sum()),
                                coverage=float(sup.mean()), sigma_FC=s_fc, sigma_OOF=s_oof,
                                ratio=s_fc / s_oof if s_oof else np.nan,
                                note="degenerate: identical training set to outer fold" if k == 4 else ""))

    # ---- deployment refit
    t0 = time.perf_counter()
    allrows = np.flatnonzero(xok)
    dep = fit_multi(model_id, X, Y, t, allrows,
                    lambda rows: deployment_alpha_splits(rows, t, fid, purge), n_jobs)
    out["fit_log"].append(dict(model=model_id, stage="deploy", fold=None, n_train_rows=len(allrows), n_val=None,
                               n_val_in_support=None, n_purged=None, seconds=round(time.perf_counter() - t0, 2),
                               alphas=";".join("nan" if np.isnan(a) else f"{a:g}" for a in dep.alphas)))
    out.update(oof=oof, resid=R, in_support=insup, fold=fid, time=t, nested=nested, legacy=legacy, full=full,
               forward_chain=fc_rows, deploy=dep, hulls=hulls)
    return out
