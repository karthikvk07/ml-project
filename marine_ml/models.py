"""Four preselected regressors and a multi-target fitting wrapper.

Ridge models (M1, M2, M4) are fitted multi-output with a per-target alpha vector
(sklearn Ridge accepts alpha of shape (n_targets,)); this is mathematically identical
to fitting each target separately with its own alpha, because Ridge solves targets
independently. All scalers/feature maps live inside the Pipeline (fold-internal).
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from .config import load_config

RIDGE_MODELS = ("M1", "M2", "M4")
MODEL_INPUTS = {"M1": "all5", "M2": "load2", "M3": "all5", "M4": "all5"}


class M4Features(TransformerMixin, BaseEstimator):
    """[speed, brake, amb1, amb2, amb3] -> 5 linear + speed^2 + brake^2 + speed*brake (8). No load x ambient."""

    def fit(self, X, y=None):
        if X.shape[1] != 5:
            raise ValueError("M4 expects 5 inputs")
        return self

    def transform(self, X):
        s, b = X[:, 0], X[:, 1]
        return np.column_stack([X, s * s, b * b, s * b])


def make_pipeline(model_id, alpha=1.0):
    if model_id in ("M1", "M2"):
        steps = [("scale_in", StandardScaler()), ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                 ("scale_feat", StandardScaler()), ("ridge", Ridge(alpha=alpha))]
    elif model_id == "M4":
        steps = [("scale_in", StandardScaler()), ("feat", M4Features()),
                 ("scale_feat", StandardScaler()), ("ridge", Ridge(alpha=alpha))]
    elif model_id == "M3":
        return HistGradientBoostingRegressor(**load_config()["hgbr_params"])
    else:
        raise ValueError(model_id)
    return Pipeline(steps)


def n_features(model, X):
    if isinstance(model, Pipeline):
        return model[:-1].transform(X[:2]).shape[1]
    return X.shape[1]


class MultiTargetModel:
    """Fitted model for all targets. groups: list of (target_indices, estimator)."""

    def __init__(self, model_id, n_targets, groups, alphas):
        self.model_id, self.n_targets, self.groups, self.alphas = model_id, n_targets, groups, alphas

    def predict(self, X):
        out = np.full((len(X), self.n_targets), np.nan)
        ok = np.isfinite(X).all(axis=1)
        if ok.any():
            for idx, est in self.groups:
                p = est.predict(X[ok])
                out[np.ix_(np.flatnonzero(ok), idx)] = p.reshape(len(p), -1)
        return out


def _missing_groups(Yok, rows):
    """Group targets by identical y-availability over `rows` (per-target y policy)."""
    groups = {}
    for j in range(Yok.shape[1]):
        key = Yok[rows, j].tobytes()
        groups.setdefault(key, []).append(j)
    return list(groups.values())


def _ridge_alpha_search(model_id, X, Y, t, rows, splits_fn):
    grid = load_config()["alpha_grid"]
    mse = np.zeros((len(grid), Y.shape[1]))
    n_used = 0
    splits = list(splits_fn(rows))
    for itr, iva in splits:
        if len(itr) < 10 or len(iva) == 0:
            continue
        n_used += 1
        for a_i, a in enumerate(grid):
            p = make_pipeline(model_id, a).fit(X[itr], Y[itr]).predict(X[iva]).reshape(len(iva), -1)
            mse[a_i] += np.mean((Y[iva] - p) ** 2, axis=0)
    mse /= max(n_used, 1)
    best = np.empty(Y.shape[1])
    for j in range(Y.shape[1]):
        m = mse[:, j].min()
        best[j] = max(a for a_i, a in enumerate(grid) if mse[a_i, j] == m)   # exact tie -> larger alpha
    return best, mse


def _fit_hgbr_one(X, y):
    return make_pipeline("M3").fit(X, y)


def fit_multi(model_id, X, Y, t, train_idx, splits_fn, n_jobs=1):
    """Fit all targets on train_idx (X complete, y per target). splits_fn(rows) -> inner splits."""
    Xok = np.isfinite(X).all(axis=1)
    Yok = np.isfinite(Y)
    base = train_idx[Xok[train_idx]]
    T = Y.shape[1]
    alphas = np.full(T, np.nan)
    groups = []
    if model_id in RIDGE_MODELS:
        for idx in _missing_groups(Yok, base):
            rows = base[Yok[base][:, idx].all(axis=1)]
            a, _ = _ridge_alpha_search(model_id, X, Y[:, idx], t, rows, splits_fn)
            est = make_pipeline(model_id, a if len(idx) > 1 else float(a[0])).fit(X[rows], Y[np.ix_(rows, idx)])
            alphas[idx] = a
            groups.append((idx, est))
    else:
        rows_j = [base[Yok[base, j]] for j in range(T)]
        ests = Parallel(n_jobs=n_jobs)(delayed(_fit_hgbr_one)(X[r], Y[r, j]) for j, r in enumerate(rows_j))
        groups = [([j], e) for j, e in enumerate(ests)]
    return MultiTargetModel(model_id, T, groups, alphas)
