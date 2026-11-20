"""Reference folds, strict symmetric purge, inner blocked CV, nested cross-fitting, forward chaining."""
import numpy as np
from .config import load_config


def fold_ids(n_rows):
    cfg = load_config()
    f = np.full(n_rows, -1, dtype=int)
    for k, (a, b) in enumerate(cfg["reference_folds"]["row_index_inclusive"]):
        f[a:b + 1] = k
    if (f < 0).any():
        raise RuntimeError("Rows not covered by any fold")
    return f


def check_fold_times(t):
    cfg = load_config()["reference_folds"]
    starts = [int(t[a]) for a, _ in cfg["row_index_inclusive"]]
    ends = [int(t[b]) for _, b in cfg["row_index_inclusive"]]
    if starts != cfg["start_times_s"] or ends != cfg["end_times_s"]:
        raise RuntimeError(f"Fold time mismatch: {starts} {ends}")


def purge_admit(t, val_start, val_end, purge):
    """Strict: admit iff t < val_start - purge OR t > val_end + purge (actual held-out span)."""
    return (t < val_start - purge) | (t > val_end + purge)


def span(t, idx):
    return float(t[idx].min()), float(t[idx].max())


def outer_split(t, fid, k, purge):
    val = np.flatnonzero(fid == k)
    vs, ve = span(t, val)
    train = np.flatnonzero((fid != k) & purge_admit(t, vs, ve, purge))
    return train, val


def nested_split(t, fid, k, j, purge):
    """Model blind to folds k AND j, validated on fold j (strict purge around fold j)."""
    val = np.flatnonzero(fid == j)
    vs, ve = span(t, val)
    train = np.flatnonzero((fid != k) & (fid != j) & purge_admit(t, vs, ve, purge))
    return train, val


def inner_blocked_splits(train_idx, t, n_blocks, purge):
    order = train_idx[np.argsort(t[train_idx], kind="stable")]
    for blk in np.array_split(order, n_blocks):
        if len(blk) == 0:
            continue
        vs, ve = span(t, blk)
        keep = purge_admit(t[order], vs, ve, purge)
        yield order[keep], blk


def forward_chain_splits(t, fid, purge, n_folds=5):
    for k in range(1, n_folds):
        val = np.flatnonzero(fid == k)
        vs, ve = span(t, val)
        train = np.flatnonzero((fid < k) & purge_admit(t, vs, ve, purge))
        yield k, train, val


def deployment_alpha_splits(idx, t, fid, purge):
    """Blocked CV over the 5 reference folds for the final (deployment) alpha choice."""
    for k in range(5):
        val = idx[fid[idx] == k]
        vs, ve = span(t, val)
        yield idx[(fid[idx] != k) & purge_admit(t[idx], vs, ve, purge)], val
