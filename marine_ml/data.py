"""Data access. Labels are kept only for post hoc evaluation and never become features."""
import io
import zipfile
import numpy as np
import pandas as pd
from .config import load_config, path, file_sha256

SCENARIO_FAMILY = {
    "AC_Fouling": "Air-cooler fouling", "AF_Clogging": "Air-filter clogging",
    "Injector_Nozzle": "Injector nozzle clogging", "Pump_Cavitation": "CW pump cavitation",
    "Turbine_Degradation": "Turbine degradation",
}


def open_zip(verify=True):
    cfg = load_config()
    p = path(cfg["data"]["zip"])
    if verify and file_sha256(p) != cfg["data"]["zip_sha256"]:
        raise RuntimeError(f"Dataset SHA256 mismatch: {p}")
    return zipfile.ZipFile(p)


def load_reference(verify=True):
    cfg = load_config()
    with open_zip(verify) as z:
        ref = pd.read_csv(z.open(cfg["data"]["reference_member"]))
    if len(ref) != cfg["data"]["reference_rows"]:
        raise RuntimeError("Reference row count mismatch")
    if not np.all(np.diff(ref["Time"].to_numpy(float)) > 0):
        raise RuntimeError("Reference time not strictly increasing")
    return ref


def scenario_members(verify=False):
    with open_zip(verify) as z:
        return sorted(n for n in z.namelist() if n.endswith(".csv") and n != "Reference_Data.csv")


def load_scenario(member, verify=False):
    with open_zip(verify) as z:
        return pd.read_csv(z.open(member))


def family_of(member):
    return SCENARIO_FAMILY.get(member.split("/")[0], "unknown")


def run_name(member):
    return member.split("/")[-1].replace(".csv", "")


def time_column(df):
    for c in ("Time_rel", "Time"):
        if c in df.columns:
            return c
    if "Time_abs" in df.columns:
        return "Time_abs"
    raise ValueError("CSV needs a time column: Time_rel, Time or Time_abs (seconds)")


def read_user_csv(file_or_bytes):
    """Read an uploaded CSV. Accepts the dataset's 1-row header or the original 3-row
    (name / symbol / unit) header. Returns (df, relative_time_seconds, notes)."""
    raw = file_or_bytes.read() if hasattr(file_or_bytes, "read") else file_or_bytes
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    df = pd.read_csv(io.BytesIO(raw))
    notes = []
    tc = time_column(df)
    if pd.to_numeric(df[tc], errors="coerce").iloc[:2].isna().any():   # 3-row header variant
        df = pd.read_csv(io.BytesIO(raw), header=0, skiprows=[1, 2])
        notes.append("Detected 3-row header (name/symbol/unit); symbol and unit rows skipped.")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    t = df[tc].to_numpy(float)
    if not np.all(np.diff(t[np.isfinite(t)]) > 0):
        raise ValueError(f"Time column '{tc}' must be strictly increasing")
    t_rel = t - t[0]
    if tc != "Time_rel":
        notes.append(f"Relative time computed from '{tc}' (first row = 0 s).")
    return df, t_rel, notes


def validate_columns(df):
    """Returns dict with missing exogenous inputs, available/missing targets."""
    from .manifest import load_targets
    cfg = load_config()
    targets = load_targets()
    return dict(
        missing_inputs=[c for c in cfg["exogenous_inputs"] if c not in df.columns],
        available_targets=[c for c in targets if c in df.columns and df[c].notna().any()],
        missing_targets=[c for c in targets if c not in df.columns or df[c].isna().all()],
    )


def design_matrix(df, which="all5"):
    cfg = load_config()
    cols = cfg["exogenous_inputs"] if which == "all5" else cfg["load_inputs"]
    bad = set(cols) & set(cfg["forbidden_predictors"])
    if bad:
        raise ValueError(f"Forbidden predictors requested: {bad}")
    return df[cols].to_numpy(float)
