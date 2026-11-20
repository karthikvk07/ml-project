"""Target manifest with hardened validation (fixes Friday audit P0-A).

Every target must belong to the authorized 38 raw-sensor inventory; derived-performance,
exogenous, time, label or unknown columns are always rejected. Provenance, adopter and a
parseable UTC timestamp are required on every row.
"""
from datetime import datetime
import pandas as pd
from .config import load_config, path

REQUIRED_COLUMNS = ["target", "approval_status", "provenance", "adopted_by", "adopted_utc"]
ALLOWED_STATUS = {"WORKING_C34A", "APPROVED"}


class ManifestError(ValueError):
    pass


def raw_inventory():
    cfg = load_config()
    return pd.read_csv(path(cfg["targets"]["raw_inventory"]))["raw_sensor"].tolist()


def validate_manifest_frame(df, reference_columns=None):
    cfg = load_config()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ManifestError(f"Manifest missing columns {missing}")
    targets = df["target"].astype(str).tolist()
    if len(targets) != cfg["targets"]["required_count"]:
        raise ManifestError(f"Need exactly {cfg['targets']['required_count']} targets, got {len(targets)}")
    if len(set(targets)) != len(targets):
        raise ManifestError("Duplicate target names")
    inv = set(raw_inventory())
    forbidden = set(cfg["forbidden_predictors"]) | set(cfg["exogenous_inputs"])
    for t in targets:
        if t in forbidden:
            raise ManifestError(f"'{t}' is exogenous/forbidden and cannot be a target")
        if t not in inv:
            raise ManifestError(f"'{t}' is not in the authorized raw-sensor inventory (derived or unknown column)")
        if reference_columns is not None and t not in reference_columns:
            raise ManifestError(f"'{t}' absent from Reference_Data.csv")
    bad_status = set(df["approval_status"]) - ALLOWED_STATUS
    if bad_status:
        raise ManifestError(f"Unusable approval status {bad_status}")
    for col in ("provenance", "adopted_by", "adopted_utc"):
        if df[col].isna().any() or (df[col].astype(str).str.strip() == "").any():
            raise ManifestError(f"Blank '{col}' in manifest")
    for ts in df["adopted_utc"].astype(str):
        try:
            datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError as e:
            raise ManifestError(f"Unparseable UTC timestamp '{ts}'") from e
    return targets


def load_targets(manifest_path=None, reference_columns=None):
    cfg = load_config()
    p = path(manifest_path or cfg["targets"]["manifest"])
    return validate_manifest_frame(pd.read_csv(p), reference_columns)
