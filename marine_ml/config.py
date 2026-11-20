import hashlib
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "system_config.json"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_IDS = ("M1", "M2", "M3", "M4")
MODEL_NAMES = {
    "M1": "M1 Five-input polynomial Ridge",
    "M2": "M2 Load-only polynomial Ridge",
    "M3": "M3 Five-input histogram gradient boosting",
    "M4": "M4 Linear-ambient / quadratic-load Ridge",
}


@lru_cache(maxsize=1)
def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def path(rel):
    return ROOT / rel


def file_sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
