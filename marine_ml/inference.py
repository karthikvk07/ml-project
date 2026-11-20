"""Load trained artifacts and analyse any CSV with the dataset schema (used by the dashboard,
scripts/predict_csv.py and the evaluation, so all three run identical code)."""
import json
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from .config import load_config, MODELS_DIR, MODEL_IDS
from .data import design_matrix
from .gate import GATE_VARS, run_gate, calibration_plan
from .manifest import load_targets
from .scoring import calibration_offset, standardize, scores, alarm_stream, episodes
from .support import Hull2D, union_contains


class ArtifactError(RuntimeError):
    pass


class Detector:
    def __init__(self, models_dir=MODELS_DIR):
        self.dir = models_dir
        card = models_dir / "model_card.json"
        if not card.exists():
            raise ArtifactError(f"No trained artifacts in {models_dir}. Run: python scripts/train.py")
        self.card = json.loads(card.read_text(encoding="utf-8"))
        self.targets = self.card["targets"]
        if self.targets != load_targets():
            raise ArtifactError("Artifact target list differs from config/targets_C34A.csv")
        sup = joblib.load(models_dir / "support.joblib")
        self.hulls = [Hull2D(vertices=v) for v in sup["hull_vertices"]]
        self.ambient = sup["ambient_ood"]
        self.thresholds = pd.read_csv(models_dir / "thresholds.csv")
        self.cfg = load_config()
        self._models = {}

    def model(self, model_id):
        if model_id not in MODEL_IDS:
            raise ValueError(model_id)
        if model_id not in self._models:
            dep = joblib.load(self.dir / model_id / "deploy.joblib")
            sc = json.loads((self.dir / model_id / "scaling.json").read_text(encoding="utf-8"))
            mu = np.array([sc["full_oof"][t]["mu"] for t in self.targets])
            sg = np.array([sc["full_oof"][t]["sigma"] for t in self.targets])
            self._models[model_id] = (dep, mu, sg)
        return self._models[model_id]

    def threshold(self, model_id, mode, cap, score_type="rms", quantile="Q99", scaling="nested", start="warm"):
        th = self.thresholds
        r = th[(th["model"] == model_id) & (th["mode"] == mode) & (th["cap"] == cap) & (th["score"] == score_type) &
               (th["scaling"] == scaling) & (th["start"] == start)]
        if len(r) != 1:
            raise ArtifactError(f"Threshold not found for {model_id}/{mode}/{cap}/{score_type}/{scaling}/{start}")
        return float(r[quantile].iloc[0]), int(r["N_raw"].iloc[0])

    # ---------- stage 1: model-independent preparation
    def prepare_run(self, df, t_rel, gate_horizon=1200.0):
        missing = [c for c in self.cfg["exogenous_inputs"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required exogenous input columns: {missing}")
        X5 = design_matrix(df, "all5")
        xok = np.isfinite(X5).all(axis=1)
        L = X5[:, :2]
        load_sup = union_contains(self.hulls, L) & xok
        amb_d, amb_flag = self.ambient.flag(df[self.cfg["ambient_inputs"]].to_numpy(float))
        Y = np.column_stack([df[c].to_numpy(float) if c in df.columns else np.full(len(df), np.nan) for c in self.targets])
        gate_ok = all(c in df.columns for c in GATE_VARS)
        gate = run_gate(t_rel, df, horizon=gate_horizon) if gate_ok else None
        return dict(t=np.asarray(t_rel, float), X5=X5, xok=xok, support=load_sup, amb_dist=amb_d, amb_ood=amb_flag,
                    Y=Y, gate=gate, gate_available=gate_ok)

    # ---------- stage 2: model predictions
    def predict(self, prep, model_id):
        dep, mu, sg = self.model(model_id)
        X = prep["X5"][:, :2] if model_id == "M2" else prep["X5"]
        X = X.copy(); X[~prep["xok"]] = np.nan
        yhat = dep.predict(X)
        return dict(model_id=model_id, yhat=yhat, R=prep["Y"] - yhat, mu=mu, sigma=sg)

    # ---------- stage 3: calibrate, score, alarm
    def score_run(self, prep, pred, mode="stability", cap=900, score_type="rms", quantile="Q99",
                  th_scaling="nested", th_start="warm"):
        model_id = pred["model_id"]
        cfg = self.cfg
        lock = prep["gate"]["lock_time"] if prep["gate"] else None
        if mode == "stability" and prep["gate"] is None:
            raise ValueError("Stability-gated calibration needs all 8 gate variables")
        plan = calibration_plan(lock, cap, cfg)
        b, win = calibration_offset(pred["R"], pred["mu"], prep["t"], prep["support"], mode, plan, cfg)
        Z = standardize(pred["R"], pred["mu"], pred["sigma"], b, prep["support"])
        s = scores(Z)
        eff_mode = "uncal" if (mode == "stability" and plan["fallback"]) else mode
        thr, n_thr = self.threshold(model_id, eff_mode, cap, score_type, quantile, th_scaling, th_start)
        active, exceed = alarm_stream(prep["t"], s[score_type], thr, plan["eval_start"],
                                      cfg["alarm"]["persist_s"], cfg["alarm"]["gap_reset_s"])
        return dict(Z=Z, b=b, cal_window=win, plan=plan, lock=lock, score=s[score_type], rms=s["rms"], top5=s["top5"],
                    n_channels=s["n_channels"], threshold=thr, threshold_mode=eff_mode, threshold_N=n_thr,
                    active=active, exceed=exceed, episodes=episodes(prep["t"], active, cfg["alarm"]["gap_reset_s"]))

    def analyze(self, df, t_rel, model_id="M4", mode="stability", cap=900, score_type="rms", quantile="Q99"):
        prep = self.prepare_run(df, t_rel)
        pred = self.predict(prep, model_id)
        res = self.score_run(prep, pred, mode, cap, score_type, quantile)
        return prep, pred, res, self.report_frame(prep, pred, res)

    def report_frame(self, prep, pred, res):
        t = prep["t"]
        rows = pd.DataFrame(dict(t_rel_s=t, score=res["score"], threshold=res["threshold"],
                                 exceeds=res["exceed"], alarm_active=res["active"],
                                 in_evaluation=t >= res["plan"]["eval_start"],
                                 load_in_support=prep["support"], ambient_knn_dist=prep["amb_dist"],
                                 ambient_out_of_reference=prep["amb_ood"], n_channels=res["n_channels"]))
        status = np.where(~rows.in_evaluation, "CALIBRATING/WARM-UP",
                 np.where(~rows.load_in_support, "UNSUPPORTED LOAD (no score)",
                 np.where(rows.alarm_active, "ALARM", np.where(rows.exceeds, "exceedance (not yet persistent)", "normal"))))
        rows["status"] = status
        rows["limitation"] = np.where(rows.ambient_out_of_reference, "ambient outside reference envelope", "") + \
            np.where((rows.n_channels > 0) & (rows.n_channels < 5), "; only 1-4 channels", "")
        return rows

    def summary(self, prep, res, model_id, mode, cap):
        t = prep["t"]
        ev = t >= res["plan"]["eval_start"]
        return dict(model=model_id, calibration_mode=mode, effective_threshold_mode=res["threshold_mode"],
                    cap_s=cap, gate_lock_s=res["lock"], fallback=res["plan"]["fallback"],
                    eval_start_s=res["plan"]["eval_start"], threshold=res["threshold"],
                    threshold_reference_samples=res["threshold_N"],
                    n_alarm_episodes=len(res["episodes"]),
                    first_alarm_s=res["episodes"][0][0] if res["episodes"] else None,
                    alarm_time_fraction=float(res["active"][ev].mean()) if ev.any() else None,
                    load_support_fraction=float(prep["support"].mean()),
                    ambient_ood_fraction=float(prep["amb_ood"].mean()),
                    channels_available=int(np.isfinite(prep["Y"]).any(axis=0).sum()))
