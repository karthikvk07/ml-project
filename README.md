# ⚓ Marine Engine Health Check (research prototype)

A web dashboard that checks recorded marine diesel engine sensor data and reports whether the engine behaved unusually compared with healthy behaviour.

> **Research prototype.** It was trained on healthy data from **one** test-bench engine (Matsui MU323DGSC). It detects deviations only, and does not classify faults. It is not a certified or real-time diagnostic tool and is not validated for other engines.

## How it works
1. Four regression models learn how 34 sensors of a **healthy** engine depend on speed, load and three ambient/cooling temperatures.
2. For new data, the app predicts the expected healthy value of each sensor and standardizes the differences (residuals).
3. A stability gate waits for the engine to settle, then calibrates to the run.
4. The combined anomaly score raises an alarm when it stays above a healthy-reference threshold (99th percentile) for at least 10 seconds.

## Key research findings (exploratory, 13 test runs)
- The usual metrics are misleading on this dataset: elapsed time alone reaches ROC-AUC = 1.0.
- Without per-run calibration, detectors are in alarm about 99% of healthy time. Calibration removes this (supported, Holm p = 0.047).
- In a placebo-controlled test, the detectors did not separate real fault onsets from fake ones in healthy data.

## Run locally
```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

## Data and licence
Marine Engine Fault Dataset v1.0, A. BahooToroody, O. Bondarenko, M. M. Abaei, Y. Niki, E. Zio (2026), Zenodo, doi:10.5281/zenodo.19857425, CC BY 4.0. A cleaned derivative is included; see `DATA_ATTRIBUTION_AND_LICENSE.md` for the modifications. The dataset creators do not endorse this project.
