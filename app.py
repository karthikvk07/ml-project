"""Marine Engine Health Check — research prototype dashboard (designed for non-specialist users).

Run:  streamlit run app.py
"""
import io
import json
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from marine_ml.config import MODEL_IDS, MODEL_NAMES, RESULTS_DIR
from marine_ml.data import read_user_csv, scenario_members, load_scenario, validate_columns, family_of
from marine_ml.inference import Detector, ArtifactError

st.set_page_config(page_title="Marine Engine Health Check", page_icon="⚓", layout="wide")

# ----------------------------------------------------------------------------- look & feel
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.status-card {border-radius: 12px; padding: 1.1rem 1.4rem; margin: .4rem 0 1rem 0;
              border-left: 8px solid; background: rgba(127,127,127,.08);}
.status-card h2 {margin: 0 0 .3rem 0; font-size: 1.6rem;}
.status-card p {margin: 0; font-size: 1.05rem; line-height: 1.5;}
.st-red   {border-color: #d64545; background: rgba(214,69,69,.10);}
.st-green {border-color: #2e9e5b; background: rgba(46,158,91,.10);}
.st-amber {border-color: #e0a100; background: rgba(224,161,0,.10);}
.step {font-weight: 600; font-size: 1.05rem; margin-top: .4rem;}
.small-note {opacity: .75; font-size: .9rem;}
</style>""", unsafe_allow_html=True)

MODEL_LABELS = {
    "M4": "Recommended — balanced model (M4)",
    "M1": "Detailed model (M1)",
    "M2": "Simple model, engine load only (M2)",
    "M3": "Machine-learning trees (M3)",
}
MODE_LABELS = {
    "stability": "Automatic — wait until the engine is stable (recommended)",
    "fixed500": "Use the first 8 minutes as 'normal'",
    "uncal": "No adjustment (not recommended)",
}
LOAD_WORDS = {"40": "40% load", "60": "60% load", "75": "75% load", "85": "85% load"}


def friendly_run(member):
    name = member.split("/")[-1].replace(".csv", "")
    fam = family_of(member)
    if "Nozzle1" in name:
        return f"{fam} · one blocked hole · 40/60/85% load"
    if "Nozzle2" in name:
        return f"{fam} · two blocked holes · changing load"
    load = next((v for k, v in LOAD_WORDS.items() if f"_{k}_" in name), "")
    return f"{fam} · {load}"


def fmt_time(s):
    if s is None or not np.isfinite(s):
        return "—"
    s = int(round(s))
    h, m = divmod(s // 60, 60)
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min {s % 60:02d} s"
    return f"{s} s"


@st.cache_resource(show_spinner="Loading the trained engine models…")
def get_detector():
    return Detector()


@st.cache_data(show_spinner=False)
def bundled_runs():
    return scenario_members()


@st.cache_data(show_spinner="Reading your file…")
def parse_csv(raw: bytes):
    return read_user_csv(raw)


def load_bundled(member):
    df = load_scenario(member)
    buf = io.BytesIO(); df.to_csv(buf, index=False)
    return buf.getvalue()


# ----------------------------------------------------------------------------- header
st.title("⚓ Marine Engine Health Check")
st.markdown("Checks recorded engine sensor data and tells you **whether the engine behaved unusually** compared "
            "with how a healthy engine of this type normally behaves.")

try:
    det = get_detector()
except ArtifactError as e:
    st.error(f"The trained models are missing. Ask your technical contact to run `python scripts/train.py`.\n\n{e}")
    st.stop()

# ----------------------------------------------------------------------------- sidebar: inputs
with st.sidebar:
    st.markdown("## Step 1 · Choose engine data")
    src = st.radio("Where is the data from?", ["An example test run", "My own CSV file"], index=0,
                   help="Example runs come from the research dataset. Your own file must use the same sensor column names.")
    raw, label = None, None
    if src == "My own CSV file":
        up = st.file_uploader("Upload a CSV file", type=["csv"])
        if up is not None:
            raw, label = up.getvalue(), up.name
    else:
        runs = bundled_runs()
        member = st.selectbox("Example test run", runs, index=runs.index("AC_Fouling/AC_Fouling_60_Load.csv"),
                              format_func=friendly_run,
                              help="Each run starts healthy; a fault was then introduced on purpose during the test.")
        raw, label = load_bundled(member), member

    st.markdown("## Step 2 · Check the results")
    st.caption("Results appear on the right. The default settings are recommended.")

    with st.expander("⚙️ Advanced settings (optional)"):
        model_id = st.selectbox("Model", MODEL_IDS, index=3, format_func=MODEL_LABELS.get)
        mode = st.selectbox("How to adjust to each run", list(MODE_LABELS), format_func=MODE_LABELS.get)
        cap = st.selectbox("Maximum wait for a stable engine", [900, 1200], index=0,
                           format_func=lambda c: f"{c // 60} minutes")
        score_type = st.selectbox("Score type", ["rms", "top5"],
                                  format_func=lambda s: {"rms": "All sensors together (recommended)",
                                                         "top5": "Five most unusual sensors"}[s])
        quantile = st.selectbox("Alarm sensitivity", ["Q99", "Q95"],
                                format_func=lambda q: {"Q99": "Normal (fewer false alarms)",
                                                       "Q95": "High (more alarms)"}[q])

    st.divider()
    st.caption("⚠️ Research prototype for one test-bench engine. Not a certified diagnostic tool.")

if raw is None:
    st.info("👈 **Start here:** choose an example test run, or upload your own CSV file, in the left panel.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Choose data**  \nPick an example run or upload a file.")
    c2.markdown("**2. Read the verdict**  \nA coloured box tells you if anything unusual happened.")
    c3.markdown("**3. Explore & download**  \nSee when it happened and which sensors changed.")
    st.stop()

# ----------------------------------------------------------------------------- validate & analyse
try:
    df, t_rel, notes = parse_csv(raw)
except Exception as e:  # noqa: BLE001
    st.error(f"We could not read this file. Please check it is a CSV with a time column.\n\nDetails: {e}")
    st.stop()
val = validate_columns(df)
if val["missing_inputs"]:
    st.error("This file is missing the operating columns the tool needs: **" + ", ".join(val["missing_inputs"]) +
             "**. Please export these sensors and try again.")
    st.stop()
try:
    prep, pred, res, rep = det.analyze(df, t_rel, model_id, mode, cap, score_type, quantile)
except ValueError as e:
    st.error(str(e)); st.stop()
summ = det.summary(prep, res, model_id, mode, cap)

ev_start = res["plan"]["eval_start"]
duration = t_rel[-1]
monitored = max(duration - ev_start, 0)
eps = res["episodes"]
alarm_frac = summ["alarm_time_fraction"] or 0.0

# data-quality assessment (plain words)
issues = []
if res["plan"]["fallback"] and mode == "stability":
    issues.append("the engine never became fully stable at the start, so the tool could not adjust to this run "
                  "— alarms may be **too frequent**")
if summ["load_support_fraction"] < 0.95:
    issues.append(f"{100 * (1 - summ['load_support_fraction']):.0f}% of the time the engine ran at a speed/load the "
                  "model was never trained on — those periods are not checked")
if len(val["missing_targets"]):
    issues.append(f"{len(val['missing_targets'])} of 34 sensors are missing from the file")
if monitored < 600:
    issues.append("the recording is very short")
quality = "Good" if not issues else "Limited"

# ----------------------------------------------------------------------------- verdict
src_name = friendly_run(label) if src == "An example test run" else label
st.markdown(f"#### Results for: {src_name}")
if eps:
    first = eps[0][0]
    longest = max(e[1] - e[0] for e in eps)
    cls, title = ("st-amber", "⚠️ Unusual behaviour detected — but treat with caution") if issues else \
                 ("st-red", "🔴 Unusual engine behaviour detected")
    body = (f"The engine started behaving differently from a healthy engine about <b>{fmt_time(first)}</b> after the "
            f"recording began. Unusual behaviour was present for <b>{100 * alarm_frac:.0f}%</b> of the checked time "
            f"(longest stretch: {fmt_time(longest)}).")
else:
    cls, title = ("st-amber", "🟡 No alarm — but results are limited") if issues else \
                 ("st-green", "🟢 No unusual behaviour detected")
    body = (f"Over the <b>{fmt_time(monitored)}</b> that were checked, the engine stayed within the range "
            f"expected for a healthy engine.")
st.markdown(f'<div class="status-card {cls}"><h2>{title}</h2><p>{body}</p></div>', unsafe_allow_html=True)
if issues:
    st.markdown("**Why caution is needed:** " + "; ".join(issues) + ".")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Recording length", fmt_time(duration))
m2.metric("Time checked", fmt_time(monitored), help=f"The first {fmt_time(ev_start)} are used to learn what 'normal' looks like for this run.")
m3.metric("Alarms raised", len(eps), help="Number of separate periods with unusual behaviour (each lasting at least 10 seconds).")
m4.metric("Data quality", quality)

# which sensors changed most
win = np.zeros(len(t_rel), bool)
for a, b in eps:
    win |= (t_rel >= a) & (t_rel <= b)
basis = "during the alarms" if win.any() else "during the checked period"
if not win.any():
    win = t_rel >= ev_start
with np.errstate(all="ignore"):
    dev = np.nanmean(np.abs(res["Z"][win]), axis=0) if win.any() else np.full(len(det.targets), np.nan)
dev_df = pd.DataFrame({"sensor": det.targets, "deviation": dev}).dropna().sort_values("deviation", ascending=False)

# ----------------------------------------------------------------------------- tabs
t1, t2, t3, t4, t5, t6 = st.tabs(["📈 Engine health over time", "🔍 Which sensors changed", "📊 Compare one sensor",
                                  "📋 Alarm list", "❓ How to read this", "🛠 Technical details"])

with t1:
    st.markdown("The line shows **how unusual the engine looked** at each moment. When it rises above the "
                "red **alarm level** for at least 10 seconds, an alarm is raised (red shading).")
    d = pd.DataFrame({"minutes": t_rel / 60, "score": rep.score})
    thr = res["threshold"]
    ymax = float(np.nanmax([thr * 3, np.nanquantile(d.score, 0.99) if d.score.notna().any() else thr * 3]))
    d["score_shown"] = d.score.clip(upper=ymax)
    layers = []
    layers.append(alt.Chart(pd.DataFrame({"x": [0], "x2": [ev_start / 60], "label": ["Learning 'normal' for this run"]}))
                  .mark_rect(opacity=0.12, color="gray").encode(x="x:Q", x2="x2:Q", tooltip="label:N"))
    if eps:
        layers.append(alt.Chart(pd.DataFrame({"x": [a / 60 for a, _ in eps], "x2": [b / 60 for _, b in eps]}))
                      .mark_rect(opacity=0.22, color="#d64545").encode(x="x:Q", x2="x2:Q"))
    layers.append(alt.Chart(d).mark_line(color="#4C78A8", strokeWidth=1.2).encode(
        x=alt.X("minutes:Q", title="Time since recording started (minutes)"),
        y=alt.Y("score_shown:Q", title="How unusual (higher = more unusual)", scale=alt.Scale(domain=[0, ymax])),
        tooltip=[alt.Tooltip("minutes:Q", format=".1f", title="minute"), alt.Tooltip("score:Q", format=".2f", title="score")]))
    layers.append(alt.Chart(pd.DataFrame({"y": [thr], "t": ["Alarm level"]})).mark_rule(color="#d64545", strokeDash=[6, 4], strokeWidth=2)
                  .encode(y="y:Q", tooltip="t:N"))
    has_lab = "Anomaly State" in df.columns and (df["Anomaly State"] == 1).any() and (df["Anomaly State"] == 0).any()
    if has_lab:
        on = t_rel[np.argmax(df["Anomaly State"].to_numpy() == 1)] / 60
        layers.append(alt.Chart(pd.DataFrame({"x": [on], "t": ["Fault introduced in the test"]}))
                      .mark_rule(color="orange", strokeWidth=2).encode(x="x:Q", tooltip="t:N"))
    st.altair_chart(alt.layer(*layers).properties(height=380).interactive(), width="stretch")
    leg = "Grey area = learning period · Red dashed line = alarm level · Red shading = alarm"
    if has_lab:
        leg += " · Orange line = when the researchers introduced the fault (for comparison only; the tool does not know this)"
    st.caption(leg)

with t2:
    st.markdown(f"Sensors that moved furthest from their expected healthy value **{basis}**. "
                "A value of 1 means a normal amount of variation; higher means more unusual.")
    st.caption("This shows *where* the change appears — it is not a diagnosis of the fault type.")
    top = dev_df.head(10)
    if len(top):
        st.altair_chart(alt.Chart(top).mark_bar(color="#d64545" if eps else "#4C78A8").encode(
            x=alt.X("deviation:Q", title="How far from normal (× typical variation)"),
            y=alt.Y("sensor:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320, labelFontSize=12)),
            tooltip=["sensor", alt.Tooltip("deviation:Q", format=".1f")]).properties(height=320),
            width="stretch")

with t3:
    avail = [c for c in det.targets if c in val["available_targets"]]
    default = dev_df.sensor.iloc[0] if len(dev_df) and dev_df.sensor.iloc[0] in avail else avail[0]
    sensor = st.selectbox("Choose a sensor", avail, index=avail.index(default),
                          help="Pre-selected: the sensor that changed most.")
    j = det.targets.index(sensor)
    ed = pd.DataFrame({"minutes": t_rel / 60, "Measured": prep["Y"][:, j], "Expected if healthy": pred["yhat"][:, j]})
    ed = ed.melt("minutes", var_name="series", value_name="value")
    st.altair_chart(alt.Chart(ed).mark_line(strokeWidth=1.2).encode(
        x=alt.X("minutes:Q", title="Time since recording started (minutes)"),
        y=alt.Y("value:Q", scale=alt.Scale(zero=False), title=sensor),
        color=alt.Color("series:N", title=None, scale=alt.Scale(domain=["Measured", "Expected if healthy"], range=["#4C78A8", "#F58518"]),
                        legend=alt.Legend(orient="top"))).properties(height=360).interactive(), width="stretch")
    st.caption("If the blue (measured) line drifts away from the orange (expected) line, this sensor is behaving unusually.")

with t4:
    if eps:
        rows = []
        for i, (a, b) in enumerate(eps, 1):
            m = (t_rel >= a) & (t_rel <= b)
            with np.errstate(all="ignore"):
                mz = np.nanmean(np.abs(res["Z"][m]), axis=0)
            topi = [k for k in np.argsort(-np.nan_to_num(mz, nan=-1))[:3] if np.isfinite(mz[k])]
            rows.append({"Alarm": i, "Started": fmt_time(a), "Ended": fmt_time(b), "Lasted": fmt_time(b - a),
                         "Sensors most affected": ", ".join(det.targets[k] for k in topi)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.success("No alarms were raised in this recording.")

with t5:
    st.markdown("""
**What does the tool do?**
It learned how 34 engine sensors (temperatures, pressures, flows) behave on a **healthy** engine at different speeds,
loads and room temperatures. For your data, it predicts what each sensor *should* read and measures how far the real
readings are from that.

**What do the colours mean?**
- 🟢 **Green** — the engine looked normal the whole time.
- 🔴 **Red** — the engine behaved unusually for at least 10 seconds at a time.
- 🟡 **Yellow** — a result is shown, but the data has limitations, so be careful.

**Why is the start of the recording grey?**
Every engine run is slightly different (room temperature, warm-up). The tool first waits for the engine to settle
and uses that period to adjust itself, so it does not raise alarms just because the day was warmer.

**What it cannot do**
- It does **not** say *what* is wrong — only *that* something looks different.
- It was trained on **one** test-bench engine. It is not validated for other engines or ships.
- In testing it often raised alarms **late** (tens of minutes after a slowly developing fault began), and it missed
  some faults entirely. Use it as a research aid, not for safety decisions.
""")

with t6:
    st.markdown(f"**Model:** {MODEL_NAMES[model_id]} · **Adjustment:** {MODE_LABELS[mode]} · "
                f"**Alarm threshold:** {res['threshold']:.2f} ({res['threshold_mode']}, {quantile}, "
                f"{res['threshold_N']:,} reference samples)")
    st.json({"stability lock (s)": res["lock"], "fallback (no adjustment)": res["plan"]["fallback"],
             "calibration window (s)": res["cal_window"], "evaluation start (s)": ev_start,
             "load in trained range (fraction)": round(summ["load_support_fraction"], 3),
             "ambient novelty fraction (strict k-NN criterion)": round(summ["ambient_ood_fraction"], 3),
             "sensor channels available": summ["channels_available"], "file notes": notes})
    if prep["gate"] is not None:
        g = pd.DataFrame({"minutes": prep["gate"]["t"] / 60, "thermal stability index": prep["gate"]["score"]})
        st.altair_chart(alt.Chart(g).mark_line().encode(x="minutes:Q", y="thermal stability index:Q").properties(height=180),
                        width="stretch")
    sj = RESULTS_DIR / "summary.json"
    if sj.exists():
        st.markdown("**Research evaluation (13 test runs, exploratory):**")
        st.dataframe(pd.DataFrame(json.loads(sj.read_text())["hypotheses"]), width="stretch", hide_index=True)
    st.caption("Data: Marine Engine Fault Dataset v1.0, BahooToroody et al., Zenodo 10.5281/zenodo.19857425, CC BY 4.0 (cleaned derivative).")

# ----------------------------------------------------------------------------- downloads
st.divider()
st.markdown("#### 📥 Save the results")
out = rep.copy()
for j, tname in enumerate(det.targets):
    out[f"expected::{tname}"] = pred["yhat"][:, j]
    out[f"z::{tname}"] = res["Z"][:, j]
txt = [f"MARINE ENGINE HEALTH CHECK — {src_name}", "", title.replace("🔴 ", "").replace("🟢 ", "").replace("🟡 ", "").replace("⚠️ ", ""),
       "", f"Recording length: {fmt_time(duration)}", f"Time checked: {fmt_time(monitored)}",
       f"Alarms raised: {len(eps)}"]
txt += [f"  Alarm {i}: {fmt_time(a)} to {fmt_time(b)}" for i, (a, b) in enumerate(eps, 1)]
txt += [f"Data quality: {quality}"] + ([f"Caution: {x}" for x in issues] if issues else [])
txt += ["", "Sensors that changed most " + basis + ":"] + [f"  - {r.sensor} ({r.deviation:.1f}x)" for r in dev_df.head(5).itertuples()]
txt += ["", f"Settings: {MODEL_LABELS[model_id]}; {MODE_LABELS[mode]}; sensitivity {quantile}",
        "", "Research prototype for one test-bench engine; not a certified diagnostic tool."]
d1, d2 = st.columns(2)
d1.download_button("📄 Download a simple summary (text)", "\n".join(txt).replace("**", "").encode(), "engine_health_summary.txt",
                   "text/plain", width="stretch")
d2.download_button("📊 Download full data report (CSV, for experts)", out.to_csv(index=False).encode(), "engine_health_report.csv",
                   "text/csv", width="stretch")
