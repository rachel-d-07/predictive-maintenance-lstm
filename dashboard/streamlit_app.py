"""Streamlit dashboard.   Run:  streamlit run app/streamlit_app.py"""
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # lets us import config.py and src/

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

import config as cfg
from src.data_loader import get_dataset
from src.health import health_report, trend_alert, degradation_profile
from src.predict import RULPredictor, parse_uploaded

st.set_page_config(page_title="Predictive Maintenance (LSTM)", page_icon="🛠️", layout="wide")
COLORS = {"HEALTHY": "#27ae60", "WARNING": "#e67e22", "CRITICAL": "#c0392b"}


# ------------------------------------------------------------------ cached loaders
@st.cache_resource(show_spinner="Loading trained model ...")
def load_predictor():
    return RULPredictor()


@st.cache_data(show_spinner="Loading dataset ...")
def load_data():
    return get_dataset()


@st.cache_data(show_spinner="Scoring all machines ...")
def score_fleet():
    _, test_df, _, _ = load_data()
    predictor = load_predictor()
    rows = []
    for unit, g in test_df.groupby("unit"):
        res = predictor.predict_series(g.sort_values("cycle")).iloc[-1]
        rep = health_report(res["pred_rul"], res["failure_prob"])
        rows.append({"machine": unit, "cycle": int(g["cycle"].max()), "pred_rul": round(res["pred_rul"], 1),
                     "failure_risk_%": round(rep["risk_pct"], 1), "status": rep["status"]})
    return pd.DataFrame(rows).sort_values("failure_risk_%", ascending=False).reset_index(drop=True)


@st.cache_data
def load_profile():
    return degradation_profile(load_data()[0])


try:
    predictor = load_predictor()
except FileNotFoundError:
    st.error("No trained model found yet. In the VS Code terminal run:  `python -m src.train`  then refresh this page.")
    st.stop()

train_df, test_df, rul_test, source = load_data()


# ------------------------------------------------------------------ helpers
def gauge(value, color, key):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"suffix": "%"},
        title={"text": f"Failure risk (next {cfg.FAILURE_HORIZON} cycles)"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 40], "color": "rgba(39,174,96,.25)"},
                         {"range": [40, 70], "color": "rgba(230,126,34,.25)"},
                         {"range": [70, 100], "color": "rgba(192,57,43,.25)"}]}))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=60, b=10))
    st.plotly_chart(fig, width="stretch", key=key)


def banner(status):
    st.markdown(f"<div style='padding:12px 16px;border-radius:8px;background:{COLORS[status]};color:white;"
                f"font-size:1.25rem;font-weight:600'>Machine status: {status}</div>", unsafe_allow_html=True)


def sensor_chart(df, sensors, key):
    if not sensors:
        return
    long = df.melt(id_vars="cycle", value_vars=sensors, var_name="sensor", value_name="value")
    long["sensor"] = long["sensor"].map(lambda s: f"{s} - {cfg.SENSOR_INFO[s]}")
    fig = px.line(long, x="cycle", y="value", facet_row="sensor", height=170 * len(sensors) + 60)
    fig.update_yaxes(matches=None, title=None)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], textangle=0, x=0.01, xanchor="left", y=a.y + 0.045))
    fig.update_layout(margin=dict(t=30))
    st.plotly_chart(fig, width="stretch", key=key)


def rul_chart(res, key, true_col=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["cycle"], y=res["pred_rul"], name="Predicted RUL", line=dict(width=3)))
    if true_col is not None:
        fig.add_trace(go.Scatter(x=res["cycle"], y=true_col, name="Actual RUL", line=dict(dash="dash")))
    fig.add_hline(y=cfg.CRITICAL_RUL, line_color="#c0392b", line_dash="dot", annotation_text="critical")
    fig.add_hline(y=cfg.WARNING_RUL, line_color="#e67e22", line_dash="dot", annotation_text="warning")
    fig.update_layout(title="Remaining Useful Life (cycles)", height=320, margin=dict(t=50, b=20),
                      xaxis_title="Cycle", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch", key=key)


def prob_chart(res, key):
    fig = go.Figure(go.Scatter(x=res["cycle"], y=res["failure_prob"] * 100, fill="tozeroy",
                               line=dict(color="#c0392b"), name="Failure risk %"))
    fig.add_hline(y=cfg.WARNING_PROB * 100, line_color="#e67e22", line_dash="dot")
    fig.add_hline(y=cfg.CRITICAL_PROB * 100, line_color="#c0392b", line_dash="dot")
    fig.update_layout(title=f"Failure probability (next {cfg.FAILURE_HORIZON} cycles, %)", height=320,
                      margin=dict(t=50, b=20), xaxis_title="Cycle", yaxis_range=[0, 100])
    st.plotly_chart(fig, width="stretch", key=key)


def render_monitor(df, res, key, true_now=None, true_curve=None):
    last = res.iloc[-1]
    rep = health_report(last["pred_rul"], last["failure_prob"])
    banner(rep["status"])
    c1, c2, c3 = st.columns([1, 1, 1.4])
    c1.metric("Predicted RUL (cycles)", f"{last['pred_rul']:.0f}",
              delta=None if true_now is None else f"actual: {true_now:.0f}", delta_color="off")
    c2.metric("Health score", f"{rep['health_score']:.0f} / 100")
    with c3:
        gauge(rep["risk_pct"], COLORS[rep["status"]], f"{key}_gauge")
    st.markdown("#### 🔔 Maintenance alerts")
    alerts = list(rep["alerts"])
    t = trend_alert(res["failure_prob"].to_numpy())
    if t:
        alerts.append(t)
    for a in alerts:
        (st.error if "CRITICAL" in a else st.warning if ("WARNING" in a or "⚠️" in a or "📈" in a) else st.success)(a)
    a, b = st.columns(2)
    with a:
        rul_chart(res, f"{key}_rul", true_curve)
    with b:
        prob_chart(res, f"{key}_prob")
    st.markdown("#### 📡 Sensor trends")
    chosen = st.multiselect("Sensors to display", cfg.FEATURES, default=["s2", "s4", "s7", "s11"], key=f"{key}_sensors",
                            format_func=lambda s: f"{s} - {cfg.SENSOR_INFO[s]}")
    sensor_chart(df, chosen, f"{key}_sensor_chart")


# ------------------------------------------------------------------ layout
st.title("🛠️ Predictive Maintenance for Industrial Machinery (LSTM)")
st.sidebar.header("About")
st.sidebar.write(f"**Data source:** {source}")
st.sidebar.write(f"**Window:** last {cfg.WINDOW_SIZE} cycles  \n**Features:** {len(cfg.FEATURES)} sensors")
st.sidebar.write(f"**Alert thresholds**  \nWarning: RUL ≤ {cfg.WARNING_RUL} or risk ≥ {cfg.WARNING_PROB:.0%}  \n"
                 f"Critical: RUL ≤ {cfg.CRITICAL_RUL} or risk ≥ {cfg.CRITICAL_PROB:.0%}")
st.sidebar.caption("RUL is capped at 125 cycles: anything above is shown as 'healthy / far from failure'.")
if "Synthetic" in source:
    st.sidebar.warning("Running on SYNTHETIC demo data. Add real C-MAPSS files to data/raw/ and retrain for real-data results.")

tab1, tab2, tab3, tab4 = st.tabs(["📡 Fleet Monitor", "🎛️ Simulation Mode", "📂 Upload Readings", "📊 Model Performance"])

# ---------------------------------------------------------------- tab 1: fleet monitor
with tab1:
    units = sorted(test_df["unit"].unique())
    c1, c2 = st.columns([1, 3])
    unit = c1.selectbox("Machine ID", units, key="mon_unit")
    g = test_df[test_df["unit"] == unit].sort_values("cycle").reset_index(drop=True)
    lo, hi = int(g["cycle"].min()), int(g["cycle"].max())
    cur = c2.slider("Observe the machine up to cycle (drag back in time)", lo, hi, hi, key="mon_cycle")
    gv = g[g["cycle"] <= cur]
    res = predictor.predict_series(gv)
    true_curve = np.minimum(float(rul_test.loc[unit]) + (hi - res["cycle"].to_numpy()), cfg.RUL_CAP)
    render_monitor(gv, res, "mon", true_now=float(true_curve[-1]), true_curve=true_curve)

    with st.expander("🏭 Whole fleet overview (sorted by failure risk)"):
        fleet = score_fleet()
        counts = fleet["status"].value_counts()
        k1, k2, k3 = st.columns(3)
        k1.metric("Healthy", int(counts.get("HEALTHY", 0)))
        k2.metric("Warning", int(counts.get("WARNING", 0)))
        k3.metric("Critical", int(counts.get("CRITICAL", 0)))
        st.dataframe(fleet, width="stretch", hide_index=True)

# ---------------------------------------------------------------- tab 2: simulation
with tab2:
    st.subheader("Simulation mode: stress a machine and watch the prediction react")
    st.caption("Pick a real machine, then change its sensor readings. Predictions update instantly. "
               "Extreme values outside the range seen in training are extrapolation, so treat them as 'what-if' only.")
    std, delta = load_profile()

    def reset_sliders():
        st.session_state["wear"] = 0.0
        for s in cfg.FEATURES:
            st.session_state[f"off_{s}"] = 0.0

    a, b = st.columns(2)
    sim_unit = a.selectbox("Base machine", sorted(test_df["unit"].unique()), key="sim_unit")
    gs = test_df[test_df["unit"] == sim_unit].sort_values("cycle").reset_index(drop=True)
    lo, hi = int(gs["cycle"].min()), int(gs["cycle"].max())
    sim_cycle = b.slider("Base cycle (end of the 30-cycle window)", lo, hi, hi, key="sim_cycle")

    st.button("↺ Reset all sliders", on_click=reset_sliders)
    wear = st.slider("Wear injection: 0 = as recorded, 1 = typical end-of-life sensor pattern",
                     0.0, 1.5, 0.0, 0.05, key="wear")
    offsets = {}
    with st.expander("Adjust individual sensors (shift in standard deviations)", expanded=True):
        cols = st.columns(4)
        for i, s in enumerate(cfg.FEATURES):
            offsets[s] = cols[i % 4].slider(f"{s} · {cfg.SENSOR_INFO[s]}", -4.0, 4.0, 0.0, 0.25, key=f"off_{s}")
    off_arr = np.array([offsets[s] for s in cfg.FEATURES])

    base_win = gs[gs["cycle"] <= sim_cycle][cfg.FEATURES].tail(cfg.WINDOW_SIZE).to_numpy(float)

    def apply(win, w):
        return win + off_arr * std + w * delta

    sim_win = apply(base_win, wear)
    r0, p0 = predictor.predict_window(base_win)
    r1, p1 = predictor.predict_window(sim_win)
    rep = health_report(r1, p1)

    banner(rep["status"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Original RUL", f"{r0:.0f}")
    m2.metric("Simulated RUL", f"{r1:.0f}", delta=f"{r1 - r0:+.0f}")
    m3.metric("Original risk", f"{p0 * 100:.0f}%")
    m4.metric("Simulated risk", f"{p1 * 100:.0f}%", delta=f"{(p1 - p0) * 100:+.0f} pts", delta_color="inverse")
    for al in rep["alerts"]:
        (st.error if "CRITICAL" in al else st.warning if ("WARNING" in al or "⚠️" in al) else st.success)(al)

    pick = st.selectbox("Compare sensor", cfg.FEATURES, index=1, key="sim_sensor",
                        format_func=lambda s: f"{s} - {cfg.SENSOR_INFO[s]}")
    j = cfg.FEATURES.index(pick)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=base_win[:, j], name="Original", mode="lines+markers"))
    fig.add_trace(go.Scatter(y=sim_win[:, j], name="Simulated", mode="lines+markers", line=dict(dash="dash")))
    fig.update_layout(height=300, xaxis_title="Position in 30-cycle window", margin=dict(t=30))
    st.plotly_chart(fig, width="stretch", key="sim_compare")

    st.markdown("##### ▶ Live degradation replay")
    st.caption("Gradually injects wear (using the sliders above as the starting point) and shows how RUL and risk respond.")
    if st.button("Run wear ramp"):
        ks = np.linspace(0, 1.5, 30)
        rr, pp = predictor.predict_windows(np.stack([apply(base_win, k) for k in ks]))
        holder = st.empty()
        for i in range(len(ks)):
            f = make_subplots(specs=[[{"secondary_y": True}]])
            f.add_trace(go.Scatter(x=ks[:i + 1], y=rr[:i + 1], name="Predicted RUL"), secondary_y=False)
            f.add_trace(go.Scatter(x=ks[:i + 1], y=pp[:i + 1] * 100, name="Failure risk %",
                                   line=dict(color="#c0392b")), secondary_y=True)
            f.update_xaxes(range=[0, 1.5], title="Injected wear")
            f.update_yaxes(range=[0, cfg.RUL_CAP + 5], title="RUL (cycles)", secondary_y=False)
            f.update_yaxes(range=[0, 100], title="Risk %", secondary_y=True)
            f.update_layout(height=340, margin=dict(t=20), legend=dict(orientation="h", y=-0.3))
            holder.plotly_chart(f, width="stretch", key=f"ramp_{i}")
            time.sleep(0.12)

# ---------------------------------------------------------------- tab 3: upload
with tab3:
    st.subheader("Upload new sensor readings")
    st.caption(f"CSV with columns: {', '.join(cfg.FEATURES)} (plus optional `unit`, `cycle`). "
               "A raw 26-column C-MAPSS .txt file also works. At least a few rows are needed; 30+ cycles is best.")
    up = st.file_uploader("Upload CSV / TXT", type=["csv", "txt"])
    samples = sorted(cfg.DATA_SAMPLE.glob("demo_*.csv"))
    sample_name = st.selectbox("...or try a sample file", ["(none)"] + [p.name for p in samples])
    src = up if up is not None else (cfg.DATA_SAMPLE / sample_name if sample_name != "(none)" else None)
    if src is None:
        st.info("Upload a file or choose a sample to see predictions.")
    else:
        try:
            df_up = parse_uploaded(src)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            df_up = None
        if df_up is not None:
            ids = sorted(df_up["unit"].unique())
            uid = st.selectbox("Machine in file", ids) if len(ids) > 1 else ids[0]
            dfu = df_up[df_up["unit"] == uid].reset_index(drop=True)
            res_u = predictor.predict_series(dfu)
            render_monitor(dfu, res_u, "up")
            st.download_button("⬇ Download predictions (CSV)", res_u.to_csv(index=False), "predictions.csv", "text/csv")

# ---------------------------------------------------------------- tab 4: performance
with tab4:
    mp = cfg.REPORTS_DIR / "metrics.json"
    if not mp.exists():
        st.info("Run `python -m src.train` to generate metrics.")
    else:
        m = json.loads(mp.read_text())
        st.caption(f"Computed on {m['n_test_engines']} held-out test engines · data: {m['data_source']}")
        r, b, c = m["rul"], m["baseline_mean_predictor"], m["failure_classification"]
        x = st.columns(4)
        x[0].metric("MAE (cycles)", f"{r['mae']:.2f}", delta=f"baseline {b['mae']:.2f}", delta_color="off")
        x[1].metric("RMSE (cycles)", f"{r['rmse']:.2f}", delta=f"baseline {b['rmse']:.2f}", delta_color="off")
        x[2].metric("R²", f"{r['r2']:.3f}")
        x[3].metric("NASA score", f"{r['nasa_score']:.0f}")
        y = st.columns(4)
        y[0].metric("Precision", f"{c['precision']:.3f}")
        y[1].metric("Recall", f"{c['recall']:.3f}")
        y[2].metric("F1-score", f"{c['f1']:.3f}")
        y[3].metric("Accuracy", f"{c['accuracy']:.3f}")
        st.caption(f"Classification = 'will fail within {c['horizon_cycles']} cycles?' · "
                   f"{c['n_failing_engines']} of {m['n_test_engines']} test engines actually were in that zone.")
        for fname in ["actual_vs_predicted_scatter", "actual_vs_predicted_by_engine", "rul_trajectories",
                      "error_histogram", "confusion_matrix", "training_curves"]:
            p = cfg.FIG_DIR / f"{fname}.png"
            if p.exists():
                st.image(str(p), caption=fname.replace("_", " "))
