"""
Stress Level Prediction — Streamlit Cloud application.
Dataset : SaYoPillow (Rachakonda et al., IEEE TCE 2021)
"""

import os
import sys
import joblib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import (
    load_and_prepare,
    FEATURE_COLS,
    FEATURE_DISPLAY,
    FEATURE_RANGES,
    LABEL_MAP,
    STRESS_COLORS,
)
from src.dl_model import StressMLP, mlp_predict

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stress Level Prediction",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Stress reduction tips per level
# ---------------------------------------------------------------------------
REDUCTION_TIPS = {
    0: [
        "Maintain your current sleep schedule — consistency is the key driver here.",
        "Continue regular physical activity; even 20 min/day preserves low cortisol.",
        "Stay hydrated and keep caffeine limited to before 2 PM.",
    ],
    1: [
        "Add 10 minutes of diaphragmatic breathing before sleep.",
        "Avoid screen exposure 45 minutes before bed — blue light suppresses melatonin.",
        "Try journaling briefly to offload cognitive load before sleep.",
        "Aim to standardise wake time, even on weekends.",
    ],
    2: [
        "Target 7.5 to 8 hours of sleep for 5 consecutive nights — recovery compounding.",
        "Introduce a pre-sleep routine: dim light, consistent time, no stimulation.",
        "Review workload. Identify the 2 tasks consuming most mental bandwidth.",
        "Consider 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s.",
        "Ensure sleeping environment is under 19 C (66 F) — optimal for deep sleep stages.",
    ],
    3: [
        "Prioritise sleep above all other recovery strategies right now.",
        "Limit alcohol completely — it fragments sleep architecture even in small quantities.",
        "Practice box breathing 3 times daily: 4s in, 4s hold, 4s out, 4s hold.",
        "Consider speaking with a physician or psychologist; self-management alone may be insufficient.",
        "Avoid checking work communications within 1 hour of sleep.",
        "Light aerobic exercise (not HIIT) reduces HPA axis activity — 30 min walk.",
    ],
    4: [
        "Contact a mental health professional or occupational health service today.",
        "Implement strict sleep hygiene: same bed/wake time, cool/dark room, no screens.",
        "Progressive muscle relaxation — tense and release each muscle group before sleep.",
        "Do not use alcohol or sleeping medication as short-term fixes without medical advice.",
        "Reach out to someone you trust. Social connection directly attenuates cortisol.",
        "Consider mindfulness-based stress reduction (MBSR) — evidence-backed 8-week protocol.",
        "If in crisis: contact your country's mental health helpline immediately.",
    ],
}

# ---------------------------------------------------------------------------
# Bootstrap — load data + pre-trained models (fast cold start)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading dataset and models…")
def bootstrap():
    data = load_and_prepare("data/SaYoPillow.csv")

    models = {}
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Load ML models
    for name in ["Random Forest", "XGBoost", "SVM (RBF)", "KNN"]:
        path = os.path.join(model_dir, f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            st.warning(f"Pre-trained model missing: {path}. Falling back to training {name}…")
            from src.ml_models import train_all_ml_models
            m = train_all_ml_models(data["X_train"], data["y_train"])
            models[name] = m[name]
            joblib.dump(models[name], path)

    # Load MLP
    mlp_path = os.path.join(model_dir, "mlp.pt")
    mlp = StressMLP()
    if os.path.exists(mlp_path):
        mlp.load_state_dict(torch.load(mlp_path, map_location="cpu", weights_only=True))
        mlp.eval()
    else:
        st.warning("Pre-trained MLP missing. Training now (one-time)…")
        from src.dl_model import train_mlp
        mlp = train_mlp(data["X_train"], data["y_train"], data["X_test"], data["y_test"])
        torch.save(mlp.state_dict(), mlp_path)
    models["MLP (PyTorch)"] = mlp

    # Cross-validation results (compute once, cache via resource)
    from src.ml_models import evaluate_models_cv
    cv = evaluate_models_cv(data["X_scaled"], data["y"])

    return data, models, cv

try:
    data, all_models, cv_results = bootstrap()
except Exception as e:
    st.error(f"Failed to start app: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align:center; margin-bottom:1.5rem;">
        <h1 style="font-family:'IBM Plex Mono', monospace; font-weight:700; color:#0a0a0a;">
            Stress Level Prediction
        </h1>
        <p style="color:#555; max-width:700px; margin:auto;">
            Five-class physiological stress classifier trained on biosignals captured during sleep.
            Predicts next-day stress level from snoring rate, respiration, body temperature,
            limb movement, blood oxygen, eye movement, sleep duration, and heart rate.
        </p>
        <p style="font-size:0.85rem; color:#888; margin-top:0.5rem;">
            Dataset: SaYoPillow — Rachakonda et al., IEEE TCE 2021
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_predict, tab_compare, tab_explain, tab_data = st.tabs(
    ["Predict", "Model Comparison", "Explainability", "Dataset"]
)

# ===========================================================================
# TAB 1 — Predict
# ===========================================================================
with tab_predict:
    st.markdown("<br>", unsafe_allow_html=True)

    model_names = list(all_models.keys())
    selected_model_name = st.selectbox(
        "Model",
        options=model_names,
        index=0,
        help="Select which trained model to use for inference.",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<p style="font-weight:600;">Sleep Metrics</p>', unsafe_allow_html=True)
        sh_val = st.slider(FEATURE_DISPLAY["sh"], *FEATURE_RANGES["sh"])
        rem_val = st.slider(FEATURE_DISPLAY["rem"], *FEATURE_RANGES["rem"])
        sr_val = st.slider(FEATURE_DISPLAY["sr"], *FEATURE_RANGES["sr"])
        lm_val = st.slider(FEATURE_DISPLAY["lm"], *FEATURE_RANGES["lm"])

    with col_right:
        st.markdown('<p style="font-weight:600;">Physiological Metrics</p>', unsafe_allow_html=True)
        rr_val = st.slider(FEATURE_DISPLAY["rr"], *FEATURE_RANGES["rr"])
        t_val = st.slider(FEATURE_DISPLAY["t"], *FEATURE_RANGES["t"])
        bo_val = st.slider(FEATURE_DISPLAY["bo"], *FEATURE_RANGES["bo"])
        hr_val = st.slider(FEATURE_DISPLAY["hr"], *FEATURE_RANGES["hr"])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Predict Stress Level", type="primary"):
        raw_input = np.array([[sr_val, rr_val, t_val, lm_val, bo_val, rem_val, sh_val, hr_val]])
        scaled_input = data["scaler"].transform(raw_input)

        model = all_models[selected_model_name]

        if selected_model_name == "MLP (PyTorch)":
            preds, proba = mlp_predict(model, scaled_input)
            pred_class = int(preds[0])
            confidence = proba[0]
        else:
            pred_class = int(model.predict(scaled_input)[0])
            confidence = model.predict_proba(scaled_input)[0]

        label = LABEL_MAP[pred_class]
        color = STRESS_COLORS[pred_class]

        st.markdown(
            f"""
            <div style="background:{color}15; border-left:5px solid {color}; padding:1rem 1.5rem; border-radius:6px; margin:1rem 0;">
                <h3 style="margin:0; color:{color};">{label}</h3>
                <p style="margin:0.25rem 0 0 0; color:#444; font-size:0.9rem;">
                    Stress class {pred_class} &nbsp;|&nbsp; Model: {selected_model_name} &nbsp;|&nbsp; Confidence: {confidence[pred_class]*100:.1f}%
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig_conf = go.Figure(
            go.Bar(
                x=[LABEL_MAP[i] for i in range(5)],
                y=[round(c * 100, 2) for c in confidence],
                marker_color=[STRESS_COLORS[i] for i in range(5)],
                text=[f"{c*100:.1f}%" for c in confidence],
                textposition="outside",
            )
        )
        fig_conf.update_layout(
            title=dict(text="Prediction Confidence", font=dict(size=13, family="IBM Plex Mono")),
            yaxis=dict(title="Confidence (%)", range=[0, 115]),
            xaxis=dict(title=""),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            margin=dict(t=40, b=20, l=10, r=10),
            height=320,
            font=dict(family="IBM Plex Sans"),
        )
        st.plotly_chart(fig_conf, use_container_width=True)

        st.markdown('<p style="font-weight:600; margin-top:1rem;">Recommendations</p>', unsafe_allow_html=True)
        for tip in REDUCTION_TIPS[pred_class]:
            st.markdown(f'<p style="margin:0.2rem 0;">• {tip}</p>', unsafe_allow_html=True)

# ===========================================================================
# TAB 2 — Model Comparison
# ===========================================================================
with tab_compare:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="font-weight:600;">Stratified 5-Fold Cross-Validation Results (ML Models)</p>', unsafe_allow_html=True)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    mlp_model = all_models["MLP (PyTorch)"]
    mlp_preds, _ = mlp_predict(mlp_model, data["X_test"])
    mlp_row = {
        "Model": "MLP (PyTorch)",
        "Accuracy Mean": round(accuracy_score(data["y_test"], mlp_preds), 4),
        "Accuracy Std": "—",
        "F1 Macro Mean": round(f1_score(data["y_test"], mlp_preds, average="macro"), 4),
        "F1 Macro Std": "—",
        "Precision Mean": round(precision_score(data["y_test"], mlp_preds, average="macro"), 4),
        "Precision Std": "—",
        "Recall Mean": round(recall_score(data["y_test"], mlp_preds, average="macro"), 4),
        "Recall Std": "—",
    }

    display_df = cv_results.copy()
    mlp_df = pd.DataFrame([mlp_row])
    display_df = pd.concat([display_df, mlp_df], ignore_index=True)
    best_acc_idx = display_df["Accuracy Mean"].idxmax()

    st.dataframe(display_df.set_index("Model"), use_container_width=True)
    st.caption(
        f"Best model by accuracy: {display_df.loc[best_acc_idx, 'Model']} "
        f"({display_df.loc[best_acc_idx, 'Accuracy Mean']:.4f})"
    )

    fig_bar = go.Figure()
    for col, color in [("Accuracy Mean", "#0a0a0a"), ("F1 Macro Mean", "#888888")]:
        vals = pd.to_numeric(display_df[col], errors="coerce")
        fig_bar.add_trace(
            go.Bar(
                name=col.replace(" Mean", ""),
                x=display_df["Model"],
                y=vals,
                marker_color=color,
                text=[f"{v:.3f}" if not np.isnan(v) else "" for v in vals],
                textposition="outside",
            )
        )
    fig_bar.update_layout(
        barmode="group",
        yaxis=dict(range=[0.8, 1.05], title="Score"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend=dict(orientation="h", y=1.12),
        margin=dict(t=30, b=10),
        height=380,
        font=dict(family="IBM Plex Sans", size=12),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    ml_df = display_df[display_df["Model"] != "MLP (PyTorch)"].copy()
    metrics = ["Accuracy Mean", "F1 Macro Mean", "Precision Mean", "Recall Mean"]
    met_disp = ["Accuracy", "F1 Macro", "Precision", "Recall"]

    fig_radar = go.Figure()
    for _, row in ml_df.iterrows():
        vals = [float(row[m]) for m in metrics]
        fig_radar.add_trace(
            go.Scatterpolar(
                r=vals + [vals[0]],
                theta=met_disp + [met_disp[0]],
                name=row["Model"],
                fill="toself",
                opacity=0.4,
            )
        )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.9, 1.0])),
        showlegend=True,
        height=420,
        font=dict(family="IBM Plex Sans"),
        margin=dict(t=30, b=10),
    )
    st.markdown('<p style="font-weight:600;">Radar — ML Models</p>', unsafe_allow_html=True)
    st.plotly_chart(fig_radar, use_container_width=True)

# ===========================================================================
# TAB 3 — Explainability
# ===========================================================================
with tab_explain:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="font-weight:600;">SHAP Feature Importance</p>', unsafe_allow_html=True)

    explain_model_name = st.selectbox(
        "Select model to explain",
        options=list(all_models.keys()),
        key="explain_select",
    )

    if st.button("Compute SHAP Values", key="shap_btn"):
        from src.explainability import compute_shap_importance

        with st.spinner("Computing SHAP values — please wait…"):
            importance = compute_shap_importance(
                explain_model_name,
                all_models[explain_model_name],
                data["X_train"],
                data["X_test"],
            )

        feat_labels = [FEATURE_DISPLAY[f] for f in FEATURE_COLS]
        order = np.argsort(importance)

        fig_shap = go.Figure(
            go.Bar(
                x=importance[order],
                y=[feat_labels[i] for i in order],
                orientation="h",
                marker_color="#0a0a0a",
                text=[f"{v:.4f}" for v in importance[order]],
                textposition="outside",
            )
        )
        fig_shap.update_layout(
            title=dict(
                text=f"Mean |SHAP| — {explain_model_name}",
                font=dict(family="IBM Plex Mono", size=12),
            ),
            xaxis=dict(title="Mean |SHAP value| (averaged across classes)"),
            yaxis=dict(title=""),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            margin=dict(t=40, b=10, l=10, r=80),
            height=380,
            font=dict(family="IBM Plex Sans", size=12),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        st.caption(
            "Feature importance shows which biosignals contribute most to stress prediction. "
            "Eye movement (REM) and blood oxygen consistently rank highest per literature."
        )
    else:
        st.info("Select a model and click Compute SHAP Values.")

# ===========================================================================
# TAB 4 — Dataset
# ===========================================================================
with tab_data:
    st.markdown("<br>", unsafe_allow_html=True)
    df = data["df"]

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Samples", len(df))
    col_s2.metric("Features", len(FEATURE_COLS))
    col_s3.metric("Stress Classes", df["sl"].nunique())

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<p style="font-weight:600;">Class Distribution</p>', unsafe_allow_html=True)
    class_counts = df["sl"].value_counts().sort_index()
    fig_dist = go.Figure(
        go.Bar(
            x=[LABEL_MAP[i] for i in class_counts.index],
            y=class_counts.values,
            marker_color=[STRESS_COLORS[i] for i in class_counts.index],
            text=class_counts.values,
            textposition="outside",
        )
    )
    fig_dist.update_layout(
        yaxis=dict(title="Count"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(t=10, b=10),
        height=300,
        font=dict(family="IBM Plex Sans", size=12),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown('<p style="font-weight:600;">Feature Distribution by Stress Level</p>', unsafe_allow_html=True)
    selected_feature = st.selectbox(
        "Feature",
        options=FEATURE_COLS,
        format_func=lambda x: FEATURE_DISPLAY[x],
        key="dist_feat",
    )

    fig_box = go.Figure()
    for level in sorted(df["sl"].unique()):
        vals = df[df["sl"] == level][selected_feature]
        fig_box.add_trace(
            go.Box(
                y=vals,
                name=LABEL_MAP[level],
                marker_color=STRESS_COLORS[level],
                boxmean=True,
            )
        )
    fig_box.update_layout(
        yaxis=dict(title=FEATURE_DISPLAY[selected_feature]),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(t=10, b=10),
        height=360,
        font=dict(family="IBM Plex Sans", size=12),
        showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown('<p style="font-weight:600;">Pearson Correlation Matrix</p>', unsafe_allow_html=True)
    corr = df[FEATURE_COLS].corr()
    feat_labels = [FEATURE_DISPLAY[f] for f in FEATURE_COLS]

    fig_corr = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=feat_labels,
            y=feat_labels,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            showscale=True,
        )
    )
    fig_corr.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(t=10, b=10),
        height=460,
        font=dict(family="IBM Plex Sans", size=11),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown('<p style="font-weight:600;">Scatter — Feature Pair</p>', unsafe_allow_html=True)
    pair_col1, pair_col2 = st.columns(2)
    with pair_col1:
        feat_x = st.selectbox("X axis", FEATURE_COLS, index=0,
                              format_func=lambda x: FEATURE_DISPLAY[x], key="px")
    with pair_col2:
        feat_y = st.selectbox("Y axis", FEATURE_COLS, index=5,
                              format_func=lambda x: FEATURE_DISPLAY[x], key="py")

    fig_scatter = px.scatter(
        df,
        x=feat_x,
        y=feat_y,
        color=df["sl"].map(LABEL_MAP),
        color_discrete_map={LABEL_MAP[k]: STRESS_COLORS[k] for k in STRESS_COLORS},
        labels={"x": FEATURE_DISPLAY[feat_x], "y": FEATURE_DISPLAY[feat_y], "color": "Stress Level"},
        opacity=0.7,
    )
    fig_scatter.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(t=10, b=10),
        height=380,
        font=dict(family="IBM Plex Sans", size=12),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    with st.expander("Raw data sample (first 50 rows)"):
        st.dataframe(df.head(50), use_container_width=True)
