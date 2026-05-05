"""
SepsisAlert — Streamlit ICU Dashboard.

Pages:
  1. Live Monitor    — patient table with risk scores
  2. Patient Detail  — SHAP chart + LLM narrative
  3. Model Stats     — AUROC, NEWS2 comparison
"""

# ---------------------------------------------------------------------------
# Path setup — must happen before src imports (Streamlit runs from repo root)
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # noqa: E402

# pylint: disable=wrong-import-position
import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.explainability.shap_explainer import explain_patient, format_for_narrative  # noqa: E402
from src.model.evaluate import news2_score  # noqa: E402
from src.model.predict import predict_batch  # noqa: E402
from src.narrative.ollama_client import OllamaClient  # noqa: E402
# pylint: enable=wrong-import-position

st.set_page_config(
    page_title="SepsisAlert",
    page_icon="!",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
# Cached loaders                                                       #
# ------------------------------------------------------------------ #

@st.cache_resource
def load_artifact():
    """Load the trained model artifact from disk."""
    path = Path("models/lightgbm_sepsis.pkl")
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_features():
    """Load the feature matrix from disk."""
    path = Path("data/processed/features.parquet")
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data
def load_cohort():
    """Load the cohort metadata from disk."""
    path = Path("data/processed/cohort.parquet")
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data
def compute_predictions(_artifact, _df):
    """Run model on all patients and return df with risk_score."""
    return predict_batch(_df, _artifact)


@st.cache_resource
def load_explainer(_artifact, _df):
    """Build SHAP explainer with background sample."""
    import shap  # pylint: disable=import-outside-toplevel
    model = _artifact["model"]
    feature_cols = _artifact["feature_cols"]
    background = _df[feature_cols].dropna().sample(min(100, len(_df)), random_state=42)
    return shap.Explainer(model.predict_proba, background)


# ------------------------------------------------------------------ #
# Styling helpers                                                      #
# ------------------------------------------------------------------ #

def risk_color(label):
    """Return a hex colour string for the given risk label."""
    return {"HIGH": "#e74c3c", "MODERATE": "#f39c12", "LOW": "#27ae60"}.get(label, "#95a5a6")


def risk_badge(label):
    """Return an HTML badge element for the given risk label."""
    color = risk_color(label)
    return (
        f'<span style="background:{color};color:white;'
        f'padding:2px 10px;border-radius:10px;font-weight:bold">'
        f'{label}</span>'
    )


# ------------------------------------------------------------------ #
# Sidebar                                                              #
# ------------------------------------------------------------------ #

def render_sidebar(predictions):
    """Render the navigation sidebar and return the selected page name."""
    st.sidebar.markdown("**SepsisAlert**")
    st.sidebar.markdown("**Early ICU Sepsis Detection**")
    st.sidebar.divider()

    page = st.sidebar.radio("Navigation", ["Live Monitor", "Patient Detail", "Model Performance"])

    if predictions is not None:
        n_high = (predictions["risk_label"] == "HIGH").sum()
        n_mod = (predictions["risk_label"] == "MODERATE").sum()
        st.sidebar.divider()
        st.sidebar.markdown("**Current Status**")
        st.sidebar.metric("High Risk", n_high, delta=None)
        st.sidebar.metric("Moderate Risk", n_mod, delta=None)

    st.sidebar.divider()
    st.sidebar.caption("Model: HistGradientBoosting | AUROC 0.895")
    st.sidebar.caption("Narrative: Ollama / mistral:7b")

    return page


# ------------------------------------------------------------------ #
# Page 1 — Live Monitor                                                #
# ------------------------------------------------------------------ #

def render_live_monitor(predictions, cohort):
    """Render the live patient monitor page."""
    st.title("ICU Live Monitor")

    if predictions is None:
        st.error("No predictions available. Run `python run_pipeline.py` first.")
        return

    # Simulate "active" patients — sample 50 for demo
    demo = predictions.sample(50, random_state=99).copy()
    demo = demo.merge(
        cohort[["stay_id", "first_careunit", "age", "gender", "intime"]],
        on="stay_id", how="left"
    )
    demo = demo.sort_values("risk_score", ascending=False).reset_index(drop=True)

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Patients", len(demo))
    col2.metric("High Risk", int((demo["risk_label"] == "HIGH").sum()))
    col3.metric("Moderate Risk", int((demo["risk_label"] == "MODERATE").sum()))
    col4.metric("Model AUROC", "0.895")

    st.divider()

    # Risk distribution bar
    counts = demo["risk_label"].value_counts().reindex(["HIGH", "MODERATE", "LOW"], fill_value=0)
    fig = go.Figure(go.Bar(
        x=counts.index,
        y=counts.values,
        marker_color=[risk_color(lbl) for lbl in counts.index],
        text=counts.values,
        textposition="auto",
    ))
    fig.update_layout(
        title="Risk Distribution — Active Patients",
        xaxis_title="Risk Level", yaxis_title="Count",
        height=250, margin={"t": 40, "b": 20},
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Patient Risk Table")
    st.caption("Sorted by risk score. Select a patient in the sidebar to see details.")

    # Format table
    display = demo[["stay_id", "first_careunit", "age", "risk_score", "risk_label"]].copy()
    display["risk_score"] = display["risk_score"].round(3)
    display.columns = ["Stay ID", "Care Unit", "Age", "Risk Score", "Risk Level"]

    def highlight_risk(row):
        """Apply row background colour based on risk level."""
        if row["Risk Level"] == "HIGH":
            return ["background-color: #fde8e8"] * len(row)
        if row["Risk Level"] == "MODERATE":
            return ["background-color: #fef9e7"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display.style.apply(highlight_risk, axis=1),
        use_container_width=True,
        height=400,
    )

    # Store selected stay_id in session
    selected = st.selectbox(
        "Select patient for detail view:",
        demo["stay_id"].astype(str).tolist(),
        index=0,
    )
    if st.button("View Patient Detail"):
        st.session_state["selected_stay_id"] = selected
        st.session_state["page"] = "Patient Detail"
        st.rerun()


# ------------------------------------------------------------------ #
# Page 2 — Patient Detail                                              #
# ------------------------------------------------------------------ #

def _render_shap_chart(feature_row, artifact, features_df, risk_score, stay_id):
    """Compute and render the SHAP horizontal bar chart for one patient."""
    if feature_row.empty:
        st.info("No feature data for this patient.")
        return

    with st.spinner("Computing SHAP explanation..."):
        try:  # pylint: disable=broad-exception-caught
            explainer = load_explainer(artifact, features_df)
            feature_cols = artifact["feature_cols"]
            fv = feature_row[feature_cols].values[0]
            explanation = explain_patient(
                explainer, fv, feature_cols, risk_score, stay_id, top_n=8
            )

            # Horizontal bar chart
            labels = [f["label"] for f in explanation.top_features]
            shap_vals = [f["shap"] for f in explanation.top_features]
            colors = [risk_color("HIGH") if v > 0 else "#27ae60" for v in shap_vals]

            fig = go.Figure(go.Bar(
                x=shap_vals,
                y=labels,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in shap_vals],
                textposition="outside",
            ))
            fig.update_layout(
                xaxis_title="SHAP value (contribution to risk)",
                height=350,
                margin={"t": 10, "b": 10},
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis={"zeroline": True, "zerolinecolor": "black", "zerolinewidth": 1},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.session_state["current_explanation"] = explanation

        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.warning(f"SHAP computation failed: {exc}")


def _render_narrative_panel():
    """Render the LLM narrative generation panel."""
    st.subheader("Clinical Narrative")
    st.caption("AI-generated explanation for bedside staff (Ollama / mistral:7b)")

    explanation = st.session_state.get("current_explanation")

    if not explanation:
        st.info("SHAP explanation will appear here after loading.")
        return

    if st.button("Generate Narrative", type="primary"):
        with st.spinner("Generating clinical narrative..."):
            try:  # pylint: disable=broad-exception-caught
                client = OllamaClient()
                if not client.is_available():
                    st.error("Ollama not running. Start with: `ollama serve`")
                else:
                    narrative = client.generate_alert(explanation)
                    st.session_state["narrative"] = narrative
            except Exception as exc:  # pylint: disable=broad-exception-caught
                st.error(f"Narrative error: {exc}")

    if "narrative" in st.session_state:
        st.info(st.session_state["narrative"])

    with st.expander("View raw SHAP summary sent to LLM"):
        st.code(format_for_narrative(explanation))


def render_patient_detail(  # pylint: disable=too-many-locals
    predictions, cohort, artifact, features_df
):
    """Render the patient detail page with SHAP chart and narrative."""
    st.title("Patient Detail")

    if predictions is None or artifact is None:
        st.error("Run pipeline first.")
        return

    # Stay ID selector
    stay_options = (
        predictions.sort_values("risk_score", ascending=False)["stay_id"].astype(str).tolist()
    )
    default = st.session_state.get("selected_stay_id", stay_options[0])
    default_idx = stay_options.index(default) if default in stay_options else 0

    stay_id = st.selectbox("Select Stay ID", stay_options, index=default_idx)
    stay_id_int = int(stay_id)

    row = predictions[predictions["stay_id"] == stay_id_int].iloc[0]
    cohort_row = cohort[cohort["stay_id"] == stay_id_int]
    feature_row = features_df[features_df["stay_id"] == stay_id_int]

    risk_score = float(row["risk_score"])
    risk_label = str(row["risk_label"])

    # Header
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric("Risk Score", f"{risk_score:.3f}")
        st.markdown(f"Risk Level: {risk_badge(risk_label)}", unsafe_allow_html=True)
    with col2:
        if not cohort_row.empty:
            cr = cohort_row.iloc[0]
            st.metric("Age", int(cr.get("age", 0)))
            st.metric("Care Unit", str(cr.get("first_careunit", "-"))[:25])
    with col3:
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score * 100,
            number={"suffix": "%", "font": {"size": 32}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color(risk_label)},
                "steps": [
                    {"range": [0, 40], "color": "#d5f5e3"},
                    {"range": [40, 60], "color": "#fef9e7"},
                    {"range": [60, 100], "color": "#fde8e8"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "value": 60},
            },
            title={"text": "Sepsis Risk"},
        ))
        fig.update_layout(height=220, margin={"t": 30, "b": 0, "l": 20, "r": 20})
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # SHAP + Narrative side by side
    col_shap, col_narr = st.columns([1, 1])

    with col_shap:
        st.subheader("Feature Contributions (SHAP)")
        st.caption("Which values drove this risk score?")
        _render_shap_chart(feature_row, artifact, features_df, risk_score, stay_id)

    with col_narr:
        _render_narrative_panel()

    st.divider()

    # Raw feature values table
    with st.expander("View all feature values"):
        if not feature_row.empty:
            feat_df = feature_row[artifact["feature_cols"]].T.reset_index()
            feat_df.columns = ["Feature", "Value"]
            feat_df["Value"] = feat_df["Value"].round(3)
            st.dataframe(feat_df, use_container_width=True)


# ------------------------------------------------------------------ #
# Page 3 — Model Performance                                           #
# ------------------------------------------------------------------ #

def render_model_performance(predictions):  # pylint: disable=too-many-locals
    """Render the model performance and benchmarking page."""
    st.title("Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("SepsisAlert AUROC", "0.895", delta="+0.281 vs NEWS2")
    col2.metric("NEWS2 AUROC", "0.614")
    col3.metric("AUPRC", "0.527")

    st.divider()

    col_roc, col_bar = st.columns(2)

    with col_roc:
        st.subheader("AUROC Comparison")

        if predictions is not None:
            from sklearn.metrics import roc_curve  # pylint: disable=import-outside-toplevel
            features_df = load_features()
            if features_df is not None:
                y_true = predictions["sepsis_label"].values
                y_score = predictions["risk_score"].values

                fpr, tpr, _ = roc_curve(y_true, y_score)
                news2_scores = features_df.apply(news2_score, axis=1).values
                fpr2, tpr2, _ = roc_curve(y_true, news2_scores)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name="SepsisAlert (0.895)",
                                         line={"color": "#2980b9", "width": 2}))
                fig.add_trace(go.Scatter(x=fpr2, y=tpr2, name="NEWS2 (0.614)",
                                         line={"color": "#e74c3c", "width": 2, "dash": "dash"}))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random",
                                         line={"color": "gray", "dash": "dot"}))
                fig.update_layout(
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    height=350,
                    legend={"x": 0.6, "y": 0.1},
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

    with col_bar:
        st.subheader("AUROC by Model")
        fig = go.Figure(go.Bar(
            x=["SepsisAlert\n(This work)", "NEWS2\n(Clinical standard)"],
            y=[0.895, 0.614],
            marker_color=["#2980b9", "#e74c3c"],
            text=["0.895", "0.614"],
            textposition="outside",
        ))
        fig.update_layout(
            yaxis={"range": [0, 1]},
            height=350,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Training Cohort Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total ICU Stays", "93,224")
    col2.metric("Sepsis Cases", "9,890 (10.6%)")
    col3.metric("Features", "43")

    st.subheader("Dataset — MIMIC-IV 3.1")
    st.markdown("""
    | Source | Table | Purpose |
    |--------|-------|---------|
    | ICU | `icustays` | Patient stays anchor |
    | ICU | `chartevents` | Vitals (HR, MAP, SpO2, Temp, RR) |
    | Hospital | `labevents` | Labs (Lactate, WBC, Creatinine, etc.) |
    | Hospital | `diagnoses_icd` | Sepsis-3 labels (ICD-10 A41.x) |
    | Hospital | `patients` | Age, gender |
    """)


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    """Entry point — load data, run agent, and render the selected page."""
    artifact = load_artifact()
    features_df = load_features()
    cohort = load_cohort()

    predictions = None
    if artifact is not None and features_df is not None:
        predictions = compute_predictions(artifact, features_df)
        if cohort is not None:
            predictions = predictions.merge(
                cohort[["stay_id", "sepsis_label"]].rename(columns={"sepsis_label": "_label"}),
                on="stay_id", how="left"
            )
            if "sepsis_label" not in predictions.columns:
                predictions["sepsis_label"] = predictions["_label"]
            predictions = predictions.drop(columns=["_label"], errors="ignore")

    page = render_sidebar(predictions)

    if page == "Live Monitor":
        render_live_monitor(predictions, cohort)
    elif page == "Patient Detail":
        render_patient_detail(predictions, cohort, artifact, features_df)
    elif page == "Model Performance":
        render_model_performance(predictions)


if __name__ == "__main__":
    main()
