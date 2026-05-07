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

from src.data.feedback import get_feedback_for_patient, save_feedback  # noqa: E402
from src.data.narrative_feedback import (  # noqa: E402
    save_narrative_feedback,
    load_few_shot_examples,
    find_similar_narratives,
)
from src.explainability.shap_explainer import explain_patient, format_for_narrative  # noqa: E402
from src.model.evaluate import news2_score  # noqa: E402
from src.model.predict import predict_batch  # noqa: E402
from src.narrative.ollama_client import OllamaClient  # noqa: E402
from src.narrative.transcribe import transcribe_audio, is_whisper_available  # noqa: E402
# pylint: enable=wrong-import-position

st.set_page_config(
    page_title="SepsisAlert",
    page_icon="!",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
# Global theme — Ocean blue (#0284c7) replaces all non-risk reds      #
# ------------------------------------------------------------------ #
st.markdown("""
<style>
/* Primary buttons */
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="baseButton-primary"] {
    background-color: #0284c7 !important;
    border-color:     #0284c7 !important;
    color: white !important;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="baseButton-primary"]:hover {
    background-color: #0369a1 !important;
    border-color:     #0369a1 !important;
}

/* Secondary / default buttons */
div.stButton > button:not([kind="primary"]) {
    border-color: #0284c7 !important;
    color:        #0284c7 !important;
}
div.stButton > button:not([kind="primary"]):hover {
    background-color: #e0f2fe !important;
}

/* Sidebar radio / nav active state */
div[data-testid="stSidebar"] .st-emotion-cache-1rtdyuf,
div[data-testid="stSidebar"] [aria-selected="true"] {
    color: #0284c7 !important;
}

/* st.tabs active underline */
button[data-baseweb="tab"][aria-selected="true"] {
    color:        #0284c7 !important;
    border-color: #0284c7 !important;
}

/* st.selectbox & st.multiselect focus ring */
div[data-baseweb="select"] > div:focus-within {
    border-color: #0284c7 !important;
    box-shadow: 0 0 0 1px #0284c7 !important;
}

/* Progress / spinner accent */
div[data-testid="stProgress"] > div > div > div {
    background-color: #0284c7 !important;
}

/* Streamlit link colour */
a { color: #0284c7 !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# Cached loaders                                                       #
# ------------------------------------------------------------------ #

@st.cache_resource
def load_artifact():
    """Load the trained model artifact from disk."""
    path = Path("models/sepsis_model.pkl")
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

    # Already sampled to 100 active patients in main()
    demo = predictions.copy()
    cohort_cols = [c for c in ["stay_id", "first_careunit", "age", "gender", "intime"]
                   if c in cohort.columns]
    demo = demo.merge(cohort[cohort_cols], on="stay_id", how="left")
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
    table_cols = [c for c in ["stay_id", "first_careunit", "age", "risk_score", "risk_label"]
                  if c in demo.columns]
    display = demo[table_cols].copy()
    display["risk_score"] = display["risk_score"].round(3)
    col_labels = {"stay_id": "Stay ID", "first_careunit": "Care Unit",
                  "age": "Age", "risk_score": "Risk Score", "risk_label": "Risk Level"}
    display.columns = [col_labels[c] for c in table_cols]

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

    # Cache SHAP explanation per patient — compute ALL features so we
    # can show both top and bottom contributors without a second call.
    shap_cache_key = f"shap_explanation_{stay_id}"

    if shap_cache_key not in st.session_state:
        with st.spinner("Computing SHAP explanation..."):
            try:  # pylint: disable=broad-exception-caught
                explainer    = load_explainer(artifact, features_df)
                feature_cols = artifact["feature_cols"]
                fv           = feature_row[feature_cols].values[0]
                # top_n = total features → get every SHAP value
                explanation  = explain_patient(
                    explainer, fv, feature_cols, risk_score, stay_id,
                    top_n=len(feature_cols),
                )
                st.session_state[shap_cache_key]      = explanation
                st.session_state["current_explanation"] = explanation
            except Exception as exc:  # pylint: disable=broad-exception-caught
                st.warning(f"SHAP computation failed: {exc}")
                return
    else:
        explanation = st.session_state[shap_cache_key]
        st.session_state["current_explanation"] = explanation

    all_features = explanation.top_features   # sorted high→low by |shap|

    # Top 8 most responsible (highest |shap|) — already first in list
    top8  = all_features[:8]
    # Bottom 8 least responsible (lowest |shap|) — last in list, reversed
    # so the smallest bar is at the bottom of the chart
    bottom8 = list(reversed(all_features[-8:]))

    # ── Colour palette ──────────────────────────────────────────
    BLUE_PRIMARY = "#0284c7"   # Ocean blue for top contributors
    BLUE_LIGHT   = "#7EC8E3"   # pale Ocean for least responsible

    def _make_bar(features, color, title):
        labels    = [f["label"] for f in features]
        shap_vals = [f["shap"]  for f in features]
        fig = go.Figure(go.Bar(
            x=shap_vals,
            y=labels,
            orientation="h",
            marker_color=color,
            text=[f"{v:+.4f}" for v in shap_vals],
            textposition="outside",
        ))
        fig.update_layout(
            xaxis_title="SHAP value (contribution to risk)",
            title=dict(text=title, font=dict(size=13, color="#374151")),
            height=320,
            margin={"t": 40, "b": 10, "l": 10, "r": 60},
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis={
                "zeroline": True,
                "zerolinecolor": "#d1d5db",
                "zerolinewidth": 2,
                "gridcolor": "#f3f4f6",
            },
            font=dict(family="Inter, sans-serif", color="#374151", size=12),
        )
        return fig

    tab_top, tab_bottom = st.tabs(["🔺 Most Responsible", "🔹 Least Responsible"])

    with tab_top:
        st.plotly_chart(
            _make_bar(top8, BLUE_PRIMARY, "Top 8 features driving this risk score"),
            use_container_width=True,
        )

    with tab_bottom:
        st.plotly_chart(
            _make_bar(bottom8, BLUE_LIGHT, "Bottom 8 features — minimal impact on risk"),
            use_container_width=True,
        )


def _get_installed_ollama_models() -> list[str]:
    """Query Ollama API for all installed models. Returns empty list if unavailable."""
    import requests  # pylint: disable=import-outside-toplevel
    try:
        import yaml  # pylint: disable=import-outside-toplevel
        with open("config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        base_url = cfg["narrative"]["ollama_base_url"]
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            return [m["name"] for m in response.json().get("models", [])]
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    return []


def _render_narrative_panel(stay_id):
    """Render the LLM narrative generation panel."""
    st.subheader("Clinical Narrative")
    st.caption("AI-generated explanation for bedside staff")

    explanation = st.session_state.get("current_explanation")

    if not explanation:
        st.info("SHAP explanation will appear here after loading.")
        return

    installed_models = _get_installed_ollama_models()

    st.markdown("""
        <style>
        .narrative-row { display: flex; align-items: center; gap: 12px; }
        .narrative-row select {
            border: 2px solid #0284c7;
            color: #0284c7;
            background: white;
            border-radius: 8px;
            padding: 6px 12px;
            font-size: 14px;
            cursor: pointer;
            outline: none;
        }
        </style>
    """, unsafe_allow_html=True)

    col_btn, col_model, col_rest = st.columns([3, 2, 3])

    with col_btn:
        generate = st.button("Generate Narrative", type="primary", use_container_width=True)

    with col_model:
        if installed_models:
            import yaml  # pylint: disable=import-outside-toplevel
            with open("config.yaml", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            default_model = cfg["narrative"]["ollama_model"]
            default_index = installed_models.index(default_model) if default_model in installed_models else 0
            selected_model = st.selectbox(
                "Model",
                options=installed_models,
                index=default_index,
                label_visibility="collapsed",
                help="Select which locally installed Ollama model to use.",
            )
        else:
            selected_model = None
            st.error("Ollama not running. Start with: `ollama serve`")

    if generate:
        if not installed_models:
            st.error("Ollama not running. Start with: `ollama serve`")
        else:
            with st.spinner(f"Generating with {selected_model}..."):
                try:  # pylint: disable=broad-exception-caught
                    import yaml  # pylint: disable=import-outside-toplevel
                    with open("config.yaml", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f)
                    cfg["narrative"]["ollama_model"] = selected_model

                    # Build patient context from two complementary sources:
                    #   Option 1 — Few-shot: highest-rated examples (style anchor)
                    #   Option 4 — RAG: most clinically similar patient (content anchor)
                    current_vec = {
                        f["label"]: f["shap"]
                        for f in explanation.top_features
                    }

                    context_parts = []

                    # Option 1 — Few-shot: top-rated examples regardless of similarity
                    few_shot = load_few_shot_examples(
                        min_rating=4,
                        max_examples=2,
                        model_used=selected_model,
                    )
                    if few_shot:
                        fs_text = "\n\n".join(
                            f"[Example rated {ex['rating']}/5]\n{ex['narrative_text']}"
                            for ex in few_shot
                        )
                        context_parts.append(
                            "Use the style and structure of these highly-rated "
                            "past narratives as a reference:\n\n" + fs_text
                        )

                    # Option 4 — RAG: most clinically similar patient
                    similar = find_similar_narratives(
                        current_shap_vector=current_vec,
                        top_n=1,
                        min_rating=4,
                        model_used=selected_model,
                    )
                    if similar:
                        ex = similar[0]
                        context_parts.append(
                            f"[Most clinically similar past patient — "
                            f"rated {ex['rating']}/5, "
                            f"SHAP similarity {ex['similarity']:.2f}]\n\n"
                            f"{ex['narrative_text']}\n\n"
                            "Use a similar level of clinical detail for the "
                            "current patient, but based only on their values."
                        )

                    patient_context = "\n\n---\n\n".join(context_parts)

                    client = OllamaClient(cfg)
                    narrative = client.generate_alert(explanation, patient_context)
                    st.session_state["narrative"] = narrative
                    st.session_state["narrative_model"] = selected_model
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    st.error(f"Narrative error: {exc}")

    if "narrative" in st.session_state:
        model_used = st.session_state.get("narrative_model", "")
        if model_used:
            st.caption(f"Generated with: {model_used}")
        st.info(st.session_state["narrative"])

        # ── Narrative quality feedback ──────────────────────────
        st.divider()
        st.caption("**How useful was this narrative?**")

        rating = st.feedback("stars", key=f"narrative_rating_{stay_id}")

        # Text area key — also used to inject audio transcription
        correction_key = f"narrative_correction_{stay_id}"

        # ── Audio feedback (sits ABOVE text area so flow is natural) ──
        if is_whisper_available():
            st.caption("🎙️ Or record your feedback:")
            audio = st.audio_input(
                "Record",
                key=f"audio_feedback_{stay_id}",
                label_visibility="collapsed",
            )
            if audio is not None:
                import hashlib  # pylint: disable=import-outside-toplevel
                audio_bytes = audio.read()
                audio_hash  = hashlib.md5(audio_bytes).hexdigest()
                processed_key = f"audio_processed_{stay_id}"

                # Only transcribe if this is a NEW recording we haven't seen yet
                if st.session_state.get(processed_key) != audio_hash:
                    with st.spinner("Transcribing audio…"):
                        import io  # pylint: disable=import-outside-toplevel
                        transcription = transcribe_audio(io.BytesIO(audio_bytes))
                    if transcription.startswith("[Error]"):
                        st.warning(transcription)
                    else:
                        st.session_state[correction_key] = transcription
                        st.session_state[processed_key]  = audio_hash
                        st.toast("Audio transcribed — review and submit below.", icon="🎙️")
                        st.rerun()
        else:
            st.caption(
                "🎙️ Audio feedback unavailable — "
                "run `pip install openai-whisper && brew install ffmpeg` to enable."
            )

        correction = st.text_area(
            "What was wrong or missing? (optional)",
            placeholder="e.g. lactate trend was misinterpreted, severity overstated…",
            height=80,
            key=correction_key,
        )

        if st.button("Submit feedback", key=f"submit_nfb_{stay_id}"):
            if rating is None:
                st.warning("Please select a star rating before submitting.")
            else:
                # st.feedback returns 0–4, convert to 1–5
                score = int(rating) + 1
                shap_vec = {
                    f["label"]: f["shap"]
                    for f in explanation.top_features
                }
                save_narrative_feedback(
                    stay_id=int(stay_id),
                    rating=score,
                    correction_note=correction or "",
                    narrative_text=st.session_state["narrative"],
                    shap_summary=format_for_narrative(explanation),
                    model_used=model_used or "",
                    shap_vector=shap_vec,
                )
                st.toast(f"Feedback saved — thank you! ({score}/5 ⭐)", icon="✅")

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
        # ── Clinician feedback buttons ──────────────────────────
        existing = get_feedback_for_patient(stay_id_int)
        if existing:
            ftype = existing["feedback_type"]
            if ftype == "confirmed_sepsis":
                st.success("✅ Labelled: Sepsis Confirmed")
            else:
                st.warning("🚩 Labelled: Flagged as Wrong Alert")

        btn_confirm, btn_flag = st.columns(2)
        with btn_confirm:
            if st.button("✅ Confirm Sepsis", type="primary", use_container_width=True):
                save_feedback(stay_id_int, "confirmed_sepsis", risk_score)
                st.toast("Saved: Sepsis confirmed ✅", icon="✅")
                st.rerun()
        with btn_flag:
            if st.button("🚩 Flag as Wrong", use_container_width=True):
                save_feedback(stay_id_int, "flagged_wrong", risk_score)
                st.toast("Saved: Alert flagged as incorrect 🚩", icon="🚩")
                st.rerun()

        # Risk gauge — bar always Ocean blue; background zones show risk regions
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score * 100,
            number={"suffix": "%", "font": {"size": 32}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0284c7"},
                "steps": [
                    {"range": [0, 40],  "color": "#d5f5e3"},
                    {"range": [40, 60], "color": "#fef9e7"},
                    {"range": [60, 100], "color": "#e8f4fd"},
                ],
                "threshold": {"line": {"color": "#64748b", "width": 4}, "value": 60},
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
        _render_narrative_panel(stay_id)

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
                                         line={"color": "#0284c7", "width": 2}))
                fig.add_trace(go.Scatter(x=fpr2, y=tpr2, name="NEWS2 (0.614)",
                                         line={"color": "#94a3b8", "width": 2, "dash": "dash"}))
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
            marker_color=["#0284c7", "#94a3b8"],
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

    # Sample the 100 "active" patients once here so sidebar and
    # Live Monitor always show the same counts.
    active_patients = None
    if predictions is not None:
        active_patients = predictions.sample(100, random_state=99).copy()

    page = render_sidebar(active_patients)

    if page == "Live Monitor":
        render_live_monitor(active_patients, cohort)
    elif page == "Patient Detail":
        render_patient_detail(predictions, cohort, artifact, features_df)
    elif page == "Model Performance":
        render_model_performance(predictions)


if __name__ == "__main__":
    main()
