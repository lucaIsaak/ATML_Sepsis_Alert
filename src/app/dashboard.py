"""
SepsisAlert Streamlit Dashboard — ICU Early Warning System

Skeleton layout. Full implementation to be added after backend is complete.

Pages:
  1. Live Monitor    — active patients, risk scores, alert triage
  2. Patient Detail  — SHAP waterfall chart + narrative for one patient
  3. Model Stats     — AUROC, NEWS2 comparison, feature importances
"""

import streamlit as st

st.set_page_config(
    page_title="SepsisAlert — ICU Early Warning",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.sidebar.title("SepsisAlert")
    st.sidebar.markdown("*Early ICU Sepsis Detection*")
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation",
        ["Live Monitor", "Patient Detail", "Model Performance"],
    )

    if page == "Live Monitor":
        render_live_monitor()
    elif page == "Patient Detail":
        render_patient_detail()
    elif page == "Model Performance":
        render_model_performance()


def render_live_monitor():
    st.title("ICU Live Monitor")
    st.info("Backend not yet connected. Run the data pipeline and model training first.")

    # Placeholder layout
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Patients", "—")
    col2.metric("High Risk", "—", delta=None)
    col3.metric("Unacknowledged Alerts", "—")
    col4.metric("Model Status", "—")

    st.divider()
    st.subheader("Patient Risk Table")
    st.caption("Patients sorted by risk score (highest first). Click a row to see details.")
    # TODO: render actual patient table from agent.get_alert_log()


def render_patient_detail():
    st.title("Patient Detail")

    stay_id = st.text_input("Stay ID", placeholder="Enter stay_id")

    if stay_id:
        st.subheader(f"Risk Assessment — Stay {stay_id}")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("Risk Score", "—")
            st.metric("Risk Level", "—")
            st.caption("Alert generated: —")

        with col2:
            st.subheader("Clinical Narrative")
            st.info("Narrative will appear here once backend is connected.")

        st.divider()
        st.subheader("SHAP Feature Contributions")
        st.caption("Which vital signs and lab values drove this risk score?")
        # TODO: render shap.plots.waterfall(explanation)


def render_model_performance():
    st.title("Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("SepsisAlert AUROC", "—")
    col2.metric("NEWS2 AUROC", "—")
    col3.metric("Alert Fatigue Reduction", "—")

    st.divider()
    st.subheader("ROC Curve — SepsisAlert vs NEWS2")
    st.info("Run model evaluation (src/model/evaluate.py) to populate charts.")

    st.subheader("Top Feature Importances")
    # TODO: render feature importance bar chart


if __name__ == "__main__":
    main()
