import streamlit as st

st.set_page_config(
    page_title="EV Broadband Impact",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# HOME PAGE
# ─────────────────────────────────────────────────────────────

st.title("EV Broadband Impact Dashboard", anchor=False)

st.markdown("""

This dashboard evaluates the potential upstream broadband impact created by electric vehicles (EVs). 
It combines EV telemetry data generation modelling, upload-time analysis, and geographic simulations 
to estimate how future EV adoption may influence network demand.

The analysis is divided into three stages:

""")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Deterministic Model", anchor=False)
    st.markdown("""
    Establishes a fixed baseline scenario using user-defined vehicle 
    behaviour, telemetry generation rates, and broadband tiers.
    """)

with col2:
    st.subheader("Monte Carlo Simulation", anchor=False)
    st.markdown("""
    Simulates a range of possible EV usage patterns and data-generation
    behaviours to capture variability and uncertainty.
    """)

with col3:
    st.subheader("State-Level Aggregation", anchor=False)
    st.markdown("""
    Scales individual vehicle results to postcode and state levels using 
    EV adoption projections.
    """)

st.markdown("---")

st.info(
    "Use the sidebar to navigate between sections."
)