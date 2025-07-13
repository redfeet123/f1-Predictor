import streamlit as st
import pandas as pd
import plotly.express as px
import fastf1
from fastf1 import Cache
from predictor import load_and_predict

st.set_page_config(page_title="🏎️ F1 Race Predictor", layout="wide")
st.title("🏁 Formula 1 Belgian Grand Prix 2025 Predictor")

# Enable FastF1 cache
Cache.enable_cache("f1_cache")

with st.sidebar:
    st.header("Prediction Settings")
    st.markdown("Predict Belgian GP results based on previous Grand Prix data using live FastF1 data.")
    if st.button("Run Belgian GP Prediction"):
        with st.spinner("Fetching previous GP data and running prediction model..."):
            results, podium, mae = load_and_predict()
            st.session_state.results = results
            st.session_state.podium = podium
            st.session_state.mae = mae

if "results" in st.session_state:
    st.subheader("📊 Predicted Results for Belgian GP")
    st.dataframe(st.session_state.results, use_container_width=True)

    st.subheader("🏆 Predicted Podium")
    st.markdown(f"🥇 **P1**: {st.session_state.podium[0]}")
    st.markdown(f"🥈 **P2**: {st.session_state.podium[1]}")
    st.markdown(f"🥉 **P3**: {st.session_state.podium[2]}")

    st.metric("📏 Model MAE (Mean Absolute Error)", f"{st.session_state.mae:.2f} seconds")

    fig = px.bar(
        st.session_state.results,
        x="Driver",
        y="PredictedRaceTime (s)",
        title="Predicted Belgian GP Race Time by Driver",
        text_auto=True
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("⬅️ Use the sidebar to run the Belgian GP prediction.")
