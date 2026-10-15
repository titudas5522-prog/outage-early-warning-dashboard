"""
Script 11: Streamlit Dashboard for Real-Time Outage Risk Monitoring
Run this with: streamlit run 11_dashboard.py

This dashboard loads the trained XGBoost model and lets the user input
current grid readings to get an instant outage risk prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timezone, timedelta

# Bangladesh Standard Time (UTC+6). Cloud servers run on UTC, so we set the offset explicitly.
BD_TIME = timezone(timedelta(hours=6))

# ==== Page Configuration ====
st.set_page_config(
    page_title="Intelligent Early Warning System",
    page_icon="⚡",
    layout="wide"
)

# ==== Load Model and Feature List ====
@st.cache_resource
def load_model():
    model = joblib.load("xgboost_outage_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_model()

# ==== Header ====
st.title("⚡ Intelligent Early Warning System")
st.subheader("Power Outage Risk Prediction Dashboard")
st.markdown("---")

# ==== Sidebar: Input Panel ====
st.sidebar.header("Current Grid & Weather Readings")

voltage = st.sidebar.slider("Voltage (V)", 190.0, 250.0, 220.0, 0.1)
current = st.sidebar.slider("Current (A)", 5.0, 40.0, 20.0, 0.1)
frequency = st.sidebar.slider("Frequency (Hz)", 48.0, 52.0, 50.0, 0.01)
load_percent = st.sidebar.slider("Load (%)", 0.0, 100.0, 60.0, 1.0)
temperature_c = st.sidebar.slider("Temperature (°C)", 10.0, 45.0, 28.0, 0.5)
humidity_percent = st.sidebar.slider("Humidity (%)", 20.0, 100.0, 70.0, 1.0)
wind_speed_ms = st.sidebar.slider("Wind Speed (m/s)", 0.0, 20.0, 3.0, 0.1)
precipitation_mm = st.sidebar.slider("Precipitation (mm)", 0.0, 50.0, 0.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("Recent History (for trend features)")
voltage_lag_1h = st.sidebar.slider("Voltage 1h ago (V)", 190.0, 250.0, 220.0, 0.1)
voltage_lag_2h = st.sidebar.slider("Voltage 2h ago (V)", 190.0, 250.0, 220.0, 0.1)
voltage_lag_3h = st.sidebar.slider("Voltage 3h ago (V)", 190.0, 250.0, 220.0, 0.1)
load_lag_1h = st.sidebar.slider("Load 1h ago (%)", 0.0, 100.0, 60.0, 1.0)
load_lag_2h = st.sidebar.slider("Load 2h ago (%)", 0.0, 100.0, 60.0, 1.0)
load_lag_3h = st.sidebar.slider("Load 3h ago (%)", 0.0, 100.0, 60.0, 1.0)

now_bd = datetime.now(BD_TIME)
current_hour = now_bd.hour
current_month = now_bd.month

# ==== Calculate Derived Features (same logic as preprocessing script) ====
weather_stress_index = (
    max(temperature_c - 25, 0) * 0.02 +
    max(humidity_percent - 70, 0) * 0.01 +
    wind_speed_ms * 0.05
)
voltage_deviation = ((voltage - 220) / 220) * 100
frequency_deviation = ((frequency - 50) / 50) * 100
load_trend = load_percent - load_lag_1h

# ==== Build Input Row (must match feature_columns order exactly) ====
input_dict = {
    "voltage": voltage,
    "current": current,
    "frequency": frequency,
    "load_percent": load_percent,
    "temperature_c": temperature_c,
    "humidity_percent": humidity_percent,
    "wind_speed_ms": wind_speed_ms,
    "precipitation_mm": precipitation_mm,
    "weather_stress_index": weather_stress_index,
    "voltage_deviation": voltage_deviation,
    "frequency_deviation": frequency_deviation,
    "load_trend": load_trend,
    "voltage_lag_1h": voltage_lag_1h,
    "load_lag_1h": load_lag_1h,
    "voltage_lag_2h": voltage_lag_2h,
    "load_lag_2h": load_lag_2h,
    "voltage_lag_3h": voltage_lag_3h,
    "load_lag_3h": load_lag_3h,
    "hour_of_day": current_hour,
    "month": current_month,
}

input_df = pd.DataFrame([input_dict])[feature_columns]

# ==== Prediction ====
risk_probability = model.predict_proba(input_df)[0][1]
risk_percent = risk_probability * 100

if risk_percent < 30:
    risk_level = "LOW"
    risk_color = "green"
elif risk_percent < 60:
    risk_level = "MEDIUM"
    risk_color = "orange"
else:
    risk_level = "HIGH"
    risk_color = "red"

# ==== Main Display ====
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Outage Probability", f"{risk_percent:.1f}%")

with col2:
    st.markdown(f"### Risk Level")
    st.markdown(f"<h1 style='color:{risk_color};'>{risk_level}</h1>", unsafe_allow_html=True)

with col3:
    st.metric("Current Hour", f"{current_hour}:00")
    st.metric("Current Month", current_month)

st.markdown("---")

# ==== Warning Message ====
if risk_level == "HIGH":
    st.error("⚠️ WARNING: High outage risk detected! Immediate action recommended.")
elif risk_level == "MEDIUM":
    st.warning("⚠️ Caution: Moderate outage risk. Monitor the situation closely.")
else:
    st.success("✅ Grid conditions are normal. Low outage risk.")

# ==== Feature Values Table ====
st.markdown("---")
st.subheader("Current Input Values")
st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

# ==== Voltage Trend Chart ====
st.markdown("---")
st.subheader("Voltage Trend (Last 3 Hours vs Now)")
trend_df = pd.DataFrame({
    "Time": ["3h ago", "2h ago", "1h ago", "Now"],
    "Voltage": [voltage_lag_3h, voltage_lag_2h, voltage_lag_1h, voltage]
})
st.line_chart(trend_df.set_index("Time"))

# ==== Footer ====
st.markdown("---")
st.caption("Intelligent Early Warning System for Power Outage Prediction | Powered by XGBoost")
