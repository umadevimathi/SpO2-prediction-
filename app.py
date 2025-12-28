import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------
# Load saved ML objects
# -----------------------------------
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# -----------------------------------
# Streamlit UI
# -----------------------------------
st.set_page_config(page_title="SpO2 Prediction", layout="centered")

st.title("🫁 SpO₂ Level Prediction System")
st.write("Predict blood oxygen saturation (SpO₂) using patient health data")

st.header("Enter Patient Details")

# -----------------------------------
# User Inputs
# -----------------------------------
heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=80)
sys_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
dia_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=50, max_value=150, value=80)
temperature = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=36.5)
data_accuracy = st.slider("Data Accuracy (%)", 50, 100, 95)

fall_detection = st.selectbox("Fall Detection", ["Yes", "No"])
hr_alert = st.selectbox("Heart Rate Alert", ["Normal", "High"])
bp_alert = st.selectbox("Blood Pressure Alert", ["Normal", "High"])
temp_alert = st.selectbox("Temperature Alert", ["Normal", "High"])

# -----------------------------------
# Prediction Button
# -----------------------------------
if st.button("Predict SpO₂ Level"):
    
    new_patient_data = {
        "Heart Rate (bpm)": heart_rate,
        "Systolic Blood Pressure (mmHg)": sys_bp,
        "Diastolic Blood Pressure (mmHg)": dia_bp,
        "Body Temperature (°C)": temperature,
        "Fall Detection": fall_detection,
        "Data Accuracy (%)": data_accuracy,
        "Heart Rate Alert": hr_alert,
        "Blood Pressure Alert": bp_alert,
        "Temperature Alert": temp_alert
    }

    # Convert to DataFrame
    new_df = pd.DataFrame([new_patient_data])

    # One-hot encoding
    new_df = pd.get_dummies(new_df)

    # Align with training features
    new_df = new_df.reindex(columns=feature_names, fill_value=0)

    # Scale data
    new_df_scaled = scaler.transform(new_df)

    # Prediction
    prediction = model.predict(new_df_scaled)

    st.success(f"🩸 Predicted SpO₂ Level: **{round(prediction[0], 2)} %**")
