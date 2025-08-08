
import streamlit as st
import numpy as np
from datetime import datetime
import tensorflow as tf
import joblib
import json
import pandas as pd # Import pandas to create DataFrame

# Load trained models, scaler, and training data means
lstm_model = tf.keras.models.load_model("arcnova_lstm_model.h5")
xgb_model = joblib.load("arcnova_xgb_model.joblib")
scaler = joblib.load("scaler.joblib")
with open('X_train_mean.json', 'r') as f:
    X_train_mean = json.load(f)

st.set_page_config(page_title="ArcNova Fusion AI", layout="wide")

# --- Styling ---
st.markdown("""
<style>
body { background-color: #0b0c10; font-family: 'Segoe UI', sans-serif; }
.stApp { background-color: #0b0c10; color: #f0f6fc; }
h1, h2, h3 { color: #58a6ff; }
.css-18e3th9 {
  background-color: #161b22 !important;
  padding: 25px;
  border-radius: 12px;
  box-shadow: 0 0 20px rgba(0, 255, 255, 0.15);
}
div.stSlider > label { font-weight: bold; color: #39ff14; }
.stButton button {
  background-color: #00f0ff; color: #000;
  font-weight: bold; border-radius: 8px;
  padding: 0.75rem 1.25rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 ArcNova Fusion Control Panel")
st.caption("AI-Powered Ignition Simulation – NASA-Class Interface")

# Define the order of columns as in the training data
# This order is crucial for correct scaling and prediction
feature_columns = list(X_train_mean.keys())


col1, col2 = st.columns(2)
with col1:
    # UI controls for the 5 features the user can adjust
    # Note: Ensure these map correctly to the column names in feature_columns
    temp = st.slider("Plasma Temp (million K)", 50, 300, 150) # Assuming this maps to 'temperature'
    pressure_val = st.slider("Pressure (Pa)", 1e6, 10e6, 5e6, format="%.0e") # Assuming this maps to 'pressure'
    field_strength = st.slider("Magnetic Field Strength (T)", 1, 10, 5) # Assuming this maps to 'magnetic_field_strength'

with col2:
    density_val = st.slider("Target/Fuel Density (e+19 / m³)", 1.0, 10.0, 5.0, 0.1) # Assuming this maps to 'target_density' and 'fuel_density'
    confinement = st.slider("Confinement Time (s)", 0.1, 1.0, 0.5, 0.05) # Assuming this maps to 'confinement_time'
    # Add radio button for model choice, keeping it for now
    model_choice = st.radio("Select AI Engine:",
                            ["LSTM – Deep Learning", "XGBoost – Traditional ML", "Hybrid Decision Core"]) # Added Hybrid option


# --------------------------------------------------------
# ARC-NOVA AI – PART 6: FUSION DECISION CORE (Integrated)
# AI Hybrid Selector – LSTM + XGBoost with Intelligence Switching
# --------------------------------------------------------

# Define trusted confidence thresholds
LSTM_THRESHOLD = 0.65     # Trust LSTM if score is higher
XGB_THRESHOLD = 0.70      # Trust XGB if score is higher

# Decision Core: Hybrid Brain Selector function (now within app.py)
def fusion_decision_core(inputs, lstm_model, xgb_model): # Pass models as arguments
    # LSTM prediction (reshape for LSTM format)
    # Ensure inputs has the correct shape (1, 21) before reshaping for LSTM
    if inputs.shape != (1, X_train_mean.__len__()): # Use the size of the mean dict for feature count
        st.error(f"Internal Error: Expected input shape (1, {X_train_mean.__len__()}) for fusion_decision_core, but got {inputs.shape}")
        return 0, "Error" # Return a default/error state

    # Reshape for LSTM (needs shape 1, timesteps, features)
    # Assuming timesteps = 5 as used in training
    reshaped = np.tile(inputs, (1, 5, 1))
    lstm_score = lstm_model.predict(reshaped)[0][0]

    # XGBoost prediction
    xgb_score = xgb_model.predict_proba(inputs)[0][1]

    # Smart decision logic
    if lstm_score > LSTM_THRESHOLD and xgb_score > XGB_THRESHOLD:
        fusion_score = (lstm_score + xgb_score) / 2
        model_used = "🤝 Hybrid AI (LSTM + XGBoost)"
    elif lstm_score > LSTM_THRESHOLD:
        fusion_score = lstm_score
        model_used = "🧠 LSTM Selected"
    elif xgb_score > XGB_THRESHOLD:
        fusion_score = xgb_score
        model_used = "🌲 XGBoost Selected"
    else:
        # If neither model is highly confident, average or use a default?
        fusion_score = (lstm_score + xgb_score) / 2 # Averaging kept
        model_used = "⚠️ Uncertain - Mixed Model"

    return fusion_score, model_used


def scale_input(temp, pressure_val, field_strength, density_val, confinement, X_train_mean, scaler, feature_columns):
    # Create a dictionary with all feature values
    # Initialize with mean values from training data
    input_data = X_train_mean.copy()

    # Update the dictionary with values from UI controls
    # Map UI controls to the correct feature names used in training data
    # Ensure correct data types and scales matching training data
    input_data['temperature'] = temp * 1e6 # Convert million K to K (assuming training data temp is in K)
    input_data['pressure'] = pressure_val # Pressure is already in Pa
    input_data['magnetic_field_strength'] = field_strength
    # Assuming density_val is for both target_density and fuel_density, in units of e+19 / m^3
    input_data['target_density'] = density_val * 1e19
    input_data['fuel_density'] = density_val * 1e19
    input_data['confinement_time'] = confinement

    # Handle categorical features based on typical values or assumptions
    # For simplicity with current UI, keep at mean (mix of categories)
    # If adding UI controls for these, update this section.
    # Example if we had a 'Magnetic Config' dropdown:
    # config = st.selectbox("Magnetic Config", ['tokamak', 'stellarator', 'reversed field pinch'])
    # input_data['magnetic_field_configuration_stellarator'] = 0
    # input_data['magnetic_field_configuration_tokamak'] = 0
    # if config == 'stellarator': input_data['magnetic_field_configuration_stellarator'] = 1
    # if config == 'tokamak': input_data['magnetic_field_configuration_tokamak'] = 1
    # (Similar logic for target_composition)


    # Handle 'unnamed:_0' - likely an index, setting to 0 for a new single prediction
    input_data['unnamed:_0'] = 0


    # Convert the dictionary to a pandas DataFrame with a single row
    # Ensure the column order matches the training data (feature_columns list)
    input_df = pd.DataFrame([input_data], columns=feature_columns)

    # Apply the loaded scaler
    scaled_input = scaler.transform(input_df)

    return scaled_input


if st.button("🚀 Predict Ignition"):
    try:
        # Scale the input using the updated function
        scaled_input_array = scale_input(
            temp, pressure_val, field_strength, density_val, confinement,
            X_train_mean, scaler, feature_columns
        )

        st.subheader("🔍 Prediction Output:")

        if model_choice == "Hybrid Decision Core":
             fusion_score, model_used = fusion_decision_core(scaled_input_array, lstm_model, xgb_model)
             # Use the result from the hybrid core
             pred_prob = fusion_score
             # model_used is already set by fusion_decision_core

        elif model_choice.startswith("LSTM"):
            # Reshape for LSTM (needs shape 1, timesteps, features)
            # Assuming timesteps = 5 as used in training
            # The scaled_input_array has shape (1, 21)
            # We need to repeat this over the timestep dimension
            reshaped_input = np.tile(scaled_input_array, (1, 5, 1))
            pred_prob = lstm_model.predict(reshaped_input)[0][0]
            model_used = "🧠 LSTM Selected"
        else: # XGBoost
            pred_prob = xgb_model.predict_proba(scaled_input_array)[0][1]
            model_used = "🌲 XGBoost Selected"

        # Determine prediction based on a threshold (e.5)
        if pred_prob > 0.5:
            st.success(f"🔥 IGNITION LIKELY – Score: {pred_prob:.2f}") # Removed model_used here
        else:
            st.error(f"❄️ NO IGNITION – Score: {pred_prob:.2f}") # Removed model_used here

        # Display the model used below the prediction result
        st.markdown(f"**AI Engine Used:** {model_used}")


        # Keep the placeholder warning about the UI being incomplete,
        # but the prediction logic should now be active.
        st.warning("⚠️ UI is incomplete: Assumes mean values for uncontrolled features. Needs explicit controls for categorical features and others.")

        st.caption(f"🕓 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.caption(f"🕓 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
