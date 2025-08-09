# app.py — ArcNova Celestial Intelligence Mode (Part 1)
import os
import time
import json
import math
import joblib
import shap
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path
from typing import Dict

# Optional imports
try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

try:
    import qiskit
    QISKIT_AVAILABLE = True
except Exception:
    QISKIT_AVAILABLE = False

# Config
st.set_page_config(page_title="ArcNova — Celestial AI Mode", layout="wide", initial_sidebar_state="expanded")
st.title("🌌 ArcNova — Celestial Intelligence Mode")
st.caption("Hybrid XAI + Gemini explanations | LSTM + XGBoost core | Explainability, chat, and optional quantum hints")

# Paths
MODEL_LSTM_PATH = "arcnova_lstm_model.h5"
MODEL_XGB_PATH = "arcnova_xgb_model.joblib"
SCALER_PATH = "scaler.joblib"
FEATURE_NAMES = ["temperature", "pressure", "magnetic_field_strength", "target_density", "fuel_density", "confinement_time"]

# Loaders
@st.cache_resource(show_spinner=False)
def load_xgb_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"XGB load failed: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_lstm_model(path):
    if tf is None:
        st.warning("TensorFlow missing; LSTM disabled.")
        return None
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        st.error(f"LSTM load failed: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_scaler(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

# Gemini helpers
def gemini_configure_from_env():
    if not GEMINI_AVAILABLE:
        return False, "google.generativeai not installed"
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        return False, "No GEMINI_API_KEY found"
    try:
        genai.configure(api_key=api_key)
        return True, "Gemini configured"
    except Exception as e:
        return False, f"Gemini config error: {e}"

def gemini_explain(prompt, model="gemini-pro", max_output_tokens=512, temperature=0.2):
    if not GEMINI_AVAILABLE:
        return None, "Gemini client not installed"
    ok, msg = gemini_configure_from_env()
    if not ok:
        return None, msg
    if len(prompt) > 4000:
        prompt = prompt[:3800] + "\n\n[truncated]"
    attempt, last_err = 0, None
    while attempt < 3:
        attempt += 1
        try:
            resp = genai.generate_text(model=model, prompt=prompt, temperature=temperature, max_output_tokens=max_output_tokens)
            return getattr(resp, "text", str(resp)), None
        except Exception as e:
            time.sleep(1.5 ** attempt)
            last_err = str(e)
    return None, f"Gemini failed after retries: {last_err}"

# Input scaling
def ui_to_feature_vector(ui_values: Dict[str, float], scaler=None):
    df = pd.DataFrame([{k: ui_values.get(k, 0) for k in FEATURE_NAMES}])
    if scaler is not None:
        try:
            return scaler.transform(df)
        except Exception:
            pass
    arr = []
    for c in FEATURE_NAMES:
        v = float(df.iloc[0][c])
        if c == "temperature":
            arr.append((v - 150) / 50)
        elif c == "pressure":
            arr.append((v - 3) / 1)
        elif c == "magnetic_field_strength":
            arr.append((v - 5) / 2)
        elif "density" in c:
            arr.append((v - 1.0) / 0.5)
        elif c == "confinement_time":
            arr.append((v - 10) / 5)
        else:
            arr.append(v)
    return np.array([arr])

# SHAP
@st.cache_resource(show_spinner=False)
def create_shap_explainer_for_xgb(model):
    try:
        return shap.Explainer(model)
    except Exception as e:
        st.warning(f"SHAP init failed: {e}")
        return None

def shap_outputs_for_input(explainer, x_input, feature_names):
    try:
        sv = explainer(x_input)
        impacts = {feature_names[i]: float(sv.values[0][i]) for i in range(len(feature_names))}
        return sv, impacts
    except Exception as e:
        return None, {"error": str(e)}

# UI sidebar
st.sidebar.header("Control Panel")
with st.sidebar.form("controls"):
    temp_ui = st.slider("Plasma Temperature (million K)", 50, 400, 150, step=5)
    pressure_ui = st.slider("Thermal Pressure (atm)", 1.0, 10.0, 3.0, step=0.1)
    field_ui = st.slider("Magnetic Field Strength (T)", 1.0, 12.0, 5.0, step=0.1)
    density_ui = st.slider("Fuel Density (g/cm³)", 0.1, 3.0, 1.0, step=0.01)
    confinement_ui = st.slider("Confinement Time (s)", 0.5, 60.0, 10.0, step=0.5)
    model_choice = st.selectbox("AI Engine", ["Hybrid Decision Core", "LSTM", "XGBoost"])
    use_gemini = st.checkbox("Enable Gemini explanations", value=True if GEMINI_AVAILABLE else False)
    quantum_mode = st.checkbox("Enable Quantum Mode", value=False and QISKIT_AVAILABLE)
    submitted = st.form_submit_button("Run Prediction & Explain")
    # app.py — ArcNova Celestial Intelligence Mode (Part 2)

# Status row
st.markdown("---")
colA, colB, colC = st.columns(3)
colA.metric("Run timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
colB.metric("Gemini available", "Yes" if GEMINI_AVAILABLE else "No")
colC.metric("Qiskit available", "Yes" if QISKIT_AVAILABLE else "No")
st.markdown("---")

if submitted:
    ui_values = {
        "temperature": float(temp_ui),
        "pressure": float(pressure_ui),
        "magnetic_field_strength": float(field_ui),
        "target_density": float(density_ui),
        "fuel_density": float(density_ui),
        "confinement_time": float(confinement_ui),
    }

    st.header("Prediction & Explanation")
    st.json(ui_values)

    xgb_model = load_xgb_model(MODEL_XGB_PATH)
    lstm_model = load_lstm_model(MODEL_LSTM_PATH)
    scaler = load_scaler(SCALER_PATH)

    x_input = ui_to_feature_vector(ui_values, scaler=scaler)

    def hybrid_decision(x_input):
        xgb_score = lstm_score = None
        chosen = None
        try:
            if xgb_model:
                xgb_score = float(xgb_model.predict_proba(x_input)[0][1])
        except Exception as e:
            st.warning(f"XGB error: {e}")
        try:
            if lstm_model:
                seq = np.tile(x_input, (1, 5, 1))
                lstm_score = float(lstm_model.predict(seq)[0][0])
        except Exception as e:
            st.warning(f"LSTM error: {e}")
        if xgb_score is not None and lstm_score is not None:
            return (lstm_score + xgb_score) / 2.0, "Hybrid", lstm_score, xgb_score
        elif xgb_score is not None:
            return xgb_score, "XGBoost", None, xgb_score
        elif lstm_score is not None:
            return lstm_score, "LSTM", lstm_score, None
        return None, "None", None, None

    fusion_score, chosen_model, lstm_score, xgb_score = hybrid_decision(x_input)

    st.subheader("🔮 Model Scores")
    st.write(f"Hybrid: {fusion_score} | LSTM: {lstm_score} | XGB: {xgb_score}")

    if fusion_score is not None:
        if fusion_score > 0.5:
            st.success(f"🔥 IGNITION LIKELY — Engine: {chosen_model}")
        else:
            st.error(f"❄️ IGNITION UNLIKELY — Engine: {chosen_model}")

    if xgb_model:
        st.markdown("## 🔍 SHAP Explanation")
        explainer = create_shap_explainer_for_xgb(xgb_model)
        if explainer:
            sv, impacts = shap_outputs_for_input(explainer, x_input, FEATURE_NAMES)
            if sv is not None:
                shap.plots.bar(sv, show=False)
                st.pyplot(bbox_inches="tight")
                st.table(pd.DataFrame.from_dict(impacts, orient='index', columns=["Impact"]).sort_values("Impact", ascending=False))
        else:
            st.info("SHAP unavailable.")

    if use_gemini:
        st.markdown("## 🧾 Gemini Explanation")
        prompt = f"INPUTS: {json.dumps(ui_values)}\nScores: {fusion_score}, {lstm_score}, {xgb_score}"
        explanation_text, err = gemini_explain(prompt)
        if explanation_text:
            st.write(explanation_text)
        else:
            st.warning(f"Gemini unavailable: {err}")

    if quantum_mode:
        st.markdown("## ⚛️ Quantum Hint")
        st.info("Quantum optimization placeholder. Requires Qiskit runtime.")

    st.markdown("---")
    if st.button("Save prediction (JSON)"):
        log = {
            "timestamp": datetime.utcnow().isoformat(),
            "inputs": ui_values,
            "fusion_score": fusion_score,
            "lstm_score": lstm_score,
            "xgb_score": xgb_score,
        }
        fname = f"arc_prediction_{int(time.time())}.json"
        Path(fname).write_text(json.dumps(log, indent=2))
        st.download_button("Download JSON", data=json.dumps(log, indent=2), file_name=fname, mime="application/json")
    st.balloons()

       
