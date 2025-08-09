# ------------------------------
# ArcNova — Celestial Mode (Part A)
# Robust, professional Streamlit app (paste Part B below)
# ------------------------------
import os
import time
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Optional heavy imports wrapped defensively
try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    import shap
except Exception:
    shap = None

# Gemini (Google Generative AI) client — optional
try:
    import google.generativeai as genai
    GEMINI_CLIENT_AVAILABLE = True
except Exception:
    genai = None
    GEMINI_CLIENT_AVAILABLE = False

# ------------------------------
# App config & UI theme (sleek, dark)
# ------------------------------
st.set_page_config(page_title="ArcNova — Celestial Intelligence", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    /* page background */
    .stApp { background: linear-gradient(180deg, #0b0f1a 0%, #07101a 100%); color: #e6f2ff; }
    /* panels */
    .card { background: rgba(20,25,30,0.7); padding: 18px; border-radius: 12px; box-shadow: 0 6px 30px rgba(0,0,0,0.5); }
    h1, h2, h3, h4 { color: #7be0ff; }
    .small-muted { color: #99b3c8; font-size:12px; }
    .metric-label { color:#bfefff; }
    .stSidebar { background: linear-gradient(180deg, #061018, #041018); }
    .btn-ignite { background: linear-gradient(90deg,#00ffd1,#00b3ff); color: #001; font-weight:700; }
    </style>
    """, unsafe_allow_html=True
)

st.title("🌌 ArcNova — Celestial Intelligence")
st.markdown("**Hybrid AI (LSTM + XGBoost)** | XAI (SHAP) | Gemini natural-language explanations — *production-minded & defensive*")
st.markdown("---")

# ------------------------------
# File paths (place your model & scaler files here)
# ------------------------------
MODEL_XGB = "arcnova_xgb_model.joblib"
MODEL_LSTM = "arcnova_lstm_model.h5"
SCALER_FILE = "scaler.joblib"
# optional: you can provide your exact training means via JSON, but we fall back to defaults
TRAIN_MEANS_FILE = "X_train_mean.json"

# ------------------------------
# Default training means fallback (used to populate missing features)
# Edit these defaults if you have better values.
# ------------------------------
DEFAULT_MEANS: Dict[str, float] = {
    "temperature": 150.0,          # million K (UI units)
    "pressure": 5e6,               # Pa
    "magnetic_field_strength": 5.0,# Tesla
    "target_density": 5.0,         # UI: e+19 / m^3
    "fuel_density": 5.0,           # UI: e+19 / m^3
    "confinement_time": 10.0,      # s
    "unnamed:_0": 0
}
# create placeholder features up to 21 total if training used 21 features
# We'll produce FEATURE_NAMES list from either JSON (if present) or defaults below.

# ------------------------------
# Helper: load JSON training-means if present
# ------------------------------
def load_train_means(path: str) -> Dict[str, float]:
    if Path(path).exists():
        try:
            with open(path, "r") as f:
                d = json.load(f)
            # make sure keys are floats
            return {k: float(v) for k, v in d.items()}
        except Exception as e:
            st.warning(f"Failed to read {path}: {e} — using fallback defaults.")
    # fallback: expand DEFAULT_MEANS into a 21-feature list
    means = DEFAULT_MEANS.copy()
    # add placeholders if required
    for i in range(7, 22):
        k = f"feat{i}"
        if k not in means:
            means[k] = 0.0
    return means

TRAIN_MEANS = load_train_means(TRAIN_MEANS_FILE)
FEATURE_NAMES = list(TRAIN_MEANS.keys())
N_FEATURES = len(FEATURE_NAMES)

# show diagnostic (non-verbose)
st.caption(f"Feature vector length (expected): {N_FEATURES} features — {len(FEATURE_NAMES)} names loaded.")

# ------------------------------
# Model & scaler loaders (non-fatal)
# ------------------------------
@st.cache_resource
def load_xgb(path=MODEL_XGB):
    if not Path(path).exists():
        st.info(f"XGBoost model not found at {path}; app will run in fallback mode.")
        return None
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.warning(f"Failed to load XGBoost ({path}): {e}")
        return None

@st.cache_resource
def load_lstm(path=MODEL_LSTM):
    if tf is None:
        return None
    if not Path(path).exists():
        st.info(f"LSTM model not found at {path}. LSTM features will be disabled.")
        return None
    try:
        m = tf.keras.models.load_model(path)
        return m
    except Exception as e:
        st.warning(f"Failed to load LSTM ({path}): {e}")
        return None

@st.cache_resource
def load_scaler(path=SCALER_FILE):
    if not Path(path).exists():
        return None
    try:
        sc = joblib.load(path)
        return sc
    except Exception as e:
        st.warning(f"Failed to load scaler ({path}): {e}")
        return None

# instantiate loaders (cached)
xgb_model = load_xgb()
lstm_model = load_lstm()
scaler = load_scaler()

# ------------------------------
# Sidebar: control form (must include form_submit_button)
# ------------------------------
st.sidebar.header("Control Panel — ArcNova Celestial")
with st.sidebar.form("controls"):
    # Visible controls (keep UX simple & technical)
    temp_mk = st.slider("Plasma Temperature (million K)", 50, 400, int(TRAIN_MEANS.get("temperature", 150)), step=5)
    pressure_pa = st.slider("Thermal Pressure (Pa)", int(1e6), int(1e7), int(TRAIN_MEANS.get("pressure", 5e6)), step=100000, format="%d")
    magT = st.slider("Magnetic Field Strength (T)", 1.0, 12.0, float(TRAIN_MEANS.get("magnetic_field_strength", 5.0)), step=0.1)
    density_ui = st.slider("Fuel Density (e+19 / m³)", 1.0, 12.0, float(TRAIN_MEANS.get("target_density", 5.0)), step=0.1)
    confinement_s = st.slider("Confinement Time (s)", 0.1, 60.0, float(TRAIN_MEANS.get("confinement_time", 10.0)), step=0.1)

    model_choice = st.selectbox("AI Engine", ["Hybrid Decision Core", "LSTM", "XGBoost"])
    enable_shap = st.checkbox("Enable SHAP XAI (XGBoost only)", value=False)
    enable_gemini = st.checkbox("Enable Gemini natural-language explanation (optional)", value=False)
    enable_quantum = st.checkbox("Enable Quantum hints (Qiskit) — conceptual", value=False)

    # Gemini API — priority: Streamlit Secrets -> env -> manual input
    gemini_key = st.secrets.get("GEMINI_API_KEY", None)
    if not gemini_key:
        gemini_key = os.getenv("GEMINI_API_KEY", None)
    if not gemini_key:
        gemini_key = st.text_input("Gemini API key (optional)", type="password")

    submitted = st.form_submit_button("🚀 Run Prediction & Explain", help="Compute prediction and XAI explanations")

# quick status bar (top)
col1, col2, col3 = st.columns([1.2, 1.0, 1.0])
col1.markdown(f"**Run time**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
col2.markdown(f"**XGBoost**: {'Loaded' if xgb_model is not None else 'Missing'}")
col3.markdown(f"**LSTM**: {'Loaded' if lstm_model is not None else 'Missing'}")
st.markdown("---")

# ------------------------------
# Utility functions (scaling, input building, hybrid core)
# ------------------------------
def build_input_dict_from_ui() -> Dict[str, Any]:
    """
    Build the full feature dictionary in the same order as FEATURE_NAMES
    using UI sliders and TRAIN_MEANS fallback for hidden features.
    """
    ui = {
        "temperature": float(temp_mk),
        "pressure": float(pressure_pa),
        "magnetic_field_strength": float(magT),
        "target_density": float(density_ui),
        "fuel_density": float(density_ui),
        "confinement_time": float(confinement_s),
    }
    # Start with TRAIN_MEANS and overwrite visible keys
    full = TRAIN_MEANS.copy()
    for k, v in ui.items():
        if k in full:
            full[k] = v
        else:
            # if feature name differs, inject visible ones and leave others untouched
            full[k] = v
    # Ensure all FEATURE_NAMES exist
    for k in FEATURE_NAMES:
        if k not in full:
            full[k] = TRAIN_MEANS.get(k, 0.0)
    return full

def df_from_feature_dict(d: Dict[str, Any]) -> pd.DataFrame:
    """
    return DataFrame shape (1, N_FEATURES) with columns = FEATURE_NAMES
    """
    row = {k: d.get(k, 0.0) for k in FEATURE_NAMES}
    return pd.DataFrame([row], columns=FEATURE_NAMES)

def safe_scale(df: pd.DataFrame) -> np.ndarray:
    """
    Transform df using scaler if available, else fallback scaling heuristic.
    Always returns numpy array shape (1, n_features_expected).
    """
    if scaler is not None:
        try:
            out = scaler.transform(df)
            return np.asarray(out, dtype=float)
        except Exception as e:
            st.warning(f"scaler.transform failed: {e} — using fallback scale.")
    # fallback: deterministic normalization centered on training means
    arr = []
    for c in FEATURE_NAMES:
        v = float(df.iloc[0][c])
        # fallback heuristics (tunable)
        if c == "temperature":
            arr.append((v - 150.0) / 50.0)
        elif c == "pressure":
            arr.append((v - 5e6) / 1e6)
        elif c == "magnetic_field_strength":
            arr.append((v - 5.0) / 2.0)
        elif "density" in c:
            arr.append((v - 5.0) / 2.0)
        elif c == "confinement_time":
            arr.append((v - 10.0) / 5.0)
        else:
            arr.append(v)
    return np.array([arr], dtype=float)

# Decision thresholds and LSTM timesteps (update to match your training if different)
LSTM_TIMESTEPS = 5
LSTM_CONF = 0.65
XGB_CONF = 0.70

def hybrid_decision_core(x_scaled: np.ndarray):
    """
    Returns (fusion_score: float 0-1, model_used: str, lstm_score, xgb_score)
    """
    lstm_score = None
    xgb_score = None

    # XGBoost path
    if xgb_model is not None:
        try:
            # if x_scaled has more features than model expects this will still work
            xgb_prob = xgb_model.predict_proba(x_scaled)[0][1]
            xgb_score = float(xgb_prob)
        except Exception as e:
            st.warning(f"XGBoost predict_proba failed: {e}")

    # LSTM path
    if lstm_model is not None and tf is not None:
        try:
            # ensure shape (1, timesteps, features)
            # If LSTM expects a different input dimensionality, user must retrain or adjust constants
            seq = np.tile(x_scaled, (1, LSTM_TIMESTEPS, 1))
            lstm_prob = lstm_model.predict(seq, verbose=0)[0][0]
            lstm_score = float(lstm_prob)
        except Exception as e:
            st.warning(f"LSTM predict failed: {e}")

    # selection logic
    model_used = "None"
    if lstm_score is not None and xgb_score is not None:
        model_used = "Hybrid"
        fusion_score = (lstm_score + xgb_score) / 2.0
    elif xgb_score is not None:
        model_used = "XGBoost"
        fusion_score = xgb_score
    elif lstm_score is not None:
        model_used = "LSTM"
        fusion_score = lstm_score
    else:
        fusion_score = None

    return fusion_score, model_used, lstm_score, xgb_score

# --------------
# End of Part A — paste Part B below this line
# --------------
# ------------------------------
# ArcNova — Celestial Mode (Part B)
# Continuation: execution, XAI, Gemini NL, export
# ------------------------------

# only run heavy stuff after user submits the form
if submitted:

    # 1) Prepare input
    feature_dict = build_input_dict_from_ui()
    input_df = df_from_feature_dict(feature_dict)
    st.subheader("🔬 Input Snapshot (features)")
    # show compact table
    display_df = pd.DataFrame.from_dict(feature_dict, orient="index", columns=["value"])
    st.table(display_df)

    # 2) Scale / normalize input
    scaled = safe_scale(input_df)
    st.caption(f"Scaled input array shape: {scaled.shape}")

    # 3) Run decision core
    fusion_score, model_used, lstm_s, xgb_s = hybrid_decision_core(scaled)

    # 4) Present results
    st.markdown("---")
    rcol1, rcol2, rcol3 = st.columns([1.4, 1.0, 1.0])
    rcol1.metric("🔥 Fusion Score", f"{fusion_score:.3f}" if fusion_score is not None else "N/A")
    rcol2.metric("Engine used", model_used)
    rcol3.metric("Timestamp (UTC)", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))

    if fusion_score is None:
        st.error("Prediction unavailable — no models loaded or prediction error occurred. Check server logs.")
    else:
        if fusion_score > 0.5:
            st.success(f"🚀 IGNITION LIKELY — score {fusion_score:.3f}")
        else:
            st.info(f"🧊 Ignition unlikely — score {fusion_score:.3f}")

    # diagnostics box
    with st.expander("Model diagnostics & raw scores", expanded=False):
        st.write({"fusion_score": fusion_score, "model_used": model_used, "lstm_score": lstm_s, "xgb_score": xgb_s})
        if xgb_model is None:
            st.info("XGBoost model not loaded — upload arcnova_xgb_model.joblib to enable.")
        if lstm_model is None:
            st.info("LSTM model not loaded or TensorFlow not installed — upload arcnova_lstm_model.h5 and include TF in requirements.")

    # 5) SHAP explainability (XGBoost only, safe)
    shap_impacts = None
    if enable_shap:
        st.markdown("### 🔍 SHAP Explanation (XGBoost)")
        if shap is None:
            st.warning("SHAP package not installed in this environment. Add 'shap' to requirements.txt to enable.")
        elif xgb_model is None:
            st.warning("XGBoost model missing — cannot compute SHAP.")
        else:
            try:
                # build explainer freshly (avoid caching unhashable model objects)
                explainer = shap.Explainer(xgb_model)
                sv = explainer(scaled)
                # convert to simple impacts dict for UI
                impacts = {}
                for i, fname in enumerate(FEATURE_NAMES[:sv.values.shape[1]]):
                    impacts[fname] = float(sv.values[0][i])
                shap_impacts = impacts
                # show bar and table (matplotlib/plot rendering handled by shap)
                try:
                    st.write("Feature impact (bar)")
                    shap.plots.bar(sv, show=False)
                    st.pyplot(bbox_inches="tight")
                except Exception:
                    # fallback: simple dataframe table
                    st.table(pd.DataFrame.from_dict(impacts, orient="index", columns=["impact"]).sort_values("impact", ascending=False))
            except Exception as e:
                st.warning(f"SHAP computation failed: {e}")

    # 6) Gemini natural-language explanation (optional)
    if enable_gemini:
        st.markdown("### 🧠 Gemini Natural-Language Explanation")
        if not (gemini_key and GEMINI_CLIENT_AVAILABLE):
            if not GEMINI_CLIENT_AVAILABLE:
                st.warning("Gemini client not installed in environment. Add 'google-generativeai' to requirements.")
            else:
                st.warning("Gemini API key not configured. Add to Streamlit Secrets (GEMINI_API_KEY) or paste in sidebar.")
        else:
            try:
                # configure gemini client
                genai.configure(api_key=gemini_key)
                # craft concise prompt
                prompt = (
                    "You are a senior fusion physicist and ML scientist. Given the following prediction data, "
                    "produce (A) a concise academic-grade explanation (3 short paragraphs), (B) a 2-sentence operator action blurb, "
                    "and (C) three prioritized tips to increase ignition probability.\n\n"
                    f"INPUTS: {json.dumps(feature_dict, default=str, indent=2)}\n\n"
                    f"PREDICTION: fusion_score={fusion_score}, model_used={model_used}, lstm={lstm_s}, xgb={xgb_s}\n\n"
                    f"SHAP_IMPACTS: {json.dumps(shap_impacts, indent=2) if shap_impacts is not None else 'N/A'}\n\n"
                    "Keep tone formal and technical for the academic section; keep operator blurb imperative and short."
                )
                # call Gemini (best-effort wrapper)
                response = genai.generate_text(model="gemini-1.5", prompt=prompt, max_output_tokens=512, temperature=0.15)
                text = response.text if hasattr(response, "text") else str(response)
                st.markdown("**Gemini output**")
                st.write(text)
            except Exception as e:
                st.warning(f"Gemini call failed: {e}")

    # 7) Optional quantum hint (light, conceptual)
    if enable_quantum:
        st.markdown("### ⚛️ Quantum Hint (conceptual)")
        st.info("Quantum mode provides conceptual recommendations (Qiskit runtime not bundled). For serious quantum runs, integrate Qiskit and an appropriate backend.")

    # 8) Save / export / download
    st.markdown("---")
    st.subheader("📦 Save & Export")
    output_record = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "inputs": feature_dict,
        "fusion_score": fusion_score,
        "model_used": model_used,
        "lstm_score": lstm_s,
        "xgb_score": xgb_s,
        "shap_impacts": shap_impacts
    }
    if st.button("Save JSON to server"):
        fname = f"arc_prediction_{int(time.time())}.json"
        try:
            Path(fname).write_text(json.dumps(output_record, indent=2))
            st.success(f"Saved {fname}")
            st.download_button("Download JSON", data=json.dumps(output_record, indent=2), file_name=fname, mime="application/json")
        except Exception as e:
            st.error(f"Failed to save JSON: {e}")

    # final celebratory UX
    st.balloons()
    st.markdown("**Execution complete.** Adjust sliders and re-run for new scenarios.")

else:
    # if not submitted show a short guidance panel
    st.info("Use the sidebar to configure reactor inputs, choose AI engine, and press *Run Prediction & Explain*.")
    st.markdown("**Tip:** Add `arcnova_xgb_model.joblib` and `scaler.joblib` to the app folder for real predictions. Add `GEMINI_API_KEY` to Streamlit Secrets to enable Gemini explanations.")
