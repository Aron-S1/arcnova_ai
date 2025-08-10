# =================== ArcNova — Celestial Mode (Part A) ===================
# Paste Part A first into app.py
# Modern, defensive Streamlit UI — prepares app, loads resources safely.

import os
import time
import json
import math
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# Optional heavy deps wrapped safely
try:
    import joblib
except Exception:
    joblib = None

try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    import shap
except Exception:
    shap = None

# Gemini (Google) client - optional
try:
    import google.generativeai as genai
    GEMINI_CLIENT = True
except Exception:
    genai = None
    GEMINI_CLIENT = False

# Qiskit optional
try:
    import qiskit
    QISKIT_AVAILABLE = True
except Exception:
    QISKIT_AVAILABLE = False

# ---------------- UI config & CSS ----------------
st.set_page_config(page_title="ArcNova — Celestial Intelligence", layout="wide", initial_sidebar_state="expanded")

# Minimal, modern dark CSS (tweak colors if you want)
st.markdown(
    """
    <style>
    :root { --bg:#0b0f14; --card:#0f1720; --muted:#9aa5b1; --accent:#7ee4ff; --success:#28c76f; }
    .stApp { background: linear-gradient(180deg, #05060a 0%, var(--bg) 100%); color: #e6eef6; }
    header .decoration {display:none;}
    .reportview-container .main .block-container{padding-top:1.25rem;}
    .card { background: var(--card); border-radius:14px; padding:18px; box-shadow: 0 6px 30px rgba(2,8,23,0.6); }
    h1, h2, h3 { color: var(--accent); margin:0; }
    .muted { color: var(--muted); font-size:0.95rem; }
    .small { font-size:0.9rem; color:var(--muted); }
    .accent-pill { background: linear-gradient(90deg,#34d5ff,#7ee4ff); padding:6px 10px; border-radius:999px; color:#001; font-weight:700; }
    .big-metric { font-size:2.8rem; font-weight:700; color:#fff; }
    .subtle { color: #aebfcc; }
    .control-label { color:#cfeeff; font-weight:600; }
    </style>
    """, unsafe_allow_html=True
)

# ---------------- Paths & defaults ----------------
MODEL_LSTM_PATH = "arcnova_lstm_model.h5"
MODEL_XGB_PATH = "arcnova_xgb_model.joblib"
SCALER_PATH = "scaler.joblib"
X_TRAIN_MEAN = "X_train_mean.json"

# Default full feature list length fallback (update this if your training had more features)
DEFAULT_FEATURE_NAMES = [
    "temperature", "pressure", "magnetic_field_strength", "target_density", "fuel_density", "confinement_time"
]
# We'll extend to hidden features automatically if scaler/model expects more.

# ---------------- Safe loaders (no crash on missing files) ----------------
def safe_load_joblib(path):
    if joblib is None:
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.info(f"Joblib load warning: can't load {path} — {e}")
        return None

def safe_load_tf_model(path):
    if tf is None:
        return None
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        st.info(f"TF load warning: can't load {path} — {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_models():
    """Load models & scaler; return dict of objects (None where missing)."""
    xgb = safe_load_joblib(MODEL_XGB_PATH)
    lstm = safe_load_tf_model(MODEL_LSTM_PATH)
    scaler = safe_load_joblib(SCALER_PATH)
    # X_train_mean.json may give feature names / defaults
    x_train_mean = {}
    try:
        if Path(X_TRAIN_MEAN).exists():
            with open(X_TRAIN_MEAN, "r") as f:
                x_train_mean = json.load(f)
    except Exception:
        x_train_mean = {}
    return {"xgb": xgb, "lstm": lstm, "scaler": scaler, "x_train_mean": x_train_mean}

resources = load_models()
xgb_model = resources["xgb"]
lstm_model = resources["lstm"]
scaler = resources["scaler"]
X_train_mean = resources["x_train_mean"]

# If X_train_mean is present, we use its keys as canonical feature order (best).
if X_train_mean:
    FEATURE_NAMES = list(X_train_mean.keys())
else:
    # Fallback: start with the visible controls and fill hidden features later if needed
    FEATURE_NAMES = DEFAULT_FEATURE_NAMES.copy()

# ---------------- Helper: deterministic fallback scaler ----------------
def fallback_scale(df: pd.DataFrame) -> np.ndarray:
    """
    Deterministic fallback scaling to keep numeric magnitudes reasonable.
    Mirrors earlier UI heuristics. Returns numpy array shape (1, n_features).
    """
    row = df.iloc[0]
    arr = []
    for c in df.columns:
        v = float(row[c])
        if c == "temperature":
            arr.append((v - 150.0) / 50.0)
        elif c == "pressure":
            # UI pressure may be in Pa or atm; if huge, we scale differently — best effort
            if v > 1e6:
                arr.append((v - 5e6) / 5e6)
            else:
                arr.append((v - 3.0) / 1.0)
        elif c == "magnetic_field_strength":
            arr.append((v - 5.0) / 2.0)
        elif "density" in c:
            arr.append((v - 1.0) / 0.5)
        elif c == "confinement_time":
            arr.append((v - 10.0) / 5.0)
        else:
            arr.append(float(v))
    return np.array([arr], dtype=float)

def transform_input_df(df: pd.DataFrame):
    """Try scaler.transform, else fallback_scale."""
    if scaler is not None:
        try:
            return scaler.transform(df)
        except Exception as e:
            st.warning(f"Scaler transform failed — using fallback: {e}")
            return fallback_scale(df)
    else:
        return fallback_scale(df)

# ---------------- Sidebar: controls (form) ----------------
with st.sidebar.form("controls", clear_on_submit=False):
    st.markdown("<div class='card'><h3 class='control-label'>Control Panel — ArcNova</h3></div>", unsafe_allow_html=True)
    st.write(" ")
    temp_ui = st.slider("Plasma Temperature (million K)", 50, 400, 150, step=5, help="Input in million Kelvin")
    pressure_ui = st.slider("Thermal Pressure (Pa)", int(1e6), int(1e7), int(5e6), step=int(1e5))
    field_ui = st.slider("Magnetic Field Strength (T)", 1.0, 12.0, 5.0, step=0.1)
    dens_ui = st.slider("Fuel / Target Density (e+19 / m³)", 0.1, 10.0, 1.0, step=0.1)
    conf_ui = st.slider("Confinement Time (s)", 0.5, 60.0, 10.0, step=0.5)
    engine_choice = st.selectbox("AI Engine", ["Hybrid Decision Core", "LSTM", "XGBoost"])
    enable_shap = st.checkbox("Enable SHAP XAI (XGBoost only)", value=False)
    enable_gemini = st.checkbox("Enable Gemini explanations (cloud key required)", value=False)
    enable_quantum = st.checkbox("Show Quantum hints (Qiskit)", value=False)
    submit_button = st.form_submit_button("🚀 Run Prediction & Explain")

# quick top row info
col1, col2, col3 = st.columns([1, 1, 1])
col1.metric("Run timestamp (UTC)", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
col2.metric("XGBoost present", "Yes" if xgb_model is not None else "No")
col3.metric("LSTM present", "Yes" if lstm_model is not None else "No")

st.markdown("---")
st.markdown("<div class='card'><h1>🌌 ArcNova — Celestial Intelligence</h1><p class='muted'>Hybrid LSTM + XGBoost core • Gemini natural-language explainability • SHAP XAI</p></div>", unsafe_allow_html=True)
st.write(" ")
# =================== End Part A =================== 
# =================== ArcNova — Celestial Mode (Part B) ===================
# Paste Part B right after Part A. This contains prediction, XAI, Gemini, export.

# ---------------- Build full feature DataFrame ----------------
def build_full_feature_df(ui_vals):
    """
    Build a DataFrame in the canonical FEATURE_NAMES order.
    If FEATURE_NAMES is shorter than expected, we automatically pad with zero-features
    so model receives consistent number of features.
    """
    base = {}
    # start with UI-supplied obvious columns (names used in training should match)
    base.update({
        "temperature": float(ui_vals["temperature"]),
        "pressure": float(ui_vals["pressure"]),
        "magnetic_field_strength": float(ui_vals["magnetic_field_strength"]),
        "target_density": float(ui_vals["target_density"]),
        "fuel_density": float(ui_vals["fuel_density"]),
        "confinement_time": float(ui_vals["confinement_time"])
    })

    # If we have a training mean dict, use its keys as canonical set (best)
    if X_train_mean:
        canonical = list(X_train_mean.keys())
    else:
        # Otherwise use existing FEATURE_NAMES and ensure at least 6 + hidden placeholders
        canonical = FEATURE_NAMES.copy()
        # ensure minimum 21 features if original project expected many features
        if len(canonical) < 21:
            # append placeholder feature names feat7..feat21
            for i in range(len(canonical)+1, 22):
                fname = f"feat{i}"
                if fname not in canonical:
                    canonical.append(fname)

    # Populate missing entries with mean or 0.0
    for fname in canonical:
        if fname not in base:
            # try training mean if available
            if X_train_mean and fname in X_train_mean:
                base[fname] = X_train_mean[fname]
            else:
                base[fname] = 0.0

    df = pd.DataFrame([base], columns=canonical)
    return df

# ---------------- Hybrid decision core ----------------
def hybrid_decision(x_input):
    """Return fusion_score, engine_name, lstm_score, xgb_score (None where not available)."""
    xgb_score = None
    lstm_score = None
    try:
        if xgb_model is not None:
            # x_input must be 2D array (1, n_features)
            xgb_score = float(xgb_model.predict_proba(x_input)[0][1])
    except Exception as e:
        st.warning(f"XGBoost prediction error: {e}")

    try:
        if lstm_model is not None:
            # Assume LSTM used a timestep stacking during training; repeat single row across timesteps
            timesteps = 5
            seq = np.tile(x_input, (1, timesteps, 1))
            lstm_score = float(lstm_model.predict(seq)[0][0])
    except Exception as e:
        st.warning(f"LSTM prediction error: {e}")

    # choose engine
    if lstm_score is not None and xgb_score is not None:
        return (lstm_score + xgb_score) / 2.0, "Hybrid", lstm_score, xgb_score
    if xgb_score is not None:
        return xgb_score, "XGBoost", None, xgb_score
    if lstm_score is not None:
        return lstm_score, "LSTM", lstm_score, None
    return None, "None", None, None

# ---------------- SHAP safe wrapper ----------------
def safe_shap_explain(x_input, feature_names):
    """
    Create a SHAP explainer and compute values if possible.
    Returns (shap_values, impacts_dict) or (None, error_message)
    """
    if shap is None:
        return None, "shap not installed"
    if xgb_model is None:
        return None, "xgboost model not loaded"
    try:
        expl = shap.Explainer(xgb_model)
        sv = expl(x_input)
        # impacts: simple numeric map
        impacts = {feature_names[i]: float(sv.values[0][i]) for i in range(len(feature_names))}
        return sv, impacts
    except Exception as e:
        return None, f"shap error: {e}"

# ---------------- Gemini helper (safe) ----------------
def gemini_generate(prompt, model_name="gemini-1.5-proto"):
    if not GEMINI_CLIENT or genai is None:
        return None, "Gemini client not installed"
    # read key from secrets or env
    gemini_key = st.secrets.get("GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return None, "Gemini API key missing (set GEMINI_API_KEY in Streamlit Secrets or env)"
    try:
        genai.configure(api_key=gemini_key)
        # modern client varies; we try common safe wrapper
        # try generate_text -> text attribute, else try generate
        try:
            resp = genai.generate_text(model=model_name, prompt=prompt, max_output_tokens=512, temperature=0.2)
            text = getattr(resp, "text", str(resp))
            return text, None
        except Exception:
            # fallback to alternate API (older clients)
            try:
                r2 = genai.create_response(model=model_name, input=prompt)
                text = r2.output_text if hasattr(r2, "output_text") else str(r2)
                return text, None
            except Exception as e:
                return None, f"Gemini call error: {e}"
    except Exception as e:
        return None, f"Gemini config error: {e}"

# ---------------- Execution: when form submitted ----------------
if submit_button:
    ui_vals = {
        "temperature": temp_ui,
        "pressure": pressure_ui,
        "magnetic_field_strength": field_ui,
        "target_density": dens_ui,
        "fuel_density": dens_ui,
        "confinement_time": conf_ui
    }

    # Build canonical DataFrame (fills hidden features)
    full_df = build_full_feature_df(ui_vals)
    st.subheader("Input Snapshot")
    st.table(full_df.T.rename(columns={0: "value"}))

    # Transform (scaler or fallback)
    try:
        transformed = transform_input_df(full_df)
    except Exception as e:
        st.error(f"Input transform failed: {e}")
        transformed = fallback_scale(full_df)

    st.caption(f"Transformed input shape: {transformed.shape}")

    # Predict
    fusion_score, engine_used, lstm_score, xgb_score = hybrid_decision(transformed)

    # Metrics row
    left, mid, right = st.columns([1, 1, 1])
    left.markdown("<div class='big-metric'>{}</div><div class='small subtle'>🔥 Fusion Score</div>".format(f"{fusion_score:.3f}" if fusion_score is not None else "N/A"), unsafe_allow_html=True)
    mid.markdown(f"<div style='font-size:1.6rem;font-weight:700'>{engine_used}</div><div class='small subtle'>Engine used</div>", unsafe_allow_html=True)
    right.markdown(f"<div style='font-size:1.1rem'>{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</div><div class='small subtle'>Timestamp (UTC)</div>", unsafe_allow_html=True)

    if fusion_score is None:
        st.error("Prediction unavailable — no models loaded or runtime error occurred. Check logs.")
    else:
        if fusion_score > 0.5:
            st.success(f"🔥 IGNITION LIKELY — Score {fusion_score:.3f}")
        else:
            st.info(f"❄️ IgnITION UNLIKELY — Score {fusion_score:.3f}")

    # Raw details
    with st.expander("Model details & raw scores", expanded=False):
        st.json({"fusion_score": fusion_score, "engine": engine_used, "lstm_score": lstm_score, "xgb_score": xgb_score})

    # ---------------- SHAP section (optional) ----------------
    if enable_shap:
        st.markdown("## 🔍 SHAP Explanation (XGBoost)")
        # We need a matching feature_names list for shap
        shap_feature_names = list(full_df.columns)
        sv, impacts_or_err = safe_shap_explain(transformed, shap_feature_names)
        if sv is None:
            st.warning(f"SHAP unavailable: {impacts_or_err}")
        else:
            try:
                st.write("Top feature impacts (by absolute SHAP value):")
                df_imp = pd.DataFrame.from_dict(impacts_or_err, orient="index", columns=["impact"])
                df_imp["abs"] = df_imp["impact"].abs()
                st.dataframe(df_imp.sort_values("abs", ascending=False).drop(columns="abs"))
                # Try bar plot
                try:
                    st.pyplot(sv.plots.bar(show=False).figure, bbox_inches="tight")
                except Exception:
                    st.info("SHAP bar plot not renderable in this environment.")
            except Exception as e:
                st.warning(f"Error rendering SHAP: {e}")

    # ---------------- Gemini natural-language explanation ----------------
    if enable_gemini:
        st.markdown("## 🧾 Gemini Natural-Language Explanation")
        # Prepare short prompt summarizing inputs, scores, and top SHAP if available
        shap_summary = {}
        if enable_shap and sv is not None:
            try:
                # Extract top 5 impacts
                impacts = {k: v for k, v in impacts_or_err.items()} if isinstance(impacts_or_err, dict) else {}
                top5 = dict(sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
                shap_summary = top5
            except Exception:
                shap_summary = {}
        prompt = (
            "You are a senior fusion physicist and ML researcher. Provide:\n\n"
            "1) A concise (3-paragraph) explanation of the model prediction and main drivers.\n"
            "2) A 2-sentence operator action blurb (imperative style).\n"
            "3) Three prioritized recommendations to increase ignition probability.\n\n"
            f"INPUTS: {json.dumps(ui_vals, indent=2)}\n"
            f"MODEL SCORES: fusion={fusion_score}, lstm={lstm_score}, xgb={xgb_score}\n"
            f"SHAP_SUMMARY: {json.dumps(shap_summary, indent=2)}\n\n"
            "Write clearly and avoid speculation. Keep each numbered section separated."
        )
        text, err = gemini_generate(prompt)
        if text:
            st.write(text)
        else:
            st.warning(f"Gemini explanation unavailable: {err}")

    # ---------------- Quantum hints (optional) ----------------
    if enable_quantum:
        st.markdown("## ⚛️ Quantum Optimization Hint")
        if QISKIT_AVAILABLE:
            st.info("Quantum hint: consider QAOA/VQE for discrete parameter search. Full runs require Qiskit runtime and task submission.")
        else:
            st.info("Qiskit not installed — quantum hints are conceptual here.")

    # ---------------- Save & export ----------------
    st.markdown("---")
    with st.expander("Save / Export Prediction"):
        export_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "inputs": full_df.to_dict(orient="records")[0],
            "fusion_score": fusion_score,
            "engine": engine_used,
            "lstm_score": lstm_score,
            "xgb_score": xgb_score
        }
        if st.button("Save prediction JSON"):
            fname = f"arc_prediction_{int(time.time())}.json"
            Path(fname).write_text(json.dumps(export_obj, indent=2))
            st.success(f"Saved {fname}")
        st.download_button("Download prediction JSON", data=json.dumps(export_obj, indent=2), file_name="arc_prediction.json", mime="application/json")

    st.success("Execution complete — adjust sliders and re-run for new scenarios.")
# End submit block

# Footer notes
st.markdown(
    """
    <div class='small muted'>
    <strong>Notes:</strong> For best fidelity upload the exact <code>scaler.joblib</code> used during training and ensure the full feature order (as in your training pipeline) is present.
    Add Gemini API key as <code>GEMINI_API_KEY</code> in Streamlit Secrets to enable natural-language explanations.
    This app uses safe fallbacks to avoid runtime crashes if optional dependencies are absent.
    </div>
    """, unsafe_allow_html=True
)
# =================== End Part B ===================
