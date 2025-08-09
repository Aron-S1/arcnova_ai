# 🌌 ArcNova — Celestial Intelligence Mode (Legendary Build)
# Hybrid XAI + Gemini explanations | LSTM + XGBoost core | Professional UI

import os
import time
import json
import joblib
import shap
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path

# Optional imports
try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import qiskit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="ArcNova — Celestial Mode",
                   layout="wide",
                   initial_sidebar_state="expanded")
st.title("🌌 ArcNova — Celestial Intelligence Mode")
st.caption("Hybrid XAI + Gemini explanations | LSTM + XGBoost core | Professional Control Panel")

# ---------------- PATHS ----------------
MODEL_LSTM_PATH = "arcnova_lstm_model.h5"
MODEL_XGB_PATH = "arcnova_xgb_model.joblib"
SCALER_PATH = "scaler.joblib"

# FULL FEATURE LIST FROM TRAINING
FEATURE_NAMES = [
    "temperature", "pressure", "magnetic_field_strength", "target_density",
    "fuel_density", "confinement_time",
    # Hidden features from training (defaults auto-filled)
    "feat7", "feat8", "feat9", "feat10", "feat11", "feat12",
    "feat13", "feat14", "feat15", "feat16", "feat17", "feat18",
    "feat19", "feat20", "feat21"
]

# ---------------- LOADERS ----------------
@st.cache_resource(show_spinner=False)
def load_xgb_model():
    try:
        return joblib.load(MODEL_XGB_PATH)
    except Exception as e:
        st.error(f"XGB load failed: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_lstm_model():
    if tf is None:
        return None
    try:
        return tf.keras.models.load_model(MODEL_LSTM_PATH)
    except Exception as e:
        st.error(f"LSTM load failed: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_scaler():
    try:
        return joblib.load(SCALER_PATH)
    except Exception:
        return None

# ---------------- GEMINI ----------------
def gemini_configure():
    if not GEMINI_AVAILABLE:
        return False, "Gemini not installed"
    key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not key:
        return False, "No GEMINI_API_KEY found"
    try:
        genai.configure(api_key=key)
        return True, "Gemini configured"
    except Exception as e:
        return False, str(e)

def gemini_explain(prompt):
    if not GEMINI_AVAILABLE:
        return None, "Gemini not installed"
    ok, msg = gemini_configure()
    if not ok:
        return None, msg
    try:
        resp = genai.generate_text(model="gemini-pro", prompt=prompt, max_output_tokens=512, temperature=0.2)
        return getattr(resp, "text", str(resp)), None
    except Exception as e:
        return None, str(e)

# ---------------- HELPERS ----------------
def build_feature_vector(ui_vals, scaler=None):
    # Fill visible sliders
    base = {
        "temperature": ui_vals["temperature"],
        "pressure": ui_vals["pressure"],
        "magnetic_field_strength": ui_vals["magnetic_field_strength"],
        "target_density": ui_vals["target_density"],
        "fuel_density": ui_vals["fuel_density"],
        "confinement_time": ui_vals["confinement_time"]
    }
    # Fill hidden features with defaults (mean=0 placeholder, adjust if needed)
    for f in FEATURE_NAMES:
        if f not in base:
            base[f] = 0.0
    df = pd.DataFrame([base])[FEATURE_NAMES]
    return scaler.transform(df) if scaler else df.to_numpy()

def hybrid_decision(x_input, xgb_model, lstm_model):
    xgb_score, lstm_score = None, None
    if xgb_model:
        try:
            xgb_score = float(xgb_model.predict_proba(x_input)[0][1])
        except Exception as e:
            st.warning(f"XGB error: {e}")
    if lstm_model:
        try:
            seq = np.tile(x_input, (1, 5, 1))
            lstm_score = float(lstm_model.predict(seq)[0][0])
        except Exception as e:
            st.warning(f"LSTM error: {e}")
    if lstm_score is not None and xgb_score is not None:
        return (lstm_score + xgb_score) / 2, "Hybrid", lstm_score, xgb_score
    elif xgb_score is not None:
        return xgb_score, "XGBoost", None, xgb_score
    elif lstm_score is not None:
        return lstm_score, "LSTM", lstm_score, None
    return None, "None", None, None

# ---------------- SIDEBAR ----------------
st.sidebar.header("Control Panel")
with st.sidebar.form("controls"):
    temp_ui = st.slider("Plasma Temp (M K)", 50, 400, 150, step=5)
    press_ui = st.slider("Pressure (atm)", 1.0, 10.0, 3.0, step=0.1)
    field_ui = st.slider("Magnetic Field (T)", 1.0, 12.0, 5.0, step=0.1)
    dens_ui = st.slider("Fuel Density (g/cm³)", 0.1, 3.0, 1.0, step=0.01)
    conf_ui = st.slider("Confinement Time (s)", 0.5, 60.0, 10.0, step=0.5)
    model_choice = st.selectbox("AI Engine", ["Hybrid Decision Core", "LSTM", "XGBoost"])
    use_gemini = st.checkbox("Enable Gemini explanations", value=True)
    use_quantum = st.checkbox("Enable Quantum Mode (Qiskit)", value=False)
    submitted = st.form_submit_button("Run Prediction & Explain")

colA, colB, colC = st.columns(3)
colA.metric("Run timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
colB.metric("Gemini available", "Yes" if GEMINI_AVAILABLE else "No")
colC.metric("Qiskit available", "Yes" if QISKIT_AVAILABLE else "No")
st.markdown("---")
# ---------------- Part 2: Prediction, XAI, Gemini, Export (Paste below Part 1) ----------------

# Load models & scaler (non-fatal)
xgb_model = load_xgb_model()
lstm_model = load_lstm_model()
scaler = load_scaler()

# Safe fallback scaler (deterministic)
def fallback_scale_array(df: pd.DataFrame) -> np.ndarray:
    """Deterministic fallback scaling used when scaler.joblib is absent.
       This keeps magnitudes reasonable and matches earlier project heuristics."""
    vals = []
    row = df.iloc[0]
    for c in FEATURE_NAMES:
        v = float(row[c])
        if c == "temperature":
            # UI inputs are in million K; keep centred at 150
            vals.append((v - 150.0) / 50.0)
        elif c == "pressure":
            vals.append((v - 3.0) / 1.0)
        elif c == "magnetic_field_strength":
            vals.append((v - 5.0) / 2.0)
        elif "density" in c:
            vals.append((v - 1.0) / 0.5)
        elif c == "confinement_time":
            vals.append((v - 10.0) / 5.0)
        else:
            # Hidden features default 0 -> stays 0
            vals.append(v)
    return np.array([vals], dtype=float)

# Helper to transform input using either real scaler or fallback
def transform_input_df(df: pd.DataFrame):
    if scaler is not None:
        try:
            arr = scaler.transform(df)
            # If scaler was trained on 21 features it returns shape (1,21)
            return arr
        except Exception as e:
            st.warning(f"Scaler transform failed, using fallback scaling: {e}")
            return fallback_scale_array(df)
    else:
        # No scaler file => deterministic fallback
        return fallback_scale_array(df)

# SHAP explainer builder (do not cache with unhashable object)
def build_shap_explainer_safe():
    """Create a SHAP explainer for XGBoost if available.
       Avoid passing the raw model into a cached function (unhashable issues)."""
    if shap is None:
        return None, "SHAP not installed"
    if xgb_model is None:
        return None, "XGBoost model not loaded"
    try:
        expl = shap.Explainer(xgb_model)  # no caching of model obj
        return expl, None
    except Exception as e:
        return None, f"SHAP init error: {e}"

# UI execution flow
if submitted:
    # Assemble UI input dict
    ui_vals = {
        "temperature": float(temp_ui),
        "pressure": float(press_ui),
        "magnetic_field_strength": float(field_ui),
        "target_density": float(dens_ui),
        "fuel_density": float(dens_ui),
        "confinement_time": float(conf_ui)
    }

    # Build full-21-feature DataFrame with hidden defaults (0.0)
    full_df = pd.DataFrame([{**ui_vals}])
    for feat in FEATURE_NAMES:
        if feat not in full_df.columns:
            full_df[feat] = 0.0
    # Reorder strictly to FEATURE_NAMES
    full_df = full_df[FEATURE_NAMES]

    st.header("Input Snapshot")
    st.table(full_df.T.rename(columns={0: "value"}))

    # Transform
    x_input = transform_input_df(full_df)  # numpy array shape (1,21 expected)

    st.caption(f"Transformed input shape: {x_input.shape}")

    # Prediction decision core
    fusion_score, engine, lstm_score, xgb_score = None, "None", None, None
    try:
        fusion_score, engine, lstm_score, xgb_score = hybrid_decision(x_input, xgb_model, lstm_model)
    except Exception as e:
        st.error(f"Prediction core failed: {e}")
        st.write(traceback.format_exc())

    # Present metrics
    st.markdown("---")
    mcol1, mcol2, mcol3 = st.columns([1.2, 1.0, 1.0])
    mcol1.metric("🔥 Fusion Score", f"{fusion_score:.3f}" if fusion_score is not None else "N/A")
    mcol2.metric("Engine", engine)
    mcol3.metric("Timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))

    if fusion_score is None:
        st.warning("Model predictions unavailable (check model files and feature alignment).")
    else:
        if fusion_score > 0.5:
            st.success(f"🔥 IGNITION LIKELY — Score {fusion_score:.3f}")
        else:
            st.error(f"❄️ IgnITION UNLIKELY — Score {fusion_score:.3f}")

    # Expanded details
    with st.expander("Model details & raw scores", expanded=True):
        st.write({"fusion_score": fusion_score, "engine": engine, "lstm_score": lstm_score, "xgb_score": xgb_score})

    # SHAP explanation (safe)
    if show_shap and xgb_model is not None and shap is not None:
        st.markdown("## 🔍 SHAP Explanation (XGBoost)")
        explainer, expl_err = build_shap_explainer_safe()
        if explainer is not None:
            try:
                sv, impacts = shap_outputs_for_input(explainer, x_input, FEATURE_NAMES)
                # Render bar (non-blocking)
                try:
                    st.write("### Feature impacts (bar)")
                    shap.plots.bar(sv, show=False)
                    st.pyplot(bbox_inches="tight")
                except Exception:
                    st.write("SHAP bar rendering fallback (matplotlib may not render in this environment).")
                # Show ranked table
                st.table(pd.DataFrame.from_dict(impacts, orient="index", columns=["Impact"]).sort_values("Impact", ascending=False))
            except Exception as e:
                st.warning(f"SHAP computation failed: {e}")
        else:
            st.info(f"SHAP unavailable: {expl_err}")
    elif show_shap:
        st.info("SHAP not available (requires shap package and XGBoost model).")

    # Gemini-powered nat-lang explanation (if available)
    if use_gemini:
        st.markdown("## 🧾 Gemini Natural-Language Explanation")
        # build prompt
        try:
            impacts_small = {}
            if shap is not None and xgb_model is not None:
                expl, _ = build_shap_explainer_safe()
                if expl is not None:
                    sv, impacts_small = shap_outputs_for_input(expl, x_input, FEATURE_NAMES)
            prompt = (
                "You are a senior fusion physicist and ML researcher.\n"
                f"INPUTS: {json.dumps(full_df.to_dict(orient='records')[0], indent=2)}\n"
                f"MODEL SCORES: fusion_score={fusion_score}, lstm={lstm_score}, xgb={xgb_score}\n"
                f"SHAP IMPACTS: {json.dumps(impacts_small, indent=2)}\n\n"
                "Write: (A) A concise 3-paragraph explanation for an academic reviewer. "
                "(B) A 2-sentence operator action blurb. (C) Three prioritized tips to improve ignition probability."
            )
            explanation_text, gem_err = gemini_explain(prompt)
            if explanation_text:
                st.markdown("### Gemini Explanation")
                st.write(explanation_text)
            else:
                st.warning(f"Gemini not available: {gem_err}")
        except Exception as e:
            st.error(f"Gemini step failed: {e}")

    # Quantum hint (light)
    if use_quantum:
        st.markdown("## ⚛️ Quantum Optimization Hint")
        if QISKIT_AVAILABLE:
            st.info("Quantum hint: consider QAOA/VQE for discrete parameter search. Full run requires Qiskit runtime.")
        else:
            st.info("Qiskit not installed — quantum mode is conceptual here.")

    # Save/export
    st.markdown("---")
    with st.expander("Save / Export Prediction"):
        if st.button("Save prediction JSON"):
            out = {
                "timestamp": datetime.utcnow().isoformat(),
                "inputs": full_df.to_dict(orient="records")[0],
                "fusion_score": fusion_score,
                "engine": engine,
                "lstm_score": lstm_score,
                "xgb_score": xgb_score,
            }
            fname = f"arc_prediction_{int(time.time())}.json"
            Path(fname).write_text(json.dumps(out, indent=2))
            st.success(f"Saved {fname}")
            st.download_button("Download prediction JSON", data=json.dumps(out, indent=2), file_name=fname, mime="application/json")

    st.success("Execution complete — adjust sliders and re-run for new scenarios.")
    st.balloons()

# Footer help note
st.markdown(
    """
    <div style="color:#99b; font-size:0.9rem">
    <strong>Notes:</strong> This app automatically supplies hidden features with safe defaults so predictions remain stable.
    For highest-fidelity results, upload the exact `scaler.joblib` used during training and ensure the full 21-feature order matches the training pipeline.
    Add Gemni key as `GEMINI_API_KEY` in Streamlit Secrets for natural-language explanations.
    </div>
    """, unsafe_allow_html=True
    )
    # ---------------- Debug/Admin Tools ----------------

st.markdown("---")
with st.expander("🔧 Debug / Admin Tools (for developer use)", expanded=False):
    debug_mode = st.checkbox("Enable developer debug mode", value=False)
    if debug_mode:
        st.info("Developer debug mode active — sensitive model info may be displayed.")

        # Show expected feature order
        st.markdown("### 📋 Expected Feature Order")
        st.write(FEATURE_NAMES)

        # Show scaler statistics if available
        if scaler is not None:
            try:
                st.markdown("### 📊 Scaler Stats")
                if hasattr(scaler, "mean_"):
                    st.write("**Means:**", list(scaler.mean_))
                if hasattr(scaler, "scale_"):
                    st.write("**Scales:**", list(scaler.scale_))
            except Exception as e:
                st.warning(f"Could not read scaler stats: {e}")
        else:
            st.warning("No scaler file loaded — using fallback scaling.")

        # Button to check transformed feature vector
        st.markdown("### 🧪 Check Features")
        if st.button("Show transformed vector for current inputs"):
            # Build the full input row with defaults
            ui_vals = {
                "temperature": float(temp_ui),
                "pressure": float(press_ui),
                "magnetic_field_strength": float(field_ui),
                "target_density": float(dens_ui),
                "fuel_density": float(dens_ui),
                "confinement_time": float(conf_ui)
            }
            full_df = pd.DataFrame([{**ui_vals}])
            for feat in FEATURE_NAMES:
                if feat not in full_df.columns:
                    full_df[feat] = 0.0
            full_df = full_df[FEATURE_NAMES]

            st.write("**Unscaled vector:**")
            st.write(full_df)

            scaled_vec = transform_input_df(full_df)
            st.write("**Scaled vector (fed to models):**")
            st.write(scaled_vec)
    
        

    
