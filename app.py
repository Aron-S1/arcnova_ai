# from your repo root
git fetch --all
git checkout 0aa345a -- app.py README.md
python -m py_compile app.py
git add app.py README.md
git commit -m "Restore clean app.py and README (remove diff markers)"
git push
# =================== ArcNova — Celestial Mode (Part A) ===================
 # Paste Part A first into app.py
 # Modern, defensive Streamlit UI — prepares app, loads resources safely.
 
 import os
 import time
 import json
-import math
-import traceback
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
@@ -56,51 +54,62 @@ st.markdown(
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
-# We'll extend to hidden features automatically if scaler/model expects more.
+
+
+def infer_expected_feature_count():
+    """Infer expected feature count from available artifacts, with conservative fallback."""
+    candidates = []
+    if X_train_mean:
+        candidates.append(len(X_train_mean.keys()))
+    if scaler is not None and hasattr(scaler, "n_features_in_"):
+        candidates.append(int(scaler.n_features_in_))
+    if xgb_model is not None and hasattr(xgb_model, "n_features_in_"):
+        candidates.append(int(xgb_model.n_features_in_))
+    return max(candidates) if candidates else 6
 
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
@@ -195,207 +204,263 @@ st.write(" ")
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
-        # Otherwise use existing FEATURE_NAMES and ensure at least 6 + hidden placeholders
+        # Otherwise use existing FEATURE_NAMES and infer needed count from loaded artifacts
         canonical = FEATURE_NAMES.copy()
-        # ensure minimum 21 features if original project expected many features
-        if len(canonical) < 21:
-            # append placeholder feature names feat7..feat21
-            for i in range(len(canonical)+1, 22):
+        expected_count = infer_expected_feature_count()
+        if len(canonical) < expected_count:
+            # append placeholder feature names featN up to expected feature count
+            for i in range(len(canonical) + 1, expected_count + 1):
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
-def hybrid_decision(x_input):
+def hybrid_decision(x_input, selected_engine):
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
 
-    # choose engine
-    if lstm_score is not None and xgb_score is not None:
-        return (lstm_score + xgb_score) / 2.0, "Hybrid", lstm_score, xgb_score
-    if xgb_score is not None:
-        return xgb_score, "XGBoost", None, xgb_score
-    if lstm_score is not None:
-        return lstm_score, "LSTM", lstm_score, None
+    # choose engine based on UI selection
+    if selected_engine == "Hybrid Decision Core":
+        if lstm_score is not None and xgb_score is not None:
+            return (lstm_score + xgb_score) / 2.0, "Hybrid", lstm_score, xgb_score
+        if xgb_score is not None:
+            return xgb_score, "XGBoost (fallback from Hybrid)", None, xgb_score
+        if lstm_score is not None:
+            return lstm_score, "LSTM (fallback from Hybrid)", lstm_score, None
+
+    if selected_engine == "XGBoost":
+        if xgb_score is not None:
+            return xgb_score, "XGBoost", None, xgb_score
+        if lstm_score is not None:
+            return lstm_score, "LSTM (fallback from XGBoost)", lstm_score, None
+
+    if selected_engine == "LSTM":
+        if lstm_score is not None:
+            return lstm_score, "LSTM", lstm_score, None
+        if xgb_score is not None:
+            return xgb_score, "XGBoost (fallback from LSTM)", None, xgb_score
+
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
+def resolve_gemini_api_key():
+    """Resolve Gemini key from Streamlit secrets or environment variables."""
+    secret_val = None
+    try:
+        # st.secrets can raise when secrets file is missing in some runtimes.
+        secret_val = st.secrets.get("GEMINI_API_KEY", None)
+    except Exception:
+        secret_val = None
+
+    env_val = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
+    return secret_val or env_val
+
+
 def gemini_generate(prompt, model_name="gemini-1.5-flash"):
     """
     Unified Gemini client wrapper — works with both old and new google-generativeai APIs.
     Falls back gracefully without crashing.
     """
     if not GEMINI_CLIENT or genai is None:
         return None, "Gemini client not installed"
 
     # Get API key
-    gemini_key = st.secrets.get("GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY")
+    gemini_key = resolve_gemini_api_key()
     if not gemini_key:
-        return None, "Gemini API key missing (set GEMINI_API_KEY in Streamlit Secrets or env)"
+        return None, "Gemini API key missing (set GEMINI_API_KEY/GOOGLE_API_KEY in Streamlit Secrets or env)"
 
     try:
         genai.configure(api_key=gemini_key)
 
-        # Try NEW API first (post-May 2024 versions)
-        try:
-            if hasattr(genai, "GenerativeModel"):
-                model = genai.GenerativeModel(model_name)
-                resp = model.generate_content(prompt)
-                if hasattr(resp, "text"):
-                    return resp.text, None
-                elif hasattr(resp, "candidates"):
-                    # Some versions store output in candidates
-                    return resp.candidates[0].content.parts[0].text, None
-                else:
+        # Allow model override and fallback through a few known names across SDK/account states.
+        env_model = os.getenv("GEMINI_MODEL")
+        model_candidates = []
+        for candidate in [env_model, model_name, "gemini-1.5-flash-latest", "gemini-1.5-pro", "gemini-2.0-flash"]:
+            if candidate and candidate not in model_candidates:
+                model_candidates.append(candidate)
+
+        # Try NEW API first (current SDK style)
+        new_api_errors = []
+        if hasattr(genai, "GenerativeModel"):
+            for candidate_model in model_candidates:
+                try:
+                    model = genai.GenerativeModel(candidate_model)
+                    resp = model.generate_content(prompt)
+                    if hasattr(resp, "text") and resp.text:
+                        return resp.text, None
+
+                    # Some versions store output in candidates/parts
+                    candidates = getattr(resp, "candidates", None) or []
+                    if candidates:
+                        content = getattr(candidates[0], "content", None)
+                        parts = getattr(content, "parts", None) if content else None
+                        if parts:
+                            first = parts[0]
+                            text = getattr(first, "text", None)
+                            if text:
+                                return text, None
+
+                    # Return a readable fallback if model responded but without plain text.
                     return str(resp), None
-        except Exception:
-            pass
+                except Exception as e:
+                    new_api_errors.append(f"{candidate_model}: {e}")
 
-        # Try OLD API next
-        try:
-            if hasattr(genai, "generate_text"):
-                resp = genai.generate_text(model=model_name, prompt=prompt,
-                                           max_output_tokens=512, temperature=0.2)
-                return getattr(resp, "text", str(resp)), None
-        except Exception:
-            pass
-
-        return None, "Gemini API call methods not supported by this version"
+        # Try OLD API next (legacy compatibility)
+        old_api_errors = []
+        if hasattr(genai, "generate_text"):
+            for candidate_model in model_candidates:
+                try:
+                    resp = genai.generate_text(
+                        model=candidate_model,
+                        prompt=prompt,
+                        max_output_tokens=512,
+                        temperature=0.2,
+                    )
+                    text = getattr(resp, "text", None)
+                    return (text if text else str(resp)), None
+                except Exception as e:
+                    old_api_errors.append(f"{candidate_model}: {e}")
+
+        return None, (
+            "Gemini API call failed for all attempted models. "
+            f"new_api_errors={'; '.join(new_api_errors) if new_api_errors else 'n/a'}; "
+            f"old_api_errors={'; '.join(old_api_errors) if old_api_errors else 'n/a'}"
+        )
     except Exception as e:
         return None, f"Gemini config/call error: {e}"
 
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
-    fusion_score, engine_used, lstm_score, xgb_score = hybrid_decision(transformed)
+    fusion_score, engine_used, lstm_score, xgb_score = hybrid_decision(transformed, engine_choice)
 
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
-            st.info(f"❄️ IgnITION UNLIKELY — Score {fusion_score:.3f}")
+            st.info(f"❄️ IGNITION UNLIKELY — Score {fusion_score:.3f}")
 
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
