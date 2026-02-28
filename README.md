# arcnova_ai
 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/README.md b/README.md
index 1284536332205a805216d16e36638fad10dc8a7e..3eede2d5c71f2f3a52ead875658ff3c4f349abd6 100644
--- a/README.md
+++ b/README.md
@@ -1 +1,64 @@
-# arcnova_ai
\ No newline at end of file
+# ArcNova AI
+
+ArcNova AI is a Streamlit application for fusion ignition scenario scoring using a hybrid inference path:
+
+- XGBoost classifier (`arcnova_xgb_model.joblib`)
+- LSTM model (`arcnova_lstm_model.h5`)
+- Optional SHAP explanations for XGBoost
+- Optional Gemini natural-language summary
+- Optional Qiskit conceptual hint panel
+
+The UI is designed to be resilient when optional dependencies or model files are missing.
+
+## Features
+
+- **Engine selection with explicit behavior**
+  - `Hybrid Decision Core`: averages LSTM + XGBoost when both are available, otherwise falls back to whichever model is available.
+  - `XGBoost`: prefers XGBoost and falls back to LSTM if XGBoost is unavailable.
+  - `LSTM`: prefers LSTM and falls back to XGBoost if LSTM is unavailable.
+- **Safe runtime loading** for optional dependencies (`tensorflow`, `shap`, `google-generativeai`, `qiskit`).
+- **Feature construction fallback** if a training schema file is unavailable.
+- **Prediction export** as JSON.
+
+## Project structure
+
+- `app.py` — Streamlit application
+- `requirements.txt` — Python dependencies
+- `arcnova_xgb_model.joblib` — XGBoost model artifact
+- `arcnova_lstm_model.h5` — LSTM model artifact
+- `scaler.joblib` — scaler used for feature transformation
+- `X_train_mean.json` *(optional)* — canonical training feature names/defaults
+
+## Setup
+
+1. Create a virtual environment.
+2. Install dependencies:
+
+```bash
+pip install -r requirements.txt
+```
+
+3. Ensure model artifacts exist in the repo root:
+   - `arcnova_xgb_model.joblib`
+   - `arcnova_lstm_model.h5`
+   - `scaler.joblib`
+
+4. (Optional) Add Gemini API key:
+   - Environment variable: `GEMINI_API_KEY`
+   - or Streamlit secrets with key `GEMINI_API_KEY`
+
+## Run
+
+```bash
+streamlit run app.py
+```
+
+## Notes on model fidelity
+
+Best prediction fidelity requires:
+
+- the exact `scaler.joblib` used during training,
+- a canonical feature schema/order,
+- and, ideally, `X_train_mean.json` containing training feature names/defaults.
+
+Without those, the app uses safe fallbacks intended for robustness over strict scientific fidelity.
 
EOF
)
