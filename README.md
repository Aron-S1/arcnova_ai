# from your repo root
git fetch --all
git checkout 0aa345a -- app.py README.md
python -m py_compile app.py
git add app.py README.md
git commit -m "Restore clean app.py and README (remove diff markers)"
git push
# arcnova_ai
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
