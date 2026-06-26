# src/train_and_predict.py
"""
House‑Price Prediction Pipeline
================================
This script demonstrates a complete workflow:

1. Load the raw CSV (dataset/Housing.csv).
2. Clean the data (drop missing rows, strip whitespace).
3. Encode categorical features with One‑Hot Encoding.
4. Scale numeric features with StandardScaler.
5. Optional feature selection (SelectKBest based on f_regression).
6. Split into train / test sets.
7. Train a RandomForestRegressor.
8. Compute regression metrics (MAE, MSE, RMSE, R²).
9. Serialize the trained model, encoder, and scaler to `models/`.
10. Provide a reusable `predict_price(payload: dict) -> float` helper.

Run the script directly to train and save artefacts, then use the
`predict_price` function (importable from `src.predict.py`) in your UI.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------------------------------------
# Paths (adjust if you change the folder layout)
# ------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "Housing.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# ------------------------------------------------------------------
# 1️⃣ Load data
# ------------------------------------------------------------------
print("Loading data …")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Raw rows: {df.shape[0]}, columns: {df.shape[1]}")

# ------------------------------------------------------------------
# 2️⃣ Basic cleaning
# ------------------------------------------------------------------
# Drop rows with any missing values (for simplicity)
df = df.dropna().reset_index(drop=True)
print(f"After dropping NA: {df.shape[0]} rows")

# ------------------------------------------------------------------
# 3️⃣ Define feature / target columns
# ------------------------------------------------------------------
# Expected column names – adjust if your CSV differs
TARGET_COL = "Price"  # target column name in the CSV
# All other columns are considered features
FEATURE_COLS = [c for c in df.columns if c != TARGET_COL]

# Separate numeric vs categorical columns
numeric_cols = df[FEATURE_COLS].select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in FEATURE_COLS if c not in numeric_cols]
print(f"Numeric: {numeric_cols}\nCategorical: {categorical_cols}")

# ------------------------------------------------------------------
# 4️⃣ Pre‑processing pipeline
# ------------------------------------------------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# Fit the preprocessing on the whole dataset (we’ll split after transformation)
X_processed = preprocess.fit_transform(X)

# Save encoder & scaler for later use (they are stored inside the ColumnTransformer)
joblib.dump(preprocess, ENCODER_PATH)
print(f"Pre‑processing pipeline saved to {ENCODER_PATH}")

# ------------------------------------------------------------------
# 5️⃣ Optional feature selection (keep top 20 features)
# ------------------------------------------------------------------
k_best = 20
selector = SelectKBest(score_func=f_regression, k=min(k_best, X_processed.shape[1]))
X_selected = selector.fit_transform(X_processed, y)
joblib.dump(selector, os.path.join(MODEL_DIR, "selector.pkl"))
print(f"Feature selector saved (top {k_best})")

# ------------------------------------------------------------------
# 6️⃣ Train / test split
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# 7️⃣ Model training (RandomForestRegressor – good default for tabular data)
# ------------------------------------------------------------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
print("Model trained.")

# ------------------------------------------------------------------
# 8️⃣ Evaluation
# ------------------------------------------------------------------
pred_test = model.predict(X_test)
mae = mean_absolute_error(y_test, pred_test)
mse = mean_squared_error(y_test, pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred_test)
print("--- Evaluation Metrics ---")
print(f"MAE : {mae:,.2f}")
print(f"MSE : {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²   : {r2:.4f}")

# ------------------------------------------------------------------
# 9️⃣ Serialize model
# ------------------------------------------------------------------
joblib.dump(model, MODEL_PATH)
print(f"Trained model saved to {MODEL_PATH}")

# ------------------------------------------------------------------
# 10️⃣ Helper for inference – importable from other modules
# ------------------------------------------------------------------
def _load_pipeline():
    """Load the preprocessing pipeline, selector and model.
    Returns (preprocess, selector, model)."""
    preprocess = joblib.load(ENCODER_PATH)
    selector = joblib.load(os.path.join(MODEL_DIR, "selector.pkl"))
    model = joblib.load(MODEL_PATH)
    return preprocess, selector, model

def predict_price(payload: dict) -> float:
    """Predict house price from a dictionary of feature values.
    The keys must match the original feature column names used during training.
    """
    preprocess, selector, model = _load_pipeline()
    # Convert payload to DataFrame (single row)
    df_input = pd.DataFrame([payload])
    # Apply same preprocessing
    X_proc = preprocess.transform(df_input)
    X_sel = selector.transform(X_proc)
    return float(model.predict(X_sel)[0])

if __name__ == "__main__":
    # Simple CLI for quick testing
    import json, sys
    if len(sys.argv) > 1:
        # Expect a JSON string as the first argument
        payload = json.loads(sys.argv[1])
    else:
        # Example payload (replace with real values)
        payload = {
            "Area": 2500,
            "Bedrooms": 3,
            "Bathrooms": 2,
            "Floors": 1,
            "Parking": "Yes",
            "Location": "City Center",
            "Furnishing Status": "Furnished",
            "Air Conditioning": "Yes",
        }
    print("Predicted price:", predict_price(payload))
