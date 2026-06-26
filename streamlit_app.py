'''
streamlit_app.py
================
Interactive Streamlit UI for house‑price prediction.

Features
--------
* Input widgets for all required house attributes.
* Loads a serialized scikit‑learn model, encoder and scaler (saved by `save_model.py`).
* Applies the same preprocessing pipeline used during training.
* Shows the predicted price formatted as Indian rupees (₹).
* Single‑file deployment – run with `streamlit run streamlit_app.py`.

Requirements
------------
`pip install streamlit joblib scikit-learn pandas`
'''"""
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ----------------------------------------------------------------------
# Helper: Load serialized artifacts
# ----------------------------------------------------------------------
def load_artifact(path: str, name: str):
    """Load a joblib‑serialized artifact; display an error if missing.

    Parameters
    ----------
    path: str
        File system path to the .pkl file.
    name: str
        Human‑readable name for error messages.
    """
    if not os.path.exists(path):
        st.error(f"❌ {name} not found at *{path}*. Ensure you have run `save_model.py` first.")
        st.stop()
    return joblib.load(path)

# ----------------------------------------------------------------------
# Paths – adjust if you saved the files elsewhere
# ----------------------------------------------------------------------
MODEL_PATH = "house_price_model.pkl"
ENCODER_PATH = "encoder.pkl"
SCALER_PATH = "scaler.pkl"

# Load artifacts once when the script starts
model = load_artifact(MODEL_PATH, "Trained model")
encoder = load_artifact(ENCODER_PATH, "Feature encoder")
scaler = load_artifact(SCALER_PATH, "Feature scaler")

# ----------------------------------------------------------------------
# Streamlit page configuration
# ----------------------------------------------------------------------
st.set_page_config(page_title="House Price Predictor", layout="centered", page_icon="🏠")

st.title("🏠 House‑Price Prediction")
st.markdown(
    "Enter the property's characteristics below and click **Predict** to see the estimated price in Indian rupees (₹)."
)

# ----------------------------------------------------------------------
# Input widgets
# ----------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=0, value=2500, step=10)
    bedrooms = st.selectbox("Bedrooms", options=[1, 2, 3, 4, 5], index=2)
    bathrooms = st.selectbox("Bathrooms", options=[1, 2, 3, 4], index=1)

with col2:
    parking = st.selectbox("Parking", options=["Yes", "No"], index=0)
    location = st.selectbox(
        "Location",
        options=["City Center", "Suburb", "Rural", "Tech Park", "Waterfront"],
        index=0,
    )
    floors = st.number_input("Floors", min_value=0, max_value=10, value=1, step=1)
    furnishing = st.selectbox(
        "Furnishing Status",
        options=["Furnished", "Semi‑Furnished", "Unfurnished"],
        index=0,
    )
    ac = st.selectbox("Air‑Conditioning", options=["Yes", "No"], index=0)

# ----------------------------------------------------------------------
# Prediction logic
# ----------------------------------------------------------------------
if st.button("🔮 Predict", type="primary"):
    # Build a DataFrame that matches the training schema
    input_data = {
        "Area": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Parking": parking,
        "Location": location,
        "Floors": floors,
        "Furnishing Status": furnishing,
        "Air Conditioning": ac,
    }
    df = pd.DataFrame([input_data])

    # Identify which columns are categorical / numerical – adjust if your pipeline differs
    cat_cols = ["Parking", "Location", "Furnishing Status", "Air Conditioning"]
    num_cols = ["Area", "Bedrooms", "Bathrooms", "Floors"]

    # Transform using the saved encoder and scaler
    try:
        X_cat = encoder.transform(df[cat_cols])
        X_num = scaler.transform(df[num_cols])
    except Exception as e:
        st.error(f"❗ Pre‑processing failed: {e}")
        st.stop()

    # Concatenate numeric and encoded categorical features
    X = np.hstack([X_num, X_cat])

    # Predict
    try:
        pred = model.predict(X)[0]
    except Exception as e:
        st.error(f"❗ Prediction failed: {e}")
        st.stop()

    # Format as Indian currency
    pred_formatted = f"₹{pred:,.0f}"
    st.success(f"**Estimated Price:** {pred_formatted}")

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.caption("Powered by Streamlit • © 2026")
