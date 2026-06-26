# save_model.py
"""Script to serialize the trained house price prediction model and preprocessing objects.

Assumes the following variables are defined in the training script or notebook:
- `model`: Trained regression model (e.g., scikit‑learn estimator)
- `encoder`: Categorical encoder (e.g., OneHotEncoder, OrdinalEncoder) used for feature preprocessing
- `scaler`: Numeric scaler (e.g., StandardScaler, MinMaxScaler) used for feature scaling

The script saves each object to a separate ``.pkl`` file using ``joblib`` for efficient
serialization. Adjust the ``OUTPUT_DIR`` path as needed.
"""
import os
import joblib

# -----------------------------------------------------------------------------
# Configuration – adjust these paths if you prefer a different location
# -----------------------------------------------------------------------------
OUTPUT_DIR = r"C:\Users\aruni\Desktop\price"  # workspace directory
MODEL_FILENAME = "house_price_model.pkl"
ENCODER_FILENAME = "house_price_encoder.pkl"
SCALER_FILENAME = "house_price_scaler.pkl"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Functions to save each component
# -----------------------------------------------------------------------------
def save_pickle(obj, filepath):
    """Serialize ``obj`` to ``filepath`` using joblib.

    Parameters
    ----------
    obj: Any
        Python object to be serialized (must be pickle‑compatible).
    filepath: str
        Destination path including file name.
    """
    joblib.dump(obj, filepath)
    print(f"Saved to {filepath}")

# -----------------------------------------------------------------------------
# Main entry point – replace the placeholder ``model``, ``encoder``, ``scaler``
# with the actual objects from your training session before running this script.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # TODO: Replace the following dummy definitions with your real objects.
    # Example:
    #   from training_module import model, encoder, scaler
    # ---------------------------------------------------------------------
    model = None      # <-- replace with your trained model instance
    encoder = None    # <-- replace with your fitted encoder instance
    scaler = None     # <-- replace with your fitted scaler instance

    if model is None or encoder is None or scaler is None:
        raise ValueError(
            "One or more objects (model, encoder, scaler) are None. "
            "Import or assign the trained objects before calling the script."
        )

    # Save each component
    save_pickle(model, os.path.join(OUTPUT_DIR, MODEL_FILENAME))
    save_pickle(encoder, os.path.join(OUTPUT_DIR, ENCODER_FILENAME))
    save_pickle(scaler, os.path.join(OUTPUT_DIR, SCALER_FILENAME))
