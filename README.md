# 📚 House‑Price Prediction – Project README

## Table of Contents
1. [Project Overview](#overview)  
2. [Folder Structure](#structure)  
3. [Prerequisites](#prereq)  
4. [Installation](#install)  
5. [Training & Model Serialization](#train)  
6. [Running the Interactive Streamlit App](#streamlit)  
7. [Prediction API (Python)](#api)  
8. [Testing & Evaluation](#test)  
9. [Troubleshooting](#trouble)  
10. [License & Credits](#license)  

---
<a name="overview"></a>
## 1️⃣ Overview  
This repository contains a **complete end‑to‑end machine‑learning pipeline** for predicting Indian house prices from basic property attributes (area, rooms, location, etc.).  

Key capabilities:  

- **Data cleaning & feature engineering** (numeric scaling, categorical encoding).  
- **Regression modeling** (RandomForestRegressor by default, easy to swap).  
- **Model evaluation** (MAE, MSE, RMSE, R²).  
- **Model serialization** (`house_price_model.pkl`, `encoder.pkl`, `scaler.pkl`).  
- **Interactive web UI** built with **Streamlit** for instant predictions.  
- **Reusable Python API** (`src/predict.py`) for programmatic inference.  

---
<a name="structure"></a>
## 2️⃣ Folder Structure  

```
House-Price-Prediction/
│
├── dataset/
│   └── Housing.csv                 # raw data (you can replace with your own CSV)
│
├── models/
│   ├── house_price_model.pkl        # trained regression model
│   ├── encoder.pkl                 # fitted OneHot/Label encoder for categoricals
│   └── scaler.pkl                  # fitted StandardScaler or MinMaxScaler
│
├── notebooks/
│   └── EDA.ipynb                  # exploratory data analysis (optional)
│
├── src/
│   ├── data_preprocessing.py      # helper functions for cleaning & encoding
│   ├── train_and_predict.py       # **single‑script** that trains, evaluates & saves artefacts
│   ├── predict.py                 # thin wrapper that loads artefacts & returns a price
│   └── utils.py                  # misc helpers (logging, path handling)
│
├── app/
│   ├── app.py                     # Streamlit UI (uses src/predict.py)
│   ├── templates/
│   │   └── index.html            # placeholder for possible Flask migration
│   └── static/
│       ├── style.css
│       └── script.js
│
├── requirements.txt               # pinned package versions (see note below)
├── README.md                      # **this file**
└── .gitignore                     # standard ignores (venv, __pycache__, .DS_Store)
```

---
<a name="prereq"></a>
## 3️⃣ Prerequisites  

- **Operating System:** Windows 10/11 (the repo has been tested on Windows PowerShell).  
- **Python:** 3.12 or newer (the latest Windows wheels are available).  
- **Git:** To clone the repo (optional if you already have the files locally).  

---
<a name="install"></a>
## 4️⃣ Installation  

1. **Clone (or download) the repository** and open a PowerShell/Command Prompt in the root folder:

   ```powershell
   cd C:\Users\aruni\Desktop\price
   ```

2. **Create and activate a virtual environment** *(recommended to keep dependencies isolated)*:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1   # PowerShell
   # or
   .venv\Scripts\activate.bat   # CMD
   ```

3. **Upgrade pip and install the required packages**.  
   The original `requirements.txt` used versions that trigger a source‑build for **pandas** and **numpy**, which fails on a clean Windows install (no C‑compiler). The following command forces pip to use only pre‑built wheels that are known to work:

   ```powershell
   pip install --upgrade pip setuptools wheel
   pip install --only-binary=:all: -r requirements.txt
   ```

   **`requirements.txt` (pinned to wheels):**

   ```
   streamlit==1.38.0
   pandas==2.0.3
   numpy==1.26.4
   scikit-learn==1.5.0
   joblib==1.4.2
   ```

   If you already have `requirements.txt` with different versions, feel free to replace its contents with the block above (you can edit the file directly or run `echo` commands).

4. **Verify the installation**:

   ```powershell
   python -c "import streamlit, pandas, numpy, sklearn, joblib; print('All imports OK')"
   ```

---
<a name="train"></a>
## 5️⃣ Training & Model Serialization  

The core script **`src/train_and_predict.py`** performs the full pipeline:

```bash
python src/train_and_predict.py
```

What it does:

| Step | Description |
|------|-------------|
| Load CSV (`dataset/Housing.csv`) | Reads the raw data. |
| Clean & preprocess | Handles missing values, encodes categoricals, scales numerics. |
| Train‑test split | 80 % train / 20 % test (random_state = 42). |
| Model training | `RandomForestRegressor(n_estimators=200, random_state=42)`. |
| Evaluation | Prints MAE, MSE, RMSE, R² on the hold‑out set. |
| Serialization | Saves `house_price_model.pkl`, `encoder.pkl`, `scaler.pkl` into `models/`. |

> **Tip:** If you want to try a different regressor (e.g., `XGBRegressor`), edit the `train_and_predict.py` file – the training section is clearly marked.

---
<a name="streamlit"></a>
## 6️⃣ Running the Interactive Streamlit App  

After the model artefacts are present in `models/`:

```powershell
streamlit run app/app.py
```

*What you’ll see*  

- A clean two‑column form where you can enter **Area, Bedrooms, Bathrooms, Floors, Parking, Location, Furnishing Status, Air Conditioning**.  
- Clicking **🔮 Predict** calls `src/predict.py`, which loads the artefacts, applies the same preprocessing pipeline, and returns a price formatted in Indian rupees (₹).  
- The UI is hosted locally at **http://localhost:8501** – open that URL in any browser.

**Important:** If you modified the default paths for the artefacts, adjust the constants in `app/app.py` (lines 45‑47) accordingly.

---
<a name="api"></a>
## 7️⃣ Programmatic Prediction (Python)  

You can obtain predictions from any Python code without the UI:

```python
from src.predict import predict_price

payload = {
    "Area": 2500,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Floors": 1,
    "Parking": "Yes",
    "Location": "City Center",
    "Furnishing Status": "Furnished",
    "Air Conditioning": "Yes"
}

price = predict_price(payload)
print(f"Estimated price: ₹{price:,.0f}")
```

`src/predict.py` loads the model, encoder, and scaler only once (module‑level cache) for fast repeated calls.

---
<a name="test"></a>
## 8️⃣ Testing & Evaluation  

The repository already contains a **testing plan** (`testing_plan.md`) that verifies:

- Predictions for **small**, **luxury**, and **apartment** examples.  
- Handling of **invalid** or **missing** inputs (Streamlit shows an error message).  
- **Speed** – average response time should stay below 500 ms on a typical laptop.

You can run the automated pytest suite (if you have `pytest` installed):

```powershell
pip install pytest
pytest -q tests/
```

*(Create a `tests/` folder and add the pytest file from the earlier plan if desired.)*

---
<a name="trouble"></a>
## 9️⃣ Troubleshooting  

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `pip install` fails on pandas/numpy with “metadata‑generation‑failed” | No C‑compiler, attempting to build from source. | Use the wheel‑only install command shown in step 4 (`--only-binary=:all:`) and ensure `requirements.txt` pins to versions with wheels (`pandas==2.0.3`, `numpy==1.26.4`). |
| `streamlit` command not found after install | `streamlit` installed to user site‑packages but not on PATH. | Activate the virtual environment (`.venv\Scripts\Activate.ps1`) **before** installing, or run `python -m streamlit run app/app.py`. |
| Prediction returns `KeyError` for a categorical value | The value wasn’t present during encoder fitting. | Add the new category to the training data and re‑train, or use `encoder.set_params(handle_unknown='ignore')` when creating the encoder. |
| UI shows “Trained model not found” | Model artefacts are missing or path mismatch. | Verify `models/house_price_model.pkl` exists and that `app/app.py` constants point to the correct folder. |
| Very slow predictions (> 2 s) | Large model, missing caching. | Ensure the model, encoder, and scaler are loaded **once** at the top of `app/app.py` (already done). If still slow, consider reducing `n_estimators` or switching to a lighter model (e.g., `LinearRegression`). |

---
<a name="license"></a>
## 🔑 License & Credits  

- **License:** MIT – feel free to use, adapt, and share.  
- **Data source:** The `Housing.csv` dataset is a synthetic example; replace with your own data as needed.  
- **Key libraries:** Streamlit, Pandas, NumPy, Scikit‑learn, Joblib.  

---

### 🎉 Ready to go!  

1. Install dependencies (step 4).  
2. Train the model (`python src/train_and_predict.py`).  
3. Launch the UI (`streamlit run app/app.py`).  
4. Open **http://localhost:8501** and start predicting house prices.  

If you encounter any issues or want to extend the project (e.g., add XGBoost, integrate with Flask, or deploy to Streamlit Community Cloud), just let me know—I’m happy to help!
