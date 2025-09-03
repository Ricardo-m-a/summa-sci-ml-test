# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Union
import joblib
import pandas as pd
import numpy as np
import os
import traceback
from fastapi.responses import JSONResponse

app = FastAPI(title="Cementos Argos Demand API")

# ------------------ ARTEFACTOS ------------------
MODELS_DIR = "./models/classifier"
model = joblib.load(os.path.join(MODELS_DIR, "ensemble_model.pkl"))
encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
threshold_dict = joblib.load(os.path.join(MODELS_DIR, "optimal_threshold.pkl"))
feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
mapped_cols = joblib.load(os.path.join(MODELS_DIR, "mapped_cols.pkl")) if os.path.exists(os.path.join(MODELS_DIR, "mapped_cols.pkl")) else []
numeric_cols = joblib.load(os.path.join(MODELS_DIR, "numeric_cols.pkl")) if os.path.exists(os.path.join(MODELS_DIR, "numeric_cols.pkl")) else []
group_stats_map = joblib.load(os.path.join(MODELS_DIR, "group_stats_map.pkl")) if os.path.exists(os.path.join(MODELS_DIR, "group_stats_map.pkl")) else {}

optimal_threshold = threshold_dict.get("threshold", 0.5)

# ------------------ AUXILIAR ------------------
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create numeric interaction features (ratios and products).
    NOTE: group-based aggregates are NOT computed here — they will be filled
    from `group_stats_map` to preserve training-time statistics.
    """
    df_eng = df.copy()
    numeric_here = df_eng.select_dtypes(include=[np.number]).columns
    if len(numeric_here) >= 2:
        for i, col1 in enumerate(numeric_here):
            for col2 in numeric_here[i+1:]:
                # Ratio and product (avoid division by zero)
                df_eng[f'{col1}_div_{col2}'] = df_eng[col1] / (df_eng[col2] + 1e-8)
                df_eng[f'{col1}_x_{col2}'] = df_eng[col1] * df_eng[col2]
    return df_eng

def map_yes_no_val(x):
    if pd.isna(x):
        return np.nan
    # If already numeric-ish
    if isinstance(x, (int, float, np.integer, np.floating)):
        return int(x)
    s = str(x).strip().upper()
    if s in ("YES", "Y", "SI", "TRUE", "1"):
        return 1
    if s in ("NO", "N", "FALSE", "0"):
        return 0
    # try convert comma thousands or numeric string
    try:
        return int(float(s.replace(",", "")))
    except:
        return x  # return original (will be handled later)

def _find_group_map_key(derived_col: str, group_map: dict):
    """
    Try to find the best matching key in group_stats_map for a derived_col.
    Tries exact match, then removes common suffixes (_x/_y), etc.
    """
    if derived_col in group_map:
        return derived_col
    # remove _x/_y variants in the cat part
    if "_by_" in derived_col:
        prefix, catpart = derived_col.split("_by_", 1)
        cat_clean = catpart.replace("_x", "").replace("_y", "")
        candidate = prefix + "_by_" + cat_clean
        if candidate in group_map:
            return candidate
    # fallback: try remove any trailing _x/_y anywhere
    alt = derived_col.replace("_x", "").replace("_y", "")
    if alt in group_map:
        return alt
    return None

# ------------------ SCHEMA (campos base) ------------------
class RequestData(BaseModel):
    autoID: Optional[str] = None
    SeniorCity: Optional[Union[int, str]] = None
    Partner: Optional[Union[int, str]] = None
    Dependents: Optional[Union[int, str]] = None
    Service1: Optional[Union[int, str]] = None
    Service2: Optional[Union[int, str]] = None
    Security: Optional[Union[int, str]] = None
    OnlineBackup: Optional[Union[int, str]] = None
    DeviceProtection: Optional[Union[int, str]] = None
    TechSupport: Optional[Union[int, str]] = None
    Contract: Optional[str] = None
    PaperlessBilling: Optional[Union[int, str]] = None
    PaymentMethod: Optional[str] = None
    Charges: Optional[Union[float, str]] = None
    Demand: Optional[Union[float, str]] = None

# ------------------ ENDPOINTS ------------------
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Cementos Argos Demand API running"}

@app.post("/predict")
def predict(request: RequestData):
    try:
        # 1) Build DataFrame from request (raw user format)
        input_dict = request.dict()
        input_df = pd.DataFrame([input_dict])

        # 2) Map binary yes/no columns detected at training
        for c in mapped_cols:
            if c in input_df.columns:
                input_df[c] = input_df[c].apply(map_yes_no_val)

        # 3) Clean Charges/Demand strings with commas
        for col in ["Charges", "Demand"]:
            if col in input_df.columns:
                v = input_df[col].iloc[0]
                if isinstance(v, str):
                    cleaned = v.replace(",", "")
                    input_df[col] = pd.to_numeric(cleaned, errors='coerce')

        # 4) Force numeric columns (from training) to numeric BEFORE create_features
        if numeric_cols:
            for col in numeric_cols:
                if col in input_df.columns:
                    input_df[col] = pd.to_numeric(input_df[col].astype(str).str.replace(",", ""), errors='coerce')

        # 5) Feature engineering (interactions)
        input_features = create_features(input_df)

        # 6) Fill group-based aggregates (mean_by_*, std_by_*) using group_stats_map saved at training
        if group_stats_map:
            # iterate over all derived keys we might need (derive from feature_cols)
            for derived_col in [c for c in feature_cols if "_by_" in c]:
                # find matching key in group_stats_map
                map_key = _find_group_map_key(derived_col, group_stats_map)
                # extract categorical column name from derived_col (suffix after _by_)
                cat_col = derived_col.split("_by_", 1)[-1]
                if map_key is not None and cat_col in input_features.columns:
                    mapping = group_stats_map.get(map_key, None)
                    # map categorical value to stat (if unseen category -> NaN)
                    input_features[derived_col] = input_features[cat_col].map(mapping)
                else:
                    # create the column with NaN if no mapping found or cat missing
                    input_features[derived_col] = np.nan

        # 7) Align to training feature columns (order + missing)
        input_features = input_features.reindex(columns=feature_cols, fill_value=np.nan)

        # 8) As a safeguard, ensure numeric_cols coerced after reindex (for imputer)
        if numeric_cols:
            for col in numeric_cols:
                if col in input_features.columns:
                    input_features[col] = pd.to_numeric(input_features[col], errors='coerce')

        # 9) Predict
        proba = model.predict_proba(input_features)[:, 1]
        pred_class = (proba >= optimal_threshold).astype(int)
        pred_label = encoder.inverse_transform(pred_class)

        return {
            "prediction": str(pred_label[0]),
            "confidence": float(proba[0]) if proba[0] >= 0.5 else float(1 - proba[0])
        }

    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "trace": tb})
