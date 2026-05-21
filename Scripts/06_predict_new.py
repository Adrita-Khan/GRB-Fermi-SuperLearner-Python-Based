"""
Script 06: Predict redshifts for new GRBs

Input:  superlearner_model.pkl
        Data/new_grbs.csv
Output: Results/predicted_redshifts.csv

Translated from R Scripts/06_predict_new.R
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.experimental import enable_iterative_imputer   # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# ── User settings ────────────────────────────────────────────
NEW_GRB_FILE = "Data/new_grbs.csv"
USE_FLUENCE  = True    # must match setting used in 01_preprocess.py
DO_MICE      = True    # impute missing values in new GRBs
# ─────────────────────────────────────────────────────────────

os.makedirs("Results", exist_ok=True)

if not os.path.exists("superlearner_model.pkl"):
    raise FileNotFoundError("superlearner_model.pkl not found. Run 05_superlearner.py first.")
if not os.path.exists(NEW_GRB_FILE):
    raise FileNotFoundError(f"Input file not found: {NEW_GRB_FILE}")

# ── Load model and metadata ───────────────────────────────────
bundle            = joblib.load("superlearner_model.pkl")
model             = bundle["model"]
feat_names        = bundle["feature_cols"]   # full feature list (with squared terms)
selected_features = bundle["selected"]       # original LASSO-selected features
print(f"Loaded trained model.")
print(f"LASSO features: {selected_features}")
print(f"Model expects {len(feat_names)} columns: {feat_names}\n")

# ── Load new GRBs ─────────────────────────────────────────────
new_df = pd.read_csv(NEW_GRB_FILE, index_col=0)
print(f"Loaded {len(new_df)} new GRBs.")

# ── Feature engineering (identical to 01_preprocess.py) ──────
new_df["log_t90"]   = np.log10(new_df["T90"])
new_df["log_epeak"] = np.log10(new_df["Epeak"])

if USE_FLUENCE:
    new_df["log_val"] = np.log10(new_df["Fluence"])
else:
    new_df["log_val"] = np.log10(new_df["Flux"])

if "spectral_model" in new_df.columns:
    is_comp = new_df["spectral_model"].str.lower() == "comp"
    new_df.loc[is_comp, "beta"] = 0.0
    print(f"COMP GRBs: {is_comp.sum()}")
else:
    is_comp = pd.Series(False, index=new_df.index)

new_df["epeak_val_ratio"] = new_df["log_epeak"] - new_df["log_val"]
new_df["log_epeak_sq"]    = new_df["log_epeak"] ** 2
new_df["alpha_epeak"]     = new_df["alpha"] * new_df["log_epeak"]

FEATURE_COLS = [
    "log_t90", "log_epeak", "log_val",
    "alpha", "beta",
    "epeak_val_ratio", "log_epeak_sq", "alpha_epeak",
]
features_new = new_df[FEATURE_COLS].copy()

# ── MICE on new data if needed ────────────────────────────────
if DO_MICE and features_new.isnull().any().any():
    print("Imputing missing values in new GRBs...")
    imputer = IterativeImputer(
        estimator    = BayesianRidge(),
        max_iter     = 20,
        random_state = 1,
    )
    arr = imputer.fit_transform(features_new.values)
    features_new = pd.DataFrame(arr, index=features_new.index,
                                 columns=FEATURE_COLS)
    # Restore COMP beta=0
    features_new.loc[is_comp, "beta"] = 0.0
else:
    features_new = features_new.dropna()

# ── Build squared terms ───────────────────────────────────────
def add_squared_terms(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    for col in list(data.columns):
        out[f"{col}Sqr"] = data[col] ** 2
    return out

feat_sel = features_new[selected_features]
X_new    = add_squared_terms(feat_sel)

# ── Align columns to training feature set ────────────────────
for col in feat_names:
    if col not in X_new.columns:
        print(f"WARNING: column '{col}' missing from new data — setting to 0.")
        X_new[col] = 0.0

X_new = X_new[feat_names]   # ensure correct column order

# ── Predict ───────────────────────────────────────────────────
print(f"Predicting redshifts for {len(X_new)} GRBs...")
pred_log10z = model.predict(X_new.values)
pred_z      = 10 ** pred_log10z - 1

# ── Output ────────────────────────────────────────────────────
results = pd.DataFrame({
    "GRB_name":          X_new.index,
    "predicted_log10z":  pred_log10z.round(4),
    "predicted_z":       pred_z.round(4),
})

if "redshift" in new_df.columns:
    results["known_redshift"] = new_df.loc[X_new.index, "redshift"].values
    mask = ~results["known_redshift"].isna()
    if mask.sum() > 1:
        obs = results.loc[mask, "known_redshift"].values
        prd = results.loc[mask, "predicted_z"].values
        r   = np.corrcoef(prd, obs)[0, 1]
        rmse = np.sqrt(np.mean((prd - obs) ** 2))
        print(f"\nValidation on known-redshift subset:")
        print(f"  Pearson r: {r:.4f}")
        print(f"  RMSE (z):  {rmse:.4f}")

results.to_csv("Results/predicted_redshifts.csv", index=False)
print(f"\nPredictions saved → Results/predicted_redshifts.csv\n")
print(results.to_string(index=False))
