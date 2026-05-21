"""
Script 01: Preprocess Fermi-GBM data + MICE imputation

Input:  Data/fermi_grb_data.csv
Output: OutputFiles/fermi_grb_preprocessed.csv   (before imputation)
        OutputFiles/fermi_grb_imputed.csv         (after imputation)
        OutputFiles/is_comp_mask.npy              (COMP GRB boolean mask)
        Plot_Output/MICE_missing_pattern.png

Translated from R Scripts/01_preprocess.R
MICE equivalent: sklearn.impute.IterativeImputer (Bayesian ridge as estimator,
mirrors R mice midastouch in spirit — predictive mean matching is not natively
available in sklearn; use the miceforest package if you need exact PMM).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer   # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# ── User settings ────────────────────────────────────────────
INPUT_FILE    = "Data/fermi_grb_data.csv"
DO_MICE       = True        # False → drop rows with any NaN
USE_FLUENCE   = True        # False → use Flux column instead
LONG_GRB_ONLY = True        # True  → keep only T90 > 2 s
MICE_SEED     = 1
N_MICE_ITER   = 20          # mirrors R mice m=20
# ─────────────────────────────────────────────────────────────

for d in ("OutputFiles", "Plot_Output"):
    os.makedirs(d, exist_ok=True)

# ── Load raw data ────────────────────────────────────────────
print(f"Loading data from {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE, index_col=0)
print(f"Loaded {len(df)} GRBs | columns: {list(df.columns)}")

# ── Filter long GRBs ─────────────────────────────────────────
if LONG_GRB_ONLY:
    df = df[df["T90"] > 2].copy()
    print(f"After T90 > 2 s filter: {len(df)} GRBs remain.")

# ── COMP model flag ──────────────────────────────────────────
# beta=0 for COMP is physically meaningful; protect it from imputation.
if "spectral_model" in df.columns:
    is_comp = df["spectral_model"].str.lower() == "comp"
    df.loc[is_comp, "beta"] = 0.0
    print(f"COMP GRBs: {is_comp.sum()} — beta forced to 0 before imputation.")
else:
    is_comp = pd.Series(False, index=df.index)
    print("No 'spectral_model' column — treating all beta values as provided.")

np.save("OutputFiles/is_comp_mask.npy", is_comp.values)

# ── Feature engineering ──────────────────────────────────────
df["log_t90"]   = np.log10(df["T90"])
df["log_epeak"] = np.log10(df["Epeak"])

if USE_FLUENCE:
    if "Fluence" not in df.columns:
        raise ValueError("Column 'Fluence' not found. Set USE_FLUENCE=False to use Flux.")
    df["log_val"] = np.log10(df["Fluence"])
else:
    if "Flux" not in df.columns:
        raise ValueError("Column 'Flux' not found.")
    df["log_val"] = np.log10(df["Flux"])

df["epeak_val_ratio"] = df["log_epeak"] - df["log_val"]
df["log_epeak_sq"]    = df["log_epeak"] ** 2
df["alpha_epeak"]     = df["alpha"] * df["log_epeak"]

# Response variable
if "redshift" in df.columns:
    df["log10z"] = np.log10(1 + df["redshift"])
    print("Response log10(1+z) created.")

# ── Validity cuts ────────────────────────────────────────────
# Mirror R cuts: invalid values → NaN so MICE can fill them
df.loc[df["Epeak"] <= 0, "log_epeak"] = np.nan
df.loc[df["T90"]   <= 0, "log_t90"]   = np.nan
df.loc[df["alpha"].abs() > 3, "alpha"] = np.nan
# beta cut only for non-COMP GRBs
df.loc[(~is_comp) & (df["beta"].abs() > 3), "beta"] = np.nan

# ── Select feature columns ───────────────────────────────────
FEATURE_COLS = [
    "log_t90", "log_epeak", "log_val",
    "alpha", "beta",
    "epeak_val_ratio", "log_epeak_sq", "alpha_epeak",
]
features = df[FEATURE_COLS].copy()

# Save pre-imputation snapshot
pre_mice = features.copy()
if "log10z" in df.columns:
    pre_mice["log10z"] = df["log10z"]
pre_mice.to_csv("OutputFiles/fermi_grb_preprocessed.csv")
print("Saved pre-imputation data → OutputFiles/fermi_grb_preprocessed.csv")

# ── Missing data plot ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
missing = features.isnull().astype(int)
sns.heatmap(missing.T, cbar=False, cmap="Blues",
            xticklabels=False, yticklabels=True, ax=ax)
ax.set_title("Missing data pattern (blue = missing)")
ax.set_xlabel("GRBs")
plt.tight_layout()
plt.savefig("Plot_Output/MICE_missing_pattern.png", dpi=150)
plt.close()
print("Missing data pattern → Plot_Output/MICE_missing_pattern.png")

# ── MICE imputation ───────────────────────────────────────────
if DO_MICE:
    print(f"Running MICE imputation ({N_MICE_ITER} iterations, BayesianRidge estimator)...")

    # Protect COMP beta=0:
    # 1. Save the COMP rows' beta values before imputation
    # 2. Run imputer on full feature matrix
    # 3. Restore beta=0 for COMP GRBs afterward
    imputer = IterativeImputer(
        estimator      = BayesianRidge(),
        max_iter       = N_MICE_ITER,
        random_state   = MICE_SEED,
        imputation_order = "random",      # closest to R mice random visit order
    )
    features_arr     = imputer.fit_transform(features.values)
    features_imputed = pd.DataFrame(features_arr, index=features.index,
                                    columns=FEATURE_COLS)

    # Restore COMP beta=0 (imputer may have changed it)
    features_imputed.loc[is_comp, "beta"] = 0.0
    print(f"beta=0 restored for {is_comp.sum()} COMP GRBs after imputation.")

else:
    print("Skipping MICE — dropping rows with any NaN.")
    features_imputed = features.dropna()

# ── Attach response and save ─────────────────────────────────
out = features_imputed.copy()
if "redshift" in df.columns:
    out["redshift"] = df.loc[out.index, "redshift"]
    out["log10z"]   = df.loc[out.index, "log10z"]

out.to_csv("OutputFiles/fermi_grb_imputed.csv")
print(f"Saved imputed data → OutputFiles/fermi_grb_imputed.csv")
print(f"Final dataset: {len(out)} GRBs, {out.shape[1]} columns.")
