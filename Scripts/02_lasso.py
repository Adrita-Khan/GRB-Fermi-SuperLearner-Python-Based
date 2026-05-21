"""
Script 02: LASSO feature selection

Input:  OutputFiles/fermi_grb_imputed.csv
Output: Plot_Output/LASSO_features.png
        OutputFiles/lasso_selected_features.json
        OutputFiles/lasso_coef_avg.csv

Translated from R Scripts/02_lasso.R

Runs LassoCV 100 times (each with a different random seed) and averages
the coefficients for stability — mirrors the R loop of 100 LASSO runs.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

os.makedirs("OutputFiles",  exist_ok=True)
os.makedirs("Plot_Output",  exist_ok=True)

N_TOP_FEATURES = 7    # mirrors R head(lassovar, 7)
N_RUNS         = 100  # mirrors R loop of 100
CV_FOLDS       = 10

# ── Load imputed data ────────────────────────────────────────
df = pd.read_csv("OutputFiles/fermi_grb_imputed.csv", index_col=0)
print(f"Loaded {len(df)} GRBs for LASSO.")

Y = df["log10z"].values
exclude = {"log10z", "redshift"}
feature_cols = [c for c in df.columns if c not in exclude]
X_base = df[feature_cols].copy()

# ── Add squared terms (mirrors R SqrTermGen) ─────────────────
def add_squared_terms(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    for col in list(data.columns):
        out[f"{col}Sqr"] = data[col] ** 2
    return out

X_expanded = add_squared_terms(X_base)
print(f"Feature matrix: {X_expanded.shape[0]} x {X_expanded.shape[1]}")
print(f"Features: {list(X_expanded.columns)}\n")

# ── Scale features for LASSO (sklearn LassoCV needs it) ──────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_expanded.values)

# ── Run LassoCV 100 times and average coefficients ───────────
print(f"Running LassoCV {N_RUNS} iterations...")
coef_matrix = np.zeros((N_RUNS, X_expanded.shape[1]))

for run in range(N_RUNS):
    lasso = LassoCV(cv=CV_FOLDS, random_state=run, max_iter=5000, n_jobs=-1)
    lasso.fit(X_scaled, Y)
    coef_matrix[run, :] = lasso.coef_

coef_avg = np.mean(coef_matrix, axis=0)
coef_abs = np.abs(coef_avg)

# ── Save coefficient table ────────────────────────────────────
coef_df = pd.DataFrame({
    "feature":   X_expanded.columns,
    "mean_coef": coef_avg,
    "mean_abs_coef": coef_abs,
}).sort_values("mean_abs_coef", ascending=False)

coef_df.to_csv("OutputFiles/lasso_coef_avg.csv", index=False)

# ── Print ranking ─────────────────────────────────────────────
print("Features ranked by mean |LASSO coefficient|:")
print(coef_df.to_string(index=False))

nonzero = coef_df[coef_df["mean_abs_coef"] > 0]
print(f"\nNon-zero LASSO features ({len(nonzero)}): "
      f"{list(nonzero['feature'])}")

n_sel = min(N_TOP_FEATURES, len(nonzero))
selected = list(nonzero["feature"].head(n_sel))
print(f"\nSelected top {n_sel} features for SuperLearner:")
print(selected)

with open("OutputFiles/lasso_selected_features.json", "w") as f:
    json.dump(selected, f, indent=2)
print("\nSaved → OutputFiles/lasso_selected_features.json")

# ── Plot ──────────────────────────────────────────────────────
plot_df = coef_df.sort_values("mean_abs_coef")
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(plot_df["feature"], plot_df["mean_abs_coef"],
        color="steelblue", edgecolor="none")
ax.set_xlabel("Mean |LASSO Coefficient| (100 runs)")
ax.set_title("LASSO Feature Importance (Fermi-GBM)")
plt.tight_layout()
plt.savefig("Plot_Output/LASSO_features.png", dpi=150)
plt.close()
print("Plot saved → Plot_Output/LASSO_features.png")
