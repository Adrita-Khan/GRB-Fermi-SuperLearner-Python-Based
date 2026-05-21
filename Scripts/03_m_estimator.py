"""
Script 03: M-estimator outlier removal

Input:  OutputFiles/fermi_grb_imputed.csv
        OutputFiles/lasso_selected_features.json
        OutputFiles/rlm_feature_set.json   (from 04_formula_search.py)
Output: OutputFiles/fermi_grb_m_est.csv
        Plot_Output/M_estimator_weights.png
        Plot_Output/M_estimator_scatter.png

Translated from R Scripts/03_m_estimator.R

Uses statsmodels RLM with HuberT norm (equivalent to MASS::rlm method="M").
GRBs whose RLM weight falls below WEIGHT_THRESHOLD are removed as outliers.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

# ── User settings ────────────────────────────────────────────
WEIGHT_THRESHOLD = 0.65   # mirrors R weight_threshold = 0.65
# ─────────────────────────────────────────────────────────────

os.makedirs("OutputFiles", exist_ok=True)
os.makedirs("Plot_Output",  exist_ok=True)

# ── Load data ────────────────────────────────────────────────
df = pd.read_csv("OutputFiles/fermi_grb_imputed.csv", index_col=0)
print(f"Loaded {len(df)} GRBs.")

with open("OutputFiles/lasso_selected_features.json") as f:
    selected_features = json.load(f)

# Load best feature set from formula search (or fall back to LASSO features)
rlm_feat_file = "OutputFiles/rlm_feature_set.json"
if os.path.exists(rlm_feat_file):
    with open(rlm_feat_file) as f:
        rlm_info = json.load(f)
    rlm_features   = rlm_info["features"]
    base_features  = rlm_info["base_features"]
    print(f"Loaded RLM feature set from {rlm_feat_file}")
else:
    print("rlm_feature_set.json not found — using LASSO features as fallback.")
    print("Run 04_formula_search.py first for a better formula.")
    rlm_features  = selected_features
    base_features = selected_features

print(f"RLM features: {rlm_features}\n")

Y  = df["log10z"].values
rc = df["redshift"].values if "redshift" in df.columns else None

# ── Build feature matrix with squared terms ──────────────────
def add_squared_terms(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    for col in list(data.columns):
        out[f"{col}Sqr"] = data[col] ** 2
    return out

def add_interaction_terms(data: pd.DataFrame, features: list) -> pd.DataFrame:
    """Add any interaction columns (name contains '_x_') from the feature list."""
    out = data.copy()
    for feat in features:
        if "_x_" in feat and feat not in out.columns:
            parts = feat.split("_x_")
            if parts[0] in out.columns and parts[1] in out.columns:
                out[feat] = out[parts[0]] * out[parts[1]]
    return out

base_df      = df[base_features].copy()
feat_df      = add_squared_terms(base_df)
feat_df      = add_interaction_terms(feat_df, rlm_features)

# Keep only the columns listed in rlm_features (some may be squared/interaction)
available = [f for f in rlm_features if f in feat_df.columns]
X_rlm = feat_df[available].values

# ── Fit RLM (M-estimator) ────────────────────────────────────
print("Fitting RLM (HuberT M-estimator)...")
X_const = sm.add_constant(X_rlm)
rlm_model  = sm.RLM(Y, X_const, M=sm.robust.norms.HuberT())
rlm_result = rlm_model.fit()
print(rlm_result.summary())

weights = rlm_result.weights
print(f"\nWeight summary: min={weights.min():.3f}  max={weights.max():.3f}  "
      f"mean={weights.mean():.3f}")
print(f"GRBs below threshold ({WEIGHT_THRESHOLD}): "
      f"{(weights < WEIGHT_THRESHOLD).sum()}")

# ── Plot: weight histogram ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(weights, bins=max(5, len(set(weights.round(1))) + 2),
        color="steelblue", edgecolor="white")
ax.axvline(WEIGHT_THRESHOLD, color="red", lw=2, ls="--",
           label=f"Threshold = {WEIGHT_THRESHOLD}")
n_outliers  = (weights < WEIGHT_THRESHOLD).sum()
n_kept      = (weights >= WEIGHT_THRESHOLD).sum()
ax.set_xlabel("M-estimator weight")
ax.set_title("M-estimator weights")
ax.legend()
ax.text(0.02, 0.95,
        f"Outliers: {n_outliers}\nRetained: {n_kept}",
        transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.7))

# List outlier GRB names
outlier_names = df.index[weights < WEIGHT_THRESHOLD].tolist()
if outlier_names:
    ax.text(0.98, 0.95,
            "Outliers:\n" + "\n".join(outlier_names),
            transform=ax.transAxes, va="top", ha="right",
            fontsize=7,
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.8))

plt.tight_layout()
plt.savefig("Plot_Output/M_estimator_weights.png", dpi=150)
plt.close()
print("Plot → Plot_Output/M_estimator_weights.png")

# ── Plot: pairwise scatter coloured by weight ─────────────────
plot_feats = base_features[:min(5, len(base_features))]   # limit to 5 for readability
n_feats    = len(plot_feats)
colors     = ["steelblue" if w >= WEIGHT_THRESHOLD else "red" for w in weights]

fig, axes = plt.subplots(n_feats, n_feats, figsize=(3 * n_feats, 3 * n_feats))
for i, fi in enumerate(plot_feats):
    for j, fj in enumerate(plot_feats):
        ax = axes[i][j]
        if i == j:
            ax.hist(df[fi], bins=15, color="steelblue", edgecolor="none")
            ax.set_title(fi, fontsize=8)
        else:
            ax.scatter(df[fj], df[fi], c=colors, s=10, alpha=0.7)
        ax.tick_params(labelsize=6)

fig.suptitle(f"Pair plot: {len(df)} GRBs — red = outliers", fontsize=10)
plt.tight_layout()
plt.savefig("Plot_Output/M_estimator_scatter.png", dpi=120)
plt.close()
print("Scatter plot → Plot_Output/M_estimator_scatter.png")

# ── Remove outliers ───────────────────────────────────────────
if outlier_names:
    print(f"\nRemoving {len(outlier_names)} outlier GRBs:")
    for name in outlier_names:
        print(f"  {name}")

kept_mask    = weights >= WEIGHT_THRESHOLD
df_clean     = df.loc[kept_mask].copy()
print(f"\nRetained {len(df_clean)} GRBs after outlier removal.")

df_clean.to_csv("OutputFiles/fermi_grb_m_est.csv")
print("Saved → OutputFiles/fermi_grb_m_est.csv")
