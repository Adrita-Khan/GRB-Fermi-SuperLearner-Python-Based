"""
Script 04: Formula / feature search for M-estimator RLM

Input:  OutputFiles/fermi_grb_imputed.csv
        OutputFiles/lasso_selected_features.json
Output: OutputFiles/rlm_feature_set.json      (best feature set for RLM)
        OutputFiles/formula_search_results.csv

Translated from R Scripts/04_formula_search.R

Strategy: start with all main effects (LASSO-selected + squared),
then greedily test adding each pairwise interaction term.
Best feature set is saved and used by 03_m_estimator.py.

NOTE: Run before 03_m_estimator.py.
"""

import os
import json
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

os.makedirs("OutputFiles", exist_ok=True)

N_FOLDS   = 5
SEED      = 1

# ── Load data ────────────────────────────────────────────────
df = pd.read_csv("OutputFiles/fermi_grb_imputed.csv", index_col=0)
with open("OutputFiles/lasso_selected_features.json") as f:
    selected_features = json.load(f)

print(f"Selected LASSO features: {selected_features}")

Y = df["log10z"].values

# ── Build feature matrix with squared terms ──────────────────
def add_squared_terms(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    for col in list(data.columns):
        out[f"{col}Sqr"] = data[col] ** 2
    return out

base_feats    = df[selected_features].copy()
feature_data  = add_squared_terms(base_feats)
feature_data["log10z"] = Y

all_feat = [c for c in feature_data.columns if c != "log10z"]
print(f"\nSearching over {len(all_feat)} features: {all_feat}\n")

# ── CV evaluation using statsmodels RLM ──────────────────────
def evaluate_features(feat_list, data, n_folds=N_FOLDS, seed=SEED):
    """5-fold CV Pearson r for an RLM fitted on feat_list."""
    kf     = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    corrs  = []
    X_cols = feat_list + ["log10z"]
    sub    = data[X_cols].dropna()
    X_arr  = sub[feat_list].values
    Y_arr  = sub["log10z"].values

    for train_idx, test_idx in kf.split(X_arr):
        X_train = sm.add_constant(X_arr[train_idx])
        X_test  = sm.add_constant(X_arr[test_idx])
        try:
            model = sm.RLM(Y_arr[train_idx], X_train,
                           M=sm.robust.norms.HuberT())
            res   = model.fit()
            pred  = X_test @ res.params
            if len(pred) > 1:
                corrs.append(np.corrcoef(pred, Y_arr[test_idx])[0, 1])
        except Exception:
            pass

    return float(np.nanmean(corrs)) if corrs else np.nan

# ── Baseline: all main effects ────────────────────────────────
base_corr = evaluate_features(all_feat, feature_data)
print(f"Baseline (all main effects): r = {base_corr:.4f}")

best_feat_list = list(all_feat)
best_corr      = base_corr

# ── Greedy search: try each pairwise interaction ──────────────
# Use only original (non-squared) features for interaction terms
results = []

for feat_a, feat_b in itertools.combinations(selected_features, 2):
    int_name = f"{feat_a}_x_{feat_b}"
    # Add the interaction column temporarily
    feature_data[int_name] = (
        feature_data[feat_a] * feature_data[feat_b]
    )
    test_feats = best_feat_list + [int_name]
    corr = evaluate_features(test_feats, feature_data)
    results.append({"term": int_name, "corr": round(corr, 5)})
    print(f"  + {int_name:<35s}  r = {corr:.4f}")
    # Clean up the temp column
    feature_data.drop(columns=[int_name], inplace=True)

# ── Pick the best interaction ─────────────────────────────────
results_df = pd.DataFrame(results).sort_values("corr", ascending=False)
results_df.to_csv("OutputFiles/formula_search_results.csv", index=False)

if not results_df.empty:
    best_row = results_df.iloc[0]
    if best_row["corr"] > best_corr:
        best_int   = best_row["term"]
        best_corr  = best_row["corr"]
        # Rebuild the interaction column for saving
        parts      = best_int.split("_x_")
        feature_data[best_int] = (
            feature_data[parts[0]] * feature_data[parts[1]]
        )
        best_feat_list.append(best_int)
        print(f"\nBest interaction added: {best_int}  (r = {best_corr:.4f})")
    else:
        print("\nNo interaction improved over baseline. Using main effects only.")

# ── Save ──────────────────────────────────────────────────────
output = {
    "features":      best_feat_list,
    "best_cv_r":     round(best_corr, 4),
    "base_features": selected_features,
}
with open("OutputFiles/rlm_feature_set.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*40}")
print(f"Best feature set (r = {best_corr:.4f}):")
for feat in best_feat_list:
    print(f"  {feat}")
print(f"Saved → OutputFiles/rlm_feature_set.json")
print(f"{'='*40}")
print("Now run 03_m_estimator.py.")
