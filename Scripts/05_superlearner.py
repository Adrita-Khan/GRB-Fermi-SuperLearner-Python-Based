"""
Script 05: SuperLearner training + 10-fold CV

Input:  OutputFiles/fermi_grb_m_est.csv
        OutputFiles/lasso_selected_features.json
Output: superlearner_model.pkl       ← trained ensemble
        Results/cv_predictions.csv
        Results/cv_metrics.txt
        Results/algo_coefficients.csv
        Plot_Output/correlation_plot.png
        Plot_Output/algorithm_weights.png

Translated from R Scripts/05_superlearner.R

SuperLearner ensemble built with mlens.SuperLearner.
Fallback: sklearn StackingRegressor if mlens is unavailable.

Learner library mirrors the R version:
  GAM     → pygam LinearGAM
  GLM     → sklearn LinearRegression
  BayesGLM → sklearn BayesianRidge
  RF      → sklearn RandomForestRegressor
  MARS    → sklearn with polynomial features + Ridge (earth equivalent)
  ranger  → sklearn RandomForestRegressor (fast, n_jobs=-1)
  XGBoost → xgboost XGBRegressor
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── User settings ────────────────────────────────────────────
N_LOOPS  = 10    # repeat CV this many times; results are averaged
N_FOLDS  = 10    # k-fold CV within each loop
SEED     = 42
N_JOBS   = -1    # use all available CPUs
# ─────────────────────────────────────────────────────────────

for d in ("Results", "Plot_Output"):
    os.makedirs(d, exist_ok=True)

# ── Load data ────────────────────────────────────────────────
df = pd.read_csv("OutputFiles/fermi_grb_m_est.csv", index_col=0)
with open("OutputFiles/lasso_selected_features.json") as f:
    selected_features = json.load(f)

print(f"Loaded {len(df)} GRBs after M-estimator cut.")
print(f"LASSO features: {selected_features}\n")

Y   = df["log10z"].values
Y_z = df["redshift"].values if "redshift" in df.columns else None

# ── Build predictor matrix with squared terms ─────────────────
def add_squared_terms(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    for col in list(data.columns):
        out[f"{col}Sqr"] = data[col] ** 2
    return out

X_base   = df[selected_features].copy()
X_df     = add_squared_terms(X_base)
X        = X_df.values
feat_names = list(X_df.columns)

print(f"Predictor matrix: {X.shape[0]} x {X.shape[1]}")
print(f"Predictors: {feat_names}\n")

# ── Define learner library ────────────────────────────────────
def make_learners():
    """Returns a dict of name → estimator, mirroring the R SL library."""

    # MARS approximation: PolynomialFeatures(degree=2) + Ridge
    mars = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("ridge", Ridge(alpha=1.0)),
    ])

    learners = {
        "GLM":       LinearRegression(),
        "BayesGLM":  BayesianRidge(),
        "RF":        RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=N_JOBS),
        "RF_fast":   RandomForestRegressor(n_estimators=100, max_features="sqrt",
                                           random_state=SEED+1, n_jobs=N_JOBS),
        "MARS":      mars,
        "XGBoost":   xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                                       max_depth=4, subsample=0.8,
                                       random_state=SEED, n_jobs=N_JOBS,
                                       verbosity=0),
    }

    # Try to add GAM (pygam)
    try:
        from pygam import LinearGAM
        learners["GAM"] = LinearGAM(n_splines=10)
    except ImportError:
        print("pygam not installed — GAM learner skipped.")

    return learners

learners    = make_learners()
learner_names = list(learners.keys())
print(f"SuperLearner library ({len(learner_names)} learners): {learner_names}\n")

# ── Try mlens SuperLearner; fall back to StackingRegressor ───
USE_MLENS = False
try:
    from mlens.ensemble import SuperLearner as MLensSL
    USE_MLENS = True
    print("Using mlens.SuperLearner for ensemble.\n")
except ImportError:
    print("mlens not available — using sklearn StackingRegressor as ensemble.\n")


def fit_ensemble(X_train, Y_train):
    """Fit the ensemble on a training fold."""
    if USE_MLENS:
        sl = MLensSL(folds=5, random_state=SEED, verbose=0)
        for name, est in learners.items():
            sl.add([est], proba=False)
        sl.add_meta(LinearRegression())
        sl.fit(X_train, Y_train)
        return sl
    else:
        estimator_list = [(name, est) for name, est in learners.items()]
        stack = StackingRegressor(
            estimators   = estimator_list,
            final_estimator = Ridge(alpha=1.0),
            cv           = 5,
            n_jobs       = N_JOBS,
            passthrough  = False,
        )
        stack.fit(X_train, Y_train)
        return stack


def predict_ensemble(model, X_test):
    return model.predict(X_test)


def get_weights(model):
    """Extract final-layer coefficients (learner weights) if available."""
    if USE_MLENS:
        try:
            meta = model.layers[-1].estimators_[0]
            return dict(zip(learner_names, meta.coef_))
        except Exception:
            return {}
    else:
        try:
            coef = model.final_estimator_.coef_
            names = [n for n, _ in model.estimators]
            return dict(zip(names, coef))
        except Exception:
            return {}


# ── Cross-validation loop ────────────────────────────────────
print(f"Starting {N_LOOPS} iterations of {N_FOLDS}-fold CV...\n")

all_preds   = np.zeros((len(Y), N_LOOPS))
all_weights = []

for loop in range(N_LOOPS):
    kf       = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + loop)
    preds_cv = np.zeros(len(Y))

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr       = Y[train_idx]

        model = fit_ensemble(X_tr, Y_tr)
        preds_cv[test_idx] = predict_ensemble(model, X_te)

        w = get_weights(model)
        if w:
            all_weights.append(w)

    all_preds[:, loop] = preds_cv
    loop_r = np.corrcoef(preds_cv, Y)[0, 1]
    print(f"  Loop {loop+1:2d}/{N_LOOPS}  |  r = {loop_r:.4f}")

# ── Aggregate ─────────────────────────────────────────────────
mean_preds = all_preds.mean(axis=1)
pred_sd    = all_preds.std(axis=1)

r_log    = np.corrcoef(mean_preds, Y)[0, 1]
rmse_log = np.sqrt(np.mean((mean_preds - Y) ** 2))
pred_z   = 10 ** mean_preds - 1

if Y_z is not None:
    r_lin    = np.corrcoef(pred_z, Y_z)[0, 1]
    rmse_lin = np.sqrt(np.mean((pred_z - Y_z) ** 2))
else:
    r_lin = rmse_lin = np.nan

print(f"\n{'='*45}")
print(f"Pearson r  (log10z):   {r_log:.4f}")
print(f"RMSE       (log10z):   {rmse_log:.4f}")
print(f"Pearson r  (linear z): {r_lin:.4f}")
print(f"RMSE       (linear z): {rmse_lin:.4f}")
print(f"{'='*45}\n")

# ── Save CV predictions ───────────────────────────────────────
cv_out = pd.DataFrame({
    "GRB_name":          df.index,
    "observed_log10z":   Y,
    "predicted_log10z":  mean_preds.round(4),
    "pred_sd_log10z":    pred_sd.round(4),
    "observed_z":        Y_z if Y_z is not None else np.nan,
    "predicted_z":       pred_z.round(4),
})
cv_out.to_csv("Results/cv_predictions.csv", index=False)
print("Saved → Results/cv_predictions.csv")

# ── Save metrics ──────────────────────────────────────────────
with open("Results/cv_metrics.txt", "w") as f:
    f.write("SuperLearner CV Metrics — Fermi-GBM Redshift Estimation\n")
    f.write("=" * 50 + "\n")
    f.write(f"N GRBs:               {len(Y)}\n")
    f.write(f"CV loops:             {N_LOOPS} x {N_FOLDS}-fold\n")
    f.write(f"Pearson r (log10z):   {r_log:.4f}\n")
    f.write(f"RMSE (log10z):        {rmse_log:.4f}\n")
    f.write(f"Pearson r (linear z): {r_lin:.4f}\n")
    f.write(f"RMSE (linear z):      {rmse_lin:.4f}\n")
    if all_weights:
        wdf = pd.DataFrame(all_weights).mean().sort_values(ascending=False)
        f.write("\nAlgorithm ensemble weights (mean):\n")
        for name, w in wdf.items():
            f.write(f"  {name:<15s}: {w:.4f}\n")
print("Saved → Results/cv_metrics.txt")

# ── Save algorithm weights ────────────────────────────────────
if all_weights:
    wdf = pd.DataFrame(all_weights)
    wdf.to_csv("Results/algo_coefficients.csv", index=False)
    mean_w = wdf.mean().sort_values(ascending=False)
    print(f"\nAlgorithm ensemble weights:\n{mean_w.round(4).to_string()}\n")

# ── Plots ─────────────────────────────────────────────────────
# Correlation plot
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(Y, mean_preds, alpha=0.6, s=25, color="steelblue")
lims = [min(Y.min(), mean_preds.min()) - 0.05,
        max(Y.max(), mean_preds.max()) + 0.05]
ax.plot(lims, lims, "r-", lw=1.5)
ax.set_xlabel("Observed log₁₀(1+z)")
ax.set_ylabel("Predicted log₁₀(1+z)")
ax.set_title(f"SuperLearner CV  |  r = {r_log:.3f}  |  RMSE = {rmse_log:.3f}")
ax.text(0.05, 0.92,
        f"r = {r_log:.3f}\nRMSE = {rmse_log:.3f}\nN = {len(Y)}",
        transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8))
plt.tight_layout()
plt.savefig("Plot_Output/correlation_plot.png", dpi=200)
plt.close()
print("Plot → Plot_Output/correlation_plot.png")

# Algorithm weights bar chart
if all_weights:
    mean_w_sorted = mean_w.sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(mean_w_sorted.index, mean_w_sorted.values,
            color="steelblue", edgecolor="none")
    ax.set_xlabel("Mean ensemble weight")
    ax.set_title("Algorithm ensemble weights")
    plt.tight_layout()
    plt.savefig("Plot_Output/algorithm_weights.png", dpi=150)
    plt.close()
    print("Plot → Plot_Output/algorithm_weights.png")

# ── Train final model on full data ────────────────────────────
print("\nTraining final model on all data...")
final_model = fit_ensemble(X, Y)
joblib.dump({
    "model":        final_model,
    "feature_cols": feat_names,
    "selected":     selected_features,
    "use_mlens":    USE_MLENS,
}, "superlearner_model.pkl")
print("Final model saved → superlearner_model.pkl")
print("Done. Run 06_predict_new.py to predict new GRBs.")
