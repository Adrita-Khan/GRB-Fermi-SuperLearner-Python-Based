# Fermi-GBM GRB Redshift Estimation — Python SuperLearner Pipeline

Python port of the R pipeline (Dainotti / Narendra et al., GRB-Web-App),
adapted for **Fermi-GBM prompt emission data**.

---

## Python equivalents of R packages

| R package       | Python equivalent used here          |
|-----------------|--------------------------------------|
| mice            | sklearn IterativeImputer (MICE)      |
| MASS::rlm       | statsmodels RLM (HuberT)             |
| glmnet (LASSO)  | sklearn LassoCV                      |
| SuperLearner    | mlens SuperLearner                   |
| GAM (mgcv)      | pygam LinearGAM                      |
| randomForest    | sklearn RandomForestRegressor        |
| xgboost         | xgboost XGBRegressor                 |
| earth (MARS)    | sklearn Pipeline + custom            |
| caret RF        | sklearn RandomForestRegressor        |
| GLM             | sklearn LinearRegression             |
| BayesGLM        | sklearn BayesianRidge                |

> **Note on SuperLearner in Python**: The `mlens` package provides a
> SuperLearner ensemble. An alternative is to use `scikit-learn`'s
> `StackingRegressor`, which is included as a fallback if mlens is not
> available (see `05_superlearner.py`).

---

## Input Data Format

Your CSV must have one row per GRB, GRB name as the index, and these columns:

| Column           | Description                                       | Required |
|------------------|---------------------------------------------------|----------|
| T90              | Duration in seconds (raw)                         | Yes      |
| Epeak            | Peak energy in keV (raw)                          | Yes      |
| Fluence          | Fluence in erg/cm² (raw). OR use Flux             | Yes*     |
| Flux             | Energy flux (raw). Used if Fluence absent         | Yes*     |
| alpha            | Low-energy spectral index                         | Yes      |
| beta             | High-energy spectral index. 0 for COMP model      | Yes      |
| spectral_model   | "BAND", "COMP", "SBPL", etc.                      | Yes      |
| redshift         | Known redshift (training data only)               | Training |

---

## Run order

```bash
pip install -r requirements.txt
python Scripts/01_preprocess.py
python Scripts/02_lasso.py
python Scripts/04_formula_search.py   # must run before 03
python Scripts/03_m_estimator.py
python Scripts/05_superlearner.py
python Scripts/06_predict_new.py
```

---

## File Map

```
fermi_superlearner_py/
├── README.md
├── requirements.txt
├── Data/
│   ├── fermi_grb_data.csv          ← YOUR TRAINING DATA
│   └── new_grbs.csv                ← GRBs to predict
├── Scripts/
│   ├── 01_preprocess.py
│   ├── 02_lasso.py
│   ├── 03_m_estimator.py
│   ├── 04_formula_search.py
│   ├── 05_superlearner.py
│   └── 06_predict_new.py
├── OutputFiles/                    ← intermediate data
├── Results/                        ← CV metrics, predictions
└── Plot_Output/                    ← diagnostic plots
```
