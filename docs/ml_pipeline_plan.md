================================================================================
                    ML CLASSIFIER - PIPELINE PLAN
                    Smart Travel Planner
================================================================================

DATE: April 29, 2026
STATUS: FINAL - APPROVED (4-Notebook Plan with SMOTE)


================================================================================
SECTION 1: COMPLETE PIPELINE ARCHITECTURE (4 NOTEBOOKS)
================================================================================

The ML phase is split across 4 sequential Jupyter notebooks. Each notebook has
a single, clear responsibility and produces deterministic artifacts that the
next notebook consumes. Random seeds are fixed at 42 everywhere for full
reproducibility.

NOTEBOOK 01 - EDA & DATA AUDIT
File: backend/ml/notebooks/01_eda_data_audit.ipynb
- Read-only analysis of the raw CSV.
- Validate shape (155 rows, 35 columns), schema, and column consistency.
- Report 1 missing value, 6 duplicate pairs, target distribution, correlations.
- Export `class_distribution_raw.csv` and `feature_summary.csv`.
- No modifications to the source data.

NOTEBOOK 02 - CLEANING + SMOTE + PREPROCESSING
File: backend/ml/notebooks/02_data_cleaning_preprocessing.ipynb
- Load raw CSV and perform cleaning:
  - Drop duplicates (by ID, then by city+country).
  - Fix the single missing seasonal_range_c value (median imputation).
  - Strip whitespace from string columns.
- Separate X and y.
- Apply SMOTENC over the entire cleaned dataset BEFORE splitting.
- Stratified train/val/test split (60/20/20) on the augmented dataset.
- Build a scikit-learn ColumnTransformer with imputers, scaler, OHE.
- Fit preprocessor on TRAIN ONLY, transform train/val/test.
- Save processed arrays and preprocessor.joblib.
- Save destinations_augmented.csv and class_distribution_augmented.csv.

NOTEBOOK 03 - BASELINE MODELS
File: backend/ml/notebooks/03_baseline_models.ipynb
- Train and compare 3 classifiers: Logistic Regression, Random Forest, XGBoost.
- 5-fold StratifiedKFold cross-validation on the training set.
- Validation set evaluation (per-class precision/recall/F1).
- Persist results to `experiments/results.csv`.
- Select best model by macro F1.

NOTEBOOK 04 - TUNING + FINAL MODEL
File: backend/ml/notebooks/04_model_tuning_final.ipynb
- GridSearchCV on the best baseline model (5-fold, optimize macro F1).
- Tuned vs baseline comparison on the validation set.
- ONE-TIME final evaluation on the held-out test set.
- Save `models/travel_classifier_final.joblib` and feature importance plot.
- Append final row to `experiments/results.csv`.


================================================================================
SECTION 2: WHY SMOTE BEFORE THE SPLIT
================================================================================

The raw dataset has only 139 unique rows with severe class imbalance (Luxury ~8).
Splitting first would leave the test set with 1-2 Luxury samples, making per-
class evaluation unreliable.

By applying SMOTE to the entire cleaned dataset BEFORE splitting, we:
- Guarantee each split (train/val/test) has enough samples of every class.
- Preserve a deterministic, stratified split downstream.
- Avoid the train-only-SMOTE pitfall where validation/test class counts remain
  too small for reliable macro F1 calculation.

LIMITATION: SMOTE can leak signal across the split. To mitigate:
- We still use stratified splitting on the augmented dataset.
- We still report per-class metrics, so any leakage shows up as suspiciously
  high test macro F1.
- The final test evaluation runs ONCE, with no further tuning.


================================================================================
SECTION 3: COLUMN GROUPS FOR THE PREPROCESSOR (NOTEBOOK 02)
================================================================================

GROUP 1 - NUMERICAL FEATURES (20 features)
avg_annual_temp_c, seasonal_range_c, cost_per_day_avg_usd, meal_budget_usd,
hotel_night_avg_usd, flight_cost_usd, museum_count, monument_count,
festival_score, beach_score, scenic_score, wellness_score, culture_score,
hiking_score, nightlife_score, family_score, luxury_score, safety_score,
tourist_density_score, adventure_sports_score

Pipeline: SimpleImputer(strategy="median") -> StandardScaler

GROUP 2 - BINARY FEATURES (2 features)
near_mountains, near_beach
Pipeline: SimpleImputer(strategy="most_frequent")

GROUP 3 - CATEGORICAL FEATURES (2 features)
region, dry_season_months
Pipeline:
  SimpleImputer(strategy="constant", fill_value="missing")
  -> OneHotEncoder(handle_unknown="ignore")

GROUP 4 - DROPPED COLUMNS (remainder="drop")
destination_id, destination_city, country, source_label_hint,
best_season, visa_requirement, english_friendly_score,
public_transport_score, latitude, longitude, travel_style (target)

TOTAL ML FEATURES: 20 + 2 + 2 = 24
TARGET: travel_style (1)
TOTAL COLUMNS IN RAW CSV: 35


================================================================================
SECTION 4: THREE CLASSIFIERS FOR COMPARISON (NOTEBOOK 03)
================================================================================

CLASSIFIER 1 - LOGISTIC REGRESSION
Why: Linear baseline, interpretable coefficients, fast training.
Params: random_state=42, max_iter=1000, class_weight="balanced", C=1.0

CLASSIFIER 2 - RANDOM FOREST
Why: Handles non-linear relationships, robust to outliers, feature importance.
Params: n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1

CLASSIFIER 3 - XGBOOST
Why: Often best on tabular data, built-in regularization, handles missing.
Params: n_estimators=100, random_state=42, eval_metric="mlogloss", max_depth=6

XGBOOST CLASS WEIGHTS:
XGBoost has no class_weight="balanced" param for multi-class. Use
sklearn.utils.class_weight.compute_sample_weight on y_train, then pass:
classifier__sample_weight=sample_weights inside fit().

Why not other models:
- SVM: slow on 24 features, hard kernel tuning.
- Neural net: too small a dataset even after SMOTE.
- KNN: distance metric is unstable with mixed numeric/binary/categorical data.


================================================================================
SECTION 5: CROSS-VALIDATION STRATEGY
================================================================================

Method:        StratifiedKFold
n_splits:      5
shuffle:       True
random_state:  42

Why 5 folds:
- 3 folds: noisy.
- 5 folds: standard bias/variance tradeoff.
- 10 folds: too slow, each fold too small.

Why stratified:
- Every fold has the same class distribution as the full training set.
- Critical because Luxury and Family classes are still smaller than Adventure
  even after SMOTE.

Metrics reported (per model):
- Accuracy mean and std across folds
- Macro F1 mean and std across folds
- Per-class precision/recall/F1 on the validation set


================================================================================
SECTION 6: HYPERPARAMETER TUNING (NOTEBOOK 04)
================================================================================

EXAMPLE GRID (Random Forest)
{
    "classifier__n_estimators":      [50, 100, 200],
    "classifier__max_depth":         [10, 20, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf":  [1, 2, 4],
    "classifier__class_weight":      [None, "balanced"]
}

EXAMPLE GRID (XGBoost)
{
    "classifier__n_estimators":  [100, 200, 300],
    "classifier__max_depth":     [3, 6, 9],
    "classifier__learning_rate": [0.01, 0.1, 0.2],
    "classifier__subsample":     [0.8, 1.0]
}

SEARCH METHOD:
- GridSearchCV
- cv=5 (StratifiedKFold)
- scoring="f1_macro"
- n_jobs=-1
- verbose=1


================================================================================
SECTION 7: EXPERIMENT TRACKING - results.csv
================================================================================

File: backend/ml/experiments/results.csv

COLUMNS:
- timestamp:       When experiment ran (YYYY-MM-DD HH:MM:SS)
- model_name:      LogisticRegression, RandomForest, XGBoost, Tuned_Model
- params:          Hyperparameters used (JSON string)
- accuracy_mean:   Mean accuracy across folds
- accuracy_std:    Std of accuracy across folds
- f1_macro_mean:   Mean macro F1 across folds
- f1_macro_std:    Std of macro F1 across folds
- val_accuracy:    Validation set accuracy
- val_f1_macro:    Validation set macro F1
- test_accuracy:   Test set accuracy (only for the final tuned model)
- test_f1_macro:   Test set macro F1 (only for the final tuned model)


================================================================================
SECTION 8: RANDOM SEEDS FOR REPRODUCIBILITY
================================================================================

LOCATION                   SEED  WHAT IT CONTROLS
train_test_split           42    Which rows go where
StratifiedKFold            42    Fold composition
LogisticRegression         42    Solver initialization
RandomForestClassifier     42    Bootstrap sampling and feature selection
XGBClassifier              42    Boosting initialization
SMOTE                      42    Synthetic sample generation


================================================================================
SECTION 9: PREVENTING OVERFITTING - CHECKLIST
================================================================================

[X] SMOTE applied before split (documented; mitigations listed in Section 2)
[X] Stratified train/val/test split
[X] Preprocessor fit on TRAIN ONLY
[X] Cross-validation on training set, not on test set
[X] GridSearchCV uses CV (not the validation set directly)
[X] Test set used ONCE at the end
[X] Class weights only on training data
[X] Fixed random seeds everywhere
[X] Identifiers dropped before the feature matrix
[X] Missing value imputation learned from training data only


================================================================================
SECTION 10: FILE STRUCTURE AFTER ALL NOTEBOOKS
================================================================================

backend/ml/
├── data/
│   ├── destinations_raw.csv              (input - 155 rows)
│   └── destinations_augmented.csv        (Notebook 02 output - balanced)
├── experiments/
│   ├── class_distribution_raw.csv        (Notebook 01)
│   ├── feature_summary.csv               (Notebook 01)
│   ├── class_distribution_augmented.csv  (Notebook 02)
│   ├── results.csv                       (Notebooks 03 and 04)
│   └── feature_importance.png            (Notebook 04)
├── models/
│   ├── preprocessor.joblib               (Notebook 02)
│   └── travel_classifier_final.joblib    (Notebook 04)
└── notebooks/
    ├── 01_eda_data_audit.ipynb
    ├── 02_data_cleaning_preprocessing.ipynb
    ├── 03_baseline_models.ipynb
    └── 04_model_tuning_final.ipynb


================================================================================
SECTION 11: SPEC REQUIREMENTS MAPPING
================================================================================

SPEC REQUIREMENT                   COVERED IN
100-200 destinations               Notebook 01 (155 rows)
scikit-learn Pipeline              Notebook 02
Preprocessing inside the pipeline  Notebook 02
Compare 3 classifiers              Notebook 03
k-fold cross-validation            Notebooks 03, 04
Accuracy + macro F1 (mean +- std)  Notebook 03
Tune at least one model            Notebook 04
Address class imbalance            Notebook 02 (SMOTE)
Per-class metrics                  Notebook 03
Track experiments in results.csv   Notebooks 03, 04
Fix seeds (random_state=42)        All notebooks
Save winner with joblib            Notebook 04


================================================================================
END OF PIPELINE PLAN DOCUMENT
================================================================================
