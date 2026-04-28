================================================================================
                    ML CLASSIFIER - PIPELINE PLAN
                    Smart Travel Planner
================================================================================

DATE: April 28, 2026
STATUS: FINAL - APPROVED


================================================================================
SECTION 1: COMPLETE PIPELINE ARCHITECTURE
================================================================================

STEP 1: LOAD RAW DATA
File: backend/ml/data/destinations_raw.csv (155 rows, messy data)

STEP 2: CLEAN DATA (Manual script, outside pipeline)
- Remove duplicates (3 rows)
- Fix typos (2 cells)
- Standardize formats (2 cells)
- Strip whitespace (3 cells)
- Save as destinations_clean.csv (150 rows)

STEP 3: TRAIN/TEST/VALIDATION SPLIT
-----------------------------------
For 150 samples, we use:
- Train: 120 rows (80%) - used for training with cross-validation
- Test: 30 rows (20%) - used ONLY once at the end for final evaluation

Cross-validation (5-fold) provides internal validation within the train set:
- Fold 1: Train 96, Validate 24
- Fold 2: Train 96, Validate 24
- Fold 3: Train 96, Validate 24
- Fold 4: Train 96, Validate 24
- Fold 5: Train 96, Validate 24

Why no separate validation set? With 150 samples, a separate validation set
would leave too few training samples (105 train, 15 val, 30 test).
Cross-validation provides robust validation within the training set.

Parameters:
- test_size = 0.2
- stratify = y (preserves class distribution)
- random_state = 42

STEP 4: SCIKIT-LEARN PIPELINE
-----------------------------
- ColumnTransformer with groups below
- Fit on X_train only
- Transform X_train and X_test
- remainder='drop' for non-specified columns (drops identifiers + future use + target)

STEP 5: CLASSIFIER COMPARISON
-----------------------------
- Logistic Regression (baseline)
- Random Forest (bagging ensemble)
- XGBoost (boosting ensemble)
- Compare with 5-fold stratified cross-validation
- Metrics: Accuracy and Macro F1 (mean + std)

STEP 6: HYPERPARAMETER TUNING
-----------------------------
- Select best performing model from Step 5
- GridSearchCV with 5-fold cross-validation
- Optimize macro F1 (not accuracy)
- Search predefined parameter grid

STEP 7: FINAL EVALUATION
------------------------
- Train best model on ALL training data (120 rows)
- Predict on test set (30 rows) - ONCE at the end
- Record final test metrics
- No further tuning based on test results

STEP 8: SAVE FINAL MODEL
------------------------
- joblib.dump(best_pipeline, 'backend/ml/models/travel_classifier.joblib')


================================================================================
SECTION 2: COLUMN GROUPS FOR PIPELINE
================================================================================

GROUP 1: NUMERICAL FEATURES (20 features)
-----------------------------------------
avg_annual_temp_c, seasonal_range_c, cost_per_day_avg_usd, meal_budget_usd,
hotel_night_avg_usd, flight_cost_usd, museum_count, monument_count,
beach_score, scenic_score, wellness_score, culture_score, hiking_score,
nightlife_score, family_score, luxury_score, safety_score,
tourist_density_score, adventure_sports_score, festival_score

Pipeline: SimpleImputer(strategy='median') -> StandardScaler

GROUP 2: BINARY FEATURES (2 features)
--------------------------------------
near_mountains, near_beach

Pipeline: SimpleImputer(strategy='most_frequent') -> StandardScaler

DEFENSE: 'most_frequent' replaces missing values with most common value (0 or 1)
This is correct because missing ≠ false (0)

GROUP 3: NOMINAL CATEGORICAL FEATURES (3 features)
--------------------------------------------------
region, dry_season_months

Note: dry_season_months is categorical, not numerical.

Pipeline: SimpleImputer(strategy='constant', fill_value='missing') -> OneHotEncoder(handle_unknown='ignore')

DEFENSE: 'missing' explicitly indicates data was missing, different from any valid category

GROUP 4: FEATURES DROPPED (remainder='drop')
-------------------------------------------
destination_id, destination_city, country, source_label_hint, best_season,
visa_requirement, english_friendly_score, public_transport_score, latitude,
longitude, travel_style (target)

Total ML features passed to pipeline: 20 + 2 + 3 = 25? Wait - let me recount.

Numerical: 20
Binary: 2
Categorical: 2 (region, dry_season_months)

20 + 2 + 2 = 24 ✓

The CSV has 35 total columns. Pipeline receives 24 features.
Remaining 11 columns are dropped: 4 identifiers + 6 future use + 1 target.


================================================================================
SECTION 3: THREE CLASSIFIERS FOR COMPARISON
================================================================================

CLASSIFIER 1: LOGISTIC REGRESSION
---------------------------------
Why: Linear baseline, interpretable coefficients, fast training
Use case: Establishes minimum performance threshold
Parameters: random_state=42, max_iter=1000, class_weight='balanced', C=1.0
Limitation: Assumes linear relationships (data may not be linear)

CLASSIFIER 2: RANDOM FOREST
---------------------------
Why: Handles non-linear relationships, robust to outliers, provides feature importance
Use case: Bagging ensemble that reduces overfitting
Parameters: n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
Limitation: Can overfit with too many trees on small dataset (120 samples)

CLASSIFIER 3: XGBOOST
---------------------
Why: Often best performance on tabular data, built-in regularization, handles missing values
Use case: Boosting ensemble that learns from previous mistakes
Parameters: n_estimators=100, random_state=42, eval_metric='mlogloss', max_depth=6

XGBOOST CLASS WEIGHT HANDLING:
XGBoost does NOT have a simple class_weight='balanced' parameter for multi-class.
Solution: Use sklearn.utils.class_weight.compute_sample_weight before fitting:
-------------------------------------------------------------------------------
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train)
best_pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
-------------------------------------------------------------------------------

This computes weights inversely proportional to class frequencies.
Limitation: XGBoost may still overfit on small dataset (120 samples may be too small)

Why not other classifiers:
- SVM: Slow on 24 features, hard to tune kernel parameters
- Neural Network: Too complex for 120 samples (would overfit severely)
- KNN: Distance metric problematic with mixed data types (numerical + categorical + binary)


================================================================================
SECTION 4: CROSS-VALIDATION STRATEGY
================================================================================

Method: StratifiedKFold
n_splits: 5
shuffle: True
random_state: 42

Why 5 folds:
- 3 folds: too noisy, high variance across folds
- 5 folds: standard choice, good balance between bias and variance
- 10 folds: too slow for 150 samples, each test fold too small (15 samples)

Why Stratified:
- Ensures each fold has same class distribution as full dataset
- Prevents fold with zero samples of rare class (Luxury has only 8 samples)
- Critical for imbalanced classification

Metrics Reported:
- Accuracy (mean ± std across 5 folds)
- Macro F1 (mean ± std across 5 folds)


================================================================================
SECTION 5: HYPERPARAMETER TUNING (Example - Random Forest)
================================================================================

PARAMETER GRID:
-------------------------------------------------------------------------------
{
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__class_weight': [None, 'balanced']
}

Why these parameters:
- n_estimators: More trees = better, but diminishing returns after 200
- max_depth: Deeper trees capture complex patterns but overfit (10-20 is safe range)
- min_samples_split: Lower values create more complex trees (2 is default)
- min_samples_leaf: Prevents overfitting on rare classes (1-4 small leaves)
- class_weight: 'balanced' directly addresses class imbalance

SEARCH METHOD: GridSearchCV
- cv = 5 (StratifiedKFold)
- scoring = 'f1_macro' (optimize macro F1, not accuracy)
- n_jobs = -1 (use all CPU cores)
- verbose = 1 (show progress)


================================================================================
SECTION 6: EXPERIMENT TRACKING (results.csv)
================================================================================

File: backend/ml/experiments/results.csv

COLUMNS:
+-------------------+-----------------------------------------------------+
| COLUMN            | DESCRIPTION                                         |
+-------------------+-----------------------------------------------------+
| timestamp         | When experiment ran (YYYY-MM-DD HH:MM:SS)          |
+-------------------+-----------------------------------------------------+
| model_name        | LogisticRegression, RandomForest, XGBoost          |
+-------------------+-----------------------------------------------------+
| params            | Hyperparameters used (JSON string)                 |
+-------------------+-----------------------------------------------------+
| accuracy_mean     | Mean accuracy across 5 folds                       |
+-------------------+-----------------------------------------------------+
| accuracy_std      | Standard deviation of accuracy across folds        |
+-------------------+-----------------------------------------------------+
| f1_macro_mean     | Mean macro F1 across 5 folds                       |
+-------------------+-----------------------------------------------------+
| f1_macro_std      | Standard deviation of macro F1 across folds        |
+-------------------+-----------------------------------------------------+
| test_f1_macro     | Final macro F1 on test set (after tuning)          |
+-------------------+-----------------------------------------------------+
| test_accuracy     | Final accuracy on test set (after tuning)          |
+-------------------+-----------------------------------------------------+

EXAMPLE ROWS:
2026-04-28 10:00:00,LogisticRegression,default,0.72,0.04,0.65,0.05,0.67,0.74
2026-04-28 10:05:00,RandomForest,default,0.78,0.03,0.71,0.04,0.73,0.80
2026-04-28 10:20:00,RandomForest,"{n_estimators=200,max_depth=15}",0.82,0.03,0.76,0.03,0.78,0.84


================================================================================
SECTION 7: RANDOM SEEDS FOR REPRODUCIBILITY
================================================================================

+---------------------------+--------------+------------------------------------------+
| LOCATION                  | SEED VALUE   | WHAT IT CONTROLS                         |
+---------------------------+--------------+------------------------------------------+
| train_test_split          | 42           | Which rows go to train vs test           |
+---------------------------+--------------+------------------------------------------+
| StratifiedKFold           | 42           | How data is folded for cross-validation  |
+---------------------------+--------------+------------------------------------------+
| LogisticRegression        | 42           | Initial random weight initialization     |
+---------------------------+--------------+------------------------------------------+
| RandomForestClassifier    | 42           | Which features considered at each split  |
+---------------------------+--------------+------------------------------------------+
| XGBClassifier             | 42           | Boosting process initialization          |
+---------------------------+--------------+------------------------------------------+

DEFENSE:
Without fixed seeds, results change every run. You cannot tell if Model A is
better than Model B, or just lucky/unlucky split. Fixed seeds ensure anyone
running your code gets IDENTICAL results (spec requirement).


================================================================================
SECTION 8: PREVENTING OVERFITTING CHECKLIST
================================================================================

[X] Train/test split BEFORE any preprocessing
[X] No information from test set used in training
[X] Cross-validation used (not single validation set)
[X] GridSearchCV uses cross-validation (not single validation)
[X] Test set used ONLY ONCE at the end
[X] Class weights only on training data
[X] Fixed random seeds for reproducibility
[X] Pipeline ensures no manual preprocessing mistakes
[X] Identifiers dropped before feature matrix
[X] Missing value imputation learned ONLY from training data


================================================================================
END OF PIPELINE PLAN DOCUMENT
================================================================================
