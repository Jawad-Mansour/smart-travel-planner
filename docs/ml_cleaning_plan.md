================================================================================
                    ML CLASSIFIER - DATA CLEANING + AUGMENTATION PLAN
                    Smart Travel Planner
================================================================================

DATE: April 29, 2026
STATUS: FINAL - APPROVED (Aligned with 4-Notebook Pipeline)


================================================================================
SECTION 1: RAW DATA ISSUES (TO BE DETECTED IN NOTEBOOK 01 - EDA)
================================================================================

The raw CSV (155 data rows) contains intentional and discovered issues to
demonstrate data engineering and cleaning skills:

+--------------------------+----------+-------------------------------------+
| ISSUE TYPE               | COUNT    | EXAMPLE                             |
+--------------------------+----------+-------------------------------------+
| Missing values           | 1 cell   | seasonal_range_c is NaN for         |
|                          |          | DST-0139 (Entebbe).                 |
+--------------------------+----------+-------------------------------------+
| Duplicate rows           | 6 pairs  | Sydney (DST-0016 / DST-0075),       |
|                          |          | Cape Town (DST-0041 / DST-0076),    |
|                          |          | Nairobi (DST-0077 / DST-0078),      |
|                          |          | Maldives (DST-0042 / DST-0079),     |
|                          |          | Santiago (DST-0080 / DST-0081),     |
|                          |          | Montevideo (DST-0082 / DST-0083).   |
+--------------------------+----------+-------------------------------------+
| Whitespace in strings    | varies   | Leading/trailing spaces in city,    |
|                          |          | country, source_label_hint columns. |
+--------------------------+----------+-------------------------------------+
| Class imbalance          | severe   | Adventure (46) / Culture (35)       |
|                          |          | dominate; Luxury (8) / Family (12)  |
|                          |          | are rare.                           |
+--------------------------+----------+-------------------------------------+

NOTE: There are NO structural defects (extra columns) in this dataset.
All 155 rows have exactly 35 fields matching the header. This was an
intentional decision to simplify the cleaning pipeline.

NOTE: There are NO appended text suffixes (",Budget", ",Culture") in
source_label_hint. This was removed per project decision.


================================================================================
SECTION 2: CLEANING + AUGMENTATION PIPELINE (NOTEBOOK 02)
================================================================================

All cleaning and augmentation runs ONCE in Notebook 02. The output is the
single source of truth for downstream notebooks (03, 04).

STEP 1 - LOAD RAW DATA
-----------------------
- Read the CSV with pandas.read_csv(). No structural repair needed since
  all 155 rows have the correct 35 fields.
- Validate column count matches header (35 columns).

STEP 2 - REMOVE DUPLICATE ROWS
------------------------------
- Drop duplicates by destination_id, keeping the first occurrence.
- Drop duplicates by (destination_city, country) as a second pass to catch
  cases where the same city appears with different IDs.
- Expected result: 139 unique rows after deduplication.

STEP 3 - FIX MISSING VALUES
---------------------------
- For seasonal_range_c (the single known missing cell at DST-0139), impute
  with the median of the column. This is a one-off manual fix, not a pipeline
  step, because there is only one such cell and the rule is documented.

DEFENSE: Pipeline-level imputation is reserved for in-pipeline median/most-
frequent/constant strategies that protect against future missing values during
inference. The DST-0139 fix is one-off and documented.

STEP 4 - STRIP WHITESPACE FROM STRING COLUMNS
---------------------------------------------
string_columns = [
    "destination_id", "destination_city", "country", "region",
    "best_season", "visa_requirement", "dry_season_months",
    "source_label_hint", "travel_style"
]
df[col] = df[col].astype(str).str.strip()

STEP 5 - VERIFY CLEANED DATA
----------------------------
- assert df.isnull().sum().sum() == 0
- assert len(df) == len(df.drop_duplicates())
- print final shape and class counts

STEP 6 - SEPARATE X AND y
-------------------------
y = df["travel_style"]
X = df.drop(columns=["travel_style"])

STEP 7 - APPLY SMOTE TO THE ENTIRE CLEANED DATASET
--------------------------------------------------
SMOTE requires a numeric matrix; categoricals must be encoded first or
SMOTENC must be used. We use SMOTENC because the dataset has both numeric
and categorical features.

from imblearn.over_sampling import SMOTENC

categorical_indices = [X.columns.get_loc(c) for c in [
    "region", "dry_season_months", "best_season", "visa_requirement",
    "destination_id", "destination_city", "country", "source_label_hint"
]]

smote = SMOTENC(
    categorical_features=categorical_indices,
    random_state=42,
    k_neighbors=3  # smaller k for rare classes
)
X_aug, y_aug = smote.fit_resample(X, y)

DEFENSE: SMOTENC is the right tool when the design matrix mixes numeric and
categorical features. Plain SMOTE would corrupt categorical values.

STEP 8 - VERIFY CLASS DISTRIBUTION
----------------------------------
- All classes should be approximately balanced (within +-5 samples).
- Save experiments/class_distribution_augmented.csv.

STEP 9 - SAVE AUGMENTED DATASET
--------------------------------
df_augmented = X_aug.copy()
df_augmented["travel_style"] = y_aug
df_augmented.to_csv("backend/ml/data/destinations_augmented.csv", index=False)


================================================================================
SECTION 3: WHY SMOTE BEFORE THE TRAIN/VAL/TEST SPLIT
================================================================================

With only 139 unique rows and 6 classes, a stratified 60/20/20 split would
leave the test set with roughly 1-2 Luxury samples, making per-class metrics
meaningless. By augmenting BEFORE the split:

- Test set has enough samples per class for reliable macro F1.
- Validation set has enough samples per class for tuning decisions.
- Stratified splitting on the augmented data still preserves balance.

LIMITATION: SMOTE can leak structure across the split, slightly inflating
test scores. We mitigate by:
- Reporting per-class metrics (leakage shows as suspiciously high macro F1).
- Running the test evaluation only ONCE at the end.
- Using cross-validation on the training set (not the test set) for tuning.


================================================================================
SECTION 4: WHAT IS HANDLED IN THE PIPELINE (NOTEBOOK 02), NOT MANUALLY
================================================================================

These transformations live inside the scikit-learn ColumnTransformer to
avoid data leakage. They are FIT on TRAIN ONLY and TRANSFORM all splits.

NUMERICAL FEATURES   ->  SimpleImputer(strategy="median") -> StandardScaler
BINARY FEATURES      ->  SimpleImputer(strategy="most_frequent")
CATEGORICAL FEATURES ->  SimpleImputer(strategy="constant",
                                       fill_value="missing")
                         -> OneHotEncoder(handle_unknown="ignore")


================================================================================
SECTION 5: EXPECTED RESULTS AFTER NOTEBOOK 02
================================================================================

+-------------------------------+----------+--------------------------+
| METRIC                        | RAW      | AFTER NOTEBOOK 02        |
+-------------------------------+----------+--------------------------+
| Total rows                    | 155      | ~270 (after SMOTE)       |
+-------------------------------+----------+--------------------------+
| Duplicate rows                | 6 pairs  | 0                        |
+-------------------------------+----------+--------------------------+
| Missing values                | 1 cell   | 0                        |
+-------------------------------+----------+--------------------------+
| Whitespace defects            | minimal  | 0                        |
+-------------------------------+----------+--------------------------+
| Smallest class size           | 8        | ~46 (post-SMOTENC)       |
|                               | (Luxury) | (all classes equal)      |
+-------------------------------+----------+--------------------------+


================================================================================
SECTION 6: NOTEBOOK 02 INPUTS / OUTPUTS
================================================================================

INPUTS:
- backend/ml/data/destinations_raw.csv

OUTPUTS:
- backend/ml/data/destinations_augmented.csv          (clean + balanced)
- backend/ml/experiments/class_distribution_augmented.csv

The dataset committed to git is destinations_raw.csv. The augmented file is
regenerated deterministically from it (random_state=42), so it can be safely
recreated by anyone running the notebook.


================================================================================
END OF CLEANING + AUGMENTATION PLAN DOCUMENT
================================================================================
