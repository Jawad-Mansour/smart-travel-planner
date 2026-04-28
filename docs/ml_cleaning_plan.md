================================================================================
                    ML CLASSIFIER - DATA CLEANING PLAN
                    Smart Travel Planner
================================================================================

DATE: April 28, 2026
STATUS: FINAL - APPROVED


================================================================================
SECTION 1: RAW DATA ISSUES (INTENTIONAL)
================================================================================

The raw CSV will contain these issues to demonstrate cleaning skills:

+-------------------------+----------+------------------------------------------+
| ISSUE TYPE              | COUNT    | EXAMPLE                                  |
+-------------------------+----------+------------------------------------------+
| Missing values          | 1 cell   | seasonal_range_c = NULL (DST-0139)       |
+-------------------------+----------+------------------------------------------+
| Duplicate rows          | 6 rows   | DST-0008, DST-0019, DST-0139 (3 pairs)   |
+-------------------------+----------+------------------------------------------+
| Outliers                | 2 rows   | hotel_night_avg_usd >= 250 (Luxury)      |
+-------------------------+----------+------------------------------------------+
| Appended text in hints  | 14 cells | ",Budget" or ",Culture" in source_hint   |
+-------------------------+----------+------------------------------------------+

RAW ROWS: 155
AFTER CLEANING: 149 rows (155 - 6 duplicates)


================================================================================
SECTION 2: CLEANING STEPS (BEFORE PIPELINE)
================================================================================

These steps run ONCE on the raw CSV before the scikit-learn pipeline.

STEP 1: LOAD RAW DATA
---------------------
import pandas as pd
df = pd.read_csv('backend/ml/data/destinations_raw.csv')

STEP 2: REMOVE DUPLICATES
-------------------------
# Remove by ID first (keeps first occurrence)
df = df.drop_duplicates(subset=['destination_id'], keep='first')

# Also check by city+country (in case IDs differ)
df = df.drop_duplicates(subset=['destination_city', 'country'], keep='first')

STEP 3: STRIP WHITESPACE
------------------------
string_columns = ['destination_id', 'destination_city', 'country', 'region',
                  'best_season', 'visa_requirement', 'dry_season_months',
                  'source_label_hint', 'travel_style']

for col in string_columns:
    if col in df.columns:
        df[col] = df[col].str.strip()

STEP 4: CLEAN SOURCE_LABEL_HINT (Remove appended labels)
---------------------------------------------------------
# Remove ",Budget" and ",Culture" suffixes from source_label_hint
df['source_label_hint'] = df['source_label_hint'].str.replace(',Budget', '', regex=False)
df['source_label_hint'] = df['source_label_hint'].str.replace(',Culture', '', regex=False)
df['source_label_hint'] = df['source_label_hint'].str.replace(',Budget', '', regex=False)

STEP 5: STANDARDIZE SEASON FORMATS
----------------------------------
def standardize_months(text):
    if pd.isna(text):
        return text
    month_map = {
        'Jan': 'January', 'Feb': 'February', 'Mar': 'March',
        'Apr': 'April', 'May': 'May', 'Jun': 'June',
        'Jul': 'July', 'Aug': 'August', 'Sep': 'September',
        'Oct': 'October', 'Nov': 'November', 'Dec': 'December'
    }
    for abbr, full in month_map.items():
        text = text.replace(abbr, full)
    return text

df['dry_season_months'] = df['dry_season_months'].apply(standardize_months)
df['best_season'] = df['best_season'].apply(standardize_months)

STEP 6: HANDLE OUTLIERS (Cap at 99th percentile)
------------------------------------------------
for col in ['cost_per_day_avg_usd', 'hotel_night_avg_usd', 'meal_budget_usd']:
    cap = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=cap)

STEP 7: VERIFY CLEANING
-----------------------
print(f"Rows after cleaning: {len(df)}")
print(f"Missing values per column:\n{df.isnull().sum()}")
print(f"Unique travel styles: {df['travel_style'].unique()}")

STEP 8: SAVE CLEANED DATA
-------------------------
df.to_csv('backend/ml/data/destinations_clean.csv', index=False)


================================================================================
SECTION 3: MISSING VALUE HANDLING (IN PIPELINE - NOT MANUAL)
================================================================================

IMPORTANT: Missing values are NOT manually imputed.

WHY: Manual imputation before train/test split causes data leakage.
The pipeline learns imputation statistics ONLY from training data.

PIPELINE STRATEGY (not manual):
- Numerical features: SimpleImputer(strategy='median')
- Binary features: SimpleImputer(strategy='most_frequent')
- Categorical features: SimpleImputer(strategy='constant', fill_value='missing')

DEFENSE:
- 'median' for numerical: Robust to outliers, preserves distribution
- 'most_frequent' for binary: Missing ≠ false, so use most common value
- 'missing' for categorical: Explicitly indicates data was missing


================================================================================
SECTION 4: EXPECTED RESULTS AFTER CLEANING
================================================================================

+-------------------------+--------------+-------------------------------+
| METRIC                  | RAW          | AFTER CLEANING                |
+-------------------------+--------------+-------------------------------+
| Total rows              | 155          | 149                           |
+-------------------------+--------------+-------------------------------+
| Duplicate rows          | 6            | 0                             |
+-------------------------+--------------+-------------------------------+
| Missing values          | 1 cell       | Same (handled in pipeline)    |
+-------------------------+--------------+-------------------------------+
| Appended text in hints  | 14 cells     | 0 (cleaned)                   |
+-------------------------+--------------+-------------------------------+


================================================================================
SECTION 5: CLEANING SCRIPT LOCATION
================================================================================

File: backend/ml/scripts/clean_data.py

Run with:
cd /c/projects/smart-travel-planner
source .venv/Scripts/activate
python backend/ml/scripts/clean_data.py


================================================================================
END OF CLEANING PLAN DOCUMENT
================================================================================
