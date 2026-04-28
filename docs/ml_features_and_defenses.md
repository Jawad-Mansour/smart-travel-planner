================================================================================
                    ML CLASSIFIER - FEATURES & DEFENSES
                    Smart Travel Planner
================================================================================

DATE: April 28, 2026
STATUS: FINAL - APPROVED


================================================================================
SECTION 1: COMPLETE FEATURE SUMMARY
================================================================================

TOTAL CSV COLUMNS: 38
ML FEATURES (used in training): 24
IDENTIFIERS (dropped before training): 4
FUTURE USE (agent only, not in ML): 6
TARGET: 1

COLUMN BREAKDOWN:
- Identifiers (4): destination_id, destination_city, country, source_label_hint
- ML Features (24): 3 climate + 3 cost + 3 culture + 11 scores + 2 binary + 2 other
- Future Use (6): best_season, visa_requirement, english_friendly_score, public_transport_score, latitude, longitude
- Target (1): travel_style

VERIFICATION: 4 + 24 + 6 + 1 = 35? WAIT - This doesn't add up to 38.
Let me recount the CSV header:

CSV HEADER (38 columns):
1.destination_id
2.destination_city
3.country
4.region
5.avg_annual_temp_c
6.seasonal_range_c
7.dry_season_months
8.cost_per_day_avg_usd
9.meal_budget_usd
10.hotel_night_avg_usd
11.flight_cost_usd
12.museum_count
13.monument_count
14.festival_score
15.beach_score
16.scenic_score
17.wellness_score
18.culture_score
19.hiking_score
20.nightlife_score
21.family_score
22.luxury_score
23.safety_score
24.tourist_density_score
25.adventure_sports_score
26.near_mountains
27.near_beach
28.best_season
29.visa_requirement
30.english_friendly_score
31.public_transport_score
32.latitude
33.longitude
34.source_label_hint
35.travel_style

COUNT: 35 columns, not 38.

CORRECTION: Total CSV columns = 35, not 38.
- Identifiers: destination_id, destination_city, country, source_label_hint = 4
- Region: region = 1 (categorical, part of ML features)
- ML Features: 24 (includes region, excludes dry_season_months? Let me recalc)

Let me recalc ML Features (24):
Climate (2): avg_annual_temp_c, seasonal_range_c
Cost (3): cost_per_day_avg_usd, meal_budget_usd, hotel_night_avg_usd
Culture (3): museum_count, monument_count, festival_score
Scores (11): beach_score, scenic_score, wellness_score, culture_score, hiking_score, nightlife_score, family_score, luxury_score, safety_score, tourist_density_score, adventure_sports_score
Binary (2): near_mountains, near_beach
Other (3): region, flight_cost_usd, dry_season_months

2+3+3+11+2+3 = 24 ✓

Future Use (6): best_season, visa_requirement, english_friendly_score, public_transport_score, latitude, longitude
Target (1): travel_style
Identifiers (4): destination_id, destination_city, country, source_label_hint

Total: 4 + 24 + 6 + 1 = 35 ✓

All documents updated to reflect 35 TOTAL COLUMNS.


================================================================================
SECTION 2: IDENTIFIERS (Dropped Before Training - No Leakage)
================================================================================

+---------------------+----------+---------------------------------------------+
| FEATURE             | TYPE     | WHY NOT USED IN ML                          |
+---------------------+----------+---------------------------------------------+
| destination_id      | String   | Unique identifier, no predictive power      |
+---------------------+----------+---------------------------------------------+
| destination_city    | String   | Name has no mathematical meaning            |
+---------------------+----------+---------------------------------------------+
| country             | String   | Redundant with region                       |
+---------------------+----------+---------------------------------------------+
| source_label_hint   | String   | Documentation only                          |
+---------------------+----------+---------------------------------------------+

DEFENSE: Including identifiers allows model to memorize destinations instead of
learning general patterns. This causes overfitting.


================================================================================
SECTION 3: TARGET COLUMN
================================================================================

+---------------------+----------+---------------------------------------------+
| FEATURE             | TYPE     | VALUES                                      |
+---------------------+----------+---------------------------------------------+
| travel_style        | String   | Adventure, Relaxation, Culture, Budget,     |
|                     | (Target) | Luxury, Family                              |
+---------------------+----------+---------------------------------------------+


================================================================================
SECTION 4: CLIMATE FEATURES (2 features)
================================================================================

+---------------------+----------+---------+----------------------------------+
| FEATURE             | TYPE     | RANGE   | WHY NEEDED                       |
+---------------------+----------+---------+----------------------------------+
| avg_annual_temp_c   | Float    | -10-35  | Directly answers "warm" from     |
|                     |          |         | user query                       |
+---------------------+----------+---------+----------------------------------+
| seasonal_range_c    | Float    | 0-60    | Temperature variability (warmest |
|                     |          |         | minus coldest)                   |
+---------------------+----------+---------+----------------------------------+

DEFENSE: User query: "2 weeks in July" and "warm" - climate features directly
answer both. dry_season_months is treated as categorical separately.


================================================================================
SECTION 5: COST FEATURES (3 features)
================================================================================

+------------------------+----------+---------+----------------------------------+
| FEATURE                | TYPE     | RANGE   | WHY NEEDED                       |
+------------------------+----------+---------+----------------------------------+
| cost_per_day_avg_usd   | Integer  | 20-400  | Average daily spend including    |
|                        |          |         | accommodation, food, activities  |
+------------------------+----------+---------+----------------------------------+
| meal_budget_usd        | Integer  | 2-20    | Budget meal (street food/cafe)   |
+------------------------+----------+---------+----------------------------------+
| hotel_night_avg_usd    | Integer  | 15-500  | Average hotel night cost         |
+------------------------+----------+---------+----------------------------------+

DEFENSE: User has "$1,500" budget for 14 days (~$107/day). These 3 features
provide enough signal without overcomplicating. Hotel range extended to 500
to allow Luxury rule to trigger via hotel price.


================================================================================
SECTION 6: CULTURE FEATURES (3 features)
================================================================================

+---------------------+----------+---------+----------------------------------+
| FEATURE             | TYPE     | RANGE   | WHY NEEDED                       |
+---------------------+----------+---------+----------------------------------+
| museum_count        | Integer  | 0-100   | Objective culture measure        |
+---------------------+----------+---------+----------------------------------+
| monument_count      | Integer  | 0-50    | Historical sites indicator       |
+---------------------+----------+---------+----------------------------------+
| festival_score      | 1-10     | 1-10    | Cultural festivals/events        |
+---------------------+----------+---------+----------------------------------+

DEFENSE: Culture style requires objective, verifiable measures. Museum and
monument counts are defensible and reproducible.


================================================================================
SECTION 7: ACTIVITY SCORES (11 features - 1-10 scale)
================================================================================

+------------------------+----------------------+----------------------------------+
| FEATURE                | WHICH STYLE          | WHY NEEDED                       |
+------------------------+----------------------+----------------------------------+
| beach_score            | Relaxation           | Beach quality for relaxation     |
+------------------------+----------------------+----------------------------------+
| scenic_score           | Relaxation, Adventure| Mountains, sunsets, views        |
+------------------------+----------------------+----------------------------------+
| wellness_score         | Relaxation, Luxury   | Spas, yoga, retreats             |
+------------------------+----------------------+----------------------------------+
| culture_score          | Culture              | Overall cultural significance    |
+------------------------+----------------------+----------------------------------+
| hiking_score           | Adventure            | Directly answers "I like hiking" |
+------------------------+----------------------+----------------------------------+
| nightlife_score        | Adventure, Party     | Bars, clubs, entertainment       |
+------------------------+----------------------+----------------------------------+
| family_score           | Family               | Kid-friendly activities          |
+------------------------+----------------------+----------------------------------+
| luxury_score           | Luxury               | High-end experiences             |
+------------------------+----------------------+----------------------------------+
| safety_score           | Family, All          | Crime rate, stability            |
+------------------------+----------------------+----------------------------------+
| tourist_density_score  | All                  | Answers "not too touristy"       |
+------------------------+----------------------+----------------------------------+
| adventure_sports_score | Adventure            | Surfing, rafting, climbing, skiing|
+------------------------+----------------------+----------------------------------+

DEFENSE: Each score maps directly to a user preference or travel style.
The 1-10 scale is intentionally linear.


================================================================================
SECTION 8: GEOGRAPHY PROXIMITY (Binary - 2 features)
================================================================================

+---------------------+----------+---------+----------------------------------+
| FEATURE             | TYPE     | VALUES  | WHY NEEDED                       |
+---------------------+----------+---------+----------------------------------+
| near_mountains      | Binary   | 0 or 1  | Hiking, skiing, scenic views     |
+---------------------+----------+---------+----------------------------------+
| near_beach          | Binary   | 0 or 1  | Swimming, sunbathing             |
+---------------------+----------+---------+----------------------------------+

DEFENSE: Binary features are objective (verifiable on map) and address the
mentor's comment: categorical values are NOT treated as ordinal numbers.


================================================================================
SECTION 9: OTHER FEATURES (3 features)
================================================================================

+---------------------+----------------------+----------------------------------+
| FEATURE             | TYPE                  | WHY NEEDED                       |
+---------------------+----------------------+----------------------------------+
| region              | Categorical          | Geographic patterns (Europe vs   |
|                     | (Europe, Asia, etc.)  | SE Asia affects cost/style)      |
+---------------------+----------------------+----------------------------------+
| flight_cost_usd     | Integer (200-2000)   | Total budget calculation         |
+---------------------+----------------------+----------------------------------+
| dry_season_months   | Categorical          | Affects travel timing            |
+---------------------+----------------------+----------------------------------+

DEFENSE: Region captures cultural/economic patterns. Flight cost is essential
for "$1,500" budget calculation. Dry season months helps with "when to book".


================================================================================
SECTION 10: FUTURE USE (In CSV for Agent - Not in ML Training)
================================================================================

+-------------------------+----------+----------------------------------------+
| FEATURE                 | TYPE     | WHEN AGENT USES IT                     |
+-------------------------+----------+----------------------------------------+
| best_season             | String   | "When should I book?" queries          |
+-------------------------+----------+----------------------------------------+
| visa_requirement        | String   | Entry requirements questions           |
+-------------------------+----------+----------------------------------------+
| english_friendly_score  | 1-10     | Solo traveler / ease of communication  |
+-------------------------+----------+----------------------------------------+
| public_transport_score  | 1-10     | Budget traveler / no rental car needed |
+-------------------------+----------+----------------------------------------+
| latitude                | Float    | Distance calculations                  |
+-------------------------+----------+----------------------------------------+
| longitude               | Float    | Distance calculations                  |
+-------------------------+----------+----------------------------------------+

DEFENSE: These features are NOT passed to the ML model. They are stored in the
CSV for the agent to use when answering specific user questions.


================================================================================
SECTION 11: MISSING VALUE HANDLING (IN PIPELINE - NOT MANUAL)
================================================================================

IMPORTANT: Missing values are NOT manually imputed.

WHY: Manual imputation before train/test split causes data leakage.

PIPELINE STRATEGY:
- Numerical features: SimpleImputer(strategy='median')
- Binary features: SimpleImputer(strategy='most_frequent')
- Categorical features: SimpleImputer(strategy='constant', fill_value='missing')

DEFENSE:
- 'most_frequent' for binary: Missing value replaced with most common value (0 or 1)
  This is better than always 0 because missing ≠ false
- 'missing' for categorical: Explicitly indicates data was missing
  Different from any valid category value


================================================================================
SECTION 12: DATA TYPES CORRECTNESS (Mentor's Comment Applied)
================================================================================

+-------------------------+--------------------------+---------------------------+
| FEATURE TYPE            | HOW MODEL SEES IT        | IS THIS CORRECT?          |
+-------------------------+--------------------------+---------------------------+
| Scores (1-10)           | Numbers (linear scale)   | YES - scale is linear by  |
|                         |                          | design                    |
+-------------------------+--------------------------+---------------------------+
| Region                  | One-hot encoded          | YES - no order assumed    |
+-------------------------+--------------------------+---------------------------+
| near_mountains (0/1)    | Binary number            | YES - just presence       |
+-------------------------+--------------------------+---------------------------+
| dry_season_months       | One-hot encoded          | YES - no order assumed    |
+-------------------------+--------------------------+---------------------------+
| missing (categorical)   | One-hot encoded          | YES - explicit category   |
+-------------------------+--------------------------+---------------------------+

CRITICAL: No categorical feature is incorrectly treated as an ordinal number.
The mentor's comment is fully addressed.


================================================================================
SECTION 13: FEATURE COUNT SUMMARY
================================================================================

+------------------------------+----------+
| CATEGORY                     | COUNT    |
+------------------------------+----------+
| Identifiers (dropped)        | 4        |
+------------------------------+----------+
| Climate                      | 2        |
+------------------------------+----------+
| Cost                         | 3        |
+------------------------------+----------+
| Culture                      | 3        |
+------------------------------+----------+
| Activity Scores              | 11       |
+------------------------------+----------+
| Geography Proximity (binary) | 2        |
+------------------------------+----------+
| Other                        | 3        |
+------------------------------+----------+
| Future Use (agent only)      | 6        |
+------------------------------+----------+
| Target                       | 1        |
+------------------------------+----------+
| TOTAL CSV COLUMNS            | 35       |
+------------------------------+----------+

ML FEATURES (used in training): 4+24? No - ML features are 24.
Breakdown: 2+3+3+11+2+3 = 24 features


================================================================================
END OF FEATURES DOCUMENT
================================================================================
