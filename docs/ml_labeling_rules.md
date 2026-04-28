================================================================================
                    ML CLASSIFIER - LABELING RULES
                    Smart Travel Planner
================================================================================

DATE: April 28, 2026
STATUS: FINAL - APPROVED


================================================================================
SECTION 1: BINARY LABELING RULES (IF-THEN IN ORDER)
================================================================================

Each destination is evaluated against these rules IN ORDER. The first rule
that matches determines travel_style.

RULE 1: ADVENTURE
-----------------
IF (hiking_score >= 7) OR (adventure_sports_score >= 7) OR (near_mountains = 1):
    THEN travel_style = "Adventure"

Justification: High hiking score, adventure sports, or mountain proximity
directly indicate outdoor activity potential.

RULE 2: RELAXATION
-----------------
IF (beach_score >= 7) OR (scenic_score >= 7) OR (wellness_score >= 7) OR (near_beach = 1):
    THEN travel_style = "Relaxation"

Justification: Beaches, scenic views (mountains/sunsets), spas/wellness,
or beach proximity indicate relaxation potential.

RULE 3: CULTURE
---------------
IF (culture_score >= 8) OR (museum_count >= 15) OR (monument_count >= 10) OR (festival_score >= 8):
    THEN travel_style = "Culture"

Justification: High culture score, many museums, many monuments,
or major festivals indicate cultural significance.

RULE 4: LUXURY
--------------
IF (luxury_score >= 8) OR (cost_per_day_avg_usd >= 200) OR (hotel_night_avg_usd >= 150):
    THEN travel_style = "Luxury"

Justification: High luxury score or high costs indicate premium experiences.
Hotel range extended to 500, so Luxury rule can trigger via hotel price.

RULE 5: FAMILY
--------------
IF (family_score >= 7) AND (safety_score >= 7):
    THEN travel_style = "Family"

Justification: Family destinations need both kid-friendly activities AND
safety. One without the other is not suitable for families.

RULE 6: BUDGET
--------------
IF (cost_per_day_avg_usd < 65) AND (meal_budget_usd < 8) AND (hotel_night_avg_usd < 35):
    THEN travel_style = "Budget"

Justification:
- Meal under $8 covers street food in SE Asia, Eastern Europe, South America
- Hotel under $35 covers hostels and budget hotels
- Daily cost under $65 = $910 for 14 days, leaving ~$590 for flights within $1500 budget

RULE 7: NIGHTLIFE / PARTY
-------------------------
IF (nightlife_score >= 8):
    THEN travel_style = "Adventure"

Justification: High nightlife score indicates party destinations, which
align with Adventure style for young travelers.

RULE 8: DEFAULT
---------------
IF no rules match:
    THEN travel_style = "Culture"

Justification: Culture is the most common default as most destinations have
some historical/cultural significance.


================================================================================
SECTION 2: VERIFICATION EXAMPLES
================================================================================

EXAMPLE 1: Bali, Indonesia
--------------------------
Scores: hiking_score=9, beach_score=8, wellness_score=9, near_mountains=1
Evaluation: Rule 1 (hiking_score>=7) -> TRUE
RESULT: Adventure

EXAMPLE 2: Swiss Alps, Switzerland
----------------------------------
Scores: scenic_score=10, hiking_score=9, near_mountains=1
Evaluation: Rule 1 (hiking_score>=7) -> TRUE
RESULT: Adventure

EXAMPLE 3: Kyoto, Japan
-----------------------
Scores: culture_score=10, museum_count=50, monument_count=30
Evaluation: Rule 3 (culture_score>=8) -> TRUE
RESULT: Culture

EXAMPLE 4: Vietnam (Budget)
---------------------------
Scores: cost_per_day_avg_usd=35, meal_budget_usd=3, hotel_night_avg_usd=20
Evaluation: Rule 6 (all conditions true) -> TRUE
RESULT: Budget

EXAMPLE 5: Ibiza, Spain
-----------------------
Scores: nightlife_score=9, beach_score=8
Evaluation: Rule 1 FALSE, Rule 2 FALSE, Rule 7 (nightlife>=8) -> TRUE
RESULT: Adventure (Party style)

EXAMPLE 6: Paris, France
------------------------
Scores: culture_score=10, museum_count=130, luxury_score=8
Evaluation: Rule 3 (culture_score>=8) -> TRUE
RESULT: Culture

EXAMPLE 7: Dubai, UAE (Luxury by hotel price)
---------------------------------------------
Scores: hotel_night_avg_usd=350, luxury_score=7, cost_per_day_avg_usd=250
Evaluation: Rule 4 (hotel_night_avg_usd>=150) -> TRUE
RESULT: Luxury


================================================================================
SECTION 3: EXPECTED CLASS DISTRIBUTION (150 destinations)
================================================================================

+----------------+----------+---------------------------------------------+
| STYLE          | COUNT    | WHY                                         |
+----------------+----------+---------------------------------------------+
| Adventure      | 45       | Most common - hiking, mountains, party      |
+----------------+----------+---------------------------------------------+
| Culture        | 35       | Common - museums, monuments, festivals      |
+----------------+----------+---------------------------------------------+
| Relaxation     | 30       | Beaches, scenery, wellness                  |
+----------------+----------+---------------------------------------------+
| Budget         | 20       | Southeast Asia, Eastern Europe, backpacker  |
+----------------+----------+---------------------------------------------+
| Family         | 12       | Theme parks, safe beaches, kid activities   |
+----------------+----------+---------------------------------------------+
| Luxury         | 8        | High cost, premium resorts                  |
+----------------+----------+---------------------------------------------+

IMBALANCE HANDLING:
- Use macro F1 (not accuracy) for model selection
- Apply class_weight='balanced' in classifiers
- Report per-class metrics for all styles


================================================================================
SECTION 4: USER INPUT MAPPING (AGENT EXTRACTION - CORRECTED)
================================================================================

The agent does NOT extract scores from user input. It extracts keywords.

For any user query, the cheap LLM extracts:

+-------------------------+-------------------------+-----------------------------------+
| USER SAYS               | AGENT EXTRACTS          | HOW DESTINATIONS ARE FOUND        |
+-------------------------+-------------------------+-----------------------------------+
| "I like hiking"         | keyword: "hiking"       | RAG retrieves destinations with   |
|                         |                         | high hiking_score (pre-computed)  |
+-------------------------+-------------------------+-----------------------------------+
| "I want beaches"        | keyword: "beach"        | RAG retrieves destinations with   |
|                         |                         | high beach_score (pre-computed)   |
+-------------------------+-------------------------+-----------------------------------+
| "History and museums"   | keywords: "history",    | RAG retrieves destinations with   |
|                         | "museums"               | high culture_score or museum_count|
+-------------------------+-------------------------+-----------------------------------+
| "$1500 budget"          | budget: 1500            | Agent calculates daily affordable:|
|                         | duration: inferred      | (1500 - flight) / days            |
+-------------------------+-------------------------+-----------------------------------+
| "Luxury resorts"        | keyword: "luxury"       | RAG retrieves destinations with   |
|                         |                         | high luxury_score                 |
+-------------------------+-------------------------+-----------------------------------+
| "Family friendly"       | keyword: "family"       | RAG retrieves destinations with   |
|                         |                         | high family_score and safety_score|
+-------------------------+-------------------------+-----------------------------------+
| "Party and nightlife"   | keywords: "party",      | RAG retrieves destinations with   |
|                         | "nightlife"             | high nightlife_score              |
+-------------------------+-------------------------+-----------------------------------+

The 24 ML features are NEVER sent to the LLM. They are only used to:
1. Train the classifier (offline)
2. Score destinations in the database (pre-computed)
3. RAG retrieves based on pre-computed scores

This makes the agent fast and cost-effective.

The score thresholds (e.g., hiking_score >= 7) are ONLY for LABELING the
dataset during creation, NOT for runtime user query interpretation.


================================================================================
END OF LABELING RULES DOCUMENT
================================================================================
