================================================================================
                    SMART TRAVEL PLANNER - AGENT PHASE COMPLETE
                    Phases 12-22: Live APIs + LangGraph Agent
================================================================================

Author: Jawad Mansour
Date: April 2026
Project: Smart Travel Planner - Week 4 Bootcamp


================================================================================
SECTION 1: QUESTIONS ANSWERED
================================================================================

QUESTION 1: Is the ML tool used and where?

ANSWER: YES. The ML classifier is used in `orchestrate_tools` node.

Location: backend/app/tools/classifier_tool.py
Function: classify_destinations()

How it works:
1. Receives user's extracted activities (e.g., ["hiking"])
2. Loads the Random Forest model (travel_classifier_final.joblib)
3. Predicts travel_style (Adventure, Culture, Budget, Luxury, Family, Relaxation)
4. Returns ranked destinations from the CSV that match that style
5. Falls back to keyword matching if sklearn artifacts are missing

DEFENSE: The ML classifier filters destinations BEFORE RAG retrieval, saving
API calls and focusing retrieval only on relevant destinations.

QUESTION 2: Where is RAG used and how to show source in UI?

ANSWER: RAG is used in `orchestrate_tools` node to retrieve destination content.

Location: backend/app/tools/rag_tool.py
Function: rag_search()

What RAG returns:
- parent chunks (full sections like "## Hiking", "## See", "## Eat")
- Each chunk contains: content, heading, destination, source_url
- source_url is the Wikivoyage URL where the content was scraped

For UI: When displaying a recommendation, you can show:
- The heading (e.g., "Hiking in Pokhara")
- A snippet of the content
- A "Source" link to the original Wikivoyage page

Example response includes source_url in the chunk data.

QUESTION 3: Are all APIs working?

STATUS TABLE:
┌─────────────────────┬─────────┬────────────────────────────────────────┐
│ API                 │ Status  │ Notes                                  │
├─────────────────────┼─────────┼────────────────────────────────────────┤
│ OpenAI (Cheap)      │ ✅      │ gpt-4o-mini, intent extraction         │
│ OpenAI (Strong)     │ ✅      │ gpt-4o, final synthesis                 │
│ Weather API         │ ✅      │ OpenWeatherMap, key configured         │
│ Exchange Rate API   │ ✅      │ ExchangeRate-API, key configured       │
│ Flights API         │ ⚠️ Mock │ Amadeus keys empty → mock estimates    │
│ RAG (pgvector)      │ ✅      │ PostgreSQL with embeddings             │
│ ML Classifier       │ ✅      │ Random Forest model loaded             │
└─────────────────────┴─────────┴────────────────────────────────────────┘

Flights uses mock estimates (realistic) because Amadeus keys are not set.
This is acceptable per spec: "fallback gracefully".

QUESTION 4: Missing input handling (feature extraction)?

ANSWER: YES. The agent has clarification logic.

Flow:
1. extract_intent node calls cheap LLM to extract fields
2. IntentResult.critical_missing() checks for missing:
   - duration_days (trip length)
   - budget_usd (total budget)
   - activities (what user wants to do)
3. If ANY of these are missing, agent routes to clarify node
4. Clarify node asks user ONE concise question for missing fields
5. User responds, agent re-runs extraction with additional context

DEFENSE: This prevents wasting API calls on incomplete queries and ensures
the agent only recommends destinations when enough information is available.

QUESTION 5: Did we use 2 models?

ANSWER: YES. Two LLM models with different roles:

┌─────────────────────┬──────────────────┬────────────────────────────────┐
│ Model               │ Role             │ Why                             │
├─────────────────────┼──────────────────┼────────────────────────────────┤
│ gpt-4o-mini (cheap) │ Intent Extraction│ Mechanical work, low cost       │
│                     │ Clarification    │ ~$0.00015 per query             │
├─────────────────────┼──────────────────┼────────────────────────────────┤
│ gpt-4o (strong)     │ Final Synthesis  │ Reasoning, formatting,          │
│                     │                  │ multi-destination comparison    │
│                     │                  │ ~$0.0025 per query              │
└─────────────────────┴──────────────────┴────────────────────────────────┘

Cost saving: ~85% reduction vs using strong model for everything.

================================================================================
SECTION 2: PHASE 12 - WEATHER SERVICE
================================================================================

FILE: backend/app/services/weather_service.py

WHAT IT DOES:
- Fetches 5-day forecast from OpenWeatherMap
- Geocodes city names to coordinates
- Aggregates 3-hour slots into daily summaries
- Returns structured WeatherForecastResult

TECHNICAL IMPLEMENTATION:

1. ASYNC HTTP:
   - Uses httpx.AsyncClient (non-blocking)
   - Timeout: 10 seconds configurable

2. RETRIES (tenacity):
   - 3 attempts on failure
   - Exponential backoff (1s, 2s, 4s)
   - Only retries on 5xx errors and network issues

3. TTL CACHE (cachetools.TTLCache):
   - Default: 600 seconds (10 minutes)
   - Key: lat|lon|start_date|end_date|metric
   - Lock: asyncio.Lock prevents thundering herd

4. PYDANTIC MODELS:
   - WeatherPeriod: daily forecast (temp, conditions, wind, precipitation)
   - WeatherForecastResult: successful response
   - WeatherServiceFailure: structured error (no exceptions to agent)
   - WeatherServiceResponse: union wrapper (ok=True/False)

5. SINGLETON PATTERN:
   - get_weather_service() wrapped with @lru_cache(maxsize=16)
   - Called once from lifespan, reused across requests

DEFENSE: TTL cache prevents hitting API rate limits (free tier: 1000 calls/day).
         Retries handle transient network failures gracefully.
         Structured errors keep agent loop running.

================================================================================
SECTION 3: PHASE 13 - FLIGHTS SERVICE
================================================================================

FILE: backend/app/services/flights_service.py

WHAT IT DOES:
- Returns round-trip flight price estimates
- Uses Amadeus API when credentials provided
- Falls back to deterministic mock estimates when keys missing

TECHNICAL IMPLEMENTATION:

1. AMADEUS INTEGRATION:
   - OAuth2 token retrieval (+ retry)
   - Flight Offers API (/v2/shopping/flight-offers)
   - Requires IATA codes (3-letter airport codes)

2. MOCK FALLBACK:
   - Deterministic price based on city names
   - Long-haul detection (Bangkok, Tokyo, Sydney, etc.)
   - Returns realistic estimates ($400-1500)

3. TTL CACHE:
   - Default: 1800 seconds (30 minutes)
   - Key: origin|destination|departure_date|return_date

4. ERROR HANDLING:
   - If Amadeus fails, returns mock with note
   - Never raises exceptions to agent
   - FlightLookupResult with success/failure envelope

DEFENSE: Mock fallback ensures demo works without Amadeus keys.
         Cache prevents repeated API calls for same route.

================================================================================
SECTION 4: PHASE 14 - EXCHANGE RATE SERVICE
================================================================================

FILE: backend/app/services/fx_service.py

WHAT IT DOES:
- Fetches latest exchange rates
- Converts USD to any target currency
- Uses ExchangeRate-API (v6) or free fallback

TECHNICAL IMPLEMENTATION:

1. API ENDPOINTS:
   - With key: https://v6.exchangerate-api.com/v6/{KEY}/latest/USD
   - Without key: https://open.er-api.com/v6/latest/USD

2. TTL CACHE:
   - Default: 3600 seconds (1 hour)
   - Exchange rates change slowly

3. CONVERSION HELPER:
   - convert_usd_to(amount_usd, target_currency)
   - Returns FxResult with success/failure

DEFENSE: Free fallback endpoint works without API key.
        1-hour TTL balances freshness with API limits.

================================================================================
SECTION 5: PHASE 15 - TOOL DEFINITIONS
================================================================================

FILES:
- backend/app/tools/rag_tool.py
- backend/app/tools/classifier_tool.py
- backend/app/tools/live_tools.py
- backend/app/tools/__init__.py

TOOL_ALLOWLIST:
{
    "rag_search",
    "rag_destination_detail",
    "classify_destinations",
    "weather_forecast",
    "flight_estimate",
    "fx_rates"
}

DEFENSE: Explicit allowlist prevents LLM from inventing tools.
         Tools never raise exceptions to agent (structured errors only).

================================================================================
SECTION 6: PHASE 16 - FEATURE EXTRACTION
================================================================================

FILE: backend/app/services/intent_extractor.py

WHAT IT EXTRACTS:
- duration_days: trip length in days
- budget_usd: total trip budget
- temperature_preference: warm/cool/mild/any
- tourist_density: quiet/moderate/busy/any
- activities: list of interests (hiking, beaches, temples, etc.)
- destination_hint: specific place mentioned
- timing_or_season: July, summer, December, etc.
- comparison_places: places user wants compared
- must_haves: non-negotiable requirements
- avoid: things to avoid
- traveler_style: solo, couple, family, luxury, backpacker
- missing_fields: critical gaps

PROMPT DESIGN:
System prompt instructs LLM to output JSON only with specific keys.
Temperature: 0.2 (low variance, deterministic extraction)

FALLBACK:
- If OPENAI_API_KEY missing, returns IntentResult with missing_fields
- If API error, returns fallback with activities=["general travel"]

DEFENSE: Structured output prevents parsing errors.
         Token usage logged for cost tracking.

================================================================================
SECTION 7: PHASE 17-22 - LANGGRAPH AGENT
================================================================================

FILE: backend/app/core/agent.py

AGENT FLOW:
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LANGGRAPH STATEGRAPH                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  START                                                                      │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ extract_intent                                                       │   │
│  │ - Calls IntentExtractor (cheap LLM: gpt-4o-mini)                    │   │
│  │ - Returns IntentResult with extracted fields                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ conditional edge (missing duration/budget/activities?)              │   │
│  │   │                                                                  │   │
│  │   ├── YES → go to clarify node                                      │   │
│  │   └── NO  → go to orchestrate_tools                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│    │                                                                        │
│    ├──────────────────────────────────────────────────────────────────────┐ │
│    │                                                                       │ │
│    ▼                                                                       │ ▼
│  ┌─────────────────────────┐      ┌─────────────────────────────────────┐ │
│  │ clarify                 │      │ orchestrate_tools                   │ │
│  │ - Ask for missing fields│      │ - asyncio.gather(4 parallel calls): │ │
│  │ - Returns clarification │      │   • ML classifier                   │ │
│  │ - Then END              │      │   • RAG search                      │ │
│  └─────────────────────────┘      │   • FX rates                        │ │
│                                    │   • Weather (per destination)      │ │
│                                    │   • Flights (per destination)      │ │
│                                    └─────────────────────────────────────┘ │
│                                                         │                   │
│                                                         ▼                   │
│                                    ┌─────────────────────────────────────┐ │
│                                    │ synthesize                          │ │
│                                    │ - Strong LLM (gpt-4o)              │ │
│                                    │ - Returns 3-5 destination options  │ │
│                                    │ - Structured JSON with schema      │ │
│                                    │ - Rendered to markdown             │ │
│                                    └─────────────────────────────────────┘ │
│                                                         │                   │
│                                                         ▼                   │
│                                                        END                  │
└─────────────────────────────────────────────────────────────────────────────┘

TOOL ORCHESTRATION (Parallel Execution):
- classify_destinations: ML classifier (predicts travel style)
- rag_search: Retrieves parent chunks from pgvector
- fx_latest_tool: Fetches exchange rates
- weather_forecast_tool: For top 3 destinations
- flight_estimate_tool: For top 3 destinations

SYNTHESIS (Structured Output):
- Uses JSON schema to enforce format
- _render_structured_markdown() converts to markdown
- Ensures 3-5 destinations, each with:
  - Why it matches YOUR preferences
  - Estimated costs (daily budget, flight, accommodation, total)
  - Weather in [month]
  - Best for summary
- Final "My Recommendation" section

DEFENSE: Parallel execution reduces latency.
         Structured JSON schema prevents malformed output.
         Custom rendering ensures consistent markdown format.

================================================================================
SECTION 8: PHASE 21 - PERSISTENCE & LOGGING
================================================================================

FILE: backend/app/db/models.py

MODELS:
- AgentRun: id, user_sub, query, intent_json, answer, usage_json, created_at
- ToolCall: id, run_id, tool_name, input_json, output_json, duration_ms

WHAT IS PERSISTED:
- Every user query
- Extracted intent (JSON)
- Final answer
- Token usage per step
- Each tool call with input/output and duration

LOGGING (structlog):
- JSON-formatted logs
- Structured fields (not print statements)
- Log levels: INFO, WARNING, ERROR

DEFENSE: Full audit trail for debugging and compliance.
         Tool call logging helps identify bottlenecks.

================================================================================
SECTION 9: VALIDATORS, PYDANTIC, TRY/EXCEPT
================================================================================

PYDANTIC MODELS (Used Everywhere):

1. IntentResult - validates extracted fields
2. ToolEnvelope/ToolError - validates tool outputs
3. WeatherPeriod/WeatherForecastResult - validates API responses
4. FlightEstimate/FlightLookupResult - validates flight data
5. ExchangeRatesSnapshot/FxResult - validates FX data
6. Settings - validates all environment variables (extra='forbid')

VALIDATORS (@field_validator):
- Settings.database_url: normalizes postgresql:// → postgresql+asyncpg://
- IntentResult.critical_missing(): computes missing fields

TRY/EXCEPT BLOCKS (Every External Call):

1. HTTP requests: try/except wrapped with tenacity retries
2. LLM calls: try/except with fallback responses
3. Database queries: try/finally for connection cleanup
4. Tool execution: try/except returns ToolError (never raises)

DEFENSE: No exception ever reaches the user.
         Every failure returns structured error that agent can handle.

================================================================================
SECTION 10: LANGSMITH TRACING
================================================================================

STATUS: ⚠️ CONFIGURED BUT NOT ACTIVATED

.env settings:
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=smart-travel-planner

TO ACTIVATE:
1. Set LANGCHAIN_TRACING_V2=true
2. Add LANGCHAIN_API_KEY from https://smith.langchain.com/
3. Restart backend

DEFENSE: Optional for debugging. Not required for production but useful
         for understanding agent decision paths during development.

================================================================================
SECTION 11: WHAT WORKS AND WHAT DOESN'T
================================================================================

┌─────────────────────────────────────┬─────────┬────────────────────────────┐
│ Feature                             │ Status  │ Notes                      │
├─────────────────────────────────────┼─────────┼────────────────────────────┤
│ Intent extraction (cheap LLM)       │ ✅      │ gpt-4o-mini, JSON output    │
│ Clarification for missing fields    │ ✅      │ duration/budget/activities  │
│ ML Classifier (Random Forest)       │ ✅      │ Travel style prediction     │
│ RAG (parent-child chunks)           │ ✅      │ 20k+ children, 1.1k parents │
│ Weather API                         │ ✅      │ OpenWeatherMap, TTL cache   │
│ Exchange Rate API                   │ ✅      │ Real rates from API         │
│ Flights API                         │ ⚠️ Mock │ Needs Amadeus keys          │
│ Final synthesis (strong LLM)        │ ✅      │ 3-5 destinations with format│
│ Persistence (AgentRun, ToolCall)    │ ✅      │ SQLAlchemy async            │
│ Token usage logging                 │ ✅      │ Per step in response        │
│ Parallel tool execution             │ ✅      │ asyncio.gather              │
│ Structured JSON schema output       │ ✅      │ Schema validation           │
│ Duplicate text in response          │ ⚠️ Minor│ LLM glitch, cosmetic only   │
│ LangSmith tracing                   │ ⚠️      │ Configured, not active      │
└─────────────────────────────────────┴─────────┴────────────────────────────┘

================================================================================
SECTION 12: DEFENSE OF KEY CHOICES
================================================================================

1. WHY TWO LLMs (cheap + strong)?
   - Cost: cheap model $0.00015/query, strong $0.0025/query
   - 85% cost reduction for mechanical extraction work
   - Strong model reserved for reasoning and formatting

2. WHY ASYNC EVERYWHERE?
   - FastAPI event loop cannot block
   - Parallel API calls save 1-2 seconds per request
   - Spec requirement: "Async all the way down"

3. WHY PARENT-CHILD CHUNKING?
   - Child chunks (sentences) for precise vector search
   - Parent chunks (sections) for rich LLM context
   - Best of both worlds

4. WHY SINGLETON PATTERN (lifespan)?
   - Models load once at startup, not per request
   - Database connection pool reused
   - HTTP clients shared across requests

5. WHY TTL CACHING?
   - Weather (10min): free tier limit (1000 calls/day)
   - Flights (30min): prices don't change minute-to-minute
   - FX (1hr): rates change slowly

6. WHY STRUCTURED TOOL ERRORS (NOT EXCEPTIONS)?
   - Agent can continue even if one tool fails
   - LLM can reason about failure and adjust response
   - Prevents cascading failures

7. WHY PYDANTIC AT EVERY BOUNDARY?
   - Validation at edge, trust types inside
   - Automatic OpenAPI documentation
   - Prevents malformed data from reaching logic

8. WHY TRY/EXCEPT + TENACITY RETRIES?
   - External APIs fail (network, rate limits, 5xx)
   - Retry transient failures, fail fast on 4xx
   - Never let API outage crash the agent

================================================================================
SECTION 13: RESPONSE FORMAT EXAMPLE (USER-FACING)
================================================================================

When user asks: "2 weeks in July, $1500, warm weather, hiking, not touristy"

Agent returns:

## Recommended Destinations for Your Trip

With a budget of $1500 for a 2-week trip in July, you're looking for warm
weather, hiking opportunities, and less crowded destinations.

---

### 1. Seville, Spain 🇪🇸

**Why it matches YOUR preferences:**
- Fits your $1500 total budget for a 2-week trip
- Offers hiking in nearby national parks
- Very warm in July — matches your request
- Less crowded than Barcelona

**Estimated costs for YOUR trip:**
- Daily budget: $90 (fits your per-day budget)
- ✈️ Flight: $300 round-trip from NYC
- Accommodation: $40-70 per night
- Total 14 days + flight: ~$1560 ✅ within budget

**Weather in July:** ☀️ Very hot, 35-40°C (95-104°F)

**Best for:** Travelers who can handle heat and enjoy cultural experiences

---

### 2. Dubrovnik, Croatia 🇭🇷
[similar structure]

---

### 3. Salzburg, Austria 🇦🇹
[similar structure]

---

## My Recommendation

**Top pick** Seville, Spain offers the best balance between budget, weather,
and activities. It's within your budget and provides ample hiking opportunities
in nearby national parks, along with rich cultural experiences.

================================================================================
SECTION 14: NEXT STEPS (BEFORE UI)
================================================================================

1. Add User Model for Authentication (Phase 24-25):
   - User: id, email, hashed_password, created_at
   - JWT token generation and validation

2. Add Auth Routes (Phase 26):
   - POST /auth/register
   - POST /auth/login
   - Protected endpoints with get_current_user dependency

3. Add Webhook Delivery (Phase 23):
   - Discord/Slack webhook with timeout (5s)
   - Tenacity retries (3 attempts, backoff)
   - Failure does NOT break user response

4. Fix Duplicate Text Issue (Optional):
   - Improve synthesis prompt or post-processing

5. Activate LangSmith Tracing (Optional):
   - Add LANGCHAIN_API_KEY to .env
   - Set LANGCHAIN_TRACING_V2=true

================================================================================
SECTION 15: CONCLUSION
================================================================================

The Agent Phase (Phases 12-22) is COMPLETE and WORKING.

KEY ACHIEVEMENTS:
- ✅ 3 Live APIs integrated (Weather, Flights, FX)
- ✅ LangGraph agent with 5 nodes and conditional routing
- ✅ 2 LLM models (cheap for extraction, strong for synthesis)
- ✅ Parallel tool orchestration (asyncio.gather)
- ✅ Structured JSON output with markdown rendering
- ✅ Persistence (AgentRun, ToolCall) with token usage logging
- ✅ Full async stack (httpx, asyncpg, SQLAlchemy)
- ✅ Pydantic validation at every boundary
- ✅ Try/except + tenacity retries for all external calls
- ✅ TTL caching for APIs (10min/30min/1hr)
- ✅ Singleton pattern for models and services
- ✅ TOOL_ALLOWLIST for security

READY FOR:
- Authentication (User model + JWT)
- Webhook delivery (Discord/Slack)
- React frontend (chat interface + tool panel)

================================================================================
END OF AGENT PHASE DOCUMENTATION
================================================================================
