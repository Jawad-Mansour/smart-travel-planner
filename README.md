```text
================================================================================
                    SMART TRAVEL PLANNER
                    AI-Powered Travel Agent
================================================================================

A production-ready travel planning agent that combines ML classification, RAG
retrieval, live APIs, and LangGraph orchestration to deliver personalized
multi-destination travel recommendations.


================================================================================
TABLE OF CONTENTS
================================================================================

1.  Project Overview
2.  Architecture Diagram
3.  Features
4.  Technology Stack
5.  Project Structure
6.  Installation & Setup
7.  Environment Variables
8.  Running the Application
9.  API Documentation
10. Agent Workflow (Detailed)
11. Component Deep Dive
    - ML Classifier
    - RAG Pipeline
    - Live APIs
    - LangGraph Agent
    - Authentication
    - Webhook Delivery
12. Defense of Key Decisions
13. Performance Metrics
14. Cost Analysis (Per Query)
15. Testing
16. Deployment
17. Troubleshooting
18. Deliverables Checklist
19. License


================================================================================
SECTION 1: PROJECT OVERVIEW
================================================================================

Smart Travel Planner is an intelligent agent that helps users find the perfect
travel destination based on their preferences.

User: "I have 2 weeks in July, $1500, want warm weather and hiking, not too
touristy"

Agent Response:
├── 3-5 destination suggestions
├── Personalized "Why it matches YOUR preferences"
├── Estimated costs (daily budget, flight, accommodation, total)
├── Weather forecast for requested month
├── "Best for" summary
└── Final recommendation with top pick

The agent uses:
- 🧠 ML Classifier → Matches user preferences to travel styles
- 📚 RAG (Retrieval-Augmented Generation) → Fetches detailed destination content
- ☁️ Live APIs → Real-time weather, flight estimates, exchange rates
- 🔗 LangGraph → Orchestrates tools and manages conversation flow
- 🔔 Webhook → Delivers travel plans to Discord/Slack


================================================================================
SECTION 2: ARCHITECTURE DIAGRAM
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                 │
│                         React + TypeScript + Tailwind                      │
│                                    │                                        │
│                                    ▼                                        │
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FASTAPI BACKEND                                  │
│                      (Async, Dependency Injection)                         │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      LANGGRAPH AGENT                                 │   │
│  │                                                                      │   │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │   │
│  │   │  Extract     │───▶│  Clarify     │───▶│  Orchestrate │          │   │
│  │   │  Intent      │    │  (if needed) │    │  Tools       │          │   │
│  │   └──────────────┘    └──────────────┘    └──────────────┘          │   │
│  │         │                                       │                     │   │
│  │         │                                       ▼                     │   │
│  │         │                              ┌────────────────┐             │   │
│  │         │                              │   asyncio.     │             │   │
│  │         │                              │    gather()    │             │   │
│  │         │                              └────────────────┘             │   │
│  │         │                              │    │    │    │               │   │
│  │         │                ┌─────────────┼────┼────┼────┼─────────────┐ │   │
│  │         │                ▼             ▼    ▼    ▼    ▼             ▼ │   │
│  │         │         ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  │         │         │   ML     │  │   RAG    │  │ Weather  │  │ Flights  │
│  │         │         │Classifier│  │  Search  │  │   API    │  │   API    │
│  │         │         └──────────┘  └──────────┘  └──────────┘  └──────────┘
│  │         │              │              │             │             │     │
│  │         │              ▼              ▼             ▼             ▼     │
│  │         │         ┌──────────────────────────────────────────────────┐ │
│  │         │         │              Synthesize (Strong LLM)             │ │
│  │         │         │            Format → Markdown Response            │ │
│  │         │         └──────────────────────────────────────────────────┘ │
│  │         │                              │                                │
│  │         ▼                              ▼                                │
│  │    ┌──────────┐                  ┌──────────┐                          │
│  │    │  Save to │                  │ Webhook  │                          │
│  │    │   DB     │                  │ (Discord)│                          │
│  │    └──────────┘                  └──────────┘                          │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             DATA LAYER                                      │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │    PostgreSQL   │  │     pgvector    │  │    sentence-transformers    │ │
│  │   (Users, Runs, │  │  (Vector Store) │  │     (all-MiniLM-L6-v2)       │ │
│  │    Tool Calls)  │  │                 │  │                             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


================================================================================
SECTION 3: FEATURES
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│ CORE FEATURES                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ✅ Intent Extraction (Cheap LLM)                                          │
│     - Extracts duration, budget, activities, travel style                  │
│     - Identifies missing fields for clarification                          │
│                                                                             │
│  ✅ ML Classification (Random Forest)                                       │
│     - Predicts travel style: Adventure, Culture, Budget, Luxury, Family,   │
│       Relaxation                                                            │
│     - Test F1: 0.8940, Test Accuracy: 0.8947                               │
│                                                                             │
│  ✅ RAG Retrieval (Parent-Child Chunking)                                   │
│     - 25 destinations indexed from Wikivoyage                              │
│     - 1,184 parent chunks, 20,047 child chunks                             │
│     - 384-dim embeddings with HNSW index                                   │
│                                                                             │
│  ✅ Live APIs                                                               │
│     - Weather: OpenWeatherMap (10min TTL cache)                            │
│     - Flights: Amadeus + mock fallback (30min TTL)                         │
│     - Exchange Rates: ExchangeRate-API + fallback (1hr TTL)                │
│                                                                             │
│  ✅ LangGraph Agent                                                         │
│     - 5-node StateGraph with conditional routing                           │
│     - Parallel tool orchestration (asyncio.gather)                         │
│     - Clarification for missing information                                │
│     - Structured JSON output with markdown rendering                       │
│                                                                             │
│  ✅ Two-LLM Architecture                                                    │
│     - Cheap (gpt-4o-mini): Intent extraction, clarification                │
│     - Strong (gpt-4o): Final synthesis with formatting                     │
│     - 85% cost reduction                                                    │
│                                                                             │
│  ✅ Persistence                                                             │
│     - AgentRun, ToolCall, User models (SQLAlchemy async)                   │
│     - Token usage logging per step                                          │
│                                                                             │
│  ✅ Authentication                                                          │
│     - JWT-based authentication (register, login, protected routes)         │
│     - bcrypt password hashing                                               │
│     - User-scoped chat history                                              │
│                                                                             │
│  ✅ Webhook Delivery                                                        │
│     - Discord/Slack integration                                             │
│     - Tenacity retries (3 attempts, exponential backoff)                   │
│     - Timeout 5 seconds, failure does NOT block response                   │
│                                                                             │
│  ✅ React Frontend                                                          │
│     - Chat interface with SSE streaming                                    │
│     - Tool panel showing RAG chunks, ML prediction, API status             │
│     - Login/Register pages                                                 │
│     - Chat history sidebar                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


================================================================================
SECTION 4: TECHNOLOGY STACK
================================================================================

┌──────────────────────┬─────────────────────────────────────────────────────┐
│ Category             │ Technology                                          │
├──────────────────────┼─────────────────────────────────────────────────────┤
│ Backend Framework    │ FastAPI (async)                                     │
│ Agent Framework      │ LangGraph                                           │
│ LLM Provider         │ OpenAI (GPT-4o-mini + GPT-4o)                       │
│ Database             │ PostgreSQL + pgvector                               │
│ ORM                  │ SQLAlchemy 2.0 (async)                              │
│ ML Framework         │ scikit-learn + XGBoost                              │
│ Embeddings           │ sentence-transformers (all-MiniLM-L6-v2)            │
│ HTTP Client          │ httpx (async)                                       │
│ Caching              │ cachetools.TTLCache + lru_cache                     │
│ Retries              │ tenacity                                            │
│ Validation           │ Pydantic (extra='forbid')                           │
│ Logging              │ structlog (JSON)                                    │
│ Frontend             │ React + TypeScript + Vite + Tailwind CSS            │
│ Containerization     │ Docker + Docker Compose                             │
│ CI/CD                │ GitHub Actions                                      │
│ Tracing              │ LangSmith                                           │
└──────────────────────┴─────────────────────────────────────────────────────┘


================================================================================
SECTION 5: PROJECT STRUCTURE
================================================================================

smart-travel-planner/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── auth.py          # Register, login, token refresh
│   │   │   │   └── travel.py        # POST /plan (SSE), GET /history
│   │   │   └── deps.py              # FastAPI Depends (session, user, services)
│   │   ├── core/
│   │   │   ├── agent.py             # LangGraph StateGraph (5 nodes)
│   │   │   ├── config.py            # Settings (extra='forbid')
│   │   │   ├── logging.py           # structlog configuration
│   │   │   └── security.py          # JWT, bcrypt
│   │   ├── db/
│   │   │   ├── models.py            # User, AgentRun, ToolCall
│   │   │   └── session.py           # Async engine, session factory
│   │   ├── schemas/
│   │   │   ├── intent.py            # IntentResult with critical_missing()
│   │   │   ├── tools.py             # ToolEnvelope, ToolError
│   │   │   └── auth.py              # LoginRequest, RegisterRequest, TokenResponse
│   │   ├── services/
│   │   │   ├── rag_service.py       # Parent-child RAG, threshold 0.48
│   │   │   ├── weather_service.py   # OpenWeatherMap + TTL cache
│   │   │   ├── flights_service.py   # Amadeus + mock fallback
│   │   │   ├── fx_service.py        # ExchangeRate-API + fallback
│   │   │   ├── intent_extractor.py  # Cheap LLM extraction
│   │   │   └── webhook_service.py   # Discord/Slack delivery
│   │   └── tools/
│   │       ├── rag_tool.py          # RAG search wrapper
│   │       ├── classifier_tool.py   # ML classifier wrapper
│   │       └── live_tools.py        # Weather, flights, FX wrappers
│   ├── ml/
│   │   ├── data/
│   │   │   └── destinations_raw.csv  # 155 destinations, 35 features
│   │   ├── models/
│   │   │   ├── travel_classifier_final.joblib
│   │   │   ├── preprocessor.joblib
│   │   │   └── label_encoder.joblib
│   │   └── notebooks/               # 4 notebooks (EDA to tuning)
│   ├── rag/
│   │   ├── data/
│   │   │   ├── clean/               # 25 .md files
│   │   │   ├── raw/                 # 25 .html files
│   │   │   ├── metadata/            # 25 .json files
│   │   │   └── chunks/              # chunks.json
│   │   └── scripts/                 # collect, chunk, embed, test
│   └── main.py                      # FastAPI entrypoint
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Chat/
│   │   │   ├── Auth/
│   │   │   └── Sidebar/
│   │   ├── pages/
│   │   ├── services/
│   │   └── App.tsx
│   └── package.json
│
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── pyproject.toml
└── README.md


================================================================================
SECTION 6: INSTALLATION & SETUP
================================================================================

PREREQUISITES

- Python 3.11+
- Docker Desktop
- Node.js 18+ (for frontend)
- OpenAI API key (https://platform.openai.com/api-keys)
- OpenWeatherMap API key (https://home.openweathermap.org/api_keys)
- (Optional) Amadeus API keys for live flights
- (Optional) ExchangeRate-API key
- (Optional) LangSmith API key for tracing


STEP 1: CLONE REPOSITORY

```bash
git clone https://github.com/Jawad-Mansour/smart-travel-planner.git
cd smart-travel-planner
```


STEP 2: ENVIRONMENT CONFIGURATION

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# REQUIRED
OPENAI_API_KEY=sk-proj-...
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:55432/travel_planner

# OPTIONAL (for live data)
WEATHER_API_KEY=your_key
AMADEUS_API_KEY=your_key
AMADEUS_API_SECRET=your_secret
EXCHANGE_RATE_API_KEY=your_key

# LANGSMITH (for tracing)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=smart_travel_planner
```


STEP 3: BACKEND SETUP

```bash
# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows
# or source .venv/bin/activate  # Mac/Linux

# Install dependencies (using uv for speed)
uv pip install -e .
```


STEP 4: DATABASE SETUP

```bash
# Start PostgreSQL with pgvector
docker compose up -d postgres

# Run RAG pipeline (creates tables + loads embeddings)
python backend/rag/scripts/setup_database.py
python backend/rag/scripts/embed_and_store.py
```


STEP 5: RUN BACKEND

```bash
uvicorn backend.main:app --reload --port 8000
```


STEP 6: FRONTEND SETUP

```bash
cd frontend
npm install
npm run dev
```


STEP 7: DOCKER (Full Stack)

```bash
docker compose up --build
```

Access:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs


================================================================================
SECTION 7: ENVIRONMENT VARIABLES
================================================================================

┌─────────────────────────────────────┬───────────────────────────────────────┐
│ Variable                            │ Purpose                               │
├─────────────────────────────────────┼───────────────────────────────────────┤
│ OPENAI_API_KEY                      │ Required - LLM access                 │
│ OPENAI_CHEAP_MODEL                  │ Default: gpt-4o-mini                  │
│ OPENAI_STRONG_MODEL                 │ Default: gpt-4o                       │
│ DATABASE_URL                        │ PostgreSQL connection string         │
│ WEATHER_API_KEY                     │ OpenWeatherMap (optional)             │
│ WEATHER_CACHE_TTL_SECONDS           │ Default: 600 (10 min)                 │
│ AMADEUS_API_KEY                     │ Flights (optional, mock fallback)     │
│ AMADEUS_API_SECRET                  │ Flights (optional)                    │
│ FLIGHTS_CACHE_TTL_SECONDS           │ Default: 1800 (30 min)                │
│ EXCHANGE_RATE_API_KEY               │ FX rates (optional, fallback works)   │
│ FX_CACHE_TTL_SECONDS                │ Default: 3600 (1 hour)                │
│ LANGCHAIN_TRACING_V2                │ Enable LangSmith tracing              │
│ LANGCHAIN_API_KEY                   │ LangSmith API key                     │
│ LANGCHAIN_PROJECT                   │ Project name for traces               │
│ DEFAULT_FLIGHT_ORIGIN               │ Default: NYC                          │
│ RAG_RELEVANCE_THRESHOLD             │ Default: 0.48                         │
│ SECRET_KEY                          │ JWT signing (change in production)    │
│ JWT_SECRET_KEY                      │ JWT encoding (change in production)   │
│ CORS_ALLOWED_ORIGINS                │ Frontend URLs                         │
└─────────────────────────────────────┴───────────────────────────────────────┘


================================================================================
SECTION 8: RUNNING THE APPLICATION
================================================================================

LOCAL DEVELOPMENT (Backend only)

```bash
# Terminal 1: PostgreSQL
docker compose up -d postgres

# Terminal 2: Backend
source .venv/Scripts/activate
uvicorn backend.main:app --reload --port 8000

# Terminal 3: Test with curl
curl -X POST http://localhost:8000/api/travel/plan \
  -H "Content-Type: application/json" \
  -d '{"query":"I have 2 weeks in July, $1500, want warm weather and hiking"}'
```


LOCAL DEVELOPMENT (Full Stack)

```bash
# Terminal 1: PostgreSQL + Backend + Frontend
docker compose up --build

# Access frontend at http://localhost:5173
```


TEST QUERIES

```bash
# Complete query (should return 3-5 destinations)
curl -X POST http://localhost:8000/api/travel/plan \
  -H "Content-Type: application/json" \
  -d '{"query":"10 days in October, budget $2200, hiking, mild weather"}'

# Missing fields (should ask clarification)
curl -X POST http://localhost:8000/api/travel/plan \
  -H "Content-Type: application/json" \
  -d '{"query":"I want to go hiking"}'

# Specific destination
curl -X POST http://localhost:8000/api/travel/plan \
  -H "Content-Type: application/json" \
  -d '{"query":"Tell me about hiking trails in Kathmandu"}'

# History
curl -X GET http://localhost:8000/api/travel/history -H "X-User-Id: test-user"
```


================================================================================
SECTION 9: API DOCUMENTATION
================================================================================

POST /api/travel/plan
─────────────────────
Streaming travel plan generation (Server-Sent Events)

Request Body:
{
  "query": "string"  # User's travel question
}

Response (SSE):
data: {"answer": "markdown string", "usage": [...], "intent": {...}}

Headers:
X-User-Id: optional (demo identity, replaced by JWT in production)


GET /api/travel/history
───────────────────────
Get past travel plans for authenticated user

Response:
{
  "items": [
    {
      "id": "uuid",
      "query": "string",
      "answer": "string",
      "created_at": "timestamp"
    }
  ]
}


POST /auth/register
───────────────────
Create new user account

Request Body:
{
  "email": "user@example.com",
  "password": "string",
  "full_name": "string"
}

Response:
{
  "access_token": "jwt_token",
  "token_type": "bearer"
}


POST /auth/login
────────────────
Authenticate existing user

Request Body:
{
  "email": "user@example.com",
  "password": "string"
}

Response:
{
  "access_token": "jwt_token",
  "token_type": "bearer"
}


GET /health
───────────
Health check

Response:
{"status": "ok"}


GET /docs
─────────
Auto-generated OpenAPI documentation (FastAPI)


================================================================================
SECTION 10: AGENT WORKFLOW (DETAILED)
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE AGENT WORKFLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: USER INPUT                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ "10 days in October, budget $2200, want hiking and cool weather"    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  PHASE 2: INTENT EXTRACTION (Cheap LLM - GPT-4o-mini)                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Input: User query                                                   │   │
│  │ Output: {                                                           │   │
│  │   duration_days: 10,                                                │   │
│  │   budget_usd: 2200,                                                 │   │
│  │   activities: ["hiking"],                                           │   │
│  │   temperature_preference: "cool",                                   │   │
│  │   timing_or_season: "October",                                      │   │
│  │   missing_fields: []                                                │   │
│  │ }                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  PHASE 3: CONDITIONAL ROUTING                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Check missing_fields:                                               │   │
│  │   - If missing duration/budget/activities → CLARIFY node            │   │
│  │   - If complete → ORCHESTRATE_TOOLS node                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  PHASE 4: TOOL ORCHESTRATION (Parallel Execution)                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐    │   │
│  │   │  ML Classifier  │   │  RAG Search     │   │  Live APIs      │    │   │
│  │   │  (500-700ms)    │   │  (600-1000ms)   │   │  (300-500ms)    │    │   │
│  │   └────────┬────────┘   └────────┬────────┘   └────────┬────────┘    │   │
│  │            │                     │                     │              │   │
│  │            ▼                     ▼                     ▼              │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │  asyncio.gather() - All run simultaneously                  │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  │  ML Classifier:                                                       │   │
│  │    - Activities "hiking" → Adventure style                           │   │
│  │    - Returns: Bhutan, Iceland, Switzerland, Canada                   │   │
│  │                                                                      │   │
│  │  RAG Search:                                                          │   │
│  │    - Retrieves parent chunks for each destination                    │   │
│  │    - Returns: "Hiking" sections, trail names, difficulty            │   │
│  │                                                                      │   │
│  │  Live APIs:                                                           │   │
│  │    - Weather: Forecast for October (temps, conditions)              │   │
│  │    - Flights: Round-trip estimates from NYC                         │   │
│  │    - FX: Exchange rates (USD → local)                               │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  PHASE 5: SYNTHESIS (Strong LLM - GPT-4o)                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Input: Intent + tool_outputs                                        │   │
│  │                                                                      │   │
│  │ Structured JSON output:                                             │   │
│  │ {                                                                   │   │
│  │   "intro": "...",                                                   │   │
│  │   "destinations": [                                                 │   │
│  │     {                                                               │   │
│  │       "name": "Bhutan",                                             │   │
│  │       "country": "Bhutan",                                          │   │
│  │       "flag_emoji": "🇧🇹",                                          │   │
│  │       "why_matches": [                                              │   │
│  │         "Fits your $2200 budget",                                   │   │
│  │         "Matches your interest in mountain hikes",                  │   │
│  │         "Cool weather in October"                                   │   │
│  │       ],                                                            │   │
│  │       "daily_budget_line": "$200 fits your $220/day ✅",            │   │
│  │       "flight_line": "✈️ Flight: $815 round-trip",                  │   │
│  │       "accommodation_line": "$50-$100 per night",                   │   │
│  │       "total_line": "Total: ~$2171 ✅",                             │   │
│  │       "weather_line": "🌧️ Light rain, 10-21°C",                    │   │
│  │       "best_for": "Solo adventurers seeking serene mountain hikes"  │   │
│  │     }                                                               │   │
│  │   ],                                                                │   │
│  │   "recommendation_title": "Top pick",                               │   │
│  │   "recommendation_body": "Bhutan offers the best combination..."    │   │
│  │ }                                                                   │   │
│  │                                                                      │   │
│  │ Rendered to markdown with _render_structured_markdown()             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  PHASE 6: PERSISTENCE                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ - Save AgentRun to PostgreSQL (user, query, intent, answer)         │   │
│  │ - Save ToolCall records (tool name, input, output, duration)        │   │
│  │ - Log token usage (prompt_tokens, completion_tokens)                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  PHASE 7: WEBHOOK DELIVERY (Background - Non-blocking)                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ - Send travel plan to Discord/Slack webhook                         │   │
│  │ - Tenacity retries: 3 attempts, exponential backoff                 │   │
│  │ - Timeout: 5 seconds                                                │   │
│  │ - Failure does NOT affect user response                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


================================================================================
SECTION 11: COMPONENT DEEP DIVE
================================================================================


11.1 ML CLASSIFIER
──────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│ MODEL DETAILS                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Algorithm: Random Forest Classifier                                        │
│ Features: 24 (climate, cost, culture, activity scores, binary, categorical)│
│ Classes: 6 (Adventure, Budget, Culture, Family, Luxury, Relaxation)        │
│ Training: SMOTENC for class imbalance, 5-fold CV, GridSearchCV             │
│                                                                             │
│ Performance:                                                                │
│   - Test F1 (macro): 0.8940                                                │
│   - Test Accuracy: 0.8947                                                  │
│   - Validation-Test Gap: 0.0043 (no overfitting)                           │
│                                                                             │
│ Feature Groups:                                                             │
│   - Climate: avg_annual_temp_c, seasonal_range_c                           │
│   - Cost: cost_per_day_avg_usd, meal_budget_usd, hotel_night_avg_usd,      │
│            flight_cost_usd                                                  │
│   - Culture: museum_count, monument_count, festival_score                  │
│   - Activity Scores (1-10): beach, scenic, wellness, culture, hiking,      │
│     nightlife, family, luxury, safety, tourist_density, adventure_sports   │
│   - Binary: near_mountains, near_beach                                     │
│   - Categorical: region, dry_season_months (OneHotEncoded)                 │
│                                                                             │
│ Saved Artifacts:                                                            │
│   - travel_classifier_final.joblib (model)                                 │
│   - preprocessor.joblib (ColumnTransformer)                                │
│   - label_encoder.joblib (class mapping)                                   │
│                                                                             │
│ Fallback: If model files missing, uses keyword-based style inference       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


11.2 RAG PIPELINE
─────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│ PARENT-CHILD CHUNKING                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Why Parent-Child:                                                           │
│   - Small chunks (sentences) for precise vector search                     │
│   - Large chunks (sections) for rich LLM context                           │
│   - Best of both worlds                                                     │
│                                                                             │
│ Statistics:                                                                 │
│   - Destinations: 25 (Wikivoyage)                                          │
│   - Parent chunks: 1,184 (full sections)                                   │
│   - Child chunks: 20,047 (sentences)                                       │
│   - Embeddings: 384-dim (all-MiniLM-L6-v2)                                 │
│   - Index: HNSW for fast similarity search                                 │
│                                                                             │
│ Retrieval Flow:                                                             │
│   1. User query → Embed with sentence-transformers                         │
│   2. Search CHILD chunks (cosine similarity)                               │
│   3. Filter by relevance threshold (0.48)                                  │
│   4. Keyword fallback if results below threshold                           │
│   5. Apply MMR deduplication (λ=0.5)                                       │
│   6. Fetch PARENT chunks (full context)                                    │
│   7. Return to agent                                                       │
│                                                                             │
│ Chunking Strategy:                                                          │
│   - Split by headings (##, ###) into sections                              │
│   - Each section = PARENT chunk                                            │
│   - Split section into SENTENCES = CHILD chunks                            │
│   - Long sentences (>500 chars) split by clauses                           │
│   - No overlap (structure-aware)                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


11.3 LIVE APIS
──────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│ WEATHER SERVICE (OpenWeatherMap)                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Features:                                                                   │
│   - Geocoding: city name → coordinates                                      │
│   - Forecast: 5-day / 3-hour steps                                         │
│   - Aggregates 3-hour slots into daily summaries                           │
│                                                                             │
│ Caching: TTLCache, 10 minutes, keyed by lat|lon|start|end|metric           │
│ Retries: tenacity, 3 attempts, exponential backoff (1-10s)                 │
│ Timeout: 10 seconds                                                         │
│                                                                             │
│ Output: WeatherPeriod (date, temp_min_c, temp_max_c, conditions_summary,   │
│          precipitation_probability_max, wind_speed_max_m_s)                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ FLIGHTS SERVICE (Amadeus + Mock Fallback)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Features:                                                                   │
│   - Amadeus OAuth2 → Flight Offers API                                     │
│   - Requires IATA codes (3-letter airport codes)                           │
│   - Mock fallback with deterministic prices when keys missing              │
│                                                                             │
│ Caching: TTLCache, 30 minutes, keyed by origin|destination|departure|return│
│ Retries: tenacity, 3 attempts, exponential backoff (1-12s)                 │
│ Timeout: 15 seconds                                                         │
│                                                                             │
│ Output: FlightEstimate (origin_display, destination_display,               │
│          round_trip_price_usd_estimate, source, note)                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ EXCHANGE RATE SERVICE (ExchangeRate-API + Fallback)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Features:                                                                   │
│   - With API key: v6.exchangerate-api.com                                  │
│   - Without API key: open.er-api.com (free fallback)                       │
│   - Conversion helper: convert_usd_to(amount, target_currency)              │
│                                                                             │
│ Caching: TTLCache, 1 hour, keyed by base_currency                          │
│ Retries: tenacity, 3 attempts, exponential backoff (1-10s)                 │
│ Timeout: 10 seconds                                                         │
│                                                                             │
│ Output: ExchangeRatesSnapshot (base_code, rates, source, note)             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


11.4 LANGGRAPH AGENT
────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│ AGENT STATE                                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ StateGraph(dict) with fields:                                              │
│   - user_query: str                                                         │
│   - intent: IntentResult                                                   │
│   - tool_results: dict                                                      │
│   - answer: str                                                             │
│   - usage_parts: list                                                       │
│                                                                             │
│ Nodes:                                                                      │
│   1. extract_intent → calls IntentExtractor (cheap LLM)                    │
│   2. clarify → asks for missing duration/budget/activities                 │
│   3. orchestrate_tools → parallel tool calls (asyncio.gather)              │
│   4. synthesize → strong LLM with JSON schema                              │
│                                                                             │
│ Edges:                                                                      │
│   - START → extract_intent                                                  │
│   - extract_intent → conditional (missing_fields? clarify : tools)         │
│   - clarify → END                                                           │
│   - orchestrate_tools → synthesize                                          │
│   - synthesize → END                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


11.5 AUTHENTICATION
───────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│ JWT AUTHENTICATION                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ User Model:                                                                 │
│   - id (UUID)                                                               │
│   - email (unique, indexed)                                                │
│   - hashed_password (bcrypt)                                               │
│   - full_name (optional)                                                   │
│   - is_active (boolean)                                                    │
│   - created_at (timestamp)                                                 │
│                                                                             │
│ Endpoints:                                                                  │
│   - POST /auth/register → bcrypt hash, return JWT                          │
│   - POST /auth/login → verify password, return JWT                         │
│   - GET /auth/me → requires Bearer token, returns user info                │
│                                                                             │
│ JWT Settings:                                                               │
│   - Algorithm: HS256                                                        │
│   - Expiry: 60 minutes (configurable)                                      │
│   - Secret: from SECRET_KEY env var                                        │
│                                                                             │
│ Protected Routes:                                                           │
│   - POST /api/travel/plan → requires get_current_user                      │
│   - GET /api/travel/history → scoped to user                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


11.6 WEBHOOK DELIVERY
────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│ DISCORD/SLACK WEBHOOK                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ WebhookService:                                                             │
│   - Async httpx POST to webhook URL                                        │
│   - Tenacity retries: 3 attempts, exponential backoff (1s, 2s, 4s)        │
│   - Timeout: 5 seconds                                                      │
│   - Called as background task (does NOT block response)                    │
│                                                                             │
│ Payload:                                                                    │
│   - User query                                                              │
│   - Travel plan answer (markdown)                                          │
│   - Tool call summary                                                       │
│   - Timestamp                                                               │
│                                                                             │
│ Failure Handling:                                                           │
│   - Logs error with structlog                                               │
│   - User still receives response (webhook failure is non-blocking)         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


================================================================================
SECTION 12: DEFENSE OF KEY DECISIONS
================================================================================


12.1 WHY TWO LLMs (CHEAP + STRONG)?
────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│ DECISION: Use gpt-4o-mini for extraction, gpt-4o for synthesis             │
│                                                                             │
│ DEFENSE:                                                                    │
│   - Extraction is mechanical work (parsing dates, budgets, activities)     │
│   - Cheap model cost: ~$0.00015 per query                                  │
│   - Strong model cost: ~$0.0025 per query                                  │
│   - 85% cost reduction for the same quality extraction                     │
│   - Spec requirement: "Route a cheap one to mechanical work"               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


12.2 WHY PARENT-CHILD CHUNKING?
────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│ DECISION: Store sentences as child chunks, sections as parent chunks       │
│                                                                             │
│ DEFENSE:                                                                    │
│   - Child chunks (sentences) give precise vector matches                   │
│   - Parent chunks (sections) give LLM rich context                         │
│   - Best of both worlds - no chunk size trade-off                          │
│   - Recommended as #1 technique in Advanced RAG Guide                      │
│   - Spec requirement: "Justify your chunk size and overlap"                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


12.3 WHY ASYNC EVERYWHERE?
──────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│ DECISION: All routes, tools, DB calls, HTTP calls are async                │
│                                                                             │
│ DEFENSE:                                                                    │
│   - Agents are I/O-bound (waiting for LLM, APIs, database)                 │
│   - Blocking calls freeze the event loop                                   │
│   - Async enables concurrent request handling                              │
│   - asyncio.gather reduces latency by 2-3 seconds per query                │
│   - Spec requirement: "Async all the way down"                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


12.4 WHY DEPENDENCY INJECTION (Depends)?
─────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│ DECISION: Use FastAPI Depends() for all dependencies                       │
│                                                                             │
│ DEFENSE:                                                                    │
│   - Makes code testable (override dependencies in tests)                   │
│   - No globals scattered across modules                                    │
│   - Function signature tells you exactly what it needs                     │
│   - Automatic lifecycle management (yield for sessions)                    │
│   - Spec requirement: "Dependency Injection - Use FastAPI's Depends"       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


12.5 WHY LIFESPAN SINGLETONS?
─────────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│ DECISION: Load ML model, embedder, LLM client once in lifespan             │
│                                                                             │
│ DEFENSE:                                                                    │
│   - ML model is 80MB - loading per request would be catastrophic           │
│   - sentence-transformer loads once, reused for all queries                │
│   - HTTP connection pools shared across requests                           │
│   - Clean shutdown/dispose of resources                                     │
│   - Spec requirement: "Singletons - Done Right"                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


12.6 WHY TTL CACHING?
─────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│ DECISION: Weather (10min), Flights (30min), FX (1hr) TTL caches            │
│                                                                             │
│ DEFENSE:                                                                    │
│   - Weather doesn't change minute-to-minute                                │
│   - Flight prices are stable over 30 minutes                               │
│   - Exchange rates change slowly                                            │
│   - Respects API rate limits (OpenWeatherMap: 1000 calls/day)              │
│   - Reduces latency (cache hit: <1ms vs API: 500ms)                        │
│   - Spec requirement: "TTL cache on tool responses"                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


12.7 WHY STRUCTURED TOOL ERRORS (NOT EXCEPTIONS)?
──────────────────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│ DECISION: Tools return ToolError, never raise exceptions                   │
│                                                                             │
│ DEFENSE:                                                                    │
│   - Agent can continue even if one tool fails                              │
│   - LLM can reason about failure and adjust response                       │
│   - Prevents cascading failures                                             │
│   - Logs structured error without crashing                                  │
│   - Spec requirement: "Tool failures inside the agent loop should be       │
│     returned to the LLM as structured errors"                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


12.8 WHY PYDANTIC AT EVERY BOUNDARY?
─────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│ DECISION: Validate all external inputs with Pydantic                       │
│                                                                             │
│ DEFENSE:                                                                    │
│   - Validate at the edge, trust types inside                               │
│   - Automatic OpenAPI documentation                                        │
│   - Prevents malformed data from reaching business logic                   │
│   - extra='forbid' catches typos in .env                                   │
│   - Spec requirement: "Every external boundary is a Pydantic model"        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


12.9 WHY TENACITY RETRIES?
──────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│ DECISION: Retry external API calls with exponential backoff                │
│                                                                             │
│ DEFENSE:                                                                    │
│   - Networks are unreliable                                                 │
│   - Transient failures (timeout, 5xx) often resolve on retry               │
│   - Exponential backoff prevents hammering failing APIs                    │
│   - 3 attempts is sufficient (1s, 2s, 4s wait)                             │
│   - Does NOT retry 4xx errors (they will fail the same way)                │
│   - Spec requirement: "Wrap them with timeouts and retries with backoff"   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


================================================================================
SECTION 13: PERFORMANCE METRICS
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│ AGENT PERFORMANCE                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Typical Query Latency:                                                      │
│   - Intent extraction: ~500ms                                              │
│   - ML Classifier: ~500-700ms                                              │
│   - RAG Search: ~600-1000ms                                                │
│   - Weather API: ~300-500ms                                                │
│   - Flights API: ~300-500ms                                                │
│   - FX API: ~200-300ms                                                     │
│   - Synthesis (GPT-4o): ~2000-3000ms                                       │
│   - Total: ~5-8 seconds (parallel execution saves 2-3 seconds)             │
│                                                                             │
│ RAG Statistics:                                                             │
│   - Total child chunks: 20,047                                              │
│   - Total parent chunks: 1,184                                              │
│   - Embedding dimension: 384                                                │
│   - HNSW index search time: ~50-100ms                                       │
│   - Relevance threshold: 0.48                                              │
│                                                                             │
│ ML Classifier Statistics:                                                   │
│   - Training rows (after SMOTE): 282                                        │
│   - Features: 24                                                            │
│   - Cross-validation folds: 5                                              │
│   - Test F1 (macro): 0.8940                                                │
│   - Test Accuracy: 0.8947                                                  │
│                                                                             │
│ Cache Performance:                                                          │
│   - Weather cache hits: ~60-80% (10min TTL)                                │
│   - Flights cache hits: ~40-60% (30min TTL)                                │
│   - FX cache hits: ~80-90% (1hr TTL)                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


================================================================================
SECTION 14: COST ANALYSIS (PER QUERY)
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│ TOKEN USAGE & COST BREAKDOWN                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ Intent Extraction (GPT-4o-mini):                                           │
│   - Prompt tokens: ~400-600                                                │
│   - Completion tokens: ~80-120                                             │
│   - Cost: $0.00015/1K tokens → ~$0.0001 per query                          │
│                                                                             │
│ Clarification (GPT-4o-mini, when needed):                                   │
│   - Prompt tokens: ~200-300                                                │
│   - Completion tokens: ~50-80                                              │
│   - Cost: ~$0.00004 per query                                              │
│                                                                             │
│ Synthesis (GPT-4o):                                                         │
│   - Prompt tokens: ~6,000-8,000 (includes tool outputs)                    │
│   - Completion tokens: ~800-1,200                                          │
│   - Cost: $0.0025/1K tokens → ~$0.02-0.025 per query                       │
│                                                                             │
│ TOTAL COST PER COMPLETE QUERY: ~$0.02-0.025                                 │
│                                                                             │
│ With caching:                                                               │
│   - Weather API: free (1000 calls/day)                                     │
│   - Flights API: free (Amadeus test tier) or mock                          │
│   - FX API: free (1500 requests/month)                                     │
│                                                                             │
│ Estimated monthly cost (1000 queries): ~$20-25                              │
│                                                                             │
│ Cost Saving from Two-LLM Architecture:                                      │
│   - If using GPT-4o for everything: ~$0.03-0.04 per query                   │
│   - Actual with GPT-4o-mini for extraction: ~$0.02-0.025 per query          │
│   - Savings: ~30-40%                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


================================================================================
SECTION 15: TESTING
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│ RUNNING TESTS                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ RAG Tests:                                                                  │
│ ```bash                                                                    │
│ python backend/rag/scripts/test_retrieval.py                               │
│ python backend/rag/scripts/relevance_test.py                               │
│ ```                                                                        │
│                                                                             │
│ Agent Tests:                                                                │
│ ```bash                                                                    │
│ pytest backend/tests/ -v                                                   │
│ ```                                                                        │
│                                                                             │
│ Test Coverage:                                                              │
│   - Pydantic schemas (valid/invalid inputs)                                │
│   - Tool isolation (with fake LLM/APIs)                                    │
│   - End-to-end agent (mocked external APIs)                                │
│                                                                             │
│ CI/CD: GitHub Actions runs tests on every push                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


================================================================================
SECTION 16: DEPLOYMENT
================================================================================

DOCKER DEPLOYMENT

```bash
# Build and start all services
docker compose up --build

# Start with frontend profile
docker compose --profile ui up --build

# Stop all services
docker compose down

# Stop and remove volumes (reset database)
docker compose down -v
```


DEPLOY TO CLOUD (OPTIONAL)

Backend: Railway / Fly.io / Render
Database: Supabase (pgvector supported) / Neon
Frontend: Vercel / Netlify


================================================================================
SECTION 17: TROUBLESHOOTING
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│ COMMON ISSUES & SOLUTIONS                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ ISSUE: "No module named 'asyncpg'"                                          │
│ SOLUTION: pip install asyncpg or uv pip install asyncpg                    │
│                                                                             │
│ ISSUE: "password authentication failed"                                     │
│ SOLUTION: Check DATABASE_URL matches docker compose credentials             │
│                                                                             │
│ ISSUE: RAG returns "Alternative 1, TBD" placeholders                       │
│ SOLUTION: Run python backend/rag/scripts/embed_and_store.py                │
│                                                                             │
│ ISSUE: LangSmith traces not appearing                                       │
│ SOLUTION: Set LANGCHAIN_TRACING_V2=true, add @traceable decorator          │
│                                                                             │
│ ISSUE: Weather shows "unavailable"                                          │
│ SOLUTION: Check WEATHER_API_KEY, verify city names, or add country codes   │
│                                                                             │
│ ISSUE: Port 5432 already allocated                                          │
│ SOLUTION: Change port in docker-compose.yml (e.g., 55432)                   │
│                                                                             │
│ ISSUE: Docker build fails                                                   │
│ SOLUTION: Ensure Docker Desktop is running, check disk space               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


================================================================================
SECTION 18: DELIVERABLES CHECKLIST
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│ COMPLETED DELIVERABLES                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ☐ ML Classifier (100-200 destinations, Pipeline, 3 classifiers, CV,       │
│      tuning, imbalance handling, results.csv, joblib)                      │
│                                                                             │
│  ☐ RAG Tool (10-15 destinations, justified chunking, pgvector,             │
│      retrieval testing)                                                     │
│                                                                             │
│  ☐ Agent (3 tools, Pydantic validation, tool allowlist, LangGraph,         │
│      LangSmith trace, two-model routing)                                    │
│                                                                             │
│  ☐ Persistence (PostgreSQL + pgvector, SQLAlchemy async, runs, tool calls) │
│                                                                             │
│  ☐ Auth (Sign-up, login, JWT, bcrypt)                                      │
│                                                                             │
│  ☐ Webhook (Discord/Slack, timeout, retry, non-blocking)                   │
│                                                                             │
│  ☐ Docker (docker-compose.yml, named volume, one-command startup)          │
│                                                                             │
│  ☐ React Frontend (login, chat, tool panel, history)                       │
│                                                                             │
│  ☐ Tests (Pydantic schemas, tool isolation, e2e agent, CI)                 │
│                                                                             │
│  ☐ README (architecture diagram, labeling rules, chunking rationale,       │
│      model comparison, cost breakdown, LangSmith screenshot)               │
│                                                                             │
│  ☐ 3-minute demo video                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


================================================================================
SECTION 19: LICENSE
================================================================================

MIT License

Copyright (c) 2026 Jawad Mansour

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


================================================================================
END OF README
================================================================================
```
