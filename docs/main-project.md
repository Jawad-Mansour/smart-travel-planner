================================================================================
                    SMART TRAVEL PLANNER - MAIN README
================================================================================

PROJECT NAME: Smart Travel Planner
AUTHOR: Jawad Mansour
DATE: April 2026


================================================================================
SECTION 1: WHAT IS THIS PROJECT?
================================================================================

This is an AI-powered travel planner that answers natural language travel queries.

Example query:
"I have two weeks off in July and around $1,500. I want somewhere warm, not too
touristy, and I like hiking."

The system returns a complete travel plan with destination recommendations,
weather, flight costs, daily budget breakdown, and a suggested itinerary.

DEFENSE: Why build this?
- Travel planning requires synthesizing multiple data sources: weather, flights,
  costs, user preferences, destination knowledge.
- No single API provides all this. An agent with multiple tools is the right
  architecture because it can decide which information to fetch and when.
- LangGraph was chosen over a simple script because travel queries are
  multi-step and conditional (e.g., if missing date, ask clarifying question).


================================================================================
SECTION 2: HOW THE SYSTEM WORKS
================================================================================

User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENT (LangGraph)                                    │
│                                                                              │
│  STEP 1: Cheap LLM (GPT-4o-mini / Claude Haiku)                             │
│          Extracts: {duration, budget, temperature preference, activities}   │
│          Identifies missing fields → asks clarification if needed           │
│                                                                              │
│  STEP 2: Tool Calls (parallel when possible)                                │
│          ├── ML Classifier: Predicts travel style (Adventure/Relaxation)   │
│          ├── RAG Retriever: Searches travel blogs for destination info     │
│          └── Live APIs: Weather + Flights + Exchange Rates                 │
│                                                                              │
│  STEP 3: Strong LLM (GPT-4o / Claude Sonnet)                                │
│          Synthesizes all tool outputs into a coherent travel plan          │
│          Handles conflicts (e.g., RAG says "great hiking", weather says rain)│
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
Webhook Delivery (Discord/Slack/Email) + User sees answer in React UI

DEFENSE: Why two LLMs?
- Cheap LLM for extraction: costs $0.00015 per query vs $0.0025 for strong LLM.
- Mechanical work (parsing dates, extracting numbers) doesn't need reasoning.
- Strong LLM only for synthesis where reasoning matters.
- This reduces cost per query by ~85% based on token counting.

DEFENSE: Why LangGraph over LangChain?
- LangGraph provides explicit state management and conditional edges.
- Travel planning has branching logic (clarify vs search vs answer).
- LangChain's simple chain is too linear for this use case.


================================================================================
SECTION 3: TECH STACK - WHY EACH CHOICE
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│ CATEGORY          │ TECHNOLOGY          │ WHY?                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Backend Framework │ FastAPI (async)     │ - Built-in async for concurrent  │
│                   │                     │   API calls (weather + flights)  │
│                   │                     │ - Automatic OpenAPI docs at /docs│
│                   │                     │ - Dependency injection built-in  │
│                   │                     │ - Fast (on par with Node/Go)     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Agent Framework   │ LangGraph           │ - State machines for conditional │
│                   │                     │   workflows                      │
│                   │                     │ - Built-in tool calling          │
│                   │                     │ - LangSmith tracing for debugging│
│                   │                     │ - Checkpoints for multi-turn     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Database          │ PostgreSQL + pgvector│ - One database for all data     │
│                   │                     │   (users, runs, embeddings)      │
│                   │                     │ - pgvector enables vector search │
│                   │                     │   without separate vector DB     │
│                   │                     │ - ACID compliance for user data  │
├─────────────────────────────────────────────────────────────────────────────┤
│ ORM               │ SQLAlchemy 2.0 async│ - Async support for non-blocking │
│                   │                     │ - Migration support via Alembic  │
│                   │                     │ - Type hints with Mypy          │
├─────────────────────────────────────────────────────────────────────────────┤
│ Package Manager   │ UV                  │ - 10-100x faster than pip       │
│                   │                     │ - Single tool replaces pip+venv │
│                   │                     │ - Reproducible uv.lock files    │
│                   │                     │ - Built-in virtual environment  │
├─────────────────────────────────────────────────────────────────────────────┤
│ LLM Provider      │ OpenAI + Anthropic  │ - OpenAI: GPT-4o-mini (cheap)   │
│                   │                     │ - Anthropic: Claude Sonnet      │
│                   │                     │   (better reasoning for synthesis)│
│                   │                     │ - Configurable via .env         │
├─────────────────────────────────────────────────────────────────────────────┤
│ ML Framework      │ scikit-learn        │ - Production-ready pipelines    │
│                   │                     │ - Cross-validation built-in     │
│                   │                     │ - GridSearchCV for hyperparam   │
│                   │                     │ - Joblib serialization          │
├─────────────────────────────────────────────────────────────────────────────┤
│ HTTP Client       │ httpx               │ - Async support (non-blocking)  │
│                   │                     │ - Native to FastAPI ecosystem   │
│                   │                     │ - Connection pooling built-in   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Caching           │ cachetools + aiocache│ - TTL cache for weather (10min)│
│                   │ + lru_cache         │ - lru_cache for model loading   │
│                   │                     │ - aiocache for async operations │
├─────────────────────────────────────────────────────────────────────────────┤
│ Frontend          │ React + TypeScript  │ - Component reusability         │
│                   │ + Vite              │ - Type safety with TypeScript   │
│                   │                     │ - Vite is faster than CRA       │
│                   │                     │ - Streaming support with SSE    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Containerization  │ Docker + Compose    │ - One command to start all      │
│                   │                     │ - Named volumes for persistence │
│                   │                     │ - Production parity locally     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Code Quality      │ ruff, black, mypy   │ - ruff: 50x faster than Flake8  │
│                   │ + pre-commit        │ - black: zero-config formatting │
│                   │                     │ - mypy: type checking prevents  │
│                   │                     │   runtime errors                │
│                   │                     │ - pre-commit: automated checks  │
└─────────────────────────────────────────────────────────────────────────────┘


================================================================================
SECTION 4: COMPLETE PHASES WITH STATUS
================================================================================

┌─────┬──────────────────────────────────────────┬────────────┐
│ #   │ PHASE NAME                               │ STATUS     │
├─────┼──────────────────────────────────────────┼────────────┤
│ 0   │ Project Skeleton & Environment           │ ✅ COMPLETE│
│ 1   │ Travel Dataset Compilation               │ ⬜ PENDING │
│ 2   │ Exploratory Data Analysis                │ ⬜ PENDING │
│ 3   │ Feature Engineering & Preprocessing      │ ⬜ PENDING │
│ 4   │ Baseline Model Training                  │ ⬜ PENDING │
│ 5   │ Model Tuning & Final Training            │ ⬜ PENDING │
│ 6   │ ML Model Loading Service                 │ ⬜ PENDING │
│ 7   │ RAG - Content Collection                 │ ⬜ PENDING │
│ 8   │ RAG - Chunking Strategy                  │ ⬜ PENDING │
│ 9   │ RAG - Embeddings & Vector Storage        │ ⬜ PENDING │
│ 10  │ RAG - Retrieval Service                  │ ⬜ PENDING │
│ 11  │ Live APIs - Weather Service              │ ⬜ PENDING │
│ 12  │ Live APIs - Flights Service              │ ⬜ PENDING │
│ 13  │ Live APIs - Exchange Rate Service        │ ⬜ PENDING │
│ 14  │ LLM Client Setup                         │ ⬜ PENDING │
│ 15  │ Agent - Tool Definitions                 │ ⬜ PENDING │
│ 16  │ Agent - Feature Extraction               │ ⬜ PENDING │
│ 17  │ Agent - LangGraph Setup                  │ ⬜ PENDING │
│ 18  │ Agent - Clarification Logic              │ ⬜ PENDING │
│ 19  │ Agent - Tool Orchestration               │ ⬜ PENDING │
│ 20  │ Agent - Final Synthesis                  │ ⬜ PENDING │
│ 21  │ Agent - Persistence & Logging            │ ⬜ PENDING │
│ 22  │ Webhook Delivery Service                 │ ⬜ PENDING │
│ 23  │ Database Models Setup                    │ ⬜ PENDING │
│ 24  │ Authentication System                    │ ⬜ PENDING │
│ 25  │ FastAPI Routes                           │ ⬜ PENDING │
│ 26  │ React Frontend                           │ ⬜ PENDING │
│ 27  │ Docker Full Stack                        │ ⬜ PENDING │
│ 28  │ Testing & CI/CD                          │ ⬜ PENDING │
│ 29  │ README & Deliverables                    │ ⬜ PENDING │
└─────┴──────────────────────────────────────────┴────────────┘


================================================================================
SECTION 5: QUICK START
================================================================================

PREREQUISITES:
- Python 3.11 (NOT 3.12 or 3.13 - pgvector and asyncpg have compatibility issues)
- Docker Desktop
- UV package manager
- GitHub account

STEP 1: Clone and Enter Project
--------------------------------------------------------------------------------
git clone https://github.com/Jawad-Mansour/smart-travel-planner.git
cd smart-travel-planner

DEFENSE: Why clone over download ZIP?
- Git preserves commit history for accountability
- Enables branch switching for feature development
- Required for GitHub Actions CI/CD later

STEP 2: Activate Virtual Environment
--------------------------------------------------------------------------------
# Windows:
source .venv/Scripts/activate

# Mac/Linux:
source .venv/bin/activate

DEFENSE: Why virtual environment?
- Isolates project dependencies from system Python
- Prevents version conflicts between projects
- UV creates reproducible .venv that matches pyproject.toml

STEP 3: Install Dependencies
--------------------------------------------------------------------------------
uv sync

DEFENSE: Why uv sync over pip install -r requirements.txt?
- uv sync reads pyproject.toml and uv.lock for deterministic installs
- 10-100x faster than pip
- Single command installs both production and dev dependencies
- Automatically creates virtual environment if missing

STEP 4: Configure Environment
--------------------------------------------------------------------------------
cp .env.example .env
# Edit .env with your actual API keys:
# - OPENAI_API_KEY: Get from platform.openai.com
# - WEATHER_API_KEY: Get from openweathermap.org
# - SECRET_KEY: Generate with `openssl rand -hex 32`
# - JWT_SECRET_KEY: Generate with `openssl rand -hex 32`

DEFENSE: Why .env.example instead of committing .env?
- .env contains secrets (API keys, passwords)
- Committing secrets is a security violation
- .env.example documents what variables are needed
- New developers copy .env.example to get started

STEP 5: Run with Docker
--------------------------------------------------------------------------------
docker compose up --build

DEFENSE: Why Docker Compose?
- One command starts backend, frontend, and PostgreSQL
- Services communicate via internal network (not localhost)
- Named volumes persist database across restarts
- Production environment matches local development

STEP 6: Access the Application
--------------------------------------------------------------------------------
- Frontend (React): http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- PostgreSQL: localhost:5432


================================================================================
SECTION 6: ENVIRONMENT VARIABLES (All Required)
================================================================================

┌─────────────────────────────────────┬───────────────────────────────────────┐
│ VARIABLE NAME                       │ PURPOSE                               │
├─────────────────────────────────────┼───────────────────────────────────────┤
│ APP_NAME                            │ Application display name             │
│ APP_ENV                             │ development/production               │
│ DEBUG                               │ Enable debug mode (True/False)       │
│ SECRET_KEY                          │ FastAPI session signing              │
│ JWT_SECRET_KEY                      │ JWT token signing                    │
│ JWT_ALGORITHM                       │ HS256 (symmetric) or RS256 (asymmetric)│
│ JWT_EXPIRY_MINUTES                  │ Token expiration time                │
│ DATABASE_URL                        │ PostgreSQL connection string        │
│ DATABASE_POOL_SIZE                  │ Connection pool size (10 default)    │
│ DATABASE_MAX_OVERFLOW               │ Extra connections when pool full     │
│ POSTGRES_USER                       │ Database username                    │
│ POSTGRES_PASSWORD                   │ Database password                    │
│ POSTGRES_DB                         │ Database name                        │
│ LLM_PROVIDER                        │ 'openai' or 'anthropic'              │
│ OPENAI_API_KEY                      │ OpenAI API key                       │
│ OPENAI_CHEAP_MODEL                  │ 'gpt-4o-mini' (extraction)          │
│ OPENAI_STRONG_MODEL                 │ 'gpt-4o' (synthesis)                │
│ ANTHROPIC_API_KEY                   │ Anthropic API key                    │
│ ANTHROPIC_CHEAP_MODEL               │ 'claude-3-haiku-20240307'           │
│ ANTHROPIC_STRONG_MODEL              │ 'claude-3-sonnet-20240229'          │
│ EMBEDDING_MODEL                     │ 'text-embedding-3-small' (1536 dims)│
│ WEATHER_API_KEY                     │ OpenWeatherMap API key               │
│ WEATHER_API_URL                     │ https://api.openweathermap.org      │
│ FLIGHTS_API_KEY                     │ Skyscanner/Amadeus API key          │
│ FX_API_KEY                          │ ExchangeRate-API key                │
│ TTL_WEATHER_SECONDS                 │ Cache weather for 600 seconds       │
│ TTL_FLIGHTS_SECONDS                 │ Cache flights for 1800 seconds      │
│ TTL_FX_SECONDS                      │ Cache exchange rates for 3600 sec   │
│ WEBHOOK_URL                         │ Discord/Slack webhook URL           │
│ WEBHOOK_TIMEOUT_SECONDS             │ 5 seconds max for webhook           │
│ WEBHOOK_MAX_RETRIES                 │ Retry 3 times with backoff          │
│ LOG_LEVEL                           │ DEBUG/INFO/WARNING/ERROR            │
│ LOG_FORMAT                          │ 'json' or 'text'                    │
│ LANGSMITH_API_KEY                   │ LangSmith tracing key               │
│ FRONTEND_URL                        │ http://localhost:5173              │
│ CORS_ALLOWED_ORIGINS                │ Comma-separated list of origins    │
│ RAG_CHUNK_SIZE                      │ 512 tokens per chunk               │
│ RAG_CHUNK_OVERLAP                   │ 128 tokens overlap                 │
│ RAG_TOP_K                           │ Return top 5 chunks                │
│ AGENT_MAX_ITERATIONS                │ Max 5 tool calls per query         │
│ AGENT_TIMEOUT_SECONDS               │ 30 seconds total runtime           │
└─────────────────────────────────────┴───────────────────────────────────────┘

DEFENSE: Why pydantic-settings over os.getenv()?
- Type validation at startup (fail fast if missing)
- Single source of truth for all configuration
- Automatic casting (e.g., DEBUG="true" → bool True)
- IDE autocomplete for settings


================================================================================
SECTION 7: PROJECT STRUCTURE WITH DEFENSES
================================================================================

smart-travel-planner/
│
├── backend/                    # WHY: Separate from frontend for independent
│   │                          #       deployment and scaling
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes/        # WHY: Each endpoint group has its own file
│   │   │   │   ├── auth.py    #       (prevents 2000-line main.py)
│   │   │   │   ├── travel.py
│   │   │   │   └── webhook.py
│   │   │   └── middlewares/   # WHY: Cross-cutting concerns (logging, rate limit)
│   │   ├── core/              # WHY: Agent logic isolated from API layer
│   │   │   ├── agent.py       #       (testable independently)
│   │   │   └── graph.py
│   │   ├── db/
│   │   │   ├── models/        # WHY: SQLAlchemy models separate from business
│   │   │   │   ├── user.py    #       logic
│   │   │   │   ├── agent_run.py
│   │   │   │   └── tool_call.py
│   │   │   └── repositories/  # WHY: Database queries in one place
│   │   ├── schemas/           # WHY: Pydantic models for API boundaries
│   │   │   ├── auth.py        #       (validation at the edge)
│   │   │   ├── travel.py
│   │   │   └── tool_inputs.py
│   │   ├── services/          # WHY: External API calls (weather, flights, LLM)
│   │   │   ├── llm_client.py  #       (injectable for testing)
│   │   │   ├── weather_service.py
│   │   │   └── webhook_service.py
│   │   ├── tools/             # WHY: Agent tools as separate modules
│   │   │   ├── ml_classifier.py
│   │   │   ├── rag_retriever.py
│   │   │   └── live_apis.py
│   │   └── utils/             # WHY: Reusable helpers (logging, cache, retry)
│   │       ├── cache.py
│   │       ├── retry.py
│   │       └── logging.py
│   │
│   ├── ml/                    # WHY: ML code isolated from API
│   │   ├── data/              # WHY: destinations.csv (version controlled)
│   │   ├── notebooks/         # WHY: EDA and experimentation (Jupyter)
│   │   ├── scripts/           # WHY: Training scripts (run separately)
│   │   ├── models/            # WHY: Saved .joblib files (gitignored)
│   │   └── experiments/       # WHY: results.csv for tracking
│   │
│   ├── rag/                   # WHY: RAG content and embeddings
│   │   ├── data/raw/          # WHY: Original Wikivoyage/blogs (gitignored)
│   │   ├── data/chunks/       # WHY: Chunked documents (gitignored)
│   │   └── scripts/           # WHY: Chunking and embedding generation
│   │
│   └── tests/                 # WHY: Tests mirror backend structure
│       ├── test_tools/        # WHY: Each tool has its own test file
│       ├── test_api/          # WHY: API endpoint tests
│       ├── test_agent/        # WHY: End-to-end agent tests
│       └── test_schemas/      # WHY: Pydantic validation tests
│
├── frontend/                  # WHY: Separate from backend, different stack
│   ├── public/                # WHY: Static assets (favicon, images)
│   └── src/
│       ├── components/        # WHY: Reusable React components
│       │   ├── auth/          # WHY: Login, Signup components
│       │   ├── chat/          # WHY: Chat interface components
│       │   ├── common/        # WHY: Button, Input, Loading
│       │   └── history/       # WHY: Past runs display
│       ├── pages/             # WHY: Top-level routes (Login, Chat, History)
│       ├── services/          # WHY: API calls to backend (axios)
│       ├── hooks/             # WHY: Custom React hooks (useAuth, useStreaming)
│       ├── context/           # WHY: React Context (AuthContext)
│       ├── types/             # WHY: TypeScript type definitions
│       └── utils/             # WHY: Formatters, constants
│
├── .github/workflows/         # WHY: CI/CD pipelines (GitHub Actions)
├── docker-compose.yml         # WHY: Orchestrates all three services
├── pyproject.toml             # WHY: Modern Python project configuration
├── uv.lock                    # WHY: Reproducible dependency locks
├── .env.example               # WHY: Template for required variables
├── .gitignore                 # WHY: Excludes venv, secrets, models
├── .pre-commit-config.yaml    # WHY: Automated code quality checks
└── .python-version            # WHY: Pins Python 3.11 for all developers


================================================================================
SECTION 8: DEFENSE OF EVERY DIRECTORY AND FILE (Phase 0)
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│ FILE/DIRECTORY          │ DEFENSE                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ backend/app/api/routes/ │ Each set of endpoints has its own router file.    │
│                         │ Prevents the 2000-line main.py anti-pattern.      │
│                         │ FastAPI routers enable modular testing.           │
├─────────────────────────────────────────────────────────────────────────────┤
│ backend/app/db/models/  │ SQLAlchemy models define database schema.        │
│                         │ Separate from repositories (separation of         │
│                         │ concerns - structure vs queries).                │
├─────────────────────────────────────────────────────────────────────────────┤
│ backend/app/db/repositories/ │ Repository pattern isolates database queries.│
│                         │ Makes it easy to swap databases or mock in tests. │
├─────────────────────────────────────────────────────────────────────────────┤
│ backend/app/schemas/    │ Pydantic models validate data at API boundaries. │
│                         │ "Validate at the edge, trust your types inside." │
│                         │ Automatic OpenAPI documentation at /docs.        │
├─────────────────────────────────────────────────────────────────────────────┤
│ backend/app/services/   │ External API calls (LLM, weather, flights) live  │
│                         │ here. Injected via Depends() for testability.    │
├─────────────────────────────────────────────────────────────────────────────┤
│ backend/app/tools/      │ Each agent tool is a separate module.            │
│                         │ Tools are: ML Classifier, RAG, Live APIs.        │
│                         │ Pydantic validation on every tool input.         │
├─────────────────────────────────────────────────────────────────────────────┤
│ backend/app/core/       │ Agent logic (LangGraph). Separated from routes   │
│                         │ because it's complex and needs its own tests.    │
├─────────────────────────────────────────────────────────────────────────────┤
│ backend/app/utils/      │ Cross-cutting concerns: caching (lru_cache, TTL),│
│                         │ retry logic (tenacity), structured logging.      │
├─────────────────────────────────────────────────────────────────────────────┤
│ backend/ml/             │ ML training is a separate concern from serving.  │
│                         │ Training scripts run offline, not in API.        │
│                         │ Prevents accidental model retraining in prod.    │
├─────────────────────────────────────────────────────────────────────────────┤
│ backend/rag/            │ RAG content collection and chunking is offline.  │
│                         │ Embeddings generated once, stored in pgvector.   │
├─────────────────────────────────────────────────────────────────────────────┤
│ backend/tests/          │ Tests mirror the source structure.               │
│                         │ pytest finds them automatically.                 │
│                         │ Required by spec: tool tests, schema tests,      │
│                         │ end-to-end test with mocked APIs.                │
├─────────────────────────────────────────────────────────────────────────────┤
│ frontend/src/components/│ React components broken down by feature.         │
│                         │ auth, chat, history, common.                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ frontend/src/services/  │ API client with axios interceptors for auth.    │
│                         │ Single place for all backend communication.       │
├─────────────────────────────────────────────────────────────────────────────┤
│ frontend/src/hooks/     │ useAuth: manages JWT in localStorage.            │
│                         │ useStreaming: Server-Sent Events for tokens.     │
├─────────────────────────────────────────────────────────────────────────────┤
│ frontend/src/context/   │ React Context for auth state (avoids prop drilling)│
├─────────────────────────────────────────────────────────────────────────────┤
│ frontend/src/types/     │ TypeScript interfaces for API requests/responses.│
│                         │ Type safety across frontend-backend boundary.    │
├─────────────────────────────────────────────────────────────────────────────┤
│ .github/workflows/      │ CI/CD: runs tests on every push.                 │
│                         │ Required by spec: "Tests run in CI on every push"│
├─────────────────────────────────────────────────────────────────────────────┤
│ docker-compose.yml      │ One command starts backend, frontend, PostgreSQL.│
│                         │ Named volume preserves data across restarts.     │
│                         │ "If a reviewer can't run 'docker compose up',    │
│                         │ you have not finished." - Spec requirement.      │
├─────────────────────────────────────────────────────────────────────────────┤
│ pyproject.toml          │ Single source of truth for dependencies.         │
│                         │ Replaces requirements.txt, setup.py, Pipfile.    │
│                         │ PEP 621 standard.                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ uv.lock                 │ Reproducible installs across all machines.       │
│                         │ Every package version is pinned.                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ .env.example            │ Documents every required environment variable.  │
│                         │ No secrets committed.                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ .gitignore              │ Excludes .venv (platform-specific), .env (secrets),│
│                         │ *.joblib (large binary models), node_modules.    │
├─────────────────────────────────────────────────────────────────────────────┤
│ .pre-commit-config.yaml │ ruff: linting (50x faster than Flake8)          │
│                         │ black: formatting (zero-config)                  │
│                         │ mypy: type checking                              │
│                         │ Prevents bad code from being committed.          │
├─────────────────────────────────────────────────────────────────────────────┤
│ .python-version         │ Pins Python 3.11 for all developers.             │
│                         │ Prevents "works on my machine" issues.           │
│                         │ Python 3.11 required for asyncpg compatibility.  │
└─────────────────────────────────────────────────────────────────────────────┘


================================================================================
SECTION 9: SECURITY DEFENSES IMPLEMENTED
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│ THREAT                   │ MITIGATION                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ API key exposure         │ .env file (gitignored), .env.example for template│
├─────────────────────────────────────────────────────────────────────────────┤
│ SQL injection            │ SQLAlchemy ORM (parameterized queries)          │
├─────────────────────────────────────────────────────────────────────────────┤
│ JWT tampering            │ HS256 signing with secrets from .env            │
├─────────────────────────────────────────────────────────────────────────────┤
│ Password leaks           │ bcrypt hashing (passlib) - never stored plain   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Stack trace leaks        │ Global exception handlers return 500 with       │
│                          │ generic message, logs have full trace           │
├─────────────────────────────────────────────────────────────────────────────┤
│ Rate limiting            │ Middleware limits requests per user/IP          │
├─────────────────────────────────────────────────────────────────────────────┤
│ CORS violations          │ CORS middleware with allowed origins from .env  │
├─────────────────────────────────────────────────────────────────────────────┤
│ Dependency vulnerabilities│ uv.lock pins versions, Dependabot alerts       │
└─────────────────────────────────────────────────────────────────────────────┘


================================================================================
SECTION 10: PERFORMANCE DEFENSES
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│ ISSUE                    │ SOLUTION                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Blocking I/O             │ Async all the way: FastAPI routes async,        │
│                          │ httpx.AsyncClient, asyncpg, SQLAlchemy async    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Repeated API calls       │ TTL caching: weather (10min), flights (30min),  │
│                          │ exchange rates (1 hour)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Model reloading on       │ lru_cache on load_model() - loads once, cached  │
│ every request            │ for all subsequent requests                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Database connection      │ Connection pool (10 connections, 20 overflow)   │
│ overhead                 │ Async session per request, auto-closed          │
├─────────────────────────────────────────────────────────────────────────────┤
│ LLM cost per query       │ Cheap LLM for extraction ($0.00015) vs strong   │
│                          │ for synthesis ($0.0025) - 85% cost reduction    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Slow downstream APIs     │ Timeout + retry with exponential backoff        │
│                          │ (tenacity)                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Large payloads           │ Streaming responses (SSE) for long LLM outputs  │
└─────────────────────────────────────────────────────────────────────────────┘


================================================================================
SECTION 11: LICENSE
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
