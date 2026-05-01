# Smart Travel Planner

An **AI travel agent** that turns natural-language trip requests into structured plans: it **extracts intent**, optionally **asks for missing details**, runs **RAG** over destination guides stored in **PostgreSQL + pgvector**, ranks places with an **ML travel-style classifier**, enriches answers with **live weather, flight estimates, and FX**, and streams results to a **React** chat UI. Optional **Discord / Slack webhooks** deliver the plan when a reply completes.

---

## What this project does (product flow)

1. A **signed-in user** opens the app and starts or resumes a **chat session**.
2. They describe a trip (budget, duration, interests, timing, constraints).
3. The **agent** decides whether it has enough structured intent to call heavy tools safely:
   - If **not**, it returns a **clarification** message and the UI collects details **inline** (no blocking modal).
   - If **yes**, it runs **classifier + RAG + live APIs** in parallel, then **synthesizes** a markdown answer (multiple destinations + recommendation).
4. Each turn is **persisted** (messages + tool logs) under that user.
5. On a **complete** plan (not a clarification-only turn), an optional **webhook** fires in the background so Discord/Slack still works even if delivery fails.

---

## End-to-end workflow (technical)

### User journey vs backend

| Step | User / UI | Backend |
|------|-----------|---------|
| 1 | Register / login | `POST /api/auth/register` or `/login` → JWT access + refresh |
| 2 | Open chat, send message | `POST /api/chat/stream` (SSE), Bearer token |
| 3 | See streaming reply | Events: `session` → `meta` → `delta` or multiple `segment` → `done` |
| 4 | Inspect “what happened” | Sidebar **Response analysis** reads last `done` payload (tools, embedding preview, RAG rows) |
| 5 | Optional channel notification | After `done`, background task posts to Discord (embed) and/or Slack (text) |

### Agent pipeline (LangGraph)

```mermaid
flowchart TD
  A[User message] --> B[extract_intent]
  B --> C{Critical fields missing?}
  C -->|yes| D[clarify LLM cheap model]
  D --> E[END stream clarification]
  C -->|no| F[orchestrate_tools parallel]
  F --> G[classifier destinations]
  F --> H[RAG pgvector search]
  F --> I[FX latest]
  F --> J[weather + flights per candidate cities]
  G --> K[synthesize LLM strong model structured JSON]
  H --> K
  I --> K
  J --> K
  K --> L[Render markdown + persist message + tool logs]
  L --> M[SSE done + optional webhooks]
```

**Routing rule:** Missing **duration, budget, activities, or preferred month/timing** sends the graph to **clarify** instead of paying for full tool orchestration — this matches the “cheap model for mechanical work” idea and avoids nonsense retrieval.

**Synthesis:** The strong model receives **structured tool JSON** and must produce **honest** markdown (budget fit, weather caveats), not a paste of raw JSON — tension between RAG and APIs is meant to surface in the prose.

---

## Tech stack

| Layer | Technology | Role |
|-------|------------|------|
| **Frontend** | React 18, Vite 5, TypeScript, Tailwind CSS | SPA: auth, SSE chat, sidebar, analysis / system-info pages |
| **Backend** | FastAPI, Uvicorn | REST + **SSE** streaming (`/api/chat/stream`) |
| **Agent** | LangGraph, OpenAI SDK **async** | State machine: intent → clarify **or** tools → synthesis |
| **LLMs** | OpenAI (`gpt-4o-mini` + `gpt-4o` by default) | Cheap: extraction / clarification; Strong: structured trip JSON → markdown |
| **DB** | PostgreSQL **16**, **pgvector** extension | Users, sessions, messages, tool logs, RAG `documents` / `chunks` with `VECTOR(384)` |
| **ORM / migrations** | SQLAlchemy 2 **async**, Alembic | `asyncpg` driver; schema versioning |
| **Embeddings** | sentence-transformers `all-MiniLM-L6-v2` (384-d) | Encode at ingest and at query time in `RAGService` |
| **ML inference** | scikit-learn joblib (`Pipeline` + preprocessor + RF winner) | Travel-style classification + destination ranking hints |
| **HTTP clients** | httpx **async** | Weather, flights, FX, webhooks |
| **Auth** | JWT access + refresh, passlib hashing | User-scoped sessions and history |
| **Containers** | Docker Compose | `pgvector/pgvector:pg16`, backend image, nginx-served frontend |
| **Observability (optional)** | LangSmith / LangChain tracing env vars | Multi-step traces for demos |
| **CI** | GitHub Actions | Tests on push (`.github/workflows/ci.yml`) |

---

## Design decisions (why we built it this way)

| Decision | Rationale |
|----------|-----------|
| **Postgres + pgvector only** | Single source of truth for users, chat, tool audit trail, and vectors — matches course spec; **no SQLite** fallback in the app DB layer (`backend/app/db/session.py`). |
| **LangGraph over a single giant prompt** | Explicit nodes and branching (clarify vs tools), easier to reason about costs and failures. |
| **Two OpenAI models** | Reduces cost: mini handles extraction/clarify; full synthesis uses the stronger model only when tools have run. Token usage is recorded per step in `usage_parts`. |
| **Tool allowlist** | Only names in `TOOL_ALLOWLIST` (`backend/app/tools/__init__.py`) are valid — avoids invented tools from the model. |
| **Three “logical” tools → six functions** | Brief asks for retrieval, classification, and live conditions; we expose **RAG**, **classifier**, **weather**, **flights**, **FX** as separate allowlisted tools for clearer logs and retries. |
| **Parent–child RAG** | Child chunks are embedded for search; parent passages are returned to the LLM for context — better coherence than tiny isolated chunks. |
| **TTL caches** | Weather / flights / FX cached (configurable seconds) to cut duplicate external calls and cost. |
| **SSE + optional `segment` events** | Full markdown is stored as **one** assistant message for DB + webhooks; the stream can deliver **destination-sized chunks** for UX. |
| **Webhooks async + tenacity** | Discord/Slack failures are retried and logged; they **never** block the HTTP response. |
| **Async everywhere on I/O paths** | FastAPI async routes, async SQLAlchemy sessions, async httpx, async OpenAI — keeps the event loop responsive under concurrent chats. |
| **pydantic-settings `Settings`** | One validated config object at startup instead of scattered `os.getenv`. |

---

## Requirements (course brief mapping)

The Week 4 brief asks for ML + RAG + agent tools + two-model routing + persistence + auth + React + webhook + Docker + solid engineering practices. Mapping:

| # | Requirement | Status | Notes |
|---|-------------|--------|-------|
| 1 | ML classifier (Pipeline, CV, ≥3 models, tuning, joblib) | Done | `notebooks/training.ipynb` → `backend/ml/models/`. Runtime: `classifier_tool.py`. Without joblib files, a **keyword fallback** runs (ship artifacts for grading). |
| 2 | RAG in **same** Postgres with **pgvector** | Done | Alembic enables `vector`; `ingest.py` + `rag_service.py`. |
| 3 | Agent (LangGraph), allowlist, validated boundaries, synthesis | Done | `agent.py`, Pydantic schemas, `TOOL_ALLOWLIST`. |
| 4 | Two models + usage visibility | Done | Config + `usage_parts` in stream / stored metadata. |
| 5 | SQLAlchemy + Alembic persistence | Done | Users, sessions, messages, `tool_call_logs`. |
| 6 | Register/login, user-scoped data | Done | JWT routes + FK from sessions to users. |
| 7 | React UI + streaming + “what the agent did” | Done | Chat + **Response analysis** accordion. |
| 8 | Webhook with timeout/retry, isolated failure | Done | `webhook_service.py`. |
| 9 | `docker compose up` full stack + DB volume | Done | `docker-compose.yml`, `pgdata`. |
| — | LangSmith screenshot | Optional | Enable tracing env vars; capture trace for README. |
| — | Demo video | Your deliverable | Record UI → agent → webhook. |

---

## Architecture (deployment view)

```mermaid
flowchart LR
  subgraph browser [Browser]
    SPA[React SPA]
  end
  subgraph compose [Docker Compose]
    FE[nginx :80 / :8080]
    API[FastAPI :8000]
    PG[(Postgres + pgvector)]
  end
  SPA --> FE
  FE -->|"/api/*"| API
  SPA -->|dev proxy /api| API
  API --> PG
  API --> OAI[OpenAI]
  API --> EXT[Weather / Flights / FX]
  API --> WH[Discord / Slack]
```

---

## Quick start

### Full stack (Docker)

```bash
cp .env.example .env
# Set OPENAI_API_KEY at minimum
docker compose up --build
```

- UI: **http://localhost** or **http://localhost:8080**
- API: **http://localhost:8000/docs**
- Load vectors: `docker compose exec backend python -m backend.ingest`

### Local dev (Postgres in Docker, API + Vite local)

```bash
docker compose up -d db
cd backend && alembic -c alembic.ini upgrade head && cd ..
# DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:55432/travel_planner
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173**.

### Alembic: `upgrade` vs `stamp`

| Command | Use when |
|---------|----------|
| **`alembic upgrade head`** | **New or empty DB** — creates tables + extension. Docker backend entrypoint runs this automatically. |
| **`alembic stamp head`** | DB schema **already** matches migrations — only fix Alembic’s version row. **Never** stamp an empty DB. |

---

## How to test the agent

Testing the agent means verifying **intent routing**, **tools**, **synthesis**, **persistence**, and optionally **webhooks**. Use the paths below in order.

### Prerequisites

- Postgres running with migrations applied (`upgrade head`).
- **`OPENAI_API_KEY`** set (otherwise you only get configuration/clarify stubs).
- **RAG useful:** run `python -m backend.ingest` once against the same `DATABASE_URL` (Docker or host).

### 1) Health and auth

1. `GET /health` → `200`, body like `{"status":"ok"}`.
2. In the UI: **Register** → **Login**, or use Swagger `POST /api/auth/register` then `POST /api/auth/login` and **Authorize** with `Bearer <access_token>`.

**Pass criteria:** JWT works; `/api/sessions` returns `200` when authenticated.

### 2) Clarification path (cheap model / no full tools)

Send a **vague** message missing budget or duration, e.g.:

> “I want to go somewhere warm.”

**Expect:**

- Stream ends with a **short assistant clarification** (no full destination grid).
- Sidebar **Response analysis** may show limited tool data (often no RAG rows).
- UI may show **inline “Quick details”** — submit budget/duration/interests/month and send again.

**Pass criteria:** No server 500; session saved; second message after inline patch triggers **full** pipeline when fields are complete.

### 3) Full agent path (tools + synthesis)

Send a **complete** request in one message, e.g.:

> “10 days in October 2026, total budget $3500, interests hiking and food, prefer mild weather, not too crowded.”

**Expect:**

- SSE: `session` → `meta` → multiple **`segment`** events (destinations + recommendation) **or** `delta` stream for clarification-style replies.
- Final **`done`** includes `tool_results`, `usage_parts`, `elapsed_seconds`.
- **Response analysis:** classifier style/confidence, **RAG** rows (if ingest ran), weather/flight lines, FX summary, embedding preview when RAG ran.

**Pass criteria:** Markdown answer lists **multiple destinations** with structured sections; analysis panel reflects **classifier + rag (+ APIs)**.

### 4) RAG sanity check

After ingest, ask something retrieval-specific:

> “What does the guide say about safety and transport in Kathmandu?”

**Pass criteria:** RAG section shows chunks with destinations/headings/snippet scores; answer text aligns with retrieved themes (not generic-only).

### 5) Webhook path

1. Set `DISCORD_WEBHOOK_URL` and/or `SLACK_WEBHOOK_URL` in `.env`.
2. Restart the API.
3. Complete a **full** plan (same as test §3 — **not** clarification-only).

**Pass criteria:** Message appears in Discord/Slack; `done.webhook_status` is `queued` when URL configured; chat still succeeds if webhook URL is wrong (check logs for warnings).

### 6) Automated tests

```bash
python -m pytest backend/tests -q
```

**Pass criteria:** Green locally and in CI.

### 7) Docker end-to-end

```bash
docker compose up --build
docker compose exec backend python -m backend.ingest
```

Then use the **browser on port 80/8080**, register, and run §2–§5.

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | `postgresql+asyncpg://…` (required) |
| `OPENAI_API_KEY` | LLM calls |
| `OPENAI_CHEAP_MODEL` / `OPENAI_STRONG_MODEL` | Routing |
| `JWT_SECRET_KEY` | JWT signing |
| `DISCORD_WEBHOOK_URL` / `SLACK_WEBHOOK_URL` | Optional outbound |
| `WEATHER_API_KEY`, Amadeus keys, `FX_*` | Live tools |
| `ML_MODELS_DIR`, `ML_DESTINATIONS_CSV` | Classifier |
| `CORS_ALLOWED_ORIGINS` | Browser origins |
| `LANGCHAIN_*` / `LANGSMITH_*` | Optional tracing |

Full list: `.env.example`.

---

## API highlights

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/auth/register`, `/login`, `/refresh` | Auth |
| GET/PATCH | `/api/auth/me`, onboarding | User profile |
| GET/POST/DELETE | `/api/sessions`, `/api/sessions/{id}` | Sessions |
| GET | `/api/sessions/{id}/messages` | History |
| POST | `/api/chat/stream` | SSE agent chat |
| GET | `/api/meta` | Static stats for UI |
| POST | `/api/travel/plan` | Legacy SSE plan |

---

## ML training & RAG

- **Training:** `notebooks/training.ipynb` — labeling rules, CV, model comparison, exports to `backend/ml/models/`.
- **Ingestion:** `backend/ingest.py` — chunks → embeddings → pgvector.
- **Deep dive:** `docs/agent_implementation_report.md`.

---

## Per-query cost

Rough **$0.02–$0.15** per full turn depending on length and models — use `usage_parts` from the last `done` event and current OpenAI pricing for an accurate number.

---

## LangSmith (optional)

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=smart-travel-planner
```

Capture a multi-tool trace screenshot for submissions.

---

## Repository layout

```
backend/app/       # API, agent, services, models
backend/alembic/   # Migrations
backend/ml/        # Data + joblib artifacts
backend/ingest.py  # RAG pipeline
frontend/src/      # React UI
docker-compose.yml
```

---

## License

Add your license here.
