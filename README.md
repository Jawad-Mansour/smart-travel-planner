## Smart Travel Planner

### RAG Chunking Strategy Rationale

The retrieval layer uses a **parent-child chunking strategy**:

- **Parent chunks** are full section-level passages (for coherent final context sent to the LLM).
- **Child chunks** are sentence-level units (for precise semantic matching during vector search).

This gives the best of both worlds:

- Fine-grained retrieval precision from short child chunks.
- Better answer quality from richer parent context, avoiding fragmented responses.
- Lower redundancy through MMR reranking before parent reconstruction.

The sequence is:
1. Scrape and clean travel content.
2. Parse sections and generate parent chunks.
3. Split into sentence children (with safeguards for very long sentences).
4. Embed only child chunks.
5. Retrieve children by vector similarity, rerank with MMR, then return unique parents.

### Backend API (Phases 12–22)

Run locally (from repo root, with dependencies installed):

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Health check: `GET /health`. Travel planner stream: `POST /api/travel/plan` with JSON `{"query":"..."}` (SSE `data:` lines). History: `GET /api/travel/history` (optional header `X-User-Id` for demo identity).

### External API cache TTLs (defaults)

| Integration | TTL | Env override |
|-------------|-----|----------------|
| Weather (OpenWeatherMap) | 10 minutes | `WEATHER_CACHE_TTL_SECONDS` |
| Flights (Amadeus or mock) | 30 minutes | `FLIGHTS_CACHE_TTL_SECONDS` |
| Exchange rates | 1 hour | `FX_CACHE_TTL_SECONDS` |

Rough LLM token costs depend on provider pricing; log usage under `usage_parts` in responses and in stored agent runs.

### Local testing (Postgres + API)

1. Copy `.env.example` to `.env` and set at least `DATABASE_URL` (and `OPENAI_API_KEY` for full LLM behavior).
2. Start Postgres: `docker compose up -d postgres` and wait until healthy.
3. Install deps: `pip install -e ".[dev]"` from repo root.
4. Run API: `uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000`.
5. Short query (expect clarification when duration/budget/activities are missing):  
   `curl -s -N -X POST http://localhost:8000/api/travel/plan -H "Content-Type: application/json" -d "{\"query\":\"I want to go hiking\"}"`
6. Full query:  
   `curl -s -N -X POST http://localhost:8000/api/travel/plan -H "Content-Type: application/json" -d "{\"query\":\"I have 2 weeks in July, $1500, want warm weather and hiking, not too touristy\"}"`
7. History:  
   `curl -s http://localhost:8000/api/travel/history -H "X-User-Id: test-user"`

RAG needs embeddings DB and chunk data from your pipeline; without it, RAG tool errors are surfaced in tool JSON but the agent still completes.
