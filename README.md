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
