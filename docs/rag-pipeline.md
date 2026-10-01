# RAG Pipeline Specification — Ingestion, Retrieval & Synthesis

## 1. Pipeline Overview

The Advanced RAG platform implements a disciplined two-stage architecture: **Asynchronous Ingestion Pipeline** and **Synchronous/Streaming RAG Query Pipeline**.

```
INGESTION:
Document Upload → Validation & Virus/Format Check → Text Extraction (PDF/DOCX/TXT) 
                → Text Normalization → Semantic/Recursive Chunking 
                → Dense Embeddings (Local/Configurable) → Qdrant Vector Upsert 
                → Relational Metadata Persistence → Status Update (COMPLETED)

QUERY & GENERATION:
User Query → Query Rewriting / History Enrichment → Dense Vector Query 
           → Qdrant Similarity Search with User Filter → Top-N Candidates 
           → Cross-Encoder Reranker → Top-K Re-ordered Context 
           → Grounded Prompt Assembly → Gemini LLM Synthesis 
           → Citation Extraction & Verification → Structured Response + Citations
```

---

## 2. Ingestion Pipeline Stages

### Stage 1: Document Upload & Validation
- **File Types**: `.pdf`, `.docx`, `.txt`.
- **Validation**:
  - MIME type verification & magic byte validation.
  - File size cap (default 25 MB).
  - Sanitization of filenames to prevent path traversal.
- **Storage**: Saved to structured local repository `storage/uploads/{user_id}/{document_uuid}/`.

### Stage 2: Text Extraction & Normalization
- **PDF**: Page-by-page text extraction with layout and page number tracking (`pypdf`).
- **DOCX**: Paragraph and heading structure extraction (`python-docx`).
- **TXT**: UTF-8 stream normalization.
- **Normalization**: Unicode cleanup (NFKC), removal of null and control characters, newline standardization (`\r\n` / `\r` to `\n`), whitespace compression while strictly preserving paragraph boundaries (`\n\n`).

### Stage 3: Hierarchical Document Chunking (Phase 8 Specification)
- **Splitting Strategy**: Deterministic `RecursiveCharacterSplitter` using hierarchical natural boundary separators:
  1. `\n\n` (Paragraph boundaries)
  2. `\n` (Line boundaries)
  3. `. ` (Period sentence boundaries)
  4. `? ` (Question sentence boundaries)
  5. `! ` (Exclamation sentence boundaries)
  6. `; ` (Semicolon boundaries)
  7. ` ` (Word boundaries)
  8. `""` (Character fallback)
- **Configurable Parameters**:
  - `chunk_size`: `500` (Characters)
  - `chunk_overlap`: `50` (Characters)
  - Validation: `0 <= chunk_overlap < chunk_size`
- **Units**: Character-based units for deterministic length bounding.
- **Token-Count Strategy**: Approximated via deterministic whitespace token counting (`len(text.split())`) in Phase 8; Phase 9 introduces embedding model tokenizers.
- **Page Boundaries & Citations**:
  - **PDF**: Page boundaries are strictly preserved locally per page so that `page_number` in `DocumentChunk` directly points to the actual page (1-based index) for precise citation lookup.
  - **TXT / DOCX**: `page_number` is set to `None` since reliable page numbers do not exist.
- **Chunk Indexing**: 0-based deterministic sequential indexing (`0, 1, 2, ...`).
- **Persistence & Idempotency**:
  - Existing chunks for a document are atomically replaced inside a database transaction before new chunks are persisted.
  - Updates `Document.total_chunks` to the exact count of generated chunks.
  - Empty or whitespace documents produce 0 chunks (`total_chunks = 0`).
  - Short documents (< `chunk_size`) produce exactly 1 chunk (`total_chunks = 1`).
- **Phase 8 Boundary & Limitations**:
  - `qdrant_point_id` remains `NULL` / `None`.
  - Dense embeddings and Qdrant vector index insertion are not executed in Phase 8 (scheduled for Phase 9 and Phase 10).

### Stage 4: Embedding Generation (Phase 9)
- Embedding interface abstraction (`EmbeddingProvider` protocol).
- Default: Local dense embeddings (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions) for zero-cost, high-speed local inference.
- Configurable switch to remote providers.

### Stage 5: Vector Store & Relational Indexing (Phase 10)
- Qdrant batch upsert with structured payload (`user_id`, `document_id`, `chunk_id`, `page_number`, `chunk_index`, `filename`, `text`).
- MySQL chunk catalog updated for relational queries.

---

## 3. Retrieval & Synthesis Pipeline Stages

### Stage 1: Query Preprocessing & Multi-turn Query Rewriting
- Resolves pronouns and conversation context from chat history into a standalone search query.

### Stage 2: Tenant-Isolated Vector Search
- Qdrant cosine similarity search strictly scoped with `user_id` filter.
- Configurable `top_k` and `score_threshold`.

### Stage 3: Reranking (Optional / Stage 2 Retrieval)
- Cross-encoder reranker scores candidate chunks against query for fine-grained semantic relevance.

### Stage 4: Grounded Context Assembly & Generation
- Strict system prompt instructs Gemini to formulate answers solely based on the provided numbered chunks.
- If context is insufficient, the system explicitly reports insufficient documentation rather than hallucinating.

### Stage 5: Structured Citations
- Extracts explicit source references linked to document IDs, filenames, page numbers, and exact chunk snippets.
