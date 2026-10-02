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

### Stage 4: Dense Embedding Generation (Phase 9 Specification)
- **Provider Interface Abstraction**: `BaseEmbeddingProvider` defines abstract contract (`provider_name`, `model_name`, `dimension`, `embed_text`, `embed_texts`, `validate_vector`, `validate_vectors`).
- **Default Local Provider**: `LocalEmbeddingProvider` using `sentence-transformers/all-MiniLM-L6-v2` for high-throughput, zero-cost local dense inference.
- **Model Loading & Lifecycle**:
  - Model loaded lazily upon first invocation and cached on the provider instance.
  - Avoids redundant per-chunk model instantiation.
- **Vector Dimension**: `384` dimensions.
- **Batch Processing**: Configurable `batch_size` (default `32`) ensuring positional 1:1 correspondence between input chunk sequences and output vectors.
- **Normalization**: Configurable L2 unit-norm normalization (`normalize_embeddings=True`) ensuring unit length for direct cosine similarity calculations.
- **Vector Validation**: Strictly checks that vectors are non-empty, match expected dimensions, contain numeric finite floating-point values, and match input batch lengths.
- **Service Orchestration**: `EmbeddingService` coordinates provider selection, batch delegation, and converts `DocumentChunk` records into in-memory `ChunkEmbedding` data structures.
- **Persistence Boundary**:
  - Embeddings are **NOT** stored in MySQL (no JSON/BLOB vector columns).
  - Qdrant indexing and collection creation are intentionally deferred to Phase 10.
  - `DocumentChunk.qdrant_point_id` remains `NULL` / `None`.

### Stage 5: Qdrant Vector Storage & Indexing (Phase 10 Specification)
- **Vector Database**: Qdrant running locally at `http://localhost:6333` with persistent Docker storage.
- **Collection Configuration**:
  - `collection_name`: Configurable (default `rag_document_chunks`).
  - `dimension`: `384` (strictly aligned with `sentence-transformers/all-MiniLM-L6-v2`).
  - `distance`: `Distance.COSINE`.
  - `initialize_collection()`: Idempotently creates collection if missing and verifies dimensional/distance parameters if existing.
- **Point ID Strategy**: Deterministic UUID string directly matching `DocumentChunk.id` to guarantee 1:1 relational correspondence and eliminate random ID drift.
- **Payload Schema**:
  - `document_id`: UUID of parent document
  - `user_id`: Tenant identifier for search filtering
  - `chunk_id`: Direct pointer to MySQL `DocumentChunk.id`
  - `chunk_index`: Integer sequential order
  - `page_number`: Integer for PDF pages, `None` for unpaged docs
  - `filename`: Original document filename
  - `file_type`: Extension (`.pdf`, `.docx`, `.txt`)
  - `token_count`: Integer token estimate
  - `chunk_text`: Extracted snippet for direct retrieval in Phase 11
- **Tenant Isolation**: All Qdrant queries, point counts, and deletions enforce strict `user_id` field filters (`FieldCondition(key="user_id", match=MatchValue(value=user_id))`).
- **Batch Upsert**: Configurable `qdrant_batch_size` (default `64`) independent from embedding batch size.
- **Idempotency & Reindexing**: Purges existing document points in Qdrant before upserting fresh vectors to prevent orphaned chunks upon document modification.
- **Deletion Synchronization**: Document deletion via `DocumentService` purges Qdrant vector points before deleting relational records and filesystem storage.
- **Relational Synchronization**: MySQL `DocumentChunk.qdrant_point_id` is populated with the deterministic point ID upon successful Qdrant upsert.
- **Phase 11 Boundary**: Semantic search, hybrid retrieval, and cross-encoder reranking are strictly deferred to Phase 11 and Phase 12.

---

## 3. Semantic Retrieval & Similarity Search (Phase 11 Specification)

### Stage 1: Query Validation & Sanitization
- **Non-Empty / Whitespace Check**: Discards empty or whitespace-only queries immediately, raising `RetrievalValidationError`.
- **Length Bounding**: Enforces configurable `max_query_length` (default 1000 characters).
- **Whitespace Normalization**: Trims leading and trailing whitespace.

### Stage 2: Query Embedding Generation
- Reuses Phase 9 `EmbeddingService` (`sentence-transformers/all-MiniLM-L6-v2`).
- Produces a single normalized 384-dimensional dense vector (`query_vector = embed_text(query)`).

### Stage 3: Strict Multi-Tenant Qdrant Vector Search
- **Mandatory Tenant Filter**: Every query applies a hard filter on `user_id == authenticated_user_id`:
  ```python
  must = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
  ```
- **Optional Document Filter**: If a `document_id` is supplied:
  ```python
  must.append(FieldCondition(key="document_id", match=MatchValue(value=document_id)))
  ```
- Cross-tenant queries with another tenant's `document_id` return 0 results safely without leaking document existence.
- **Top-K Retrieval**: Configurable `top_k` (default `5`, validated `1 <= top_k <= 100`).
- **Cosine Distance Scoring**: Qdrant executes cosine distance search, returning scores normalized to `[0.0, 1.0]`.

### Stage 4: Similarity Threshold & Payload Validation
- **Score Filtering**: Filters results having `score >= similarity_threshold` (default `0.5`, configurable per-request).
- **Payload Schema Validation**:
  - Validates `chunk_id`, `document_id`, `user_id`, `chunk_text`, `filename`, `file_type`.
  - Malformed points missing essential fields are skipped safely with warning logs rather than crashing the API.
  - Payloads with mismatched `user_id` are discarded as security anomalies.
- **Payload-Derived Citations**: Reads metadata directly from Qdrant payloads to eliminate N+1 relational database lookups.

### Stage 5: Ranking & Structured Response
- Results are sorted in descending order of similarity score.
- 1-indexed `rank` (`1, 2, 3, ...`) is assigned after threshold filtering.
- Returns structured `RetrievalResponse`:
  ```json
  {
    "query": "search text",
    "results": [
      {
        "chunk_id": "uuid",
        "document_id": "uuid",
        "user_id": "user-uuid",
        "chunk_index": 0,
        "chunk_text": "text content",
        "page_number": 1,
        "filename": "doc.pdf",
        "file_type": ".pdf",
        "token_count": 45,
        "score": 0.89,
        "rank": 1
      }
    ],
    "total_retrieved": 1,
    "top_k": 5,
    "similarity_threshold": 0.5
  }
  ```
- **Empty Result Handling**: If no vectors match, returns `results = []` and `total_retrieved = 0` without invoking the LLM.

### Phase 11 Boundaries & Deferred Functionality
The following capabilities are strictly deferred to subsequent phases:
- **Phase 12**: Hybrid Search (Dense + Sparse BM25 retrieval).
- **Phase 13**: Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`).
- **Phase 14**: Conversational Query Rewriting & History Enrichment.
- **Phase 15**: Grounded Context Assembly & Gemini LLM Synthesis.
- **Phase 16**: Structured Citation Extraction & Verification.
- **Phase 17**: Streaming Chat & Evaluation Framework.

