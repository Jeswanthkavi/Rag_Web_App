# RAG Pipeline Specification — Ingestion, Retrieval & Synthesis

## 1. Pipeline Overview

The Advanced RAG platform implements a disciplined two-stage architecture: **Asynchronous Ingestion Pipeline** and **Synchronous RAG Query Pipeline**.

```
INGESTION PIPELINE:
Document Upload → Validation & Virus/Format Check → Text Extraction (PDF/DOCX/TXT) 
                → Text Normalization → Semantic/Recursive Chunking 
                → Dense Embeddings (Local/Configurable) → Qdrant Vector Upsert 
                → Relational Metadata Persistence → Status Update (COMPLETED)

QUERY & GENERATION PIPELINE (Phase 15 & 16 Updated):
JWT Authentication → Conversation Validation → Bounded Conversation History
                   → Conversational Query Rewriting (if history present) [Phase 15]
                   → Dense Semantic Retrieval in Qdrant (Candidate Pool: retrieval_top_k=20)
                   → Cross-Encoder Reranking (Re-orders candidates -> final_top_k=5) [Phase 16]
                   → Structured Context Construction ([Source 1], [Source 2])
                   → Grounded LLM Prompt Assembly (Answers ORIGINAL User Query)
                   → Gemini LLM Generation → Citation Verification
                   → User & Assistant Message Persistence → Structured RAGResponse
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
- **Token-Count Strategy**: Approximated via deterministic whitespace token counting (`len(text.split())`).
- **Page Boundaries & Citations**:
  - **PDF**: Page boundaries are strictly preserved locally per page so that `page_number` in `DocumentChunk` directly points to the actual page (1-based index) for precise citation lookup.
  - **TXT / DOCX**: `page_number` is set to `None` since reliable page numbers do not exist.
- **Chunk Indexing**: 0-based deterministic sequential indexing (`0, 1, 2, ...`).
- **Persistence & Idempotency**:
  - Existing chunks for a document are atomically replaced inside a database transaction before new chunks are persisted.
  - Updates `Document.total_chunks` to the exact count of generated chunks.

### Stage 4: Dense Embedding Generation (Phase 9 Specification)
- **Provider Interface Abstraction**: `BaseEmbeddingProvider` defines abstract contract (`provider_name`, `model_name`, `dimension`, `embed_text`, `embed_texts`, `validate_vector`, `validate_vectors`).
- **Default Local Provider**: `LocalEmbeddingProvider` using `sentence-transformers/all-MiniLM-L6-v2` for high-throughput, zero-cost local dense inference.
- **Vector Dimension**: `384` dimensions.
- **Batch Processing**: Configurable `batch_size` (default `32`) ensuring positional 1:1 correspondence between input chunk sequences and output vectors.
- **Normalization**: Configurable L2 unit-norm normalization (`normalize_embeddings=True`) ensuring unit length for direct cosine similarity calculations.

### Stage 5: Qdrant Vector Storage & Indexing (Phase 10 Specification)
- **Vector Database**: Qdrant running locally at `http://localhost:6333` with persistent Docker storage.
- **Collection Configuration**:
  - `collection_name`: Configurable (default `rag_document_chunks`).
  - `dimension`: `384` (strictly aligned with `sentence-transformers/all-MiniLM-L6-v2`).
  - `distance`: `Distance.COSINE`.
- **Point ID Strategy**: Deterministic UUID string directly matching `DocumentChunk.id` to guarantee 1:1 relational correspondence.
- **Payload Schema**:
  - `document_id`, `user_id`, `chunk_id`, `chunk_index`, `page_number`, `filename`, `file_type`, `token_count`, `chunk_text`.
- **Tenant Isolation**: All Qdrant queries, point counts, and deletions enforce strict `user_id` field filters (`FieldCondition(key="user_id", match=MatchValue(value=user_id))`).

---

## 3. Conversational Query Rewriting (Phase 15 Specification)

### Objective
Transforms an ambiguous or follow-up conversational query (e.g., *"What about the battery?"*) into a standalone retrieval query (e.g., *"What is the warranty coverage or warranty period for the battery?"*) using prior conversation turns.

### Key Architecture & Principles
- **Dual Query Representation**:
  - `original_query`: The user's exact input, used for:
    - User turn persistence in conversation memory
    - The LLM prompt `=== USER QUESTION ===` section to ensure direct answering
    - The API response `query` field
  - `retrieval_query`: The standalone reformulated query, used exclusively for:
    - Dense vector embedding generation and Qdrant candidate search
    - Cross-encoder reranker scoring
- **Provider Abstraction**:
  - `BaseQueryRewriter`: Abstract interface defining `.rewrite(query, conversation_history)`.
  - `GeminiQueryRewriter`: Production implementation calling Gemini REST API with zero temperature (`temperature=0.0`).
  - `FakeQueryRewriter`: Deterministic provider for unit and integration testing.
- **Rewriting Rules & Guardrails**:
  1. Produces a standalone search query that can be understood in isolation.
  2. Resolves pronouns (*it, they, this, that, its, their*) and vague references (*the previous issue, the battery*).
  3. Preserves domain terms, technical keywords, product names, numbers, and dates.
  4. Does NOT answer the question.
  5. Does NOT invent facts unsupported by the conversation history.
  6. Returns ONLY the rewritten query text.
- **Standalone Query Bypass**:
  - If there is no prior conversation history (e.g. first turn) or history is whitespace, rewriting is bypassed immediately without calling the LLM provider.
- **Prompt Injection Defense**:
  - Conversation history is explicitly delimited as `=== UNTRUSTED CONVERSATION HISTORY ===` and marked as passive data.
  - The model instruction strictly forbids executing commands embedded inside past messages.
- **Safe Fallback Behavior**:
  - If the rewriter fails, times out, or returns an empty string, the system gracefully falls back: `retrieval_query = original_query`.
  - The overall RAG request never fails solely due to a query rewriting error.

---

## 4. Candidate Retrieval & Cross-Encoder Reranking (Phase 16 Specification)

### Objective
Improves retrieval precision by retrieving a larger candidate pool of dense vector matches and reranking them with a cross-encoder model.

### Key Architecture & Principles
- **Candidate Pool vs Final Count**:
  - `retrieval_top_k` (default `20`): Number of candidate chunks retrieved from Qdrant via dense cosine similarity.
  - `final_top_k` (default `5`): Final number of top reranked chunks supplied to prompt context construction and the LLM.
- **Score Semantics & Preservation**:
  - `dense_score`: Cosine similarity score from vector search. Preserved in `RerankResult.dense_score`.
  - `rerank_score`: Relevance score predicted by the cross-encoder (`query ↔ chunk_text`). Preserved in `RerankResult.rerank_score`.
  - `rank`: 1-indexed final rank position (`1..final_top_k`).
- **Provider Abstraction**:
  - `BaseReranker`: Abstract interface defining `.rerank(query, candidates, top_k)`.
  - `LocalReranker`: Local cross-encoder provider (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with lazy loading and batch inference.
  - `FakeReranker`: Deterministic mock reranker for testing supporting custom scoring functions and simulated failures.
- **Safe Fallback Behavior**:
  - If the reranker is disabled or encounters an inference error, the system gracefully falls back to the original dense retrieval candidate ordering sliced to `final_top_k`.
- **Tenant Isolation**:
  - Reranking strictly evaluates the pre-filtered candidates returned by tenant-isolated Qdrant search. The reranker never accesses the database or vector store directly.

---

## 5. Grounded Generation, Citations & Memory (Phase 12, 13 & 14)

### Context Assembly & Prompt Formatting
`ContextBuilder` formats reranked results into deterministic source blocks:
```text
[Source 1]
Document: system_specs.pdf (Page 4)
Chunk ID: chunk-uuid-1
Content:
The battery warranty covers 2 years of defect-free operation.

[Source 2]
Document: terms.docx
Chunk ID: chunk-uuid-2
Content:
Replacement battery units are available at certified service centers.
```

### Prompt Structure & Anti-Injection Delimiters
```text
=== CONVERSATION HISTORY (Contextual Only, Not Authoritative Source) ===
User: What is the device warranty?
Assistant: The device warranty is 2 years [Source 1].
=== END OF CONVERSATION HISTORY ===

=== RETRIEVED CONTEXT ===
[Source 1]
Document: system_specs.pdf (Page 4)
Chunk ID: chunk-uuid-1
Content:
The battery warranty covers 2 years of defect-free operation.
=== END OF CONTEXT ===

=== USER QUESTION ===
What about the battery?
```

### Citation Verification
- `CitationParser` extracts source markers (`[Source 1]`, `[Source 2]`).
- Verifies indices against actual retrieved chunks (`1..len(reranked_results)`).
- Rejects fabricated or out-of-bounds citations.
- Populates citation metadata (`document_id`, `chunk_id`, `filename`, `page_number`, `chunk_index`) strictly from authoritative candidate records.

---

## 6. Strict Phase Boundaries

- **Phase 15 (Implemented)**: Conversational Query Rewriting with Gemini & Fake providers, prompt injection guards, and fallback.
- **Phase 16 (Implemented)**: Cross-Encoder Candidate Reranking with Local & Fake providers, dense/rerank score preservation, and fallback.
- **Phase 17 (NOT Implemented)**: Hybrid Retrieval (Dense + BM25/Sparse), Reciprocal Rank Fusion (RRF), and Score Fusion belong strictly to Phase 17.
- **Phase 18 (NOT Implemented)**: Precision@K, Recall@K, MRR, and full automated evaluation framework belong strictly to Phase 18.
