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
- **PDF**: Page-by-page text extraction with layout and page number tracking.
- **DOCX**: Paragraph and heading structure extraction.
- **TXT**: UTF-8 stream normalization.
- **Normalization**: Unicode cleanup, removal of null bytes, newline normalization, and whitespace standardization.

### Stage 3: Configurable Chunking
- Recursive character splitting with sliding token/character windows and overlap (default `chunk_size=500`, `chunk_overlap=50`).
- Preservation of page metadata and section headings per chunk.

### Stage 4: Embedding Generation
- Embedding interface abstraction (`EmbeddingProvider` protocol).
- Default: Local dense embeddings (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions) for zero-cost, high-speed local inference.
- Configurable switch to remote providers.

### Stage 5: Vector Store & Relational Indexing
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
