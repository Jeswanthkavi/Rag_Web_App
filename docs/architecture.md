# System Architecture — Advanced RAG Platform

## 1. Architectural Overview

The **Advanced RAG Platform** is an enterprise-grade, multi-tenant Retrieval-Augmented Generation system designed for document intelligence, semantic retrieval, grounded synthesis, and interactive conversational analysis.

```mermaid
graph TD
    subgraph Client Layer
        FE[React + TypeScript Frontend]
    end

    subgraph API Gateway / Backend Layer
        FastAPI[FastAPI Application]
        AuthMid[JWT Auth Middleware & Dependency]
        RouterAuth[/api/v1/auth]
        RouterDocs[/api/v1/documents]
        RouterChat[/api/v1/chat]
        RouterConv[/api/v1/conversations]
    end

    subgraph Service & Orchestration Layer
        AuthSvc[Auth Service]
        IngestSvc[Document Ingestion & Chunking Service]
        EmbedSvc[Embedding Service]
        VectorSvc[Vector Service]
        RetrievalSvc[Hybrid Retrieval & Reranker Service]
        RAGSvc[Grounded RAG & Citation Service]
    end

    subgraph Storage & Infrastructure Layer
        MySQL[(MySQL 8.0: App State & Metadata)]
        Qdrant[(Qdrant: Vector Database)]
        Gemini[Google Gemini API]
        LocalDisk[Secure Local File Storage]
    end

    FE -->|HTTP/REST| FastAPI
    FastAPI --> AuthMid
    AuthMid --> RouterAuth & RouterDocs & RouterChat & RouterConv
    RouterAuth --> AuthSvc
    RouterDocs --> IngestSvc
    RouterChat --> RAGSvc
    
    IngestSvc --> LocalDisk
    IngestSvc --> EmbedSvc
    EmbedSvc --> VectorSvc
    VectorSvc --> Qdrant
    IngestSvc --> MySQL
    
    RAGSvc --> RetrievalSvc
    RetrievalSvc --> VectorSvc
    RAGSvc --> Gemini
    RAGSvc --> MySQL
```

---

## 2. Core Subsystems

### 2.1 Backend Application (`backend/`)
- **FastAPI Core**: High-performance asynchronous REST API framework.
- **SQLAlchemy 2.0 & Alembic**: Declarative relational ORM with strict migration tracking against MySQL.
- **Pydantic v2 & Pydantic-Settings**: Type-safe schema validation, request/response serialization, and environment configuration management.
- **Service Layer Pattern**: Business logic decoupled from API routing controllers and database models.
- **Repository Layer Pattern**: Abstracted database query operations with strict user-level tenancy filters.

### 2.2 Relational Data Storage (MySQL 8.0)
Stores structured application state, access control, document metadata, chunk text references, and conversation history:
- `users`: User identity, credentials (Argon2 / Bcrypt hashed), tenancy attributes.
- `documents`: File catalog, extraction status, mime types, file sizes, processing telemetry.
- `document_chunks`: Extracted text chunks, page offsets, chunk indices, token counts, vector references.
- `conversations`: Chat sessions tied to specific user contexts.
- `messages`: Chronological conversation turns (user prompts, model completions, citations, token stats).

### 2.3 Vector Database (Qdrant)
Stores high-dimensional dense embeddings derived from document chunks:
- Collection: `rag_document_chunks`
- Payload Schema:
  - `user_id` (Indexed keyword filter — **guarantees multi-tenant isolation**)
  - `document_id` (Indexed keyword filter)
  - `chunk_id` (Unique chunk identifier)
  - `page_number` (Integer page reference)
  - `chunk_index` (Integer chunk sequence)
  - `filename` (String original source name)
  - `text` (String chunk payload)

### 2.4 LLM & Grounding Engine (Google Gemini)
- Utilizes Google Gemini models for query reformulation, conversational synthesis, and strict context-grounded response generation.
- Strict system prompts enforce that generation answers strictly from retrieved source context with cited references.

---

## 3. Strict Multi-Tenant Security & Isolation

1. **User Data Isolation**:
   - Every database query for documents, chunks, conversations, and messages enforces `WHERE user_id = :current_user_id`.
   - Every Qdrant vector retrieval query enforces a required `Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])`.
   - Cross-tenant document retrieval is structurally impossible at both database and vector index levels.

2. **Secret Hygiene**:
   - Zero credentials, passwords, or API keys are hardcoded in source code or documentation.
   - All sensitive settings are loaded via `.env` through Pydantic-Settings.
   - Passwords hashed using standard cryptographic algorithms.
