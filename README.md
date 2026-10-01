# Enterprise Advanced RAG Platform

A production-grade, multi-tenant Retrieval-Augmented Generation (RAG) platform with document ingestion, asynchronous processing, local dense embeddings, vector retrieval in Qdrant, metadata filtering, cross-encoder reranking, query rewriting, grounded generation via Google Gemini, citation verification, evaluation benchmarks, and modern full-stack architecture.

---

## Architecture & System Overview

```
[ Frontend: React + TypeScript ]
               │
               ▼ (REST API / JWT Auth)
[ Backend: FastAPI + SQLAlchemy 2.0 ]
   ├── Document Ingestion & Text Processing (PDF / DOCX / TXT)
   ├── Local Embeddings (Sentence-Transformers)
   ├── Vector Store & Tenancy Isolation (Qdrant)
   ├── Reranking & Query Rewriting
   └── Grounded Generation & Citation Verification (Google Gemini)
               │
   ┌───────────┴───────────┐
   ▼                       ▼
[ MySQL 8.0 ]       [ Qdrant Vector DB ]
(App State & Metadata)  (Vector Embeddings & Payloads)
```

For in-depth architecture and design documents, see:
- [Architecture Overview](file:///c:/Users/JESWANTHKAVI/Desktop/advanced-rag/docs/architecture.md)
- [RAG Ingestion & Query Pipeline](file:///c:/Users/JESWANTHKAVI/Desktop/advanced-rag/docs/rag-pipeline.md)
- [REST API Specification](file:///c:/Users/JESWANTHKAVI/Desktop/advanced-rag/docs/api.md)
- [Evaluation Framework](file:///c:/Users/JESWANTHKAVI/Desktop/advanced-rag/docs/evaluation.md)

---

## Tech Stack

| Component | Technology | Description |
|---|---|---|
| **Backend API** | Python 3.12, FastAPI, Pydantic v2 | High performance asynchronous REST API |
| **Relational DB** | MySQL 8.0, SQLAlchemy 2.0, Alembic | Relational metadata, user accounts, and conversation state |
| **Vector DB** | Qdrant | Dense vector search with tenant-isolated payload filtering |
| **LLM Provider** | Google Gemini API (`gemini-2.5-flash`) | Context-grounded synthesis and query rewriting |
| **Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`) | Modular local dense vector generation (384-d) |
| **Frontend** | React, TypeScript, Vite | Modern responsive user interface |

---

## Infrastructure & Environment Verification

### Prerequisites
- Python 3.12+ (Virtual environment in `.venv/`)
- MySQL 8.0 (Databases: `advanced_rag`, `advanced_rag_test`)
- Qdrant Vector DB (Port `6333`)
- Docker Desktop

### Running Environment Verification
To verify your environment, services, and connectivity:

```powershell
.venv\Scripts\python.exe scripts\verify_environment.py
```

Expected output:
- Python 3.12+ and `.venv` verification: `[PASS]`
- Git & Docker daemon verification: `[PASS]`
- MySQL connectivity (`advanced_rag` & `advanced_rag_test`): `[PASS]`
- Qdrant connectivity (`http://localhost:6333`): `[PASS]`
- Environment variables presence: `[PASS]`

---

## Project Structure

```
advanced-rag/
├── .env.example              # Sanitized environment configuration template
├── .gitignore                # Git ignore rules for secrets, virtualenv, and caches
├── README.md                 # Project master overview
├── docs/                     # Comprehensive architecture and technical specs
│   ├── architecture.md       # High-level system architecture and multi-tenant security
│   ├── rag-pipeline.md       # Document ingestion and RAG generation lifecycles
│   ├── api.md                # REST API design specifications
│   └── evaluation.md         # Retrieval and generation evaluation framework
├── scripts/                  # DevOps, tooling, and verification scripts
│   └── verify_environment.py # Environment and infrastructure validation script
├── backend/                  # FastAPI backend application (Implemented in subsequent phases)
└── frontend/                 # React TypeScript client (Implemented in Phase 19)
```

---

## Security & Multi-Tenancy Principles

1. **Strict Tenant Isolation**: All database queries and vector similarity searches strictly filter by authenticated `user_id`. No cross-tenant document or vector leakage is permitted.
2. **Zero Secret Exposure**: Passwords, API keys, and JWT tokens are managed strictly via `.env` and Pydantic settings. No secret values are logged or committed.
3. **Grounded Synthesis**: LLM generation strictly relies on retrieved document context, preventing ungrounded hallucinations.
