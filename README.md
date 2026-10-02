# Advanced RAG Platform

A production-grade, local-first, multi-tenant Retrieval-Augmented Generation (RAG) platform featuring hierarchical document ingestion, local dense vector embeddings, BM25 lexical search, Reciprocal Rank Fusion (RRF), Cross-Encoder reranking, conversational memory with query rewriting, grounded LLM synthesis via Google Gemini, verified inline citations, security hardening, performance optimizations, and full Docker orchestration.

---

## 1. Architecture Overview

```text
User / Browser (Next.js 14 + React 18 + TypeScript)
                       │
                       ▼ (REST API / JWT Authentication)
          FastAPI Application (Python 3.12)
   ┌───────────────────┼───────────────────┐
   ▼                   ▼                   ▼
Document Ingestion   Hybrid Retrieval     Conversational RAG
 - PDF/DOCX/TXT       - Local Dense MiniLM - Query Rewriting
 - Hierarchical Chunk - BM25 Lexical Cache - Context Builder
 - SHA-256 Dedupl.    - RRF Fusion (k=60)  - Gemini LLM
                      - Cross-Encoder Rerank Verified Citations
   │                   │                   │
   └───────────┬───────┴───────────────────┘
               ▼
  ┌────────────────────────┐
  │ MySQL 8.0 & Qdrant DB  │
  └────────────────────────┘
```

For in-depth architecture and operational documentation:
- [Architecture & Isolation](docs/architecture.md)
- [RAG Ingestion & Hybrid Pipeline](docs/rag-pipeline.md)
- [REST API Documentation](docs/api.md)
- [Evaluation Framework](docs/evaluation.md)
- [Production Deployment Guide](docs/deployment.md)
- [Backup & Disaster Recovery Procedures](docs/backup_recovery.md)

---

## 2. Tech Stack

- **Backend API**: Python 3.12, FastAPI, Pydantic v2, SQLAlchemy 2.0
- **Frontend Client**: Next.js 14, React 18, TypeScript, CSS Modules
- **Relational Storage**: MySQL 8.0 with Alembic schema migrations
- **Vector Storage**: Qdrant (384-dimensional dense vectors with cosine similarity)
- **Local Dense Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (Local, zero API cost)
- **Lexical Keyword Search**: `BM25Okapi` with thread-safe `TenantBM25Cache`
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (Local inference)
- **LLM Synthesis**: Google Gemini (`gemini-2.5-flash`) with strict anti-injection grounding
- **Security**: Argon2id password hashing, JWT authentication, sliding-window rate limiting, OWASP headers, payload size bounds

---

## 3. Prerequisites

- Python 3.12+ (Virtual environment in `.venv/`)
- Node.js 20+ and npm
- Docker Engine 24+ and Docker Compose v2+
- MySQL 8.0 server (local or containerized)
- Qdrant Vector DB (local or containerized)

---

## 4. Environment Variables Setup

Copy `.env.example` to create your active `.env`:

```bash
cp .env.example .env
```

Key environment variables:
| Variable | Default / Example | Purpose |
| :--- | :--- | :--- |
| `DATABASE_URL` | `mysql+pymysql://root:@localhost:3306/advanced_rag` | Primary MySQL database URL |
| `TEST_DATABASE_URL` | `mysql+pymysql://root:@localhost:3306/advanced_rag_test` | Dedicated isolated test database |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant vector database HTTP URL |
| `GEMINI_API_KEY` | `your_api_key_here` | Google Gemini API Key for synthesis |
| `JWT_SECRET_KEY` | `secure_random_key` | Secret key for signing user JWTs |
| `SECRET_KEY` | `secure_random_key` | Application core cryptographic secret |

---

## 5. How to Start Infrastructure (MySQL & Qdrant)

Using Docker Compose to run backing services:

```bash
docker compose up -d mysql qdrant
```

Verify service connectivity:
```bash
.venv\Scripts\python.exe scripts\verify_environment.py
```

---

## 6. How to Run Database Migrations

Apply all schema revisions to the current head:

```bash
# Using Alembic CLI
.venv\Scripts\alembic upgrade head
```

To downgrade or test rollback:
```bash
.venv\Scripts\alembic downgrade -1
.venv\Scripts\alembic upgrade head
```

---

## 7. How to Start Backend

```bash
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Docs: `http://localhost:8000/docs`
- Health Probe: `http://localhost:8000/health`
- Readiness Probe: `http://localhost:8000/ready`

---

## 8. How to Start Frontend

```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000` in your browser.

---

## 9. How to Run Tests

### Backend Unit & Integration Tests (349+ tests)
```bash
.venv\Scripts\python.exe -m pytest -q
```

### Frontend Unit & Component Tests (9+ tests)
```bash
cd frontend
npm test
```

### Frontend Production Build Validation
```bash
cd frontend
npm run build
```

---

## 10. How to Run Benchmarks & Evaluation

### Retrieval & RAG Latency Benchmark
```bash
.venv\Scripts\python.exe scripts\run_benchmark.py
```
Reports Mean, Median, and P95 latency for Dense embedding, BM25 lexical retrieval, RRF hybrid fusion, and Cross-Encoder reranking.

### Comprehensive RAG Quality Evaluation
```bash
.venv\Scripts\python.exe scripts\run_evaluation.py
```
Measures Precision@K, Recall@K, MRR, NDCG@K, Faithfulness, Answer Relevance, and Citation Precision/Recall across test query sets.

---

## 11. How to Run Full Stack with Docker Compose

Build and launch the complete stack (MySQL, Qdrant, FastAPI Backend, Next.js Frontend):

```bash
docker compose up -d --build
```

Check status and logs:
```bash
docker compose ps
docker compose logs -f backend
```

---

## 12. Known Limitations

1. **Local Model Cold Start**: Initial inference requests for SentenceTransformer and Cross-Encoder load model weights into CPU memory (~100-300ms on first invocation), subsequent queries run in-memory via singleton cache.
2. **LLM API Latency**: Gemini 2.5 Flash API calls are subject to external network latency (~800ms - 2500ms).
3. **Synchronous Ingestion Limits**: File ingestion parses files synchronously in background threads. Files above 25MB are rejected by upload validation to prevent resource exhaustion.
