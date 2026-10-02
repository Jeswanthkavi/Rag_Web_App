# Production Deployment & Operation Guide

## 1. Overview

The Advanced RAG Platform is built to run reliably locally and in containerized environments with zero vendor lock-in.

### Architecture Stack
- **API Server**: FastAPI (Python 3.12, Uvicorn, Non-root user)
- **Frontend**: Next.js 14 (React 18, TypeScript, Multi-stage Node 20 build)
- **Relational DB**: MySQL 8.0 (Tenant isolation, metadata, conversations, messages, chunk catalog)
- **Vector DB**: Qdrant (384-dimensional dense vectors with cosine similarity and tenant payload filtering)
- **Local ML**: SentenceTransformers (`all-MiniLM-L6-v2`) and Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
- **LLM Provider**: Google Gemini (`gemini-2.5-flash`) with structured grounded prompting

---

## 2. Prerequisites
- Docker Engine 24+ & Docker Compose v2+
- OR Local environment: Python 3.12, Node.js 20+, MySQL 8.0, Qdrant

---

## 3. Deployment Procedure (Step-by-Step)

### Step 1: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Ensure you provide a valid `GEMINI_API_KEY` and secure random strings for `SECRET_KEY` and `JWT_SECRET_KEY`.

### Step 2: Start Storage & Database Services
```bash
docker compose up -d mysql qdrant
```

### Step 3: Run Database Migrations
Before starting the backend, apply all Alembic schema migrations:
```bash
# If running with local Python:
alembic upgrade head

# Or inside a one-off backend migration container:
docker compose run --rm backend alembic upgrade head
```

### Step 4: Start Backend and Frontend Services
```bash
docker compose up -d --build backend frontend
```

### Step 5: Verify Health and Readiness
```bash
# Liveness probe:
curl -f http://localhost:8000/health

# Readiness probe (verifies MySQL & Qdrant connectivity):
curl -f http://localhost:8000/ready
```

---

## 4. Health & Observability Endpoints

| Endpoint | Method | Purpose | Normal Response |
| :--- | :--- | :--- | :--- |
| `/health` | GET | Process liveness check | `{"status": "healthy", "service": "Advanced-RAG-Platform"}` (200 OK) |
| `/ready` | GET | Infrastructure dependency readiness | `{"status": "ready", "database": "connected", "vector_store": "connected"}` (200 OK) |
| `/api/v1/rag/query` | POST | Grounded RAG Query with Latency Telemetry | Structured `RAGResponse` with `latency_metrics` breakdown |

---

## 5. Security & Isolation Verification
- **Tenant Isolation**: All database queries strictly join/filter on `user_id`. Qdrant queries use `must: [FieldCondition(key="user_id", match=MatchValue(value=user_id))]`.
- **BM25 In-Memory Isolation**: Tokenized corpus and index models are cached in `TenantBM25Cache` keyed strictly by `(user_id, document_id)`.
- **OWASP Headers**: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Strict-Transport-Security`, `Content-Security-Policy`.
- **Rate Limiting**: Sliding window in-memory rate limiter per client IP.
- **Request Tracing**: All requests are tagged with `X-Request-ID` and logged structured in JSON format.
