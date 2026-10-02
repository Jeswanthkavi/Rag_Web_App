# REST API Specification

All endpoints are versioned under `/api/v1` and return standardized JSON responses.

---

## 1. Authentication & Users (`/api/v1/auth`, `/api/v1/users`)

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `POST` | `/api/v1/auth/register` | Register a new user | No |
| `POST` | `/api/v1/auth/login` | Authenticate and obtain JWT access token | No |
| `GET` | `/api/v1/users/me` | Fetch current authenticated user profile | Yes (Bearer JWT) |

---

## 2. Document Management (`/api/v1/documents`)

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `POST` | `/api/v1/documents/upload` | Upload a PDF/DOCX/TXT document for ingestion | Yes |
| `GET` | `/api/v1/documents` | List all documents owned by current user | Yes |
| `GET` | `/api/v1/documents/{document_id}` | Get status and metadata of a specific document | Yes |
| `DELETE` | `/api/v1/documents/{document_id}` | Delete document, chunks, and Qdrant vectors | Yes |

---

## 3. Semantic Retrieval (`/api/v1/retrieval`)

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `POST` | `/api/v1/retrieval/search` | Execute tenant-isolated semantic similarity search across document chunks | Yes (Bearer JWT) |

### Request Body (`POST /api/v1/retrieval/search`)
```json
{
  "query": "natural language query string",
  "top_k": 5,
  "similarity_threshold": 0.5,
  "document_id": "optional-document-uuid"
}
```

### Response Body (`200 OK`)
```json
{
  "query": "natural language query string",
  "results": [
    {
      "chunk_id": "c1-uuid",
      "document_id": "doc-uuid",
      "user_id": "user-uuid",
      "chunk_index": 0,
      "chunk_text": "Extracted text content...",
      "page_number": 1,
      "filename": "document.pdf",
      "file_type": ".pdf",
      "token_count": 45,
      "score": 0.892,
      "rank": 1
    }
  ],
  "total_retrieved": 1,
  "top_k": 5,
  "similarity_threshold": 0.5
}
```

---

## 4. Grounded RAG Generation (`/api/v1/rag`)

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `POST` | `/api/v1/rag/query` | Execute grounded RAG answer generation against user's documents | Yes (Bearer JWT) |

### Request Body (`POST /api/v1/rag/query`)
```json
{
  "query": "How is the system deployed?",
  "top_k": 5,
  "similarity_threshold": 0.5,
  "document_id": "optional-document-uuid"
}
```

### Response Body (`200 OK`)
```json
{
  "query": "How is the system deployed?",
  "answer": "Based on the retrieved context, the system uses microservices deployed on Kubernetes with auto-scaling.",
  "retrieved_context_count": 2,
  "model_name": "gemini-2.5-flash"
}
```

---

## 5. Conversations & Messages (`/api/v1/conversations`) - Phase 14

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/api/v1/conversations` | List user's conversation sessions | Yes |
| `POST` | `/api/v1/conversations` | Create a new conversation session | Yes |
| `GET` | `/api/v1/conversations/{conversation_id}` | Retrieve conversation turn history | Yes |
| `DELETE` | `/api/v1/conversations/{conversation_id}` | Delete a conversation and its messages | Yes |

---

## 6. Health & Observability (`/api/v1/health`)

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/api/v1/health` | Service liveness check | No |
| `GET` | `/api/v1/health/readiness` | Probes MySQL & Qdrant connectivity | No |
