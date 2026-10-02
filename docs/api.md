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

## 4. Conversations & Messages (`/api/v1/conversations`)

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/api/v1/conversations` | List user's conversation sessions | Yes |
| `POST` | `/api/v1/conversations` | Create a new conversation session | Yes |
| `GET` | `/api/v1/conversations/{conversation_id}` | Retrieve conversation turn history | Yes |
| `DELETE` | `/api/v1/conversations/{conversation_id}` | Delete a conversation and its messages | Yes |

---

## 4. Grounded Chat & RAG (`/api/v1/chat`)

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `POST` | `/api/v1/chat/query` | Execute grounded RAG query with optional filters & citation metadata | Yes |
| `POST` | `/api/v1/chat/stream` | Stream grounded RAG generation (Server-Sent Events) | Yes |

---

## 5. Health & Observability (`/api/v1/health`)

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/api/v1/health` | Service liveness check | No |
| `GET` | `/api/v1/health/readiness` | Probes MySQL & Qdrant connectivity | No |
