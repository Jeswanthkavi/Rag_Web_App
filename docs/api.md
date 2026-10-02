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

## 4. Grounded RAG Generation & Citations (`/api/v1/rag`)

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `POST` | `/api/v1/rag/query` | Execute grounded RAG answer generation with citation verification and conversation memory | Yes (Bearer JWT) |

### Request Body (`POST /api/v1/rag/query`)
```json
{
  "query": "What is the warranty period for the solar battery?",
  "top_k": 5,
  "similarity_threshold": 0.5,
  "document_id": "optional-document-uuid",
  "conversation_id": "optional-conversation-uuid"
}
```

### Response Body (`200 OK`)
```json
{
  "query": "What is the warranty period for the solar battery?",
  "answer": "According to the warranty guide [Source 1], the battery includes a 10-year replacement warranty.",
  "retrieved_context_count": 2,
  "citations": [
    {
      "citation_id": "source_1",
      "document_id": "doc-123-uuid",
      "chunk_id": "chunk-456-uuid",
      "filename": "warranty_guide.docx",
      "page_number": null,
      "chunk_index": 0
    }
  ],
  "conversation_id": "conv-uuid-1",
  "message_id": "msg-assistant-uuid",
  "model_name": "gemini-2.5-flash"
}
```

---

## 5. Conversations & Messages (`/api/v1/conversations`)

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/api/v1/conversations` | List user's conversation sessions | Yes (Bearer JWT) |
| `POST` | `/api/v1/conversations` | Create a new conversation session | Yes (Bearer JWT) |
| `GET` | `/api/v1/conversations/{conversation_id}` | Retrieve conversation detail with chronological message history | Yes (Bearer JWT) |
| `PATCH` | `/api/v1/conversations/{conversation_id}` | Update conversation title | Yes (Bearer JWT) |
| `DELETE` | `/api/v1/conversations/{conversation_id}` | Delete a conversation and cascade delete its messages | Yes (Bearer JWT) |

### Request Body (`POST /api/v1/conversations`)
```json
{
  "title": "Solar Research Project"
}
```

### Response Body (`200 OK` / `GET /api/v1/conversations/{id}`)
```json
{
  "id": "conv-uuid-1",
  "user_id": "user-uuid-1",
  "title": "Solar Research Project",
  "created_at": "2026-10-02T10:00:00Z",
  "updated_at": "2026-10-02T10:05:00Z",
  "messages": [
    {
      "id": "msg-1",
      "conversation_id": "conv-uuid-1",
      "role": "USER",
      "content": "What is the solar efficiency?",
      "citations": null,
      "token_metrics": null,
      "created_at": "2026-10-02T10:00:00Z"
    },
    {
      "id": "msg-2",
      "conversation_id": "conv-uuid-1",
      "role": "ASSISTANT",
      "content": "Solar panel efficiency is 22% [Source 1].",
      "citations": [
        {
          "citation_id": "source_1",
          "document_id": "doc-uuid",
          "chunk_id": "chunk-uuid",
          "filename": "solar_specs.pdf",
          "page_number": 3,
          "chunk_index": 0
        }
      ],
      "token_metrics": {
        "prompt_tokens": 120,
        "completion_tokens": 15
      },
      "created_at": "2026-10-02T10:00:02Z"
    }
  ]
}
```

---

## 6. Health & Observability (`/api/v1/health`)

| Method | Endpoint | Description | Auth Required |
|---|---|---|---|
| `GET` | `/api/v1/health` | Service liveness check | No |
| `GET` | `/api/v1/health/readiness` | Probes MySQL & Qdrant connectivity | No |
