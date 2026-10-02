# Production Deployment & Operation Guide

## 1. Overview & Architecture Stack

The Advanced RAG Platform is architected for dual-mode execution: local development/self-hosted Docker and cloud-native serverless/container deployment.

```
+-------------------------------------------------------------+
|                     TARGET CLOUD ARCHITECTURE               |
+-------------------------------------------------------------+
                            Internet
                               |
                               v
                     Vercel Edge Network
                   Next.js 14 Web Frontend
                               |
                               v (HTTPS / Authorization Bearer)
                   Cloud FastAPI API Server
                               |
       +-----------------------+-----------------------+
       |                       |                       |
       v                       v                       v
 Managed MySQL 8.0       Qdrant Cloud        Cloud Object Storage
 (Schema & Metadata)  (Vector Collection)      (S3 / R2 / GCS)
                               |
                               v
                   OpenRouter / NVIDIA Nemotron
                      (nvidia/nemotron-3-ultra-550b-a55b:free)
```

---

## 2. Environment Configuration Matrix

Configuration is strictly divided into **Frontend Public Variables** (exposed to browsers) and **Backend-Only Secrets** (never accessible to the frontend).

### A. Frontend Environment Configuration (`frontend/.env.production` or Vercel Environment)

| Variable | Scope | Description | Example Placeholder |
| :--- | :--- | :--- | :--- |
| `NEXT_PUBLIC_API_BASE_URL` | Public (Client-side) | Full base URL to the deployed backend API | `https://api.yourdomain.com/api/v1` |

> [!CAUTION]
> **Zero Backend Secrets Rule**: Never set `OPENROUTER_API_KEY`, `JWT_SECRET_KEY`, `DATABASE_URL`, or `QDRANT_API_KEY` in the frontend environment. Any variable prefixed with `NEXT_PUBLIC_` is readable by end-users.

---

### B. Backend Environment Configuration (Cloud Platform / Container Secrets)

| Variable | Required in Prod | Description | Placeholder / Format |
| :--- | :--- | :--- | :--- |
| `APP_ENV` | Yes | Runtime environment flag | `production` |
| `DEBUG` | Yes | Disable debug tracebacks | `False` |
| `PORT` | Auto / Yes | Container listen port | `8000` |
| `HOST` | Yes | Host bind address | `0.0.0.0` |
| `SECRET_KEY` | Yes | Cryptographic secret for internal tokens | `<64-character-random-hex>` |
| `JWT_SECRET_KEY` | Yes | Secret used to sign user JWT access tokens | `<64-character-random-hex>` |
| `FRONTEND_URL` | Yes | Production frontend origin for CORS | `https://your-app.vercel.app` |
| `ALLOWED_ORIGINS` | Optional | Comma-separated list of additional origins | `https://your-app.vercel.app,https://admin.yourdomain.com` |
| `DATABASE_URL` | Yes | Managed MySQL connection URL | `mysql+pymysql://<user>:<password>@<managed-mysql-host>:3306/<database>?ssl_ca=<path>` |
| `QDRANT_URL` | Yes | Qdrant Cloud cluster endpoint | `https://<cluster-id>.<region>.gcp.cloud.qdrant.io:6333` |
| `QDRANT_API_KEY` | Yes | Qdrant Cloud access API key | `<qdrant-cloud-api-key>` |
| `QDRANT_COLLECTION_NAME` | Yes | Collection name | `rag_document_chunks` |
| `OPENROUTER_API_KEY` | Yes | OpenRouter API authorization key | `sk-or-v1-...` |
| `OPENROUTER_MODEL` | Yes | Default generation model | `nvidia/nemotron-3-ultra-550b-a55b:free` |
| `OPENROUTER_BASE_URL` | Yes | Base completions URL | `https://openrouter.ai/api/v1` |
| `OPENROUTER_TIMEOUT` | No | Timeout in seconds | `30.0` |
| `UPLOAD_STORAGE_DIR` | Yes (Local/Mount) | Base directory for physical file uploads | `/app/storage/uploads` |

---

## 3. Local Development vs Production Deployment

| Aspect | Local Development | Cloud Production |
| :--- | :--- | :--- |
| **Frontend Host** | Next.js dev server on `http://localhost:3000` | Vercel Serverless CDN (`https://*.vercel.app`) |
| **Backend Host** | Uvicorn reload on `http://localhost:8000` | Container runtime (Render / AWS ECS / Cloud Run / Kubernetes) |
| **Relational Database** | Local Docker MySQL (`localhost:3306`) | Managed Cloud MySQL (AWS RDS / PlanetScale / Aiven) with SSL |
| **Vector Database** | Local Docker Qdrant (`http://localhost:6333`) | Qdrant Cloud Managed Cluster with HTTPS & API Key |
| **Document Storage** | Local directory (`./storage/uploads`) | Durable Shared Mount or Cloud Object Storage (S3 / R2 / GCS) |
| **CORS Policy** | `http://localhost:3000`, `http://127.0.0.1:3000` | Strict origin matching `FRONTEND_URL` (No `*` wildcards allowed) |
| **JWT Secrets** | Dev fallback defaults allowed | **Fails fast on startup** if default secrets are detected |

---

## 4. File Storage & Cloud Durability Requirements

### Ephemeral Container Advisory
Cloud container platforms (Render, Cloud Run, AWS Fargate) utilize ephemeral local filesystems that are wiped on redeployment or horizontal scaling. 

1. **Document Storage Abstraction**: The backend employs `BaseStorageProvider` (`backend/app/storage/base.py`) which decouples document access from disk paths.
2. **Production Storage Strategy**:
   - **Option A (Persistent Volume)**: Mount a durable network volume (e.g. AWS EFS or Render Disk) at `/app/storage/uploads`.
   - **Option B (Object Storage)**: Configure an S3-compatible bucket (AWS S3, Cloudflare R2, Google Cloud Storage) using the `BaseStorageProvider` interface.

---

## 5. Privacy & Rate Limiting Disclaimers

### Upstream OpenRouter / NVIDIA Nemotron Privacy Notice
The default model `nvidia/nemotron-3-ultra-550b-a55b:free` operates on OpenRouter's free public inference tier.
- **Data Sensitivity**: Free-tier public models may log prompts and completions according to upstream provider terms.
- **Prohibited Data**: Do NOT upload classified, HIPAA-regulated, PCI-sensitive, or proprietary business documents when using the free tier.
- **Paid Tier Migration**: For production enterprise compliance, upgrade to paid private routing models (e.g., `anthropic/claude-3.5-sonnet`, `openai/gpt-4o`, or dedicated enterprise NVIDIA endpoints) by setting `OPENROUTER_MODEL` and providing a funded `OPENROUTER_API_KEY`.

### Rate Limiting Architecture
1. **Application Rate Limiter**: Enforces sliding-window limits per client IP (Login: 30/min, Register: 15/min, RAG: 120/min). In multi-instance deployments, each node tracks limits in-memory; for clustered deployments, Redis-backed tracking may be layered on.
2. **Upstream Rate Limits**: OpenRouter free-tier endpoints enforce concurrency and RPM limits. The backend handles upstream HTTP 429 and transient overload payloads gracefully without leaking request headers or API tokens.

---

## 6. Pre-Deployment Step-by-Step Checklist

### Step 1: Database Migration
Before directing live traffic to the backend, run Alembic migrations against the clean production database:
```bash
alembic upgrade head
```
*Current head revision*: `d998d17cd59d` (Initial schema with users, documents, document_chunks, conversations, messages).

### Step 2: Health & Readiness Verification
```bash
# Liveness probe (Returns 200 process status):
curl -f https://api.yourdomain.com/health

# Readiness probe (Verifies MySQL & Qdrant connectivity):
curl -f https://api.yourdomain.com/ready
```

### Step 3: Frontend Deployment on Vercel
1. Link GitHub repository to Vercel.
2. Set Root Directory to `frontend`.
3. Set Environment Variable: `NEXT_PUBLIC_API_BASE_URL=https://api.yourdomain.com/api/v1`.
4. Deploy and verify authentication and document workflows.
