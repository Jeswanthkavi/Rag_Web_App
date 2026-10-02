"""Application Configuration Management.

Provides type-safe, environment-driven settings using Pydantic Settings v2.
Enforces multi-tier configuration validation, secret masking, and safe startup checks.
"""

from functools import lru_cache
from typing import List, Optional, Union
import re
import urllib.parse
from pydantic import (
    BaseModel,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseModel):
    """Core application and server settings."""
    app_name: str = Field(default="Advanced-RAG-Platform", description="Application display name")
    app_env: str = Field(default="development", description="Runtime environment: development, staging, production, test")
    debug: bool = Field(default=True, description="Debug mode flag")
    api_v1_prefix: str = Field(default="/api/v1", description="Prefix for API version 1 routes")
    host: str = Field(default="0.0.0.0", description="API host bind address")
    port: int = Field(default=8000, ge=1, le=65535, description="API port")
    secret_key: SecretStr = Field(
        default=SecretStr("default-dev-secret-key-change-in-prod-12345678"),
        description="Application secret key for internal crypto operations"
    )
    frontend_url: Optional[str] = Field(
        default=None,
        description="Configurable production frontend origin (e.g. https://your-app.vercel.app)"
    )
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins"
    )

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                import json
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


class DatabaseSettings(BaseModel):
    """Relational database connection and pooling settings (MySQL)."""
    database_url: str = Field(
        default="mysql+pymysql://root:@localhost:3306/advanced_rag",
        description="SQLAlchemy database URL for application database"
    )
    test_database_url: Optional[str] = Field(
        default="mysql+pymysql://root:@localhost:3306/advanced_rag_test",
        description="SQLAlchemy database URL for test database"
    )
    db_pool_size: int = Field(default=10, ge=1, le=100, description="SQLAlchemy connection pool size")
    db_max_overflow: int = Field(default=20, ge=0, le=100, description="SQLAlchemy pool max overflow")
    db_pool_recycle: int = Field(default=3600, ge=60, description="SQLAlchemy connection recycle in seconds")
    db_echo: bool = Field(default=False, description="Echo SQL statements to stdout")

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("DATABASE_URL cannot be empty")
        if not (v.startswith("mysql://") or v.startswith("mysql+pymysql://") or v.startswith("sqlite://")):
            raise ValueError("DATABASE_URL must be a valid MySQL or SQLite URL")
        return v.strip()


class QdrantSettings(BaseModel):
    """Vector database connection settings (Qdrant)."""
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant HTTP URL")
    qdrant_api_key: Optional[SecretStr] = Field(default=None, description="Optional Qdrant Cloud / API Key")
    qdrant_collection_name: str = Field(default="rag_document_chunks", description="Target vector collection name")
    qdrant_timeout: float = Field(default=10.0, gt=0, description="Qdrant client timeout in seconds")
    qdrant_batch_size: int = Field(default=64, ge=1, le=500, description="Batch size for Qdrant upserts")
    prefer_grpc: bool = Field(default=False, description="Prefer gRPC over HTTP")

    @field_validator("qdrant_url")
    @classmethod
    def validate_qdrant_url(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("QDRANT_URL cannot be empty")
        parsed = urllib.parse.urlparse(v)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid QDRANT_URL: {v}")
        return v.strip().rstrip("/")


class OpenRouterSettings(BaseModel):
    """OpenRouter / NVIDIA Nemotron LLM configuration."""
    openrouter_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="OpenRouter API Key"
    )
    openrouter_model: str = Field(
        default="nvidia/nemotron-3-ultra-550b-a55b:free",
        description="OpenRouter model identifier"
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL"
    )
    openrouter_temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature")
    openrouter_max_tokens: int = Field(default=2048, ge=64, le=8192, description="Maximum completion tokens")
    openrouter_timeout: float = Field(default=30.0, gt=0, description="OpenRouter API request timeout in seconds")
    openrouter_app_name: str = Field(default="Advanced-RAG-Platform", description="Application display name for OpenRouter header")
    openrouter_http_referer: str = Field(default="http://localhost:8000", description="Application HTTP referer for OpenRouter header")


class GeminiSettings(BaseModel):
    """Google Gemini LLM configuration."""
    gemini_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Google Gemini API Key"
    )
    gemini_model: str = Field(default="gemini-3.8-flash", description="Gemini model identifier")
    gemini_temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature")
    gemini_max_tokens: int = Field(default=2048, ge=64, le=8192, description="Maximum completion tokens")
    gemini_timeout: float = Field(default=30.0, gt=0, description="Gemini API request timeout in seconds")


class AuthSettings(BaseModel):
    """Authentication and JWT configuration."""
    jwt_secret_key: SecretStr = Field(
        default=SecretStr("default-jwt-secret-key-change-in-prod-87654321"),
        description="Secret key for signing JWT tokens"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    access_token_expire_minutes: int = Field(default=60, ge=5, le=43200, description="JWT expiration in minutes")
    password_min_length: int = Field(default=8, ge=6, description="Minimum password length")


class StorageSettings(BaseModel):
    """File upload and document storage limits."""
    upload_storage_dir: str = Field(default="./storage/uploads", description="Directory to store uploaded documents")
    max_file_size_mb: int = Field(default=25, ge=1, le=100, description="Maximum upload file size in megabytes")
    allowed_file_extensions: List[str] = Field(
        default=[".pdf", ".docx", ".txt"],
        description="Allowed document file extensions"
    )

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @field_validator("allowed_file_extensions", mode="before")
    @classmethod
    def parse_allowed_extensions(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                import json
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [ext.strip().lower() for ext in v.split(",") if ext.strip()]
        return [ext.lower() for ext in v]


class EmbeddingSettings(BaseModel):
    """Embedding model configuration."""
    embedding_provider: str = Field(default="local", description="Embedding provider: local, gemini, etc.")
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model name/path for dense vector generation"
    )
    embedding_dimension: int = Field(default=384, ge=64, le=4096, description="Dimension of generated vectors")
    embedding_batch_size: int = Field(default=32, ge=1, le=256, description="Batch size for embedding generation")
    normalize_embeddings: bool = Field(default=True, description="Whether to normalize dense embeddings")


class QueryRewritingSettings(BaseModel):
    """Conversational query rewriting configuration."""
    enable_query_rewriting: bool = Field(default=True, description="Enable query rewriting for conversational follow-ups")
    query_rewriter_model: str = Field(default="nvidia/nemotron-3-ultra-550b-a55b:free", description="Model name for query rewriting")
    query_rewriter_temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Temperature for query rewriting")
    query_rewriter_max_history_messages: int = Field(default=5, ge=1, le=20, description="Max history messages for rewriting prompt")
    query_rewriter_max_query_length: int = Field(default=500, ge=50, le=2000, description="Max length of rewritten query")
    query_rewriter_timeout: float = Field(default=10.0, gt=0, description="Query rewriting request timeout in seconds")


class HybridRetrievalSettings(BaseModel):
    """Hybrid retrieval (Dense + BM25/Lexical + RRF) configuration."""
    enable_hybrid_retrieval: bool = Field(default=True, description="Enable hybrid dense and lexical retrieval")
    dense_top_k: int = Field(default=20, ge=1, le=100, description="Number of candidates retrieved from dense vector search")
    lexical_top_k: int = Field(default=20, ge=1, le=100, description="Number of candidates retrieved from lexical search")
    fusion_candidate_limit: int = Field(default=20, ge=1, le=100, description="Maximum fused candidates passed to reranker")
    rrf_k: int = Field(default=60, ge=1, le=200, description="Reciprocal Rank Fusion smoothing constant")
    bm25_k1: float = Field(default=1.5, ge=0.1, le=5.0, description="BM25 term frequency saturation parameter")
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0, description="BM25 document length normalization parameter")


class RetrievalSettings(BaseModel):
    """Text chunking and retrieval parameters."""
    chunk_size: int = Field(default=500, ge=50, le=4000, description="Character chunk target size")
    chunk_overlap: int = Field(default=50, ge=0, le=1000, description="Overlap characters between chunks")
    retrieval_top_k: int = Field(default=20, ge=1, le=100, description="Number of candidate chunks for initial dense retrieval")
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum cosine similarity score")
    enable_reranking: bool = Field(default=True, description="Enable cross-encoder reranking")
    reranking_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder reranking model name"
    )
    reranking_top_k: int = Field(default=5, ge=1, le=50, description="Final number of top reranked chunks")

    @model_validator(mode="after")
    def validate_chunk_params(self) -> "RetrievalSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        if self.reranking_top_k > self.retrieval_top_k:
            raise ValueError("reranking_top_k cannot exceed retrieval_top_k")
        return self


class ConversationSettings(BaseModel):
    """Conversation memory and multi-turn context settings."""
    max_history_messages: int = Field(default=10, ge=1, le=100, description="Maximum number of historical conversation turns to supply as context")


class SecuritySettings(BaseModel):
    """Security hardening, rate limiting, and protection parameters."""
    rate_limit_enabled: bool = Field(default=True, description="Enable request rate limiting")
    rate_limit_login: int = Field(default=30, ge=1, le=1000, description="Max login requests per minute")
    rate_limit_register: int = Field(default=15, ge=1, le=1000, description="Max registration requests per minute")
    rate_limit_upload: int = Field(default=60, ge=1, le=1000, description="Max document upload requests per minute")
    rate_limit_rag: int = Field(default=120, ge=1, le=1000, description="Max RAG query requests per minute")
    rate_limit_general: int = Field(default=600, ge=1, le=5000, description="Max general API requests per minute")
    max_request_body_bytes: int = Field(default=30 * 1024 * 1024, ge=1024, description="Max request body size in bytes")
    enable_security_headers: bool = Field(default=True, description="Inject OWASP security response headers")


class LoggingSettings(BaseModel):
    """Application logging configuration."""
    log_level: str = Field(default="INFO", description="Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    log_format: str = Field(default="json", description="Log format: json or text")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        upper = v.upper()
        if upper not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"Invalid log_level: {v}")
        return upper



class Settings(BaseSettings):
    """Unified application settings combining all configuration sections.
    
    Reads from environment variables and an optional .env file.
    Provides safe data masking to prevent secret leaks in logs and telemetry.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # Core App
    app_name: str = "Advanced-RAG-Platform"
    app_env: str = "development"
    debug: bool = True
    api_v1_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    secret_key: SecretStr = SecretStr("default-dev-secret-key-change-in-prod-12345678")
    frontend_url: Optional[str] = None
    allowed_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # Database
    database_url: str = "mysql+pymysql://root:@localhost:3306/advanced_rag"
    test_database_url: Optional[str] = "mysql+pymysql://root:@localhost:3306/advanced_rag_test"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_recycle: int = 3600
    db_echo: bool = False

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[SecretStr] = None
    qdrant_collection_name: str = "rag_document_chunks"
    qdrant_timeout: float = 10.0
    qdrant_batch_size: int = 64
    prefer_grpc: bool = False

    # OpenRouter (Primary LLM)
    openrouter_api_key: SecretStr = SecretStr("")
    openrouter_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_temperature: float = 0.2
    openrouter_max_tokens: int = 2048
    openrouter_timeout: float = 30.0
    openrouter_app_name: str = "Advanced-RAG-Platform"
    openrouter_http_referer: str = "http://localhost:8000"

    # Gemini (Secondary / Optional Legacy)
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-3.8-flash"
    gemini_temperature: float = 0.2
    gemini_max_tokens: int = 2048
    gemini_timeout: float = 30.0

    # Auth
    jwt_secret_key: SecretStr = SecretStr("default-jwt-secret-key-change-in-prod-87654321")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    password_min_length: int = 8

    # Storage
    upload_storage_dir: str = "./storage/uploads"
    max_file_size_mb: int = 25
    allowed_file_extensions: List[str] = [".pdf", ".docx", ".txt"]

    # Embedding
    embedding_provider: str = "local"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_batch_size: int = 32
    normalize_embeddings: bool = True

    # Retrieval & Reranking
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 20
    score_threshold: float = 0.5
    enable_reranking: bool = True
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranking_top_k: int = 5

    # Hybrid Retrieval (Phase 17)
    enable_hybrid_retrieval: bool = True
    dense_top_k: int = 20
    lexical_top_k: int = 20
    fusion_candidate_limit: int = 20
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Query Rewriting
    enable_query_rewriting: bool = True
    query_rewriter_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    query_rewriter_temperature: float = 0.0
    query_rewriter_max_history_messages: int = 5
    query_rewriter_max_query_length: int = 500
    query_rewriter_timeout: float = 10.0

    # Conversation Memory
    max_history_messages: int = 10

    # Security Hardening (Phase 20)
    rate_limit_enabled: bool = True
    rate_limit_login: int = 30
    rate_limit_register: int = 15
    rate_limit_upload: int = 60
    rate_limit_rag: int = 120
    rate_limit_general: int = 600
    max_request_body_bytes: int = 30 * 1024 * 1024
    enable_security_headers: bool = True

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"


    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _parse_origins(cls, v: Union[str, List[str]]) -> List[str]:
        return AppSettings.parse_allowed_origins(v)

    @field_validator("allowed_file_extensions", mode="before")
    @classmethod
    def _parse_extensions(cls, v: Union[str, List[str]]) -> List[str]:
        return StorageSettings.parse_allowed_extensions(v)

    @field_validator("database_url")
    @classmethod
    def _validate_db_url(cls, v: str) -> str:
        return DatabaseSettings.validate_database_url(v)

    @field_validator("qdrant_url")
    @classmethod
    def _validate_qdrant_url(cls, v: str) -> str:
        return QdrantSettings.validate_qdrant_url(v)

    @model_validator(mode="after")
    def _validate_cross_field_rules(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        if self.reranking_top_k > self.retrieval_top_k:
            raise ValueError("reranking_top_k cannot exceed retrieval_top_k")
        if self.frontend_url and self.frontend_url.strip():
            url = self.frontend_url.strip()
            if url not in self.allowed_origins:
                self.allowed_origins.append(url)
        return self

    # Sub-section property accessors for modular service dependency injection
    @property
    def app(self) -> AppSettings:
        return AppSettings(
            app_name=self.app_name,
            app_env=self.app_env,
            debug=self.debug,
            api_v1_prefix=self.api_v1_prefix,
            host=self.host,
            port=self.port,
            secret_key=self.secret_key,
            frontend_url=self.frontend_url,
            allowed_origins=self.allowed_origins,
        )

    @property
    def database(self) -> DatabaseSettings:
        return DatabaseSettings(
            database_url=self.database_url,
            test_database_url=self.test_database_url,
            db_pool_size=self.db_pool_size,
            db_max_overflow=self.db_max_overflow,
            db_pool_recycle=self.db_pool_recycle,
            db_echo=self.db_echo,
        )

    @property
    def qdrant(self) -> QdrantSettings:
        return QdrantSettings(
            qdrant_url=self.qdrant_url,
            qdrant_api_key=self.qdrant_api_key,
            qdrant_collection_name=self.qdrant_collection_name,
            qdrant_timeout=self.qdrant_timeout,
            qdrant_batch_size=self.qdrant_batch_size,
            prefer_grpc=self.prefer_grpc,
        )

    @property
    def openrouter(self) -> OpenRouterSettings:
        return OpenRouterSettings(
            openrouter_api_key=self.openrouter_api_key,
            openrouter_model=self.openrouter_model,
            openrouter_base_url=self.openrouter_base_url,
            openrouter_temperature=self.openrouter_temperature,
            openrouter_max_tokens=self.openrouter_max_tokens,
            openrouter_timeout=self.openrouter_timeout,
            openrouter_app_name=self.openrouter_app_name,
            openrouter_http_referer=self.openrouter_http_referer,
        )

    @property
    def gemini(self) -> GeminiSettings:
        return GeminiSettings(
            gemini_api_key=self.gemini_api_key,
            gemini_model=self.gemini_model,
            gemini_temperature=self.gemini_temperature,
            gemini_max_tokens=self.gemini_max_tokens,
            gemini_timeout=self.gemini_timeout,
        )

    @property
    def auth(self) -> AuthSettings:
        return AuthSettings(
            jwt_secret_key=self.jwt_secret_key,
            jwt_algorithm=self.jwt_algorithm,
            access_token_expire_minutes=self.access_token_expire_minutes,
            password_min_length=self.password_min_length,
        )

    @property
    def storage(self) -> StorageSettings:
        return StorageSettings(
            upload_storage_dir=self.upload_storage_dir,
            max_file_size_mb=self.max_file_size_mb,
            allowed_file_extensions=self.allowed_file_extensions,
        )

    @property
    def embedding(self) -> EmbeddingSettings:
        return EmbeddingSettings(
            embedding_provider=self.embedding_provider,
            embedding_model_name=self.embedding_model_name,
            embedding_dimension=self.embedding_dimension,
            embedding_batch_size=self.embedding_batch_size,
            normalize_embeddings=self.normalize_embeddings,
        )

    @property
    def retrieval(self) -> RetrievalSettings:
        return RetrievalSettings(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            retrieval_top_k=self.retrieval_top_k,
            score_threshold=self.score_threshold,
            enable_reranking=self.enable_reranking,
            reranking_model=self.reranking_model,
            reranking_top_k=self.reranking_top_k,
        )

    @property
    def hybrid_retrieval(self) -> HybridRetrievalSettings:
        return HybridRetrievalSettings(
            enable_hybrid_retrieval=self.enable_hybrid_retrieval,
            dense_top_k=self.dense_top_k,
            lexical_top_k=self.lexical_top_k,
            fusion_candidate_limit=self.fusion_candidate_limit,
            rrf_k=self.rrf_k,
            bm25_k1=self.bm25_k1,
            bm25_b=self.bm25_b,
        )

    @property
    def query_rewriting(self) -> QueryRewritingSettings:
        return QueryRewritingSettings(
            enable_query_rewriting=self.enable_query_rewriting,
            query_rewriter_model=self.query_rewriter_model,
            query_rewriter_temperature=self.query_rewriter_temperature,
            query_rewriter_max_history_messages=self.query_rewriter_max_history_messages,
            query_rewriter_max_query_length=self.query_rewriter_max_query_length,
            query_rewriter_timeout=self.query_rewriter_timeout,
        )

    @property
    def conversation(self) -> ConversationSettings:
        return ConversationSettings(
            max_history_messages=self.max_history_messages,
        )

    @property
    def security(self) -> SecuritySettings:
        return SecuritySettings(
            rate_limit_enabled=self.rate_limit_enabled,
            rate_limit_login=self.rate_limit_login,
            rate_limit_register=self.rate_limit_register,
            rate_limit_upload=self.rate_limit_upload,
            rate_limit_rag=self.rate_limit_rag,
            rate_limit_general=self.rate_limit_general,
            max_request_body_bytes=self.max_request_body_bytes,
            enable_security_headers=self.enable_security_headers,
        )

    @property
    def logging(self) -> LoggingSettings:
        return LoggingSettings(
            log_level=self.log_level,
            log_format=self.log_format,
        )

    @staticmethod
    def _mask_url_password(url: str) -> str:
        """Sanitizes database or HTTP credentials from URLs for safe logging."""
        if not url:
            return ""
        try:
            parsed = urllib.parse.urlsplit(url)
            if parsed.password:
                netloc = f"{parsed.username}:***@{parsed.hostname}"
                if parsed.port:
                    netloc += f":{parsed.port}"
                return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
        except Exception:
            pass
        return re.sub(r":([^/@]+)@", ":***@", url)

    def safe_summary(self) -> dict:
        """Returns a sanitized dictionary representation masking all secrets."""
        return {
            "app": {
                "name": self.app_name,
                "env": self.app_env,
                "debug": self.debug,
                "api_v1_prefix": self.api_v1_prefix,
                "host": self.host,
                "port": self.port,
                "allowed_origins": self.allowed_origins,
            },
            "database": {
                "database_url": self._mask_url_password(self.database_url),
                "test_database_url": self._mask_url_password(self.test_database_url or ""),
                "pool_size": self.db_pool_size,
                "max_overflow": self.db_max_overflow,
            },
            "qdrant": {
                "url": self.qdrant_url,
                "collection_name": self.qdrant_collection_name,
                "api_key_configured": bool(self.qdrant_api_key and self.qdrant_api_key.get_secret_value()),
                "batch_size": self.qdrant_batch_size,
                "prefer_grpc": self.prefer_grpc,
            },
            "openrouter": {
                "model": self.openrouter_model,
                "base_url": self.openrouter_base_url,
                "temperature": self.openrouter_temperature,
                "max_tokens": self.openrouter_max_tokens,
                "api_key_configured": bool(self.openrouter_api_key and self.openrouter_api_key.get_secret_value()),
            },
            "gemini": {
                "model": self.gemini_model,
                "temperature": self.gemini_temperature,
                "max_tokens": self.gemini_max_tokens,
                "api_key_configured": bool(self.gemini_api_key and self.gemini_api_key.get_secret_value()),
            },
            "auth": {
                "jwt_algorithm": self.jwt_algorithm,
                "access_token_expire_minutes": self.access_token_expire_minutes,
                "password_min_length": self.password_min_length,
            },
            "storage": {
                "upload_dir": self.upload_storage_dir,
                "max_file_size_mb": self.max_file_size_mb,
                "allowed_extensions": self.allowed_file_extensions,
            },
            "embedding": {
                "provider": self.embedding_provider,
                "model_name": self.embedding_model_name,
                "dimension": self.embedding_dimension,
                "normalize_embeddings": self.normalize_embeddings,
            },
            "retrieval": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "retrieval_top_k": self.retrieval_top_k,
                "score_threshold": self.score_threshold,
                "enable_reranking": self.enable_reranking,
                "reranking_model": self.reranking_model,
                "reranking_top_k": self.reranking_top_k,
            },
            "hybrid_retrieval": {
                "enabled": self.enable_hybrid_retrieval,
                "dense_top_k": self.dense_top_k,
                "lexical_top_k": self.lexical_top_k,
                "fusion_candidate_limit": self.fusion_candidate_limit,
                "rrf_k": self.rrf_k,
            },
            "query_rewriting": {
                "enabled": self.enable_query_rewriting,
                "model": self.query_rewriter_model,
                "temperature": self.query_rewriter_temperature,
                "max_history_messages": self.query_rewriter_max_history_messages,
                "max_query_length": self.query_rewriter_max_query_length,
            },
            "conversation": {
                "max_history_messages": self.max_history_messages,
            },
            "security": {
                "rate_limit_enabled": self.rate_limit_enabled,
                "rate_limit_login": self.rate_limit_login,
                "rate_limit_register": self.rate_limit_register,
                "rate_limit_upload": self.rate_limit_upload,
                "rate_limit_rag": self.rate_limit_rag,
                "rate_limit_general": self.rate_limit_general,
                "max_request_body_bytes": self.max_request_body_bytes,
                "enable_security_headers": self.enable_security_headers,
            },
            "logging": {
                "log_level": self.log_level,
                "log_format": self.log_format,
            }
        }



def validate_startup_config(settings: Settings) -> bool:
    """Performs non-destructive validation of critical startup settings.
    
    Returns True if configuration is valid; raises ValueError on critical misconfiguration.
    """
    errors = []
    
    # Check Database URL
    if not settings.database_url:
        errors.append("DATABASE_URL is required and cannot be empty")
        
    # Check Qdrant URL
    if not settings.qdrant_url:
        errors.append("QDRANT_URL is required and cannot be empty")
        
    # Check OpenRouter API Key
    if not settings.openrouter_api_key.get_secret_value():
        errors.append("OPENROUTER_API_KEY is required and cannot be empty")
        
    # Check Chunk parameters
    if settings.chunk_overlap >= settings.chunk_size:
        errors.append("CHUNK_OVERLAP must be strictly less than CHUNK_SIZE")

    # Production-specific safety hardening
    if settings.app_env.lower() in ("production", "prod"):
        if settings.jwt_secret_key.get_secret_value() == "default-jwt-secret-key-change-in-prod-87654321":
            errors.append("JWT_SECRET_KEY must be explicitly configured with a secure random secret in production")
        if settings.secret_key.get_secret_value() == "default-dev-secret-key-change-in-prod-12345678":
            errors.append("SECRET_KEY must be explicitly configured with a secure random secret in production")
        if "*" in settings.allowed_origins:
            errors.append("Wildcard '*' is forbidden in ALLOWED_ORIGINS when running in production")
        
    if errors:
        raise ValueError(f"Startup configuration validation failed: {'; '.join(errors)}")
        
    return True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton instance of application settings."""
    return Settings()
