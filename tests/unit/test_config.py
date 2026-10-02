"""Unit tests for configuration management (Phase 2)."""

import pytest
from pydantic import ValidationError, SecretStr
from backend.app.core.config import (
    Settings,
    AppSettings,
    DatabaseSettings,
    QdrantSettings,
    OpenRouterSettings,
    GeminiSettings,
    AuthSettings,
    StorageSettings,
    EmbeddingSettings,
    RetrievalSettings,
    LoggingSettings,
    validate_startup_config,
)


class TestConfigDefaults:
    """Test standard default configuration values."""

    def test_default_settings_instantiation(self):
        settings = Settings()
        assert settings.app_name == "Advanced-RAG-Platform"
        assert settings.api_v1_prefix == "/api/v1"
        assert settings.qdrant_collection_name == "rag_document_chunks"
        assert settings.embedding_dimension == 384
        assert settings.chunk_size == 500
        assert settings.chunk_overlap == 50
        assert settings.retrieval_top_k == 20
        assert settings.reranking_top_k == 5
        assert settings.enable_reranking is True
        assert settings.enable_query_rewriting is True
        assert settings.score_threshold == 0.5
        assert settings.openrouter_model == "nvidia/nemotron-3-ultra-550b-a55b:free"
        assert settings.openrouter_temperature == 0.2
        assert settings.openrouter_max_tokens == 2048

    def test_modular_subsection_properties(self):
        settings = Settings()
        assert isinstance(settings.app, AppSettings)
        assert isinstance(settings.database, DatabaseSettings)
        assert isinstance(settings.qdrant, QdrantSettings)
        assert isinstance(settings.openrouter, OpenRouterSettings)
        assert isinstance(settings.gemini, GeminiSettings)
        assert isinstance(settings.auth, AuthSettings)
        assert isinstance(settings.storage, StorageSettings)
        assert isinstance(settings.embedding, EmbeddingSettings)
        assert isinstance(settings.retrieval, RetrievalSettings)
        assert isinstance(settings.logging, LoggingSettings)
        from backend.app.core.config import QueryRewritingSettings
        assert isinstance(settings.query_rewriting, QueryRewritingSettings)


class TestConfigParsingAndValidation:
    """Test validation rules, custom parsing, and constraint checks."""

    def test_allowed_origins_parsing(self):
        # Test comma-separated string
        app_cfg = AppSettings(allowed_origins="http://localhost:3000,http://app.example.com")
        assert app_cfg.allowed_origins == ["http://localhost:3000", "http://app.example.com"]

        # Test list of strings
        app_cfg2 = AppSettings(allowed_origins=["http://localhost:3000"])
        assert app_cfg2.allowed_origins == ["http://localhost:3000"]

    def test_allowed_file_extensions_parsing(self):
        storage_cfg = StorageSettings(allowed_file_extensions=".PDF, .Docx, .txt")
        assert storage_cfg.allowed_file_extensions == [".pdf", ".docx", ".txt"]
        assert storage_cfg.max_file_size_bytes == 25 * 1024 * 1024

    def test_invalid_database_url(self):
        with pytest.raises(ValidationError):
            DatabaseSettings(database_url="postgres://user:pass@localhost/db")

        with pytest.raises(ValidationError):
            DatabaseSettings(database_url="")

    def test_invalid_qdrant_url(self):
        with pytest.raises(ValidationError):
            QdrantSettings(qdrant_url="not-a-valid-url")

    def test_invalid_log_level(self):
        with pytest.raises(ValidationError):
            LoggingSettings(log_level="INVALID_LEVEL")

    def test_chunk_overlap_must_be_less_than_chunk_size(self):
        with pytest.raises(ValidationError) as excinfo:
            RetrievalSettings(chunk_size=200, chunk_overlap=200)
        assert "chunk_overlap must be strictly less than chunk_size" in str(excinfo.value)

        with pytest.raises(ValidationError) as excinfo2:
            Settings(chunk_size=200, chunk_overlap=300)
        assert "chunk_overlap must be strictly less than chunk_size" in str(excinfo2.value)

    def test_reranking_top_k_constraint(self):
        with pytest.raises(ValidationError) as excinfo:
            RetrievalSettings(retrieval_top_k=5, reranking_top_k=10)
        assert "reranking_top_k cannot exceed retrieval_top_k" in str(excinfo.value)


class TestSecretMaskingAndSecurity:
    """Verify secrets are masked and never exposed in string representations."""

    def test_secret_str_masking(self):
        settings = Settings(
            openrouter_api_key=SecretStr("super-secret-openrouter-key-12345"),
            secret_key=SecretStr("app-secret-67890"),
        )
        assert "super-secret-openrouter-key-12345" not in str(settings.openrouter_api_key)
        assert "super-secret-openrouter-key-12345" not in repr(settings.openrouter_api_key)
        assert settings.openrouter_api_key.get_secret_value() == "super-secret-openrouter-key-12345"

    def test_safe_summary_masks_credentials(self):
        settings = Settings(
            database_url="mysql+pymysql://admin:secretpassword123@localhost:3306/advanced_rag",
            openrouter_api_key=SecretStr("my-openrouter-key"),
        )
        summary = settings.safe_summary()
        
        # Verify database password is masked
        assert "secretpassword123" not in str(summary)
        assert "admin:***@localhost:3306/advanced_rag" in summary["database"]["database_url"]
        
        # Verify API key is represented only as boolean status
        assert "my-openrouter-key" not in str(summary)
        assert summary["openrouter"]["api_key_configured"] is True


class TestStartupValidation:
    """Test validation of required settings at application startup."""

    def test_startup_validation_success(self):
        valid_settings = Settings(
            database_url="mysql+pymysql://root:@localhost:3306/advanced_rag",
            qdrant_url="http://localhost:6333",
            openrouter_api_key=SecretStr("valid-openrouter-key-value"),
        )
        assert validate_startup_config(valid_settings) is True

    def test_startup_validation_fails_on_empty_openrouter_key(self):
        invalid_settings = Settings(
            database_url="mysql+pymysql://root:@localhost:3306/advanced_rag",
            qdrant_url="http://localhost:6333",
            openrouter_api_key=SecretStr(""),
        )
        with pytest.raises(ValueError) as exc:
            validate_startup_config(invalid_settings)
        assert "OPENROUTER_API_KEY is required" in str(exc.value)


class TestEnvironmentVariableLoading:
    """Verify environment variables override defaults properly."""

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("APP_NAME", "Custom-RAG-Engine")
        monkeypatch.setenv("PORT", "9090")
        monkeypatch.setenv("CHUNK_SIZE", "800")
        monkeypatch.setenv("CHUNK_OVERLAP", "100")
        monkeypatch.setenv("DEBUG", "False")

        settings = Settings()
        assert settings.app_name == "Custom-RAG-Engine"
        assert settings.port == 9090
        assert settings.chunk_size == 800
        assert settings.chunk_overlap == 100
        assert settings.debug is False

    def test_frontend_url_merged_into_allowed_origins(self):
        settings = Settings(frontend_url="https://rag-app.vercel.app")
        assert "https://rag-app.vercel.app" in settings.allowed_origins
        assert "https://rag-app.vercel.app" in settings.app.allowed_origins

    def test_production_fails_fast_on_default_jwt_secret(self):
        prod_settings = Settings(
            app_env="production",
            database_url="mysql+pymysql://root:@localhost:3306/advanced_rag",
            qdrant_url="http://localhost:6333",
            openrouter_api_key=SecretStr("real-openrouter-key"),
            jwt_secret_key=SecretStr("default-jwt-secret-key-change-in-prod-87654321"),
        )
        with pytest.raises(ValueError) as exc:
            validate_startup_config(prod_settings)
        assert "JWT_SECRET_KEY must be explicitly configured" in str(exc.value)

    def test_production_fails_fast_on_wildcard_cors(self):
        prod_settings = Settings(
            app_env="production",
            database_url="mysql+pymysql://root:@localhost:3306/advanced_rag",
            qdrant_url="http://localhost:6333",
            openrouter_api_key=SecretStr("real-openrouter-key"),
            jwt_secret_key=SecretStr("super-strong-random-prod-jwt-secret-xyz-12345"),
            secret_key=SecretStr("super-strong-random-prod-app-secret-abc-67890"),
            allowed_origins=["*"],
        )
        with pytest.raises(ValueError) as exc:
            validate_startup_config(prod_settings)
        assert "Wildcard '*' is forbidden in ALLOWED_ORIGINS" in str(exc.value)

    def test_production_passes_with_valid_config(self):
        prod_settings = Settings(
            app_env="production",
            database_url="mysql+pymysql://root:@localhost:3306/advanced_rag",
            qdrant_url="http://localhost:6333",
            openrouter_api_key=SecretStr("real-openrouter-key"),
            jwt_secret_key=SecretStr("super-strong-random-prod-jwt-secret-xyz-12345"),
            secret_key=SecretStr("super-strong-random-prod-app-secret-abc-67890"),
            allowed_origins=["https://rag-app.vercel.app"],
        )
        assert validate_startup_config(prod_settings) is True

