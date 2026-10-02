"""FastAPI Dependencies for database sessions, services, and authentication."""

from typing import Generator, Optional
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from backend.app.core.config import get_settings
from backend.app.core.security import decode_access_token
from backend.app.db.session import get_db
from backend.app.models.user import User
from backend.app.repositories.user_repository import UserRepository
from backend.app.repositories.document_repository import DocumentRepository
from backend.app.repositories.document_chunk_repository import DocumentChunkRepository
from backend.app.services.auth_service import AuthService
from backend.app.services.document_service import DocumentService
from backend.app.chunking.service import ChunkingService
from backend.app.embeddings.service import EmbeddingService
from backend.app.vectorstore.service import VectorStoreService
from backend.app.retrieval.service import RetrievalService
from backend.app.llm.base import BaseLLMProvider
from backend.app.llm.gemini import GeminiProvider
from backend.app.rag.service import RAGService

settings = get_settings()

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.api_v1_prefix}/auth/login",
    auto_error=False,
)


def get_user_repository(db: Session = Depends(get_db)) -> UserRepository:
    """Dependency for providing an instance of UserRepository."""
    return UserRepository(db)


def get_document_repository(db: Session = Depends(get_db)) -> DocumentRepository:
    """Dependency for providing an instance of DocumentRepository."""
    return DocumentRepository(db)


def get_document_chunk_repository(db: Session = Depends(get_db)) -> DocumentChunkRepository:
    """Dependency for providing an instance of DocumentChunkRepository."""
    return DocumentChunkRepository(db)


def get_chunking_service() -> ChunkingService:
    """Dependency for providing an instance of ChunkingService."""
    return ChunkingService()


def get_embedding_service() -> EmbeddingService:
    """Dependency for providing an instance of EmbeddingService."""
    return EmbeddingService()


def get_vector_store_service(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> VectorStoreService:
    """Dependency for providing an instance of VectorStoreService."""
    return VectorStoreService(embedding_service=embedding_service)


def get_auth_service(
    user_repo: UserRepository = Depends(get_user_repository),
) -> AuthService:
    """Dependency for providing an instance of AuthService."""
    return AuthService(user_repo)


def get_document_service(
    doc_repo: DocumentRepository = Depends(get_document_repository),
    vector_service: VectorStoreService = Depends(get_vector_store_service),
) -> DocumentService:
    """Dependency for providing an instance of DocumentService."""
    return DocumentService(doc_repo=doc_repo, vector_service=vector_service)


from backend.app.query_rewriting.service import QueryRewritingService
from backend.app.reranking.service import RerankingService
from backend.app.services.conversation_service import ConversationService


def get_conversation_service(db: Session = Depends(get_db)) -> ConversationService:
    """Dependency for providing an instance of ConversationService."""
    return ConversationService(db=db)


def get_query_rewriting_service() -> QueryRewritingService:
    """Dependency for providing an instance of QueryRewritingService."""
    return QueryRewritingService()


def get_reranking_service() -> RerankingService:
    """Dependency for providing an instance of RerankingService."""
    return RerankingService()


def get_retrieval_service(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> RetrievalService:
    """Dependency for providing an instance of RetrievalService."""
    return RetrievalService(embedding_service=embedding_service)


def get_llm_provider() -> BaseLLMProvider:
    """Dependency for providing an instance of BaseLLMProvider."""
    return GeminiProvider()


def get_rag_service(
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    llm_provider: BaseLLMProvider = Depends(get_llm_provider),
    conversation_service: ConversationService = Depends(get_conversation_service),
    query_rewriting_service: QueryRewritingService = Depends(get_query_rewriting_service),
    reranking_service: RerankingService = Depends(get_reranking_service),
) -> RAGService:
    """Dependency for providing an instance of RAGService."""
    return RAGService(
        retrieval_service=retrieval_service,
        llm_provider=llm_provider,
        conversation_service=conversation_service,
        query_rewriting_service=query_rewriting_service,
        reranking_service=reranking_service,
    )


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    user_repo: UserRepository = Depends(get_user_repository),
) -> User:
    """Extracts, validates JWT access token, and returns the authenticated user entity.
    
    Raises 401 Unauthorized on missing, invalid, or expired tokens.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: Optional[str] = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject identifier.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = user_repo.get_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Ensures the authenticated user account is in active status."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account.",
        )
    return current_user
