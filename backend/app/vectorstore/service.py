"""VectorStore service coordinating embedding generation, Qdrant upserts, and MySQL point ID synchronization."""

import logging
from typing import List, Optional
from sqlalchemy.orm import Session

from backend.app.embeddings.service import EmbeddingService
from backend.app.models.document import Document
from backend.app.models.document_chunk import DocumentChunk
from backend.app.repositories.document_chunk_repository import DocumentChunkRepository
from backend.app.vectorstore.client import create_qdrant_client
from backend.app.vectorstore.config import VectorStoreConfig
from backend.app.vectorstore.models import (
    IndexingResult,
    VectorPoint,
    VectorStoreError,
    VectorStoreHealth,
)
from backend.app.vectorstore.repository import QdrantVectorRepository

logger = logging.getLogger("app.vectorstore.service")


class VectorStoreService:
    """Service managing document vectorization, Qdrant point persistence, and relational synchronization."""

    def __init__(
        self,
        vector_repo: Optional[QdrantVectorRepository] = None,
        embedding_service: Optional[EmbeddingService] = None,
        config: Optional[VectorStoreConfig] = None,
    ):
        self.config = config or VectorStoreConfig.from_settings()
        if vector_repo:
            self.vector_repo = vector_repo
        else:
            client = create_qdrant_client(self.config)
            self.vector_repo = QdrantVectorRepository(client, self.config)

        self.embedding_service = embedding_service or EmbeddingService()

    def initialize_collection(self, collection_name: Optional[str] = None) -> bool:
        """Ensures the Qdrant collection is created with correct dimensions and distance metric."""
        return self.vector_repo.initialize_collection(
            collection_name=collection_name,
            dimension=self.embedding_service.dimension,
        )

    def check_health(self, collection_name: Optional[str] = None) -> VectorStoreHealth:
        """Executes health diagnostic on the underlying Qdrant instance and collection."""
        return self.vector_repo.check_health(collection_name)

    def index_document(
        self,
        document: Document,
        db: Session,
        collection_name: Optional[str] = None,
    ) -> IndexingResult:
        """Embeds and upserts all chunks for a document into Qdrant and updates MySQL qdrant_point_id."""
        if not document or not document.id or not document.user_id:
            raise ValueError("Valid Document with persistent id and user_id is required for indexing.")

        target_collection = collection_name or self.config.collection_name

        # 1. Initialize collection idempotently
        self.initialize_collection(target_collection)

        # 2. Retrieve persisted chunks for this document
        chunk_repo = DocumentChunkRepository(db)
        chunks = chunk_repo.get_by_document(document.id, document.user_id)

        if not chunks:
            # If document has no chunks, ensure any previous vectors are cleared
            self.vector_repo.delete_by_document(
                document_id=document.id,
                user_id=document.user_id,
                collection_name=target_collection,
            )
            return IndexingResult(
                document_id=document.id,
                user_id=document.user_id,
                indexed_chunks_count=0,
                collection_name=target_collection,
                point_ids=[],
            )

        # 3. Generate dense embeddings for all chunks in batch
        chunk_embeddings = self.embedding_service.embed_chunks(chunks)

        # 4. Construct VectorPoint objects with deterministic IDs and metadata payloads
        vector_points: List[VectorPoint] = []
        point_ids: List[str] = []

        for chunk, emb in zip(chunks, chunk_embeddings):
            point_id = str(chunk.id)  # Deterministic UUID string
            payload = {
                "document_id": str(document.id),
                "user_id": str(document.user_id),
                "chunk_id": str(chunk.id),
                "chunk_index": int(chunk.chunk_index),
                "page_number": chunk.page_number,
                "filename": document.filename,
                "file_type": document.file_type,
                "token_count": chunk.token_count,
                "chunk_text": chunk.chunk_text,
            }
            vector_points.append(
                VectorPoint(
                    id=point_id,
                    vector=emb.vector,
                    payload=payload,
                )
            )
            point_ids.append(point_id)

        try:
            # 5. Clear previous vectors for this document (idempotency / reindexing cleanup)
            self.vector_repo.delete_by_document(
                document_id=document.id,
                user_id=document.user_id,
                collection_name=target_collection,
            )

            # 6. Batch upsert vectors to Qdrant
            upserted_count = self.vector_repo.upsert_points(
                points=vector_points,
                collection_name=target_collection,
            )

            # 7. Synchronize point IDs back to MySQL document_chunks
            for chunk, p_id in zip(chunks, point_ids):
                chunk.qdrant_point_id = p_id
                db.add(chunk)

            db.commit()
            for chunk in chunks:
                db.refresh(chunk)

            logger.info(
                f"Document {document.id} successfully indexed: {upserted_count} points in '{target_collection}'"
            )

            return IndexingResult(
                document_id=document.id,
                user_id=document.user_id,
                indexed_chunks_count=upserted_count,
                collection_name=target_collection,
                point_ids=point_ids,
            )

        except Exception as e:
            db.rollback()
            sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
            logger.error(f"Indexing failed for Document {document.id}: {sanitized}")
            raise VectorStoreError(f"Document indexing failed: {sanitized}") from e

    def delete_document_vectors(
        self,
        document_id: str,
        user_id: str,
        collection_name: Optional[str] = None,
    ) -> int:
        """Deletes all Qdrant vector points associated with a specific document and tenant user."""
        target_collection = collection_name or self.config.collection_name
        return self.vector_repo.delete_by_document(
            document_id=document_id,
            user_id=user_id,
            collection_name=target_collection,
        )
