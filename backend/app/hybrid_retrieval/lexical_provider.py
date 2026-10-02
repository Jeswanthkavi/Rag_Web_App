"""SQL & BM25 backed lexical retrieval provider with strict tenant isolation."""

import logging
from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.session import get_db, SessionLocal
from backend.app.hybrid_retrieval.base import BaseLexicalRetriever, LexicalRetrievalError
from backend.app.hybrid_retrieval.bm25 import BM25Okapi, tokenize
from backend.app.hybrid_retrieval.config import HybridRetrievalConfig
from backend.app.models.document import Document
from backend.app.models.document_chunk import DocumentChunk
from backend.app.retrieval.models import RetrievalResult

logger = logging.getLogger("app.hybrid_retrieval.lexical")


class SQLLexicalRetriever(BaseLexicalRetriever):
    """Production lexical retriever executing BM25 scoring over tenant-scoped SQL document chunks."""

    def __init__(
        self,
        db: Optional[Session] = None,
        config: Optional[HybridRetrievalConfig] = None,
    ):
        self._db = db
        self.config = config or HybridRetrievalConfig.from_settings()

    @property
    def provider_name(self) -> str:
        return "sql_bm25"

    def _get_session(self) -> Session:
        if self._db is not None:
            return self._db
        return SessionLocal()

    def search(
        self,
        query: str,
        user_id: str,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Performs BM25 lexical keyword search across chunks owned strictly by user_id."""
        if not user_id or not user_id.strip():
            raise LexicalRetrievalError("user_id is required for tenant-isolated lexical search.")

        if not query or not query.strip():
            return []

        clean_user_id = user_id.strip()
        clean_doc_id = document_id.strip() if document_id and document_id.strip() else None
        limit = top_k if top_k is not None else self.config.lexical_top_k

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        session = self._get_session()
        close_session = self._db is None

        try:
            # 1. Fetch chunks strictly scoped to the authenticated tenant
            stmt = (
                select(DocumentChunk, Document.filename, Document.file_type)
                .join(Document, DocumentChunk.document_id == Document.id)
                .where(DocumentChunk.user_id == clean_user_id)
            )
            if clean_doc_id:
                stmt = stmt.where(DocumentChunk.document_id == clean_doc_id)

            rows = session.execute(stmt).all()
            if not rows:
                return []

            # 2. Tokenize corpus chunks
            chunks = []
            tokenized_corpus = []
            for chunk_obj, filename, file_type in rows:
                chunks.append((chunk_obj, filename, file_type))
                tokenized_corpus.append(tokenize(chunk_obj.chunk_text))

            # 3. Score corpus using BM25
            bm25 = BM25Okapi(
                corpus=tokenized_corpus,
                k1=self.config.bm25_k1,
                b=self.config.bm25_b,
            )
            scores = bm25.get_scores(query_tokens)

            # 4. Pair chunks with non-zero scores
            scored_candidates = []
            for idx, ((chunk_obj, filename, file_type), score) in enumerate(zip(chunks, scores)):
                if score > 0.0:
                    scored_candidates.append((score, chunk_obj, filename, file_type))

            # 5. Sort descending by BM25 score
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            sliced = scored_candidates[:limit]

            # 6. Format into standard RetrievalResult objects
            results: List[RetrievalResult] = []
            for rank_idx, (score, chunk_obj, filename, file_type) in enumerate(sliced, start=1):
                results.append(
                    RetrievalResult(
                        chunk_id=chunk_obj.id,
                        document_id=chunk_obj.document_id,
                        user_id=chunk_obj.user_id,
                        chunk_index=chunk_obj.chunk_index,
                        chunk_text=chunk_obj.chunk_text,
                        page_number=chunk_obj.page_number,
                        filename=filename,
                        file_type=file_type,
                        token_count=chunk_obj.token_count,
                        score=float(score),
                        rank=rank_idx,
                    )
                )

            return results

        except Exception as e:
            sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
            logger.error(f"SQL BM25 search failed for user '{clean_user_id}': {sanitized}")
            raise LexicalRetrievalError(f"Lexical retrieval failed: {sanitized}") from e
        finally:
            if close_session:
                session.close()
