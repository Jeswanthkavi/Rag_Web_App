"""Grounded RAG generation service orchestrating semantic retrieval, context assembly, and LLM synthesis."""

import logging
import time
from typing import Optional

from backend.app.llm.base import BaseLLMProvider, LLMProviderError
from backend.app.llm.gemini import GeminiProvider
from backend.app.rag.config import RAGConfig
from backend.app.rag.context_builder import ContextBuilder
from backend.app.rag.models import (
    RAGError,
    RAGQuery,
    RAGResponse,
    RAGValidationError,
)
from backend.app.rag.prompts import PromptBuilder
from backend.app.retrieval.models import RetrievalError, RetrievalValidationError
from backend.app.retrieval.service import RetrievalService

logger = logging.getLogger("app.rag.service")


class RAGService:
    """Production RAG service coordinating retrieval and grounded LLM answer generation."""

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        config: Optional[RAGConfig] = None,
        context_builder: Optional[ContextBuilder] = None,
        prompt_builder: Optional[PromptBuilder] = None,
    ):
        self.config = config or RAGConfig.from_settings()
        self.retrieval_service = retrieval_service or RetrievalService()
        self.llm_provider = llm_provider or GeminiProvider()
        self.context_builder = context_builder or ContextBuilder()
        self.prompt_builder = prompt_builder or PromptBuilder()

    def validate_query(self, query: str) -> str:
        """Validates and trims user query string."""
        if query is None:
            raise RAGValidationError("Query cannot be None.")
        
        trimmed = query.strip()
        if not trimmed:
            raise RAGValidationError("Query string cannot be empty or whitespace-only.")
        
        if len(trimmed) > self.config.max_query_length:
            raise RAGValidationError(
                f"Query length ({len(trimmed)}) exceeds maximum allowed length of {self.config.max_query_length} characters."
            )
        
        return trimmed

    def generate_answer(
        self,
        query: str,
        user_id: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        document_id: Optional[str] = None,
    ) -> RAGResponse:
        """Executes grounded RAG generation for an authenticated user query.
        
        Workflow:
        1. Validate query and tenant identity.
        2. Perform tenant-isolated semantic retrieval via RetrievalService.
        3. If no matching chunks are retrieved, immediately return no_context_message without calling LLM.
        4. Assemble structured context from retrieved chunks.
        5. Construct grounding prompt with prompt injection barriers.
        6. Invoke LLMProvider (Gemini) to generate grounded completion.
        7. Return structured RAGResponse.
        """
        # 1. Validation
        sanitized_query = self.validate_query(query)

        if not user_id or not user_id.strip():
            raise RAGValidationError("user_id is required for tenant-isolated RAG generation.")
        clean_user_id = user_id.strip()
        clean_doc_id = document_id.strip() if document_id and document_id.strip() else None

        # 2. Semantic Retrieval
        try:
            start_retrieval = time.perf_counter()
            retrieval_response = self.retrieval_service.search(
                query=sanitized_query,
                user_id=clean_user_id,
                top_k=top_k or self.config.top_k,
                similarity_threshold=similarity_threshold if similarity_threshold is not None else self.config.similarity_threshold,
                document_id=clean_doc_id,
            )
            retrieval_duration = time.perf_counter() - start_retrieval
            logger.info(
                f"Retrieved {len(retrieval_response.results)} chunks in {retrieval_duration:.3f}s for user '{clean_user_id}'"
            )
        except RetrievalValidationError as e:
            raise RAGValidationError(str(e)) from e
        except (RetrievalError, Exception) as e:
            sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
            logger.error(f"Retrieval step failed during RAG query: {sanitized}")
            raise RAGError(f"Retrieval failed: {sanitized}") from e

        # 3. No-Context Guard
        if not retrieval_response.results:
            logger.info(
                f"No relevant context retrieved for query '{sanitized_query[:30]}...'. Returning deterministic no-context response."
            )
            model_name = getattr(self.llm_provider, "model_name", None)
            if model_name is not None and not isinstance(model_name, str):
                model_name = str(model_name)

            return RAGResponse(
                query=sanitized_query,
                answer=self.config.no_context_message,
                retrieved_context_count=0,
                model_name=model_name,
            )

        # 4. Context Assembly
        formatted_context = self.context_builder.build_context(retrieval_response.results)

        # 5. Prompt Construction
        user_prompt = self.prompt_builder.build_user_prompt(
            query=sanitized_query,
            formatted_context=formatted_context,
        )

        # 6. LLM Generation
        try:
            start_llm = time.perf_counter()
            llm_response = self.llm_provider.generate(
                prompt=user_prompt,
                system_instruction=self.prompt_builder.system_instruction,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            llm_duration = time.perf_counter() - start_llm
            logger.info(
                f"LLM generation succeeded in {llm_duration:.3f}s (model={llm_response.model_name}, chunks={len(retrieval_response.results)})"
            )

            if not llm_response.text or not llm_response.text.strip():
                logger.warning("LLM provider returned an empty completion string.")
                return RAGResponse(
                    query=sanitized_query,
                    answer=self.config.no_context_message,
                    retrieved_context_count=len(retrieval_response.results),
                    model_name=llm_response.model_name,
                )

            return RAGResponse(
                query=sanitized_query,
                answer=llm_response.text.strip(),
                retrieved_context_count=len(retrieval_response.results),
                model_name=llm_response.model_name,
            )

        except LLMProviderError as e:
            sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
            logger.error(f"LLM generation failed: {sanitized}")
            raise RAGError(f"Answer generation failed: {sanitized}") from e
        except Exception as e:
            sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
            logger.error(f"Unexpected error during RAG generation: {sanitized}")
            raise RAGError(f"RAG generation failed: {sanitized}") from e
