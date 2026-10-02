"""Grounded RAG generation service orchestrating conversational query rewriting, dense retrieval, reranking, citations, and LLM synthesis."""

import logging
import time
from typing import List, Optional

from backend.app.citations.models import Citation
from backend.app.citations.parser import CitationParser
from backend.app.hybrid_retrieval.base import HybridRetrievalError
from backend.app.hybrid_retrieval.service import HybridRetrievalService
from backend.app.llm.base import BaseLLMProvider, LLMProviderError
from backend.app.llm.gemini import GeminiProvider
from backend.app.query_rewriting.service import QueryRewritingService
from backend.app.rag.config import RAGConfig
from backend.app.rag.context_builder import ContextBuilder
from backend.app.rag.models import (
    RAGError,
    RAGQuery,
    RAGResponse,
    RAGValidationError,
)
from backend.app.rag.prompts import PromptBuilder
from backend.app.reranking.service import RerankingService
from backend.app.retrieval.models import RetrievalError, RetrievalValidationError
from backend.app.retrieval.service import RetrievalService
from backend.app.services.conversation_service import (
    ConversationNotFoundError,
    ConversationService,
)

logger = logging.getLogger("app.rag.service")


class RAGService:
    """Production RAG service coordinating query rewriting, hybrid retrieval, reranking, memory, citations, and LLM synthesis."""

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        config: Optional[RAGConfig] = None,
        context_builder: Optional[ContextBuilder] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        citation_parser: Optional[CitationParser] = None,
        conversation_service: Optional[ConversationService] = None,
        query_rewriting_service: Optional[QueryRewritingService] = None,
        reranking_service: Optional[RerankingService] = None,
        hybrid_retrieval_service: Optional[HybridRetrievalService] = None,
    ):
        self.config = config or RAGConfig.from_settings()
        self.retrieval_service = retrieval_service or RetrievalService()
        self.llm_provider = llm_provider or GeminiProvider()
        self.context_builder = context_builder or ContextBuilder()
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.citation_parser = citation_parser or CitationParser()
        self.conversation_service = conversation_service
        self.query_rewriting_service = query_rewriting_service or QueryRewritingService()
        self.reranking_service = reranking_service or RerankingService()
        self.hybrid_retrieval_service = hybrid_retrieval_service




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
        conversation_id: Optional[str] = None,
    ) -> RAGResponse:
        """Executes grounded RAG generation with query rewriting, dense retrieval, reranking, citations, and conversation history.
        
        Pipeline:
        1. Validate original user query and tenant identity.
        2. If conversation_id is supplied, validate ownership, load bounded history, and record original user turn.
        3. Perform conversational query rewriting (if history present) to generate standalone retrieval_query.
        4. Perform tenant-isolated dense semantic retrieval in Qdrant using retrieval_query with retrieval_top_k candidate pool.
        5. If no matching chunks are retrieved, return deterministic no-context response (0 LLM cost).
        6. Rerank retrieved candidate chunks against retrieval_query to select final top_k chunks.
        7. Assemble structured context with deterministic source labels ([Source 1], [Source 2]).
        8. Construct grounded prompt with conversation history and anti-injection demarcations, supplying original_query as USER QUESTION.
        9. Invoke LLMProvider (Gemini) to generate grounded completion with citations.
        10. Parse and verify citations against authoritative retrieved results.
        11. If conversation_id is supplied, record assistant turn with verified citations.
        12. Return structured RAGResponse.
        """
        # 1. Validation
        sanitized_query = self.validate_query(query)

        if not user_id or not user_id.strip():
            raise RAGValidationError("user_id is required for tenant-isolated RAG generation.")
        clean_user_id = user_id.strip()
        clean_doc_id = document_id.strip() if document_id and document_id.strip() else None
        clean_conv_id = conversation_id.strip() if conversation_id and conversation_id.strip() else None

        total_start = time.perf_counter()
        latencies: dict = {}

        # 2. Conversation Memory Integration (if conversation_id provided)
        t0 = time.perf_counter()
        conversation_history_text: Optional[str] = None
        if clean_conv_id:
            if not self.conversation_service:
                raise RAGError("Conversation service is not configured for conversation-linked RAG queries.")
            try:
                # Load prior history before saving current turn
                conversation_history_text = self.conversation_service.get_bounded_history_for_prompt(
                    conversation_id=clean_conv_id,
                    user_id=clean_user_id,
                    max_messages=self.config.max_history_messages,
                )
                # Record the incoming user query (original question preserved)
                self.conversation_service.record_user_message(
                    conversation_id=clean_conv_id,
                    user_id=clean_user_id,
                    content=sanitized_query,
                )
            except ConversationNotFoundError as e:
                raise RAGValidationError(str(e)) from e
        latencies["memory_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # 3. Conversational Query Rewriting
        t0 = time.perf_counter()
        rewritten_info = self.query_rewriting_service.rewrite_query(
            query=sanitized_query,
            conversation_history=conversation_history_text,
        )
        retrieval_query = rewritten_info.retrieval_query or sanitized_query
        latencies["query_rewriting_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # 4. Candidate Retrieval (Hybrid Dense + Lexical Fusion or Dense Only)
        dense_top_k = self.config.retrieval_top_k
        if top_k is not None:
            dense_top_k = max(dense_top_k, top_k * 4)

        try:
            start_retrieval = time.perf_counter()
            if self.config.enable_hybrid_retrieval and self.hybrid_retrieval_service:
                retrieval_response = self.hybrid_retrieval_service.search(
                    query=retrieval_query,
                    user_id=clean_user_id,
                    top_k=dense_top_k,
                    similarity_threshold=similarity_threshold if similarity_threshold is not None else self.config.similarity_threshold,
                    document_id=clean_doc_id,
                )
            else:
                retrieval_response = self.retrieval_service.search(
                    query=retrieval_query,
                    user_id=clean_user_id,
                    top_k=dense_top_k,
                    similarity_threshold=similarity_threshold if similarity_threshold is not None else self.config.similarity_threshold,
                    document_id=clean_doc_id,
                )
            retrieval_duration = time.perf_counter() - start_retrieval
            latencies["retrieval_ms"] = round(retrieval_duration * 1000, 2)
            logger.info(
                f"Retrieved {len(retrieval_response.results)} candidates in {retrieval_duration:.3f}s for user '{clean_user_id}' (query: '{retrieval_query[:40]}...')"
            )
        except (RetrievalValidationError, RAGValidationError) as e:
            raise RAGValidationError(str(e)) from e
        except (RetrievalError, HybridRetrievalError, Exception) as e:
            sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
            logger.error(f"Retrieval step failed during RAG query: {sanitized}")
            raise RAGError(f"Retrieval failed: {sanitized}") from e

        # 5. No-Context Guard
        if not retrieval_response.results:
            logger.info(
                f"No relevant context retrieved for query '{sanitized_query[:30]}...'. Returning deterministic no-context response."
            )
            model_name = getattr(self.llm_provider, "model_name", None)
            if model_name is not None and not isinstance(model_name, str):
                model_name = str(model_name)

            assistant_msg_id: Optional[str] = None
            if clean_conv_id and self.conversation_service:
                assistant_msg = self.conversation_service.record_assistant_message(
                    conversation_id=clean_conv_id,
                    user_id=clean_user_id,
                    content=self.config.no_context_message,
                    citations=[],
                )
                assistant_msg_id = assistant_msg.id

            latencies["total_rag_ms"] = round((time.perf_counter() - total_start) * 1000, 2)
            return RAGResponse(
                query=sanitized_query,
                answer=self.config.no_context_message,
                retrieved_context_count=0,
                citations=[],
                conversation_id=clean_conv_id,
                message_id=assistant_msg_id,
                model_name=model_name,
                rewritten_query=retrieval_query if rewritten_info.was_rewritten else None,
                latency_metrics=latencies,
            )

        # 6. Candidate Reranking
        t0 = time.perf_counter()
        final_top_k = top_k if top_k is not None else self.config.final_top_k
        reranked_results = self.reranking_service.rerank(
            query=retrieval_query,
            candidates=retrieval_response.results,
            top_k=final_top_k,
        )

        if not reranked_results:
            # Safe fallback if reranked results list is somehow empty
            reranked_results = [
                c for c in retrieval_response.results[:final_top_k]
            ]
        latencies["reranking_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # 7. Context Assembly
        t0 = time.perf_counter()
        formatted_context = self.context_builder.build_context(reranked_results)
        latencies["context_building_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # 8. Prompt Construction (User's original question is answered)
        user_prompt = self.prompt_builder.build_user_prompt(
            query=sanitized_query,
            formatted_context=formatted_context,
            conversation_history=conversation_history_text,
        )

        # 9. LLM Generation
        try:
            start_llm = time.perf_counter()
            llm_response = self.llm_provider.generate(
                prompt=user_prompt,
                system_instruction=self.prompt_builder.system_instruction,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            llm_duration = time.perf_counter() - start_llm
            latencies["generation_ms"] = round(llm_duration * 1000, 2)
            logger.info(
                f"LLM generation succeeded in {llm_duration:.3f}s (model={llm_response.model_name}, chunks={len(reranked_results)})"
            )

            answer_text = llm_response.text.strip() if llm_response.text else ""
            if not answer_text:
                logger.warning("LLM provider returned an empty completion string.")
                answer_text = self.config.no_context_message

            # 10. Citation Extraction & Verification
            t0 = time.perf_counter()
            citations: List[Citation] = []
            if self.config.enable_citations and answer_text != self.config.no_context_message:
                citations = self.citation_parser.parse_and_verify(
                    answer_text=answer_text,
                    retrieved_results=reranked_results,
                )
            latencies["citations_ms"] = round((time.perf_counter() - t0) * 1000, 2)

            # 11. Record Assistant Message in Conversation History
            assistant_msg_id: Optional[str] = None
            if clean_conv_id and self.conversation_service:
                citations_payload = [c.model_dump() for c in citations] if citations else []
                token_metrics = {
                    "prompt_tokens": llm_response.prompt_tokens,
                    "completion_tokens": llm_response.completion_tokens,
                }
                assistant_msg = self.conversation_service.record_assistant_message(
                    conversation_id=clean_conv_id,
                    user_id=clean_user_id,
                    content=answer_text,
                    citations=citations_payload,
                    token_metrics=token_metrics,
                )
                assistant_msg_id = assistant_msg.id

            latencies["total_rag_ms"] = round((time.perf_counter() - total_start) * 1000, 2)
            return RAGResponse(
                query=sanitized_query,
                answer=answer_text,
                retrieved_context_count=len(reranked_results),
                citations=citations,
                conversation_id=clean_conv_id,
                message_id=assistant_msg_id,
                model_name=llm_response.model_name,
                rewritten_query=retrieval_query if rewritten_info.was_rewritten else None,
                latency_metrics=latencies,
            )

        except LLMProviderError as e:
            sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
            logger.error(f"LLM generation failed: {sanitized}")
            raise RAGError(f"Answer generation failed: {sanitized}") from e
        except Exception as e:
            sanitized = str(e).replace("\\", "/").split(":")[-1].strip()
            logger.error(f"Unexpected error during RAG generation: {sanitized}")
            raise RAGError(f"RAG generation failed: {sanitized}") from e
