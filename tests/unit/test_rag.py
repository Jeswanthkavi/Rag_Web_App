"""Unit tests for Grounded RAG configuration, context builder, prompt builder, and RAGService."""

import pytest
from unittest.mock import MagicMock, patch
from typing import List

from backend.app.llm.base import LLMProviderError, LLMResponse
from backend.app.llm.fake_provider import FakeLLMProvider
from backend.app.rag.config import RAGConfig, DEFAULT_NO_CONTEXT_MESSAGE
from backend.app.rag.context_builder import ContextBuilder
from backend.app.rag.models import (
    RAGError,
    RAGQuery,
    RAGResponse,
    RAGValidationError,
)
from backend.app.rag.prompts import PromptBuilder, GROUNDED_SYSTEM_INSTRUCTION
from backend.app.rag.service import RAGService
from backend.app.retrieval.models import (
    RetrievalError,
    RetrievalResponse,
    RetrievalResult,
    RetrievalValidationError,
)


class TestRAGConfig:
    """Tests for RAGConfig validation and defaults."""

    def test_default_config_valid(self):
        cfg = RAGConfig()
        assert cfg.top_k == 5
        assert cfg.similarity_threshold == 0.5
        assert cfg.temperature == 0.2
        assert cfg.max_tokens == 2048
        assert cfg.max_query_length == 1000
        assert cfg.no_context_message == DEFAULT_NO_CONTEXT_MESSAGE

    def test_invalid_top_k_raises(self):
        with pytest.raises(ValueError, match="top_k must be between 1 and 100"):
            RAGConfig(top_k=0)
        with pytest.raises(ValueError, match="top_k must be between 1 and 100"):
            RAGConfig(top_k=101)

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            RAGConfig(temperature=-0.1)
        with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
            RAGConfig(temperature=2.5)

    def test_invalid_max_tokens_raises(self):
        with pytest.raises(ValueError, match="max_tokens must be between 64 and 8192"):
            RAGConfig(max_tokens=32)
        with pytest.raises(ValueError, match="max_tokens must be between 64 and 8192"):
            RAGConfig(max_tokens=10000)

    def test_invalid_max_query_length_raises(self):
        with pytest.raises(ValueError, match="max_query_length must be between 10 and 10000"):
            RAGConfig(max_query_length=5)


class TestRAGQueryModel:
    """Tests for RAGQuery request schema validation."""

    def test_valid_rag_query(self):
        q = RAGQuery(query="What is neural search?", top_k=3, similarity_threshold=0.6)
        assert q.query == "What is neural search?"
        assert q.top_k == 3
        assert q.similarity_threshold == 0.6
        assert q.document_id is None

    def test_empty_query_raises_validation_error(self):
        with pytest.raises(ValueError):
            RAGQuery(query="")

    def test_invalid_top_k_bounds(self):
        with pytest.raises(ValueError):
            RAGQuery(query="Valid query", top_k=0)
        with pytest.raises(ValueError):
            RAGQuery(query="Valid query", top_k=51)

    def test_invalid_threshold_bounds(self):
        with pytest.raises(ValueError):
            RAGQuery(query="Valid query", similarity_threshold=-0.1)
        with pytest.raises(ValueError):
            RAGQuery(query="Valid query", similarity_threshold=1.1)


class TestContextBuilder:
    """Tests for ContextBuilder context formatting and metadata preservation."""

    def test_empty_results_returns_empty_string(self):
        builder = ContextBuilder()
        assert builder.build_context([]) == ""

    def test_paged_and_unpaged_chunks_formatting(self):
        builder = ContextBuilder()
        results = [
            RetrievalResult(
                chunk_id="c1-uuid",
                document_id="doc-1",
                user_id="u1",
                chunk_index=0,
                chunk_text="First chunk from page 2.",
                page_number=2,
                filename="document.pdf",
                file_type=".pdf",
                token_count=10,
                score=0.92,
                rank=1,
            ),
            RetrievalResult(
                chunk_id="c2-uuid",
                document_id="doc-2",
                user_id="u1",
                chunk_index=0,
                chunk_text="Second chunk from unpaged text.",
                page_number=None,
                filename="notes.txt",
                file_type=".txt",
                token_count=8,
                score=0.85,
                rank=2,
            ),
        ]

        context = builder.build_context(results)
        assert "[Source 1]" in context
        assert "Document: document.pdf (Page 2)" in context
        assert "Chunk ID: c1-uuid" in context
        assert "First chunk from page 2." in context

        assert "[Source 2]" in context
        assert "Document: notes.txt" in context
        assert "(Page" not in context.split("[Source 2]")[1]
        assert "Chunk ID: c2-uuid" in context
        assert "Second chunk from unpaged text." in context


class TestPromptBuilder:
    """Tests for PromptBuilder system instructions, citations, and conversation history."""

    def test_system_instruction_contains_grounding_and_citation_rules(self):
        builder = PromptBuilder()
        assert "ONLY the factual information present in the RETRIEVED CONTEXT" in builder.system_instruction
        assert "CITATION REQUIREMENT" in builder.system_instruction
        assert "UNTRUSTED CONTENT DEFENSE" in builder.system_instruction
        assert "CONVERSATION HISTORY" in builder.system_instruction
        assert "I do not have enough information" in builder.system_instruction

    def test_user_prompt_demarcations_without_history(self):
        builder = PromptBuilder()
        prompt = builder.build_user_prompt(
            query="What is the refund deadline?",
            formatted_context="[Source 1]\nDocument: policy.pdf\nContent: Refunds are 30 days.",
        )
        assert "=== RETRIEVED CONTEXT ===" in prompt
        assert "Refunds are 30 days." in prompt
        assert "=== END OF CONTEXT ===" in prompt
        assert "=== USER QUESTION ===" in prompt
        assert "What is the refund deadline?" in prompt
        assert "=== CONVERSATION HISTORY" not in prompt

    def test_user_prompt_demarcations_with_history(self):
        builder = PromptBuilder()
        prompt = builder.build_user_prompt(
            query="What about shipping fees?",
            formatted_context="[Source 1]\nDocument: policy.pdf\nContent: Shipping is free for returns.",
            conversation_history="User: What is the refund deadline?\nAssistant: 30 days [Source 1].",
        )
        assert "=== CONVERSATION HISTORY" in prompt
        assert "User: What is the refund deadline?" in prompt
        assert "Assistant: 30 days [Source 1]." in prompt
        assert "=== END OF CONVERSATION HISTORY ===" in prompt
        assert "=== RETRIEVED CONTEXT ===" in prompt
        assert "=== USER QUESTION ===" in prompt
        assert "What about shipping fees?" in prompt


class TestRAGServiceUnit:
    """Unit tests for RAGService orchestration, query validation, no-context guard, and error handling."""

    def test_validate_query_trims_whitespace(self):
        service = RAGService(
            retrieval_service=MagicMock(),
            llm_provider=FakeLLMProvider(),
        )
        res = service.validate_query("   What is grounded RAG?   \n")
        assert res == "What is grounded RAG?"

    def test_validate_query_empty_or_whitespace_raises(self):
        service = RAGService(
            retrieval_service=MagicMock(),
            llm_provider=FakeLLMProvider(),
        )
        with pytest.raises(RAGValidationError, match="cannot be empty or whitespace-only"):
            service.validate_query("")

        with pytest.raises(RAGValidationError, match="cannot be empty or whitespace-only"):
            service.validate_query("   \t  ")

        with pytest.raises(RAGValidationError, match="cannot be None"):
            service.validate_query(None)

    def test_validate_query_length_limit(self):
        service = RAGService(
            retrieval_service=MagicMock(),
            llm_provider=FakeLLMProvider(),
            config=RAGConfig(max_query_length=30),
        )
        with pytest.raises(RAGValidationError, match="exceeds maximum allowed length"):
            service.validate_query("A" * 31)

    def test_missing_user_id_raises(self):
        service = RAGService(
            retrieval_service=MagicMock(),
            llm_provider=FakeLLMProvider(),
        )
        with pytest.raises(RAGValidationError, match="user_id is required"):
            service.generate_answer(query="Valid query", user_id="")

    def test_no_context_guard_returns_message_without_calling_llm(self):
        mock_retrieval = MagicMock()
        mock_retrieval.search.return_value = RetrievalResponse(
            query="unknown topic",
            results=[],
            total_retrieved=0,
            top_k=5,
            similarity_threshold=0.5,
        )

        mock_llm = MagicMock()

        service = RAGService(
            retrieval_service=mock_retrieval,
            llm_provider=mock_llm,
        )

        resp: RAGResponse = service.generate_answer(query="unknown topic", user_id="user-123")

        # Verify LLM was NOT called
        mock_llm.generate.assert_not_called()

        assert resp.query == "unknown topic"
        assert resp.answer == DEFAULT_NO_CONTEXT_MESSAGE
        assert resp.retrieved_context_count == 0

    def test_successful_grounded_generation(self):
        mock_retrieval = MagicMock()
        mock_retrieval.search.return_value = RetrievalResponse(
            query="What is the refund period?",
            results=[
                RetrievalResult(
                    chunk_id="c1",
                    document_id="d1",
                    user_id="user-1",
                    chunk_index=0,
                    chunk_text="All customers have 30 calendar days to request a full refund.",
                    page_number=1,
                    filename="refund_policy.pdf",
                    file_type=".pdf",
                    token_count=15,
                    score=0.91,
                    rank=1,
                )
            ],
            total_retrieved=1,
            top_k=5,
            similarity_threshold=0.5,
        )

        fake_llm = FakeLLMProvider(
            model_name="gemini-2.5-flash",
            custom_response="The refund period is 30 calendar days.",
        )

        service = RAGService(
            retrieval_service=mock_retrieval,
            llm_provider=fake_llm,
        )

        resp = service.generate_answer(
            query="What is the refund period?",
            user_id="user-1",
            top_k=3,
            similarity_threshold=0.6,
            document_id="d1",
        )

        # Verify retrieval parameters passed correctly
        mock_retrieval.search.assert_called_once_with(
            query="What is the refund period?",
            user_id="user-1",
            top_k=3,
            similarity_threshold=0.6,
            document_id="d1",
        )

        # Verify LLM prompt and response
        assert fake_llm.last_prompt is not None
        assert "All customers have 30 calendar days" in fake_llm.last_prompt
        assert "What is the refund period?" in fake_llm.last_prompt
        assert fake_llm.last_system_instruction is not None
        assert "ONLY the factual information" in fake_llm.last_system_instruction

        assert resp.answer == "The refund period is 30 calendar days."
        assert resp.retrieved_context_count == 1
        assert resp.model_name == "gemini-2.5-flash"

    def test_empty_llm_response_handled_safely(self):
        mock_retrieval = MagicMock()
        mock_retrieval.search.return_value = RetrievalResponse(
            query="test query",
            results=[
                RetrievalResult(
                    chunk_id="c1",
                    document_id="d1",
                    user_id="user-1",
                    chunk_index=0,
                    chunk_text="Some relevant text.",
                    filename="doc.txt",
                    file_type=".txt",
                    score=0.8,
                    rank=1,
                )
            ],
            total_retrieved=1,
            top_k=5,
            similarity_threshold=0.5,
        )

        mock_llm = MagicMock()
        mock_llm.model_name = "test-model"
        mock_llm.generate.return_value = LLMResponse(text="   ", model_name="test-model")

        service = RAGService(
            retrieval_service=mock_retrieval,
            llm_provider=mock_llm,
        )

        resp = service.generate_answer(query="test query", user_id="user-1")
        assert resp.answer == DEFAULT_NO_CONTEXT_MESSAGE
        assert resp.retrieved_context_count == 1

    def test_retrieval_failure_raises_rag_error(self):
        mock_retrieval = MagicMock()
        mock_retrieval.search.side_effect = RetrievalError("Qdrant unavailable")

        service = RAGService(
            retrieval_service=mock_retrieval,
            llm_provider=FakeLLMProvider(),
        )

        with pytest.raises(RAGError, match="Retrieval failed"):
            service.generate_answer(query="test", user_id="user-1")

    def test_llm_failure_raises_rag_error(self):
        mock_retrieval = MagicMock()
        mock_retrieval.search.return_value = RetrievalResponse(
            query="test",
            results=[
                RetrievalResult(
                    chunk_id="c1",
                    document_id="d1",
                    user_id="user-1",
                    chunk_index=0,
                    chunk_text="Sample text.",
                    filename="doc.txt",
                    file_type=".txt",
                    score=0.8,
                    rank=1,
                )
            ],
            total_retrieved=1,
            top_k=5,
            similarity_threshold=0.5,
        )

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = LLMProviderError("Gemini quota exceeded")

        service = RAGService(
            retrieval_service=mock_retrieval,
            llm_provider=mock_llm,
        )

        with pytest.raises(RAGError, match="Answer generation failed"):
            service.generate_answer(query="test", user_id="user-1")

    def test_rag_generation_extracts_and_verifies_citations(self):
        mock_retrieval = MagicMock()
        mock_retrieval.search.return_value = RetrievalResponse(
            query="What is solar efficiency?",
            results=[
                RetrievalResult(
                    chunk_id="chunk-101",
                    document_id="doc-202",
                    user_id="user-1",
                    chunk_index=0,
                    chunk_text="Solar cells achieve 22% conversion efficiency.",
                    page_number=5,
                    filename="specs.pdf",
                    file_type=".pdf",
                    token_count=10,
                    score=0.92,
                    rank=1,
                )
            ],
            total_retrieved=1,
            top_k=5,
            similarity_threshold=0.5,
        )

        fake_llm = FakeLLMProvider(
            custom_response="Solar cells achieve 22% conversion efficiency [Source 1]."
        )

        service = RAGService(
            retrieval_service=mock_retrieval,
            llm_provider=fake_llm,
        )

        resp = service.generate_answer(query="What is solar efficiency?", user_id="user-1")

        assert len(resp.citations) == 1
        cit = resp.citations[0]
        assert cit.citation_id == "source_1"
        assert cit.document_id == "doc-202"
        assert cit.chunk_id == "chunk-101"
        assert cit.filename == "specs.pdf"
        assert cit.page_number == 5
        assert cit.chunk_index == 0

    def test_rag_generation_with_conversation_memory(self):
        mock_retrieval = MagicMock()
        mock_retrieval.search.return_value = RetrievalResponse(
            query="What about battery degradation?",
            results=[
                RetrievalResult(
                    chunk_id="chunk-b",
                    document_id="doc-b",
                    user_id="user-1",
                    chunk_index=2,
                    chunk_text="Battery capacity degrades by 2% per year under normal conditions.",
                    page_number=1,
                    filename="battery.pdf",
                    file_type=".pdf",
                    token_count=12,
                    score=0.90,
                    rank=1,
                )
            ],
            total_retrieved=1,
            top_k=5,
            similarity_threshold=0.5,
        )

        fake_llm = FakeLLMProvider(
            custom_response="Battery capacity degrades by 2% annually [Source 1]."
        )

        mock_conv_service = MagicMock()
        mock_conv_service.get_bounded_history_for_prompt.return_value = (
            "User: What is the warranty?\nAssistant: 10 years."
        )
        mock_assistant_msg = MagicMock()
        mock_assistant_msg.id = "msg-assistant-123"
        mock_conv_service.record_assistant_message.return_value = mock_assistant_msg

        service = RAGService(
            retrieval_service=mock_retrieval,
            llm_provider=fake_llm,
            conversation_service=mock_conv_service,
        )

        resp = service.generate_answer(
            query="What about battery degradation?",
            user_id="user-1",
            conversation_id="conv-123",
        )

        # Verify user message recorded
        mock_conv_service.record_user_message.assert_called_once_with(
            conversation_id="conv-123",
            user_id="user-1",
            content="What about battery degradation?",
        )

        # Verify assistant message recorded with citations
        mock_conv_service.record_assistant_message.assert_called_once()

        # Verify prompt had conversation history
        assert "=== CONVERSATION HISTORY" in fake_llm.last_prompt
        assert "User: What is the warranty?" in fake_llm.last_prompt

        # Verify response metadata
        assert resp.conversation_id == "conv-123"
        assert resp.message_id == "msg-assistant-123"
        assert len(resp.citations) == 1
