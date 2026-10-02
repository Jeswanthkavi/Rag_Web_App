"""Unit tests for Citation domain models, parser, and verification."""

import pytest
from backend.app.citations.models import Citation
from backend.app.citations.parser import CitationParser
from backend.app.retrieval.models import RetrievalResult


@pytest.fixture
def sample_retrieval_results():
    return [
        RetrievalResult(
            chunk_id="chunk-1-uuid",
            document_id="doc-1-uuid",
            user_id="user-123",
            chunk_index=0,
            chunk_text="The solar panel efficiency is 22 percent under standard test conditions.",
            page_number=3,
            filename="solar_specs.pdf",
            file_type=".pdf",
            token_count=12,
            score=0.95,
            rank=1,
        ),
        RetrievalResult(
            chunk_id="chunk-2-uuid",
            document_id="doc-2-uuid",
            user_id="user-123",
            chunk_index=4,
            chunk_text="Battery warranty covers replacement for 10 years or 10000 cycles.",
            page_number=None,
            filename="warranty_guide.docx",
            file_type=".docx",
            token_count=10,
            score=0.88,
            rank=2,
        ),
        RetrievalResult(
            chunk_id="chunk-3-uuid",
            document_id="doc-1-uuid",
            user_id="user-123",
            chunk_index=1,
            chunk_text="Operating temperature range is -20C to 60C.",
            page_number=4,
            filename="solar_specs.pdf",
            file_type=".pdf",
            token_count=8,
            score=0.82,
            rank=3,
        ),
    ]


class TestCitationParser:
    """Tests for citation extraction, validation, and metadata mapping."""

    def test_extract_source_indices_standard_format(self):
        parser = CitationParser()
        text = "Efficiency is 22% [Source 1] and warranty is 10 years [Source 2]."
        indices = parser.extract_source_indices(text)
        assert indices == [1, 2]

    def test_extract_source_indices_bracket_number_format(self):
        parser = CitationParser()
        text = "Operating range is -20C to 60C [3] with 22% efficiency [1]."
        indices = parser.extract_source_indices(text)
        assert indices == [3, 1]

    def test_extract_source_indices_case_insensitive_and_colon(self):
        parser = CitationParser()
        text = "Fact A [source: 1] and Fact B [SOURCE 2]."
        indices = parser.extract_source_indices(text)
        assert indices == [1, 2]

    def test_extract_source_indices_deduplication(self):
        parser = CitationParser()
        text = "Claim 1 [Source 1], Claim 2 [Source 2], Claim 3 [Source 1]."
        indices = parser.extract_source_indices(text)
        assert indices == [1, 2]

    def test_extract_source_indices_empty_or_no_citations(self):
        parser = CitationParser()
        assert parser.extract_source_indices("") == []
        assert parser.extract_source_indices("Just a plain answer with no markers.") == []

    def test_parse_and_verify_valid_citations(self, sample_retrieval_results):
        parser = CitationParser()
        answer = "Solar panels achieve 22% efficiency [Source 1] with a 10-year warranty [Source 2]."
        citations = parser.parse_and_verify(answer, sample_retrieval_results)

        assert len(citations) == 2
        
        # Verify first citation mapped to result 0
        c1 = citations[0]
        assert c1.citation_id == "source_1"
        assert c1.document_id == "doc-1-uuid"
        assert c1.chunk_id == "chunk-1-uuid"
        assert c1.filename == "solar_specs.pdf"
        assert c1.page_number == 3
        assert c1.chunk_index == 0

        # Verify second citation mapped to result 1
        c2 = citations[1]
        assert c2.citation_id == "source_2"
        assert c2.document_id == "doc-2-uuid"
        assert c2.chunk_id == "chunk-2-uuid"
        assert c2.filename == "warranty_guide.docx"
        assert c2.page_number is None
        assert c2.chunk_index == 4

    def test_parse_and_verify_rejects_hallucinated_out_of_bounds_citations(self, sample_retrieval_results):
        parser = CitationParser()
        # Source 99 and Source 4 do not exist in sample_retrieval_results (len=3)
        answer = "Claim from [Source 1], hallucinated [Source 99], and out of range [Source 4]."
        citations = parser.parse_and_verify(answer, sample_retrieval_results)

        assert len(citations) == 1
        assert citations[0].citation_id == "source_1"
        assert citations[0].document_id == "doc-1-uuid"

    def test_parse_and_verify_rejects_zero_or_negative_indices(self, sample_retrieval_results):
        parser = CitationParser()
        answer = "Zero source [Source 0] and valid source [Source 3]."
        citations = parser.parse_and_verify(answer, sample_retrieval_results)

        assert len(citations) == 1
        assert citations[0].citation_id == "source_3"
        assert citations[0].chunk_id == "chunk-3-uuid"

    def test_parse_and_verify_empty_results_returns_empty_list(self):
        parser = CitationParser()
        citations = parser.parse_and_verify("Claim [Source 1]", [])
        assert citations == []

    def test_citation_metadata_derived_strictly_from_retrieval(self, sample_retrieval_results):
        parser = CitationParser()
        # Even if the LLM hallucinated different text or page in the answer body,
        # the citation metadata object is built purely from the authoritative RetrievalResult
        answer = "The fake manual on page 999 says something [Source 1]."
        citations = parser.parse_and_verify(answer, sample_retrieval_results)

        assert len(citations) == 1
        assert citations[0].filename == "solar_specs.pdf"
        assert citations[0].page_number == 3
        assert citations[0].document_id == "doc-1-uuid"
