"""Unit tests for RecursiveCharacterSplitter and text chunking logic."""

import pytest
from backend.app.chunking.config import ChunkingConfig
from backend.app.chunking.splitter import (
    RecursiveCharacterSplitter,
    ChunkData,
    count_tokens_approx,
)
from backend.app.ingestion.base import ExtractedPage


class TestRecursiveCharacterSplitter:
    """Test suite covering hierarchical splitting, boundaries, overlap, and edge cases."""

    def test_invalid_configuration_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            RecursiveCharacterSplitter(chunk_size=0, chunk_overlap=0)

        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            RecursiveCharacterSplitter(chunk_size=500, chunk_overlap=-1)

        with pytest.raises(ValueError, match="chunk_overlap.*strictly less than chunk_size"):
            RecursiveCharacterSplitter(chunk_size=500, chunk_overlap=500)

        with pytest.raises(ValueError, match="chunk_overlap.*strictly less than chunk_size"):
            RecursiveCharacterSplitter(chunk_size=500, chunk_overlap=600)

    def test_chunking_config_dataclass_validation(self):
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size=-10, chunk_overlap=0)

        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)

        cfg = ChunkingConfig(chunk_size=300, chunk_overlap=30)
        assert cfg.chunk_size == 300
        assert cfg.chunk_overlap == 30

    def test_token_approximation(self):
        assert count_tokens_approx("") == 0
        assert count_tokens_approx("   ") == 0
        assert count_tokens_approx(None) == 0
        assert count_tokens_approx("Hello world this is a test") == 6
        assert count_tokens_approx("Line one\nLine two\tLine three") == 6

    def test_empty_and_whitespace_text(self):
        splitter = RecursiveCharacterSplitter(chunk_size=500, chunk_overlap=50)
        assert splitter.split_text("") == []
        assert splitter.split_text("   \n\t   \n  ") == []

    def test_short_text_single_chunk(self):
        splitter = RecursiveCharacterSplitter(chunk_size=500, chunk_overlap=50)
        text = "This is a short single sentence document."
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_exact_chunk_size_text(self):
        splitter = RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=10)
        text = "a" * 50
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert len(chunks[0]) == 50

    def test_long_paragraph_splitting(self):
        splitter = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=20)
        # Create a single long sentence without spaces
        long_word = "a" * 250
        chunks = splitter.split_text(long_word)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_paragraph_boundary_preservation(self):
        splitter = RecursiveCharacterSplitter(chunk_size=120, chunk_overlap=20)
        p1 = "First paragraph containing important topic details about information retrieval."
        p2 = "Second paragraph discussing vector databases and dense index structures."
        p3 = "Third paragraph explaining reranking models and cross-encoders."
        text = f"{p1}\n\n{p2}\n\n{p3}"
        
        chunks = splitter.split_text(text)
        assert len(chunks) >= 3
        assert p1 in chunks[0]
        assert p2 in chunks[1]
        assert p3 in chunks[2]

    def test_sentence_boundary_splitting(self):
        splitter = RecursiveCharacterSplitter(chunk_size=80, chunk_overlap=15)
        s1 = "This is the first sentence."
        s2 = "Here is the second sentence."
        s3 = "And this is the third sentence."
        text = f"{s1} {s2} {s3}"

        chunks = splitter.split_text(text)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 80

    def test_overlap_behavior(self):
        splitter = RecursiveCharacterSplitter(chunk_size=60, chunk_overlap=20)
        p1 = "Alpha beta gamma delta epsilon zeta eta theta."
        p2 = "Iota kappa lambda mu nu xi omicron pi rho sigma."
        text = f"{p1}\n\n{p2}"

        chunks = splitter.split_text(text)
        assert len(chunks) >= 2
        # Check that adjacent chunks exist and have non-empty text
        for chunk in chunks:
            assert len(chunk) <= 60

    def test_deterministic_output(self):
        splitter = RecursiveCharacterSplitter(chunk_size=150, chunk_overlap=30)
        sample = (
            "Advanced Retrieval-Augmented Generation enhances generative AI by grounding "
            "model responses in verified documents. Hierarchical chunking breaks documents "
            "into structured pieces while maintaining context through strategic overlap."
        )
        run_1 = splitter.split_text(sample)
        run_2 = splitter.split_text(sample)
        run_3 = splitter.split_text(sample)
        assert run_1 == run_2 == run_3

    def test_split_document_pages_multi_page_pdf(self):
        splitter = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=20)
        pages = [
            ExtractedPage(page_number=1, text="Page 1: Overview of modern neural search architectures."),
            ExtractedPage(page_number=2, text="Page 2: Vector databases like Qdrant provide fast HNSW indexing."),
            ExtractedPage(page_number=3, text="   "),  # Empty whitespace page
            ExtractedPage(page_number=4, text="Page 4: Re-ranking models reorder the top candidates."),
        ]

        chunks: list[ChunkData] = splitter.split_document_pages(pages)
        assert len(chunks) == 3

        # Chunk 0
        assert chunks[0].chunk_index == 0
        assert chunks[0].page_number == 1
        assert "Page 1" in chunks[0].text
        assert chunks[0].token_count > 0

        # Chunk 1
        assert chunks[1].chunk_index == 1
        assert chunks[1].page_number == 2
        assert "Page 2" in chunks[1].text

        # Chunk 2 (Page 3 was empty, so chunk 2 corresponds to page 4)
        assert chunks[2].chunk_index == 2
        assert chunks[2].page_number == 4
        assert "Page 4" in chunks[2].text

    def test_split_document_pages_unpaged_text_docx(self):
        splitter = RecursiveCharacterSplitter(chunk_size=100, chunk_overlap=20)
        pages = [
            ExtractedPage(page_number=None, text="Unpaged text file content split across multiple chunks.")
        ]
        chunks = splitter.split_document_pages(pages)
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].page_number is None
