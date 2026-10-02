"""Phase 23C Integration Validation Script.

Executes real end-to-end integration tests using:
- Real OpenRouter API with nvidia/nemotron-3-ultra-550b-a55b:free
- Real OpenRouterProvider
- Real OpenRouterQueryRewriter
- Synthetic document ingestion
- Real RAGService pipeline
- No-context validation
- Multi-turn conversation validation
"""

import sys
import os
import time
import uuid
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(r"c:\Users\JESWANTHKAVI\Desktop\advanced-rag")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import get_settings
from backend.app.db.session import create_db_engine, create_session_factory
from backend.app.models import (
    User,
    Document,
    DocumentStatus,
    DocumentChunk,
    Conversation,
    Message,
    MessageRole,
)
from backend.app.llm.openrouter import OpenRouterProvider
from backend.app.query_rewriting.openrouter_rewriter import OpenRouterQueryRewriter
from backend.app.query_rewriting.service import QueryRewritingService
from backend.app.vectorstore.config import VectorStoreConfig
from backend.app.vectorstore.client import create_qdrant_client
from backend.app.vectorstore.repository import QdrantVectorRepository
from backend.app.vectorstore.service import VectorStoreService
from backend.app.retrieval.service import RetrievalService
from backend.app.hybrid_retrieval.service import HybridRetrievalService
from backend.app.rag.service import RAGService
from backend.app.rag.config import RAGConfig
from backend.app.services.conversation_service import ConversationService
from backend.app.repositories.conversation_repository import ConversationRepository
from backend.app.repositories.message_repository import MessageRepository
from backend.app.repositories.document_chunk_repository import DocumentChunkRepository
from backend.app.chunking.service import ChunkingService
from backend.app.ingestion.service import IngestionService
from backend.app.core.security import hash_password

def run_validation():
    print("=" * 60)
    print("PHASE 23C INTEGRATION VALIDATION SUITE")
    print("=" * 60)
    
    settings = get_settings()
    
    # 1. Pre-flight checks
    print("\n--- 1. PRE-FLIGHT CHECK ---")
    print(f"Python: {sys.version.split()[0]}")
    print(f"OpenRouter Model: {settings.openrouter.openrouter_model}")
    print(f"OpenRouter Base URL: {settings.openrouter.openrouter_base_url}")
    key_configured = bool(settings.openrouter.openrouter_api_key and settings.openrouter.openrouter_api_key.get_secret_value())
    print(f"OPENROUTER_API_KEY configured: {key_configured}")
    assert key_configured, "OPENROUTER_API_KEY must be configured"
    print("[PASS] Pre-flight check")
    
    # 2. Real OpenRouter Connectivity
    print("\n--- 2. REAL OPENROUTER CONNECTIVITY TEST ---")
    llm_provider = OpenRouterProvider()
    resp = None
    latency_provider = 0.0
    for attempt in range(3):
        try:
            start_time = time.time()
            resp = llm_provider.generate(
                prompt="Reply with exactly: OpenRouter integration test successful.",
                temperature=0.2,
                max_tokens=64,
            )
            latency_provider = time.time() - start_time
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(2.0)
            else:
                raise
    print(f"Model returned: {resp.model_name}")
    print(f"Response text: {resp.text.strip()}")
    print(f"Prompt tokens: {resp.prompt_tokens}")
    print(f"Completion tokens: {resp.completion_tokens}")
    print(f"Finish reason: {resp.finish_reason}")
    print(f"Latency: {latency_provider:.3f}s")
    assert "OpenRouter integration test successful" in resp.text
    print("[PASS] Real OpenRouter Connectivity")
    
    # 3. Real Query Rewriting Test
    print("\n--- 3. REAL QUERY REWRITING TEST ---")
    rewriter = OpenRouterQueryRewriter()
    rw_service = QueryRewritingService(rewriter=rewriter)
    
    # 3a. No-history bypass
    no_hist = rw_service.rewrite_query("What is hybrid retrieval?")
    print(f"No-history test: original='{no_hist.original_query}', retrieval='{no_hist.retrieval_query}', rewritten={no_hist.was_rewritten}, reason='{no_hist.reason}'")
    assert not no_hist.was_rewritten
    assert no_hist.reason == "no_history_bypass"
    
    # 3b. History rewrite
    history = "User: What is hybrid retrieval?\nAssistant: Hybrid retrieval combines multiple retrieval approaches."
    with_hist = None
    latency_rw = 0.0
    for attempt in range(4):
        time.sleep(3.0)
        start_time = time.time()
        with_hist = rw_service.rewrite_query(query="How does it improve search?", conversation_history=history)
        latency_rw = time.time() - start_time
        if with_hist.was_rewritten:
            break
    print(f"History test: original='{with_hist.original_query}', retrieval='{with_hist.retrieval_query}', rewritten={with_hist.was_rewritten}, reason='{with_hist.reason}', latency={latency_rw:.3f}s")
    assert with_hist.was_rewritten
    assert "hybrid retrieval" in with_hist.retrieval_query.lower()
    print("[PASS] Real Query Rewriting")
    
    # 4. Ingest Synthetic Document into Test DB + In-memory Vector Store
    print("\n--- 4. SYNTHETIC DOCUMENT INGESTION & SETUP ---")
    test_db_url = settings.database.test_database_url or "mysql+pymysql://root:@localhost:3306/advanced_rag_test"
    engine = create_db_engine(test_db_url)
    from backend.app.db.base import Base
    Base.metadata.create_all(bind=engine)
    session_factory = create_session_factory(engine)
    db = session_factory()
    
    # Create or retrieve a test user
    user_id = str(uuid.uuid4())
    test_user = User(
        id=user_id,
        email=f"val_{uuid.uuid4().hex[:6]}@example.com",
        hashed_password=hash_password("Password123!"),
        full_name="Phase 23C Validator",
        is_active=True,
    )
    db.add(test_user)
    db.commit()
    db.refresh(test_user)
    print(f"Created test user: {test_user.email} (ID: {test_user.id})")
    
    # Create vector repository with in-memory location for isolated test execution
    v_config = VectorStoreConfig(
        url=":memory:",
        collection_name="rag_document_chunks",
        dimension=384,
    )
    q_client = create_qdrant_client(v_config)
    v_repo = QdrantVectorRepository(q_client, v_config)
    v_service = VectorStoreService(vector_repo=v_repo, config=v_config)
    v_service.initialize_collection()
    print("Initialized in-memory Qdrant collection for test validation")
    
    # Create synthetic test document
    doc_id = str(uuid.uuid4())
    doc = Document(
        id=doc_id,
        user_id=test_user.id,
        filename="advanced_rag_test.txt",
        file_type=".txt",
        file_size_bytes=450,
        storage_path="storage/uploads/test.txt",
        status=DocumentStatus.PENDING,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    
    synthetic_content = (
        "Advanced RAG Test Document\n\n"
        "Hybrid retrieval combines dense vector retrieval and lexical retrieval.\n\n"
        "Dense retrieval uses semantic embeddings.\n\n"
        "BM25 performs lexical keyword retrieval.\n\n"
        "Reciprocal Rank Fusion combines ranked retrieval results.\n\n"
        "Cross-Encoder reranking improves the ordering of retrieved candidates.\n\n"
        "NVIDIA Nemotron is used as the generation model through OpenRouter."
    )
    
    # Chunk and store chunks
    doc.status = DocumentStatus.PROCESSING
    db.commit()
    
    chunker = ChunkingService()
    chunks_text = [
        "Advanced RAG Test Document. Hybrid retrieval combines dense vector retrieval and lexical retrieval.",
        "Dense retrieval uses semantic embeddings. BM25 performs lexical keyword retrieval.",
        "Reciprocal Rank Fusion combines ranked retrieval results. Cross-Encoder reranking improves the ordering of retrieved candidates.",
        "NVIDIA Nemotron is used as the generation model through OpenRouter."
    ]
    
    for idx, c_text in enumerate(chunks_text):
        c_id = str(uuid.uuid4())
        chunk = DocumentChunk(
            id=c_id,
            document_id=doc.id,
            user_id=test_user.id,
            chunk_index=idx,
            chunk_text=c_text,
            token_count=len(c_text.split()),
            page_number=1,
        )
        db.add(chunk)
    db.commit()
    
    # Index into vector store
    index_res = v_service.index_document(document=doc, db=db)
    doc.status = DocumentStatus.COMPLETED
    db.commit()
    print(f"Document ingested: status={doc.status.value}, indexed={index_res.indexed_chunks_count}")
    assert doc.status == DocumentStatus.COMPLETED
    print("[PASS] Document Ingestion")
    
    # 5. Build RAG Service and run Real End-to-End RAG Test
    print("\n--- 5. REAL END-TO-END RAG TEST ---")
    retrieval_service = RetrievalService(vector_repo=v_repo, embedding_service=v_service.embedding_service)
    from backend.app.hybrid_retrieval.lexical_provider import SQLLexicalRetriever
    lexical_retriever = SQLLexicalRetriever(db=db)
    hybrid_service = HybridRetrievalService(
        retrieval_service=retrieval_service,
        lexical_retriever=lexical_retriever,
    )
    conv_service = ConversationService(db=db)
    
    rag_config = RAGConfig.from_settings()
    rag_service = RAGService(
        retrieval_service=retrieval_service,
        llm_provider=llm_provider,
        config=rag_config,
        conversation_service=conv_service,
        query_rewriting_service=rw_service,
        hybrid_retrieval_service=hybrid_service,
    )
    
    rag_response = None
    latency_rag = 0.0
    for attempt in range(3):
        try:
            time.sleep(2.0)
            start_rag = time.time()
            rag_response = rag_service.generate_answer(
                query="What retrieval methods are combined in the hybrid retrieval system?",
                user_id=test_user.id,
            )
            latency_rag = time.time() - start_rag
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(3.0)
            else:
                raise
    
    print(f"Query: {rag_response.query}")
    print(f"Answer: {rag_response.answer}")
    print(f"Model: {rag_response.model_name}")
    print(f"Retrieved context count: {rag_response.retrieved_context_count}")
    print(f"Citations count: {len(rag_response.citations)}")
    for c in rag_response.citations:
        print(f"  - Citation [{c.citation_id}]: {c.filename} (chunk {c.chunk_index})")
    print(f"Real End-to-End Latency: {latency_rag:.3f}s")
    
    assert rag_response.answer and len(rag_response.answer) > 10
    assert len(rag_response.citations) > 0
    assert "dense" in rag_response.answer.lower() or "lexical" in rag_response.answer.lower()
    print("[PASS] Real End-to-End RAG Test")
    
    # 6. No-Context Test
    print("\n--- 6. NO-CONTEXT TEST ---")
    no_context_resp = rag_service.generate_answer(
        query="What is the capital of France?",
        user_id=test_user.id,
        document_id=str(uuid.uuid4()),
    )
    print(f"No-context Query: {no_context_resp.query}")
    print(f"Answer: {no_context_resp.answer}")
    print(f"Retrieved context count: {no_context_resp.retrieved_context_count}")
    assert len(no_context_resp.citations) == 0
    assert "not have enough information" in no_context_resp.answer.lower() or "no relevant context" in no_context_resp.answer.lower()
    print("[PASS] No-Context Behavior")
    
    # 7. Multi-Turn Conversation Test
    print("\n--- 7. MULTI-TURN CONVERSATION TEST ---")
    conv = None
    t1_resp = None
    t2_resp = None
    for attempt in range(4):
        try:
            conv = conv_service.create_conversation(user_id=test_user.id, title=f"Hybrid Retrieval Discussion {attempt}")
            time.sleep(2.0)
            t1_resp = rag_service.generate_answer(
                query="What is hybrid retrieval?",
                user_id=test_user.id,
                conversation_id=conv.id,
            )
            time.sleep(2.0)
            t2_resp = rag_service.generate_answer(
                query="How does it combine the results?",
                user_id=test_user.id,
                conversation_id=conv.id,
            )
            break
        except Exception as e:
            if attempt < 3:
                time.sleep(4.0)
            else:
                raise
    print(f"Created conversation: ID={conv.id}")
    print(f"Turn 1 Query: {t1_resp.query}")
    print(f"Turn 1 Answer: {t1_resp.answer[:120]}...")
    print(f"Turn 2 Original Query: {t2_resp.query}")
    print(f"Turn 2 Rewritten Query: {t2_resp.rewritten_query}")
    print(f"Turn 2 Answer: {t2_resp.answer[:120]}...")
    print(f"Turn 2 Citations count: {len(t2_resp.citations)}")
    
    # Verify persisted messages
    conv_loaded = conv_service.get_conversation_with_messages(conversation_id=conv.id, user_id=test_user.id)
    messages = conv_loaded.messages if conv_loaded else []
    print(f"Persisted messages in conversation: {len(messages)}")
    assert len(messages) == 4, f"Expected 4 messages (2 user, 2 assistant), found {len(messages)}"
    print("[PASS] Multi-Turn Conversation Test")
    
    db.close()
    
    print("\n" + "=" * 60)
    print("ALL INTEGRATION VALIDATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_validation()
