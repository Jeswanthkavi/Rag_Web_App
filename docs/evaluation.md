# RAG Evaluation Framework

## 1. Objectives

The evaluation framework objectively measures retrieval precision and generation faithfulness across test benchmarks to avoid hallucination and improve retrieval efficacy.

---

## 2. Evaluation Metrics

### 2.1 Retrieval Metrics
- **Precision@K**: Fraction of retrieved chunks among top-$K$ that contain relevant ground truth facts.
- **Recall@K**: Fraction of ground truth relevant chunks successfully retrieved in top-$K$.
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of the first relevant chunk in retrieval results.
- **Hit Rate@K**: Proportion of queries where at least one ground-truth relevant chunk was retrieved.

### 2.2 RAG Generation & Groundedness Metrics
- **Context Relevance**: Measures whether retrieved chunks are concise and directly relevant to the query.
- **Faithfulness (Groundedness)**: Assesses whether every claim in the generated response can be directly inferred from the retrieved chunks.
- **Answer Relevance**: Assesses how directly and completely the generated answer addresses the user query.
- **Citation Correctness**: Verification of every citation bracket `[1]`, `[2]` against the referenced chunk.

### 2.3 System Latency Telemetry
- **Ingestion Latency**: Time to extract text, chunk, embed, and store in Qdrant.
- **Retrieval Latency**: Milliseconds for vector search + cross-encoder rerank.
- **LLM Generation Latency**: Milliseconds from prompt dispatch to completion response.

---

## 3. Evaluation Dataset Structure

Benchmark evaluation records are structured in JSON format:

```json
[
  {
    "id": "eval_001",
    "question": "What is the primary optimization goal of model quantization in the document?",
    "document_filename": "deep_learning_optimization.pdf",
    "expected_chunks": ["chunk_uuid_102"],
    "reference_answer": "Model quantization reduces the precision of weights and activations to lower memory footprint and latency.",
    "metadata_filters": {
      "topic": "quantization"
    }
  }
]
```
