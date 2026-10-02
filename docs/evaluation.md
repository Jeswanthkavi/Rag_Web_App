# Evaluation Framework Specification (Phase 18)

## 1. Overview

The Evaluation Framework provides reproducible, multi-dimensional benchmarking for the Advanced RAG platform. It evaluates four foundational dimensions:
1. **Retrieval Performance** (Precision@K, Recall@K, Mean Reciprocal Rank, Context Relevance)
2. **Answer Generation Quality** (Answer Relevance, Faithfulness / Groundedness)
3. **Citation Integrity** (Valid Citation Rate, Citation Precision, Citation Coverage)
4. **Latency Telemetry** (Mean, Median, p95, Min, Max for retrieval, reranking, generation, and total RAG)

---

## 2. Metrics Definitions & Mathematical Formulations

### 2.1 Retrieval Metrics

- **Precision@K**:
  $$\text{Precision@K} = \frac{|\text{Retrieved Chunk IDs in top } K \cap \text{Expected Chunk IDs}|}{K}$$
  Supports configurable $K \in \{1, 3, 5, 10\}$.

- **Recall@K**:
  $$\text{Recall@K} = \frac{|\text{Retrieved Chunk IDs in top } K \cap \text{Expected Chunk IDs}|}{|\text{Expected Chunk IDs}|}$$
  *Edge Case Handling*: If $|\text{Expected Chunk IDs}| = 0$, returns `1.0` if 0 items were retrieved, otherwise `0.0`.

- **Mean Reciprocal Rank (MRR)**:
  $$\text{MRR} = \frac{1}{N} \sum_{i=1}^N \frac{1}{\text{rank}_i}$$
  where $\text{rank}_i$ is the position of the *first* relevant chunk in sample $i$'s retrieval results. If no relevant chunk appears, reciprocal rank is `0.0`.

- **Context Relevance**:
  $$\text{Context Relevance} = \frac{|\text{Retrieved Chunks} \cap \text{Relevant Chunks}|}{|\text{Retrieved Chunks}|}$$

### 2.2 Generation & Groundedness Metrics

- **Answer Relevance**:
  Computes token-overlap Jaccard similarity between the synthesized answer and the target ground-truth answer.
  $$\text{Relevance} = \frac{|T_{\text{generated}} \cap T_{\text{expected}}|}{|T_{\text{generated}} \cup T_{\text{expected}}|}$$

- **Faithfulness / Groundedness**:
  Measures the proportion of content words in the generated answer directly supported by the retrieved context chunks.
  $$\text{Faithfulness} = \frac{|T_{\text{generated}} \cap T_{\text{context}}|}{|T_{\text{generated}}|}$$

- **Judge Provider Abstraction**:
  - `BaseJudgeProvider`: Abstract contract for evaluation judges.
  - `DeterministicJudge`: Fast, deterministic token-overlap evaluation (no LLM costs).
  - `FakeJudge`: Configurable fixed-score judge for deterministic unit testing.

### 2.3 Citation Metrics

- **Valid Citation Rate**:
  Proportion of generated citation references that map to actual context chunks retrieved for the query ($1..N$).
- **Citation Precision**:
  Fraction of generated citations that match expected authoritative source files or chunk IDs.
- **Citation Coverage**:
  Fraction of expected citation sources present in the synthesized answer.

### 2.4 Latency Metrics

Measures execution duration with monotonic high-precision timers (`time.perf_counter()`):
- `retrieval_latency_ms`
- `rerank_latency_ms`
- `generation_latency_ms`
- `total_latency_ms`

Descriptive aggregates: Mean, Median, 95th Percentile (p95), Minimum, and Maximum.

---

## 3. Dataset Format

The evaluation runner accepts synthetic in-memory fixtures, JSON files, or JSONL files.

### 3.1 JSON / JSONL Schema

```json
{
  "sample_id": "eval_001",
  "query": "What is the warranty coverage period for the lithium battery?",
  "expected_chunk_ids": ["chunk_battery_warranty_01"],
  "expected_answer": "The lithium battery includes a 10-year replacement warranty covering manufacturing defects.",
  "expected_citations": ["chunk_battery_warranty_01"],
  "metadata": {
    "category": "warranty",
    "difficulty": "easy"
  }
}
```

---

## 4. Running Evaluations

### 4.1 CLI Execution

```bash
# Run against synthetic dataset
.venv\Scripts\python.exe scripts\run_evaluation.py

# Run against custom JSON/JSONL dataset
.venv\Scripts\python.exe scripts\run_evaluation.py --dataset path/to/dataset.jsonl --output report.json
```

### 4.2 Programmatic Usage

```python
from app.evaluation.dataset import EvaluationDataset
from app.evaluation.runner import EvaluationRunner

dataset = EvaluationDataset.from_synthetic()
runner = EvaluationRunner()

# Retrieval-only evaluation
report = runner.evaluate_retrieval(
    dataset=dataset,
    search_fn=my_search_function,
)
print(report.model_dump_json(indent=2))
```

---

## 5. Limitations of Deterministic Evaluation

1. **Token Overlap vs Semantic Understanding**: `DeterministicJudge` uses token Jaccard similarity and vocabulary intersection. While fast and reproducible without API costs, it may penalize valid paraphrases.
2. **Context Grounding Heuristic**: Token-level faithfulness measures lexical support rather than full logical entailment.
3. **Future Extension**: An LLM-as-a-Judge provider implementing `BaseJudgeProvider` can be plugged in for semantic grading in future rounds.

---

## 6. Strict Phase Boundaries

- **Phase 17 (Implemented)**: Hybrid Retrieval (Dense + Lexical BM25) and Reciprocal Rank Fusion.
- **Phase 18 (Implemented)**: Evaluation Framework.
- **Phase 19 (NOT Implemented)**: Frontend (React/Next.js).
- **Phase 20 (NOT Implemented)**: Security hardening beyond this round.
- **Phase 21 (NOT Implemented)**: Performance optimization.
- **Phase 22 (NOT Implemented)**: Production readiness.
