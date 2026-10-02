"""Evaluation dataset loaders and synthetic test dataset."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from backend.app.evaluation.models import EvaluationSample


SYNTHETIC_EVALUATION_DATA = [
    {
        "sample_id": "eval_001",
        "query": "What is the warranty coverage period for the lithium battery?",
        "expected_chunk_ids": ["chunk_battery_warranty_01"],
        "expected_answer": "The lithium battery includes a 10-year replacement warranty covering manufacturing defects.",
        "expected_citations": ["chunk_battery_warranty_01"],
        "metadata": {"category": "warranty", "difficulty": "easy"},
    },
    {
        "sample_id": "eval_002",
        "query": "How does the Kubernetes auto-scaler handle traffic spikes?",
        "expected_chunk_ids": ["chunk_k8s_autoscaling_02"],
        "expected_answer": "The Kubernetes Horizontal Pod Autoscaler scales replica counts based on CPU and memory thresholds.",
        "expected_citations": ["chunk_k8s_autoscaling_02"],
        "metadata": {"category": "infrastructure", "difficulty": "medium"},
    },
    {
        "sample_id": "eval_003",
        "query": "What are the return policy requirements and refund turnaround times?",
        "expected_chunk_ids": ["chunk_refund_policy_01", "chunk_refund_timeline_02"],
        "expected_answer": "Customers must return items within 30 days in original packaging; refunds are processed within 5 business days.",
        "expected_citations": ["chunk_refund_policy_01", "chunk_refund_timeline_02"],
        "metadata": {"category": "policy", "difficulty": "medium"},
    },
    {
        "sample_id": "eval_004",
        "query": "What is the PostgreSQL database failover architecture and RPO target?",
        "expected_chunk_ids": ["chunk_db_failover_01"],
        "expected_answer": "PostgreSQL utilizes PgBouncer connection pooling with asynchronous replica promotion providing an RPO under 5 minutes.",
        "expected_citations": ["chunk_db_failover_01"],
        "metadata": {"category": "database", "difficulty": "hard"},
    },
    {
        "sample_id": "eval_005",
        "query": "What are the solar inverter operating temperature limits?",
        "expected_chunk_ids": ["chunk_solar_inverter_03"],
        "expected_answer": "The solar inverter operates between -20C and +50C with thermal throttling above 45C.",
        "expected_citations": ["chunk_solar_inverter_03"],
        "metadata": {"category": "hardware", "difficulty": "easy"},
    },
]


class EvaluationDataset:
    """Dataset container managing evaluation test samples."""

    def __init__(self, samples: Sequence[EvaluationSample], name: str = "custom_dataset"):
        self.samples = list(samples)
        self.name = name

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> EvaluationSample:
        return self.samples[idx]

    @classmethod
    def from_synthetic(cls) -> "EvaluationDataset":
        """Loads built-in synthetic test evaluation dataset."""
        samples = [EvaluationSample(**item) for item in SYNTHETIC_EVALUATION_DATA]
        return cls(samples=samples, name="synthetic_evaluation_dataset")

    @classmethod
    def create_synthetic(cls) -> "EvaluationDataset":
        """Alias for from_synthetic."""
        return cls.from_synthetic()

    @classmethod
    def from_json_file(cls, filepath: str | Path) -> "EvaluationDataset":
        """Loads evaluation dataset from a JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            samples = [EvaluationSample(**item) for item in data]
            return cls(samples=samples, name=path.stem)
        elif isinstance(data, dict) and "samples" in data:
            samples = [EvaluationSample(**item) for item in data["samples"]]
            name = data.get("name", path.stem)
            return cls(samples=samples, name=name)
        else:
            raise ValueError(f"Invalid JSON evaluation dataset format in {path}")

    @classmethod
    def from_json(cls, filepath: str | Path) -> "EvaluationDataset":
        """Alias for from_json_file."""
        return cls.from_json_file(filepath)

    @classmethod
    def from_jsonl_file(cls, filepath: str | Path) -> "EvaluationDataset":
        """Loads evaluation dataset from a JSONL file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset file not found: {path}")

        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                    samples.append(EvaluationSample(**obj))
                except Exception as e:
                    raise ValueError(f"Invalid JSONL line {line_idx} in {path}: {e}") from e

        return cls(samples=samples, name=path.stem)

    @classmethod
    def from_jsonl(cls, filepath: str | Path) -> "EvaluationDataset":
        """Alias for from_jsonl_file."""
        return cls.from_jsonl_file(filepath)

    def save_json(self, filepath: str | Path) -> None:
        """Saves dataset samples to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [s.model_dump() for s in self.samples]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def save_jsonl(self, filepath: str | Path) -> None:
        """Saves dataset samples to a JSONL file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for s in self.samples:
                f.write(s.model_dump_json() + "\n")

