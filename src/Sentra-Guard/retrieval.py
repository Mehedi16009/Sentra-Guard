from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievalQueryResult:
    scores: np.ndarray
    neighbors: Optional[List[List[Dict[str, Any]]]] = None


class FaissKnowledgeBase:
    def __init__(self, embeddings: np.ndarray, metadata: pd.DataFrame) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata must have equal length.")

        self.embeddings = embeddings.astype("float32", copy=False)
        self.metadata = metadata.reset_index(drop=True).copy()
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def query_from_embeddings(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
        return_neighbors: bool = False,
    ) -> RetrievalQueryResult:
        if query_embeddings.ndim != 2:
            raise ValueError("Query embeddings must be a 2D array.")

        similarities, indices = self.index.search(query_embeddings.astype("float32", copy=False), top_k)
        scores: List[float] = []
        all_neighbors: List[List[Dict[str, Any]]] = []

        for sim_row, idx_row in zip(similarities, indices):
            neighbor_rows: List[Dict[str, Any]] = []
            numerator = 0.0
            denominator = 0.0

            for similarity, idx in zip(sim_row, idx_row):
                if idx == -1:
                    continue
                label = int(self.metadata.iloc[idx]["label"])
                numerator += float(similarity) * float(label)
                denominator += float(similarity)

                if return_neighbors:
                    entry = self.metadata.iloc[idx].to_dict()
                    entry["similarity"] = float(similarity)
                    neighbor_rows.append(entry)

            if abs(denominator) < 1e-12:
                score = float(self.metadata.iloc[idx_row[idx_row != -1]]["label"].mean()) if np.any(idx_row != -1) else 0.0
            else:
                score = numerator / denominator

            scores.append(float(score))
            if return_neighbors:
                all_neighbors.append(neighbor_rows)

        return RetrievalQueryResult(
            scores=np.asarray(scores, dtype=np.float32),
            neighbors=all_neighbors if return_neighbors else None,
        )

    def add_example(self, embedding: np.ndarray, metadata_row: Dict[str, Any]) -> None:
        embedding = embedding.reshape(1, -1).astype("float32", copy=False)
        self.index.add(embedding)
        self.embeddings = np.vstack([self.embeddings, embedding])
        self.metadata = pd.concat([self.metadata, pd.DataFrame([metadata_row])], ignore_index=True)

    def save(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(output_path / "knowledge_base.index"))
        self.metadata.to_csv(output_path / "knowledge_base.csv", index=False)
        np.save(output_path / "knowledge_base_embeddings.npy", self.embeddings)


class SemanticRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = device
        self.encoder = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str], batch_size: int = 64, show_progress_bar: bool = True) -> np.ndarray:
        embeddings = self.encoder.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32", copy=False)

    def build_knowledge_base(
        self,
        embeddings: np.ndarray,
        texts: Sequence[str],
        labels: Sequence[int],
        languages: Sequence[str],
        sample_ids: Sequence[str],
        sources: Optional[Sequence[str]] = None,
    ) -> FaissKnowledgeBase:
        metadata = pd.DataFrame(
            {
                "sample_id": list(sample_ids),
                "text": list(texts),
                "label": [int(label) for label in labels],
                "language": list(languages),
                "source": list(sources) if sources is not None else ["train"] * len(texts),
            }
        )
        return FaissKnowledgeBase(embeddings=embeddings, metadata=metadata)
