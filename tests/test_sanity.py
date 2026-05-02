from __future__ import annotations

import numpy as np
import pandas as pd

from sentra_guard.data import extract_text_based_jailbreak_rows, map_prompt_type_to_label
from sentra_guard.fusion import compute_final_score, generate_weight_grid, uncertainty_mask
from sentra_guard.retrieval import FaissKnowledgeBase


def test_prompt_type_mapping() -> None:
    assert map_prompt_type_to_label("attack") == 1
    assert map_prompt_type_to_label("no_attack") == 0
    assert map_prompt_type_to_label("benign") == 0
    assert map_prompt_type_to_label("harmful") == 1


def test_extract_text_based_jailbreak_rows() -> None:
    frame = pd.DataFrame(
        {
            "jailbreak_query": ["a", "b", "c", ""],
            "transfer_from_llm": [True, False, False, False],
            "format": ["image", "template", "logic", "image"],
        }
    )
    filtered = extract_text_based_jailbreak_rows(frame)
    assert filtered["text"].tolist() == ["a", "b", "c"]
    assert filtered["label"].tolist() == [1, 1, 1]


def test_fusion_equation() -> None:
    classifier_scores = np.array([0.8, 0.2], dtype=np.float32)
    retrieval_scores = np.array([0.6, 0.4], dtype=np.float32)
    zero_shot_scores = np.array([0.4, 0.6], dtype=np.float32)
    final_scores = compute_final_score(classifier_scores, retrieval_scores, zero_shot_scores, (0.5, 0.25, 0.25))
    np.testing.assert_allclose(final_scores, np.array([0.65, 0.35], dtype=np.float32))


def test_uncertainty_mask() -> None:
    scores = np.array([0.50, 0.57, 0.59, 0.40], dtype=np.float32)
    mask = uncertainty_mask(scores, theta=0.50, delta=0.08)
    assert mask.tolist() == [True, True, False, False]


def test_weight_grid_constraints() -> None:
    grid = generate_weight_grid(("classifier", "retrieval", "zero_shot"), step=0.25)
    assert all(abs(sum(weights) - 1.0) < 1e-8 for weights in grid)
    assert all(all(weight >= 0.25 for weight in weights) for weights in grid)


def test_retrieval_score_equation() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    metadata = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "text": ["harmful exemplar", "benign exemplar"],
            "label": [1, 0],
            "language": ["en", "en"],
            "source": ["train", "train"],
        }
    )
    kb = FaissKnowledgeBase(embeddings=embeddings, metadata=metadata)
    queries = np.array([[0.75, 0.25]], dtype=np.float32)
    result = kb.query_from_embeddings(queries, top_k=2, return_neighbors=False)
    expected = (0.75 * 1 + 0.25 * 0) / (0.75 + 0.25)
    np.testing.assert_allclose(result.scores, np.array([expected], dtype=np.float32))
