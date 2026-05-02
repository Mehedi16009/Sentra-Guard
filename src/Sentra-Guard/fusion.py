from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


BRANCH_NAMES = ("classifier", "retrieval", "zero_shot")


@dataclass(frozen=True)
class VariantSpec:
    name: str
    active_branches: Tuple[str, ...]


VARIANT_SPECS: Tuple[VariantSpec, ...] = (
    VariantSpec(name="classifier_only", active_branches=("classifier",)),
    VariantSpec(name="retrieval_only", active_branches=("retrieval",)),
    VariantSpec(name="zero_shot_only", active_branches=("zero_shot",)),
    VariantSpec(name="classifier_retrieval", active_branches=("classifier", "retrieval")),
    VariantSpec(name="classifier_zero_shot", active_branches=("classifier", "zero_shot")),
    VariantSpec(name="retrieval_zero_shot", active_branches=("retrieval", "zero_shot")),
    VariantSpec(name="full_sentra_guard", active_branches=("classifier", "retrieval", "zero_shot")),
)


def variant_map() -> Dict[str, VariantSpec]:
    return {spec.name: spec for spec in VARIANT_SPECS}


def compute_final_score(
    classifier_scores: np.ndarray,
    retrieval_scores: np.ndarray,
    zero_shot_scores: np.ndarray,
    weights: Tuple[float, float, float],
) -> np.ndarray:
    w1, w2, w3 = weights
    return (w1 * classifier_scores) + (w2 * retrieval_scores) + (w3 * zero_shot_scores)


def uncertainty_mask(final_scores: np.ndarray, theta: float, delta: float) -> np.ndarray:
    return np.abs(final_scores - theta) < delta


def generate_weight_grid(
    active_branches: Sequence[str],
    step: float = 0.05,
) -> List[Tuple[float, float, float]]:
    active_branches = tuple(active_branches)
    active_set = set(active_branches)

    if not active_set.issubset(set(BRANCH_NAMES)):
        raise ValueError(f"Unknown branch names: {active_branches}")

    if len(active_branches) == 1:
        if active_branches[0] == "classifier":
            return [(1.0, 0.0, 0.0)]
        if active_branches[0] == "retrieval":
            return [(0.0, 1.0, 0.0)]
        return [(0.0, 0.0, 1.0)]

    values = np.round(np.arange(0.0, 1.0 + 1e-9, step), 2)
    min_active_weight = round(step, 2)
    weight_grid: List[Tuple[float, float, float]] = []

    for w1 in values:
        for w2 in values:
            w3 = round(1.0 - w1 - w2, 2)
            if w3 < -1e-9 or w3 > 1.0:
                continue
            candidate = {"classifier": round(float(w1), 2), "retrieval": round(float(w2), 2), "zero_shot": round(float(w3), 2)}

            if abs(sum(candidate.values()) - 1.0) > 1e-8:
                continue

            valid = True
            for branch in BRANCH_NAMES:
                value = candidate[branch]
                if branch in active_set and value < min_active_weight:
                    valid = False
                    break
                if branch not in active_set and abs(value) > 1e-9:
                    valid = False
                    break
            if not valid:
                continue

            weight_grid.append((candidate["classifier"], candidate["retrieval"], candidate["zero_shot"]))

    deduplicated = sorted(set(weight_grid))
    if not deduplicated:
        raise RuntimeError(f"No weight combinations generated for active branches: {active_branches}")
    return deduplicated
