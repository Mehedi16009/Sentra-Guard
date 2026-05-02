from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from .config import ExperimentConfig
from .fusion import VARIANT_SPECS, compute_final_score, generate_weight_grid


def attack_success_rate(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    harmful_mask = y_true == 1
    if harmful_mask.sum() == 0:
        return float("nan")
    return float(np.mean(y_pred[harmful_mask] == 0))


def safe_roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_pr_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def evaluate_scored_frame(scored_frame: pd.DataFrame, dataset_name: str, variant: str) -> Dict[str, Any]:
    y_true = scored_frame["label"].to_numpy(dtype=np.int32)
    y_pred = scored_frame["pred_label"].to_numpy(dtype=np.int32)
    y_score = scored_frame["final_score"].to_numpy(dtype=np.float32)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "dataset": dataset_name,
        "variant": variant,
        "n": int(len(scored_frame)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "asr": attack_success_rate(y_true, y_pred),
        "latency_ms": float(scored_frame["end_to_end_latency_ms"].mean()),
        "roc_auc": safe_roc_auc(y_true, y_score),
        "pr_auc": safe_pr_auc(y_true, y_score),
        "hitl_rate": float(scored_frame["needs_hitl"].mean()),
    }


def normalize_active_weights(weights: Tuple[float, float, float]) -> Tuple[float, float, float]:
    total = sum(weights)
    return tuple(weight / total if total > 0 else 0.0 for weight in weights)


def default_ablation_weights(cfg: ExperimentConfig) -> Dict[str, Tuple[float, float, float]]:
    return {
        "classifier_only": (1.0, 0.0, 0.0),
        "retrieval_only": (0.0, 1.0, 0.0),
        "zero_shot_only": (0.0, 0.0, 1.0),
        "classifier_retrieval": normalize_active_weights((cfg.w1, cfg.w2, 0.0)),
        "classifier_zero_shot": normalize_active_weights((cfg.w1, 0.0, cfg.w3)),
        "retrieval_zero_shot": normalize_active_weights((0.0, cfg.w2, cfg.w3)),
        "full_sentra_guard": (cfg.w1, cfg.w2, cfg.w3),
    }


def apply_variant(scored_frame: pd.DataFrame, variant_name: str, weights: Tuple[float, float, float], theta: float, delta: float) -> pd.DataFrame:
    frame = scored_frame.copy()
    frame["final_score"] = compute_final_score(
        classifier_scores=frame["classifier_score"].to_numpy(dtype=np.float32),
        retrieval_scores=frame["retrieval_score"].to_numpy(dtype=np.float32),
        zero_shot_scores=frame["zero_shot_score"].to_numpy(dtype=np.float32),
        weights=weights,
    )
    frame["pred_label"] = (frame["final_score"].to_numpy() >= theta).astype(int)
    frame["pred_label_name"] = np.where(frame["pred_label"] == 1, "harmful", "benign")
    frame["needs_hitl"] = np.abs(frame["final_score"].to_numpy() - theta) < delta
    frame["variant"] = variant_name
    frame["w1"] = float(weights[0])
    frame["w2"] = float(weights[1])
    frame["w3"] = float(weights[2])

    latency = 0.0
    if weights[0] > 0:
        latency += float(frame["classifier_latency_ms"].iloc[0])
    if weights[1] > 0:
        latency += float(frame["retrieval_latency_ms"].iloc[0])
    if weights[2] > 0:
        latency += float(frame["zero_shot_latency_ms"].iloc[0])
    frame["end_to_end_latency_ms"] = latency
    return frame


def run_ablation_study(datasets: Dict[str, pd.DataFrame], cfg: ExperimentConfig) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset_name, scored_frame in datasets.items():
        for variant_name, weights in default_ablation_weights(cfg).items():
            variant_frame = apply_variant(
                scored_frame=scored_frame,
                variant_name=variant_name,
                weights=weights,
                theta=cfg.theta,
                delta=cfg.delta,
            )
            rows.append(evaluate_scored_frame(variant_frame, dataset_name, variant_name))
    return pd.DataFrame(rows).sort_values(["dataset", "f1"], ascending=[True, False]).reset_index(drop=True)


def run_threshold_sweep(val_scored: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    scores = compute_final_score(
        classifier_scores=val_scored["classifier_score"].to_numpy(dtype=np.float32),
        retrieval_scores=val_scored["retrieval_score"].to_numpy(dtype=np.float32),
        zero_shot_scores=val_scored["zero_shot_score"].to_numpy(dtype=np.float32),
        weights=(cfg.w1, cfg.w2, cfg.w3),
    )
    for theta in cfg.thresholds:
        temp = val_scored.copy()
        temp["final_score"] = scores
        temp["pred_label"] = (scores >= theta).astype(int)
        temp["needs_hitl"] = np.abs(scores - theta) < cfg.delta
        row = evaluate_scored_frame(temp, "D1_val", "full_sentra_guard_default_weights")
        row["theta"] = float(theta)
        row["w1"] = float(cfg.w1)
        row["w2"] = float(cfg.w2)
        row["w3"] = float(cfg.w3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        by=["f1", "recall", "precision", "roc_auc", "theta"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def run_weight_search(val_scored: pd.DataFrame, cfg: ExperimentConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    classifier_scores = val_scored["classifier_score"].to_numpy(dtype=np.float32)
    retrieval_scores = val_scored["retrieval_score"].to_numpy(dtype=np.float32)
    zero_shot_scores = val_scored["zero_shot_score"].to_numpy(dtype=np.float32)

    rows: List[Dict[str, Any]] = []
    best_row: Dict[str, Any] | None = None

    for weights in generate_weight_grid(("classifier", "retrieval", "zero_shot"), step=cfg.weight_grid_step):
        combined = compute_final_score(
            classifier_scores=classifier_scores,
            retrieval_scores=retrieval_scores,
            zero_shot_scores=zero_shot_scores,
            weights=weights,
        )
        for theta in cfg.thresholds:
            temp = val_scored.copy()
            temp["final_score"] = combined
            temp["pred_label"] = (combined >= theta).astype(int)
            temp["needs_hitl"] = np.abs(combined - theta) < cfg.delta

            row = evaluate_scored_frame(temp, "D1_val", "weight_search")
            row["theta"] = float(theta)
            row["w1"] = float(weights[0])
            row["w2"] = float(weights[1])
            row["w3"] = float(weights[2])
            rows.append(row)

            if best_row is None:
                best_row = row
            else:
                challenger = (row["f1"], row["recall"], row["precision"], row["roc_auc"], -abs(theta - 0.5))
                incumbent = (best_row["f1"], best_row["recall"], best_row["precision"], best_row["roc_auc"], -abs(best_row["theta"] - 0.5))
                if challenger > incumbent:
                    best_row = row

    if best_row is None:
        raise RuntimeError("Weight search did not produce any candidates.")

    result = pd.DataFrame(rows).sort_values(
        by=["f1", "recall", "precision", "roc_auc", "theta"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    return result, best_row


def finalize_predictions(scored_frame: pd.DataFrame, dataset_name: str, weights: Tuple[float, float, float], theta: float, delta: float) -> pd.DataFrame:
    frame = scored_frame.copy()
    frame["dataset"] = dataset_name
    frame["final_score"] = compute_final_score(
        classifier_scores=frame["classifier_score"].to_numpy(dtype=np.float32),
        retrieval_scores=frame["retrieval_score"].to_numpy(dtype=np.float32),
        zero_shot_scores=frame["zero_shot_score"].to_numpy(dtype=np.float32),
        weights=weights,
    )
    frame["pred_label"] = (frame["final_score"].to_numpy() >= theta).astype(int)
    frame["pred_label_name"] = np.where(frame["pred_label"] == 1, "harmful", "benign")
    frame["needs_hitl"] = np.abs(frame["final_score"].to_numpy() - theta) < delta
    frame["selected_w1"] = float(weights[0])
    frame["selected_w2"] = float(weights[1])
    frame["selected_w3"] = float(weights[2])
    frame["selected_theta"] = float(theta)
    frame["selected_delta"] = float(delta)
    return frame


def build_confusion_dataframe(predictions_by_dataset: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset_name, frame in predictions_by_dataset.items():
        cm = confusion_matrix(frame["label"], frame["pred_label"], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append(
            {
                "dataset": dataset_name,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return pd.DataFrame(rows)
