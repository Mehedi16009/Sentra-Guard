from __future__ import annotations

import gc
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import ExperimentConfig
from .fusion import compute_final_score, uncertainty_mask
from .retrieval import FaissKnowledgeBase
from .train import TransformerHarmClassifier


def cleanup_memory(logger: Any | None = None, tag: str = "") -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if logger is not None and tag:
        logger.info("[cleanup] %s", tag)


def amp_autocast():
    return torch.cuda.amp.autocast(enabled=torch.cuda.is_available())


class OptimizedZeroShotScorer:
    def __init__(self, cfg: ExperimentConfig, logger: Any | None = None) -> None:
        self.cfg = cfg
        self.logger = logger
        self.model_name = cfg.selected_zero_shot_model_name
        self.device = cfg.device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.candidate_labels = ["harmful", "benign"]
        self.hypothesis_template = "The user's intent is {}."
        self.cache: Dict[str, float] = {}
        self.entailment_idx, self.contradiction_idx = self._infer_nli_indices()

    def _infer_nli_indices(self) -> Tuple[int, int]:
        label2id = {str(key).lower(): int(value) for key, value in self.model.config.label2id.items()}
        id2label = {int(key): str(value).lower() for key, value in self.model.config.id2label.items()}

        entailment_idx = None
        contradiction_idx = None

        for label_name, idx in label2id.items():
            if "entail" in label_name:
                entailment_idx = idx
            if "contrad" in label_name:
                contradiction_idx = idx

        if entailment_idx is None or contradiction_idx is None:
            for idx, label_name in id2label.items():
                if "entail" in label_name:
                    entailment_idx = idx
                if "contrad" in label_name:
                    contradiction_idx = idx

        if entailment_idx is None or contradiction_idx is None:
            raise ValueError(
                f"Could not infer entailment/contradiction indices from zero-shot model config. "
                f"label2id={self.model.config.label2id}, id2label={self.model.config.id2label}"
            )
        return entailment_idx, contradiction_idx

    @torch.inference_mode()
    def predict_scores(self, texts: Sequence[str], split_name: str) -> np.ndarray:
        unique_texts = list(dict.fromkeys(texts))
        uncached = [text for text in unique_texts if text not in self.cache]

        if self.logger is not None:
            self.logger.info(
                "[zero-shot:%s] cache_hit=%d | cache_miss=%d",
                split_name,
                len(unique_texts) - len(uncached),
                len(uncached),
            )

        total = len(uncached)
        idx = 0
        batch_size = self.cfg.zero_shot_batch_size
        start_time = time.perf_counter()

        while idx < total:
            bs = min(batch_size, total - idx)
            batch_texts = uncached[idx : idx + bs]
            try:
                premises: List[str] = []
                hypotheses: List[str] = []
                for text in batch_texts:
                    for label in self.candidate_labels:
                        premises.append(text)
                        hypotheses.append(self.hypothesis_template.format(label))

                encoded = self.tokenizer(
                    premises,
                    hypotheses,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max(128, self.cfg.classifier_max_length),
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}

                with amp_autocast():
                    logits = self.model(**encoded).logits

                entailment_logits = logits[:, self.entailment_idx].float().reshape(len(batch_texts), len(self.candidate_labels))
                label_probs = torch.softmax(entailment_logits, dim=-1).cpu().numpy().astype(np.float32)
                harmful_probs = label_probs[:, 0]

                for text, score in zip(batch_texts, harmful_probs):
                    self.cache[text] = float(score)

                idx += bs
                if self.logger is not None:
                    elapsed = time.perf_counter() - start_time
                    rate = idx / max(elapsed, 1e-8)
                    eta = (total - idx) / max(rate, 1e-8)
                    self.logger.info(
                        "[zero-shot:%s] processed=%d/%d | elapsed=%.1fs | eta=%.1fs | bs=%d",
                        split_name,
                        idx,
                        total,
                        elapsed,
                        eta,
                        bs,
                    )
            except RuntimeError as exc:
                if torch.cuda.is_available() and "out of memory" in str(exc).lower() and bs > 1:
                    cleanup_memory(self.logger, f"zero-shot oom retry on {split_name}")
                    batch_size = max(1, bs // 2)
                    if self.logger is not None:
                        self.logger.warning("[zero-shot:%s] OOM; reducing batch size to %d", split_name, batch_size)
                    continue
                raise

        return np.asarray([self.cache[text] for text in texts], dtype=np.float32)


def maybe_cap_split(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    max_eval_samples: Optional[int],
    seed: int,
    logger: Any | None = None,
    split_name: str = "",
) -> Tuple[pd.DataFrame, np.ndarray]:
    if max_eval_samples is None or len(frame) <= max_eval_samples:
        return frame.reset_index(drop=True).copy(), embeddings

    sampled = frame.sample(n=max_eval_samples, random_state=seed).sort_index()
    sampled_embeddings = embeddings[sampled.index.to_numpy()]
    sampled = sampled.reset_index(drop=True)
    if logger is not None:
        logger.info("[%s] applied eval cap: %d/%d", split_name, len(sampled), len(frame))
    return sampled, sampled_embeddings


def score_split(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    split_name: str,
    cfg: ExperimentConfig,
    classifier: TransformerHarmClassifier,
    knowledge_base: FaissKnowledgeBase,
    zero_shot_scorer: OptimizedZeroShotScorer,
    logger: Any,
    weights: Optional[Tuple[float, float, float]] = None,
    theta: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    weights = weights or (cfg.w1, cfg.w2, cfg.w3)
    theta = cfg.theta if theta is None else theta
    frame_eval, embeddings_eval = maybe_cap_split(
        frame,
        embeddings,
        cfg.max_eval_samples,
        cfg.seed,
        logger=logger,
        split_name=split_name,
    )
    texts = frame_eval["normalized_text"].tolist()

    classifier_bs = 64 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 20 * (1024 ** 3) else 32

    classifier_start = time.perf_counter()
    classifier_scores = classifier.predict_proba(
        texts=texts,
        batch_size=classifier_bs if torch.cuda.is_available() else max(16, cfg.batch_size),
        split_name=split_name,
        logger=logger,
    )
    classifier_total = time.perf_counter() - classifier_start
    classifier_latency_ms = (classifier_total / max(len(frame_eval), 1)) * 1000.0
    cleanup_memory(logger, f"{split_name} classifier")

    retrieval_start = time.perf_counter()
    retrieval_scores = knowledge_base.query_from_embeddings(
        embeddings_eval,
        top_k=cfg.top_k,
        return_neighbors=False,
    ).scores
    retrieval_total = time.perf_counter() - retrieval_start
    retrieval_latency_ms = (retrieval_total / max(len(frame_eval), 1)) * 1000.0
    logger.info("[retrieval:%s] processed=%d/%d | elapsed=%.1fs | eta=0.0s | bs=reuse", split_name, len(frame_eval), len(frame_eval), retrieval_total)
    cleanup_memory(logger, f"{split_name} retrieval")

    zero_shot_start = time.perf_counter()
    zero_shot_scores = zero_shot_scorer.predict_scores(texts=texts, split_name=split_name)
    zero_shot_total = time.perf_counter() - zero_shot_start
    zero_shot_latency_ms = (zero_shot_total / max(len(frame_eval), 1)) * 1000.0
    cleanup_memory(logger, f"{split_name} zero-shot")

    final_scores = compute_final_score(
        classifier_scores=classifier_scores,
        retrieval_scores=retrieval_scores,
        zero_shot_scores=zero_shot_scores,
        weights=weights,
    )
    pred_label = (final_scores >= theta).astype(int)
    needs_hitl = uncertainty_mask(final_scores, theta=theta, delta=cfg.delta)

    total_time_sec = classifier_total + retrieval_total + zero_shot_total
    end_to_end_latency_ms = classifier_latency_ms + retrieval_latency_ms + zero_shot_latency_ms

    scored = frame_eval[
        ["sample_id", "text", "normalized_text", "label", "source", "detected_language", "normalization_status"]
    ].copy()
    scored["split"] = split_name
    scored["classifier_score"] = classifier_scores
    scored["retrieval_score"] = retrieval_scores
    scored["zero_shot_score"] = zero_shot_scores
    scored["final_score"] = final_scores
    scored["pred_label"] = pred_label
    scored["pred_label_name"] = np.where(scored["pred_label"] == 1, "harmful", "benign")
    scored["needs_hitl"] = needs_hitl
    scored["w1"] = float(weights[0])
    scored["w2"] = float(weights[1])
    scored["w3"] = float(weights[2])
    scored["theta"] = float(theta)
    scored["delta"] = float(cfg.delta)
    scored["classifier_latency_ms"] = float(classifier_latency_ms)
    scored["retrieval_latency_ms"] = float(retrieval_latency_ms)
    scored["zero_shot_latency_ms"] = float(zero_shot_latency_ms)
    scored["end_to_end_latency_ms"] = float(end_to_end_latency_ms)
    scored["classifier_time_sec"] = float(classifier_total)
    scored["retrieval_time_sec"] = float(retrieval_total)
    scored["zero_shot_time_sec"] = float(zero_shot_total)
    scored["total_time_sec"] = float(total_time_sec)

    profile = {
        "split": split_name,
        "n": int(len(frame_eval)),
        "classifier_time_sec": float(classifier_total),
        "retrieval_time_sec": float(retrieval_total),
        "zero_shot_time_sec": float(zero_shot_total),
        "total_time_sec": float(total_time_sec),
        "classifier_latency_ms": float(classifier_latency_ms),
        "retrieval_latency_ms": float(retrieval_latency_ms),
        "zero_shot_latency_ms": float(zero_shot_latency_ms),
        "end_to_end_latency_ms": float(end_to_end_latency_ms),
    }
    logger.info(
        "[profile:%s] classifier=%.2fs | retrieval=%.2fs | zero-shot=%.2fs | total=%.2fs | latency=%.2f ms/prompt",
        split_name,
        classifier_total,
        retrieval_total,
        zero_shot_total,
        total_time_sec,
        end_to_end_latency_ms,
    )
    return scored, profile
