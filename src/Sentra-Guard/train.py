from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .config import ExperimentConfig, build_torch_generator, seed_worker


class PromptDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer: AutoTokenizer, max_length: int) -> None:
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = [int(label) for label in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(values[idx]) for key, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    accuracy = float(np.mean(y_true == y_pred))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def amp_autocast():
    return torch.cuda.amp.autocast(enabled=torch.cuda.is_available())


class TransformerHarmClassifier:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.checkpoint_dir = cfg.checkpoints_dir / "classifier"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.classifier_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.classifier_model_name,
            num_labels=2,
        ).to(self.device)

    def _build_loader(self, frame: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
        dataset = PromptDataset(
            texts=frame["normalized_text"].tolist(),
            labels=frame["label"].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.cfg.classifier_max_length,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            worker_init_fn=seed_worker,
            generator=build_torch_generator(self.cfg.seed),
            pin_memory=torch.cuda.is_available(),
        )

    def _run_epoch(self, loader: DataLoader, optimizer: AdamW | None = None, scheduler: Any | None = None) -> Dict[str, float]:
        is_train = optimizer is not None
        self.model.train() if is_train else self.model.eval()

        losses: List[float] = []
        all_probs: List[float] = []
        all_labels: List[int] = []

        for batch in loader:
            batch = {key: value.to(self.device) for key, value in batch.items()}
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(is_train):
                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy()
            losses.append(float(loss.item()))
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

        preds = (np.asarray(all_probs) >= 0.5).astype(int)
        metrics = binary_metrics(all_labels, preds)
        metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
        return metrics

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, logger: Any) -> pd.DataFrame:
        train_loader = self._build_loader(train_df, batch_size=self.cfg.batch_size, shuffle=True)
        val_loader = self._build_loader(val_df, batch_size=self.cfg.batch_size, shuffle=False)

        optimizer = AdamW(self.model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        total_steps = max(1, len(train_loader) * self.cfg.epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(total_steps * self.cfg.warmup_ratio)),
            num_training_steps=total_steps,
        )

        best_val_f1 = -1.0
        patience = 0
        history: List[Dict[str, Any]] = []

        for epoch in range(1, self.cfg.epochs + 1):
            train_metrics = self._run_epoch(train_loader, optimizer=optimizer, scheduler=scheduler)
            val_metrics = self._run_epoch(val_loader)

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
            }
            history.append(row)
            logger.info("Epoch metrics: %s", json.dumps(row))

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                patience = 0
                self.model.save_pretrained(self.checkpoint_dir)
                self.tokenizer.save_pretrained(self.checkpoint_dir)
            else:
                patience += 1
                if patience >= self.cfg.early_stopping_patience:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                    break

        self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint_dir).to(self.device)
        return pd.DataFrame(history)

    @torch.inference_mode()
    def predict_proba(self, texts: Sequence[str], batch_size: int, split_name: str, logger: Any | None = None) -> np.ndarray:
        self.model.eval()
        total = len(texts)
        results = np.zeros(total, dtype=np.float32)
        start_time = __import__("time").perf_counter()
        idx = 0
        current_batch_size = batch_size

        while idx < total:
            bs = min(current_batch_size, total - idx)
            batch_texts = list(texts[idx : idx + bs])
            try:
                encoded = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.cfg.classifier_max_length,
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with amp_autocast():
                    logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy().astype(np.float32)
                results[idx : idx + bs] = probs
                idx += bs
                if logger is not None:
                    elapsed = __import__("time").perf_counter() - start_time
                    rate = idx / max(elapsed, 1e-8)
                    eta = (total - idx) / max(rate, 1e-8)
                    logger.info(
                        "[classifier:%s] processed=%d/%d | elapsed=%.1fs | eta=%.1fs | bs=%d",
                        split_name,
                        idx,
                        total,
                        elapsed,
                        eta,
                        bs,
                    )
            except RuntimeError as exc:
                if torch.cuda.is_available() and "out of memory" in str(exc).lower() and bs > 1:
                    torch.cuda.empty_cache()
                    current_batch_size = max(1, bs // 2)
                    if logger is not None:
                        logger.warning("[classifier:%s] OOM; reducing batch size to %d", split_name, current_batch_size)
                    continue
                raise
        return results
