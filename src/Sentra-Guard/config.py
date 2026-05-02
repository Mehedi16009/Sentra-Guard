from __future__ import annotations

import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch


def default_translation_models() -> Dict[str, str]:
    return {
        "bn": "Helsinki-NLP/opus-mt-bn-en",
        "es": "Helsinki-NLP/opus-mt-es-en",
        "hi": "Helsinki-NLP/opus-mt-hi-en",
        "ar": "Helsinki-NLP/opus-mt-ar-en",
        "zh": "Helsinki-NLP/opus-mt-zh-en",
    }


@dataclass
class ExperimentConfig:
    experiment_name: str = "sentra_guard_reproduction"
    output_dir: str = "./artifacts"

    dataset_source: str = "huggingface"
    d1_dataset_name: str = "YinkaiW/harmbench-dataset"
    d2_dataset_name: str = "JailbreakV-28K/JailBreakV-28k"
    d2_dataset_config: str = "JailBreakV_28K"
    d1_path: Optional[str] = None
    d2_path: Optional[str] = None

    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15

    classifier_model_name: str = "distilbert-base-uncased"
    sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    zero_shot_model_name: str = "facebook/bart-large-mnli"
    zero_shot_fast_model_name: str = "MoritzLaurer/deberta-v3-base-zeroshot-v2"
    zero_shot_runtime_option: str = "default"

    learning_rate: float = 2e-5
    epochs: int = 5
    batch_size: int = 16
    top_k: int = 5

    w1: float = 0.50
    w2: float = 0.25
    w3: float = 0.25
    theta: float = 0.50
    delta: float = 0.08

    classifier_max_length: int = 64
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 2

    seed: int = 42
    deterministic: bool = True
    num_workers: int = 0
    max_eval_samples: Optional[int] = None

    embedding_batch_size: int = 128
    translation_batch_size: int = 16
    zero_shot_batch_size: int = 8
    threshold_min: float = 0.30
    threshold_max: float = 0.80
    threshold_step: float = 0.05
    weight_grid_step: float = 0.05

    text_column_candidates: Tuple[str, ...] = (
        "prompt",
        "text",
        "content",
        "instruction",
        "input",
        "query",
        "message",
        "user_prompt",
        "sentence",
        "question",
        "jailbreak_query",
    )
    label_column_candidates: Tuple[str, ...] = (
        "label",
        "labels",
        "class",
        "target",
        "is_harmful",
        "binary_label",
        "prompt_type",
    )
    role_column_candidates: Tuple[str, ...] = (
        "role",
        "speaker",
        "author",
        "message_role",
    )
    translation_models: Dict[str, str] = field(default_factory=default_translation_models)

    def __post_init__(self) -> None:
        if self.dataset_source not in {"huggingface", "local"}:
            raise ValueError("dataset_source must be 'huggingface' or 'local'.")
        if self.dataset_source == "local":
            if not self.d1_path or not self.d2_path:
                raise ValueError("Local mode requires both d1_path and d2_path.")
        if self.zero_shot_runtime_option not in {"default", "fast"}:
            raise ValueError("zero_shot_runtime_option must be 'default' or 'fast'.")
        if abs(self.train_size + self.val_size + self.test_size - 1.0) > 1e-8:
            raise ValueError("train_size + val_size + test_size must equal 1.")

        for path in (
            self.output_root,
            self.cache_dir,
            self.logs_dir,
            self.checkpoints_dir,
            self.retrieval_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def output_root(self) -> Path:
        return Path(self.output_dir)

    @property
    def cache_dir(self) -> Path:
        return self.output_root / "cache"

    @property
    def logs_dir(self) -> Path:
        return self.output_root / "logs"

    @property
    def checkpoints_dir(self) -> Path:
        return self.output_root / "checkpoints"

    @property
    def retrieval_dir(self) -> Path:
        return self.output_root / "retrieval"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def device_index(self) -> int:
        return 0 if torch.cuda.is_available() else -1

    @property
    def thresholds(self) -> np.ndarray:
        return np.round(
            np.arange(self.threshold_min, self.threshold_max + 1e-9, self.threshold_step),
            2,
        )

    @property
    def selected_zero_shot_model_name(self) -> str:
        return self.zero_shot_fast_model_name if self.zero_shot_runtime_option == "fast" else self.zero_shot_model_name

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)


def set_global_determinism(seed: int = 42, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def build_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_experiment_logger(output_dir: str | Path, name: str = "sentra_guard") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    log_path = Path(output_dir) / "logs" / f"{name}.log"
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
