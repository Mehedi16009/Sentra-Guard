from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset
from langdetect import DetectorFactory, LangDetectException, detect
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import MarianMTModel, MarianTokenizer

from .config import ExperimentConfig

DetectorFactory.seed = 42

CONTROL_CHAR_RE = __import__("re").compile(r"[\u200b-\u200f\u202a-\u202e\u2060\u2066-\u2069]")
MULTISPACE_RE = __import__("re").compile(r"\s+")
TEXT_PREFIX_BLACKLIST = ("system:", "[system]", "<system>")


def clean_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_CHAR_RE.sub(" ", text)
    text = text.replace("\x00", " ")
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def canonicalize_language(language: str) -> str:
    language = (language or "unknown").strip().lower()
    if language.startswith("zh"):
        return "zh"
    if language.startswith("bn"):
        return "bn"
    if language.startswith("es"):
        return "es"
    if language.startswith("hi"):
        return "hi"
    if language.startswith("ar"):
        return "ar"
    if language.startswith("en"):
        return "en"
    return language


def detect_language_safe(text: str) -> str:
    try:
        return canonicalize_language(detect(text[:512]))
    except LangDetectException:
        return "unknown"


def infer_text_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    column_map = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in column_map:
            return column_map[candidate.lower()]
    return None


def infer_label_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    column_map = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in column_map:
            return column_map[candidate.lower()]
    return None


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported local dataset format: {path.suffix}")


def map_prompt_type_to_label(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    value_str = str(value).strip().lower()
    harmful = {"attack", "harmful", "unsafe", "jailbreak", "1", "true"}
    benign = {"benign", "no_attack", "safe", "0", "false"}
    if value_str in harmful:
        return 1
    if value_str in benign:
        return 0
    return None


def coerce_binary_label(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value > 0)
    if isinstance(value, float):
        return int(value > 0)
    return map_prompt_type_to_label(value)


def extract_text_based_jailbreak_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "jailbreak_query" not in frame.columns:
        raise ValueError("JailBreakV frame must contain 'jailbreak_query'.")

    format_norm = (
        frame["format"].astype(str).str.strip().str.lower().str.replace(r"[_-]+", " ", regex=True)
        if "format" in frame.columns
        else pd.Series("", index=frame.index)
    )
    transfer_mask = (
        frame["transfer_from_llm"].fillna(False).astype(bool)
        if "transfer_from_llm" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    text_format_mask = format_norm.isin({"template", "persuade", "logic"})
    mask = transfer_mask | text_format_mask

    if mask.sum() == 0:
        mask = frame["jailbreak_query"].astype(str).str.strip().astype(bool)

    result = frame[mask].copy()
    result["text"] = result["jailbreak_query"].astype(str)
    result["label"] = 1
    return result


def load_d1_harmbench_frame(cfg: ExperimentConfig, logger: Any) -> pd.DataFrame:
    logger.info("Loading D1 dataset: %s", cfg.d1_dataset_name if cfg.dataset_source == "huggingface" else cfg.d1_path)
    if cfg.dataset_source == "huggingface":
        dataset = load_dataset(cfg.d1_dataset_name)
        frames: List[pd.DataFrame] = []
        for split_name, split_dataset in dataset.items():
            split_frame = split_dataset.to_pandas()
            split_frame["_hf_split"] = split_name
            frames.append(split_frame)
        raw = pd.concat(frames, ignore_index=True)
    else:
        raw = read_table(Path(cfg.d1_path or ""))
        raw["_hf_split"] = "local"

    text_col = infer_text_column(raw.columns, cfg.text_column_candidates)
    label_col = "prompt_type" if "prompt_type" in raw.columns else infer_label_column(raw.columns, cfg.label_column_candidates)

    if text_col is None:
        raise ValueError("Could not detect HarmBench text column.")

    label_series = raw[label_col].map(coerce_binary_label) if label_col else pd.Series([None] * len(raw))
    split_fallback = raw["_hf_split"].astype(str).str.strip().str.lower().map({"attack": 1, "no_attack": 0})

    frame = pd.DataFrame(
        {
            "sample_id": [f"D1_{idx:07d}" for idx in range(len(raw))],
            "text": raw[text_col].map(clean_text),
            "label": label_series.fillna(split_fallback),
            "source": raw["_hf_split"].map(lambda value: f"D1_{value}"),
            "_hf_split": raw["_hf_split"],
        }
    )
    frame = frame[frame["text"].astype(bool)]
    frame = frame[frame["label"].isin([0, 1])].reset_index(drop=True)
    frame["label"] = frame["label"].astype(int)
    logger.info("Loaded D1 rows: %d", len(frame))
    return frame


def load_d2_jailbreakv_frame(cfg: ExperimentConfig, logger: Any) -> pd.DataFrame:
    logger.info("Loading D2 dataset: %s", cfg.d2_dataset_name if cfg.dataset_source == "huggingface" else cfg.d2_path)
    if cfg.dataset_source == "huggingface":
        dataset = load_dataset(cfg.d2_dataset_name, cfg.d2_dataset_config)
        if isinstance(dataset, DatasetDict):
            split_name = cfg.d2_dataset_config if cfg.d2_dataset_config in dataset else next(iter(dataset.keys()))
            raw = dataset[split_name].to_pandas()
        elif isinstance(dataset, HFDataset):
            raw = dataset.to_pandas()
        else:
            raise TypeError(f"Unsupported D2 dataset type: {type(dataset)!r}")
    else:
        raw = read_table(Path(cfg.d2_path or ""))

    filtered = extract_text_based_jailbreak_rows(raw)
    frame = filtered[["text", "label"]].copy()
    frame["sample_id"] = [f"D2H_{idx:07d}" for idx in range(len(frame))]
    frame["source"] = "D2_JailBreakV_text"
    frame = frame[["sample_id", "text", "label", "source"]].reset_index(drop=True)
    logger.info("Loaded D2 text-based harmful rows: %d", len(frame))
    return frame


class MultilingualNormalizer:
    def __init__(self, translation_models: Dict[str, str], device: torch.device, batch_size: int = 16) -> None:
        self.translation_models = translation_models
        self.device = device
        self.batch_size = batch_size
        self.model_cache: Dict[str, MarianMTModel] = {}
        self.tokenizer_cache: Dict[str, MarianTokenizer] = {}
        self.text_cache: Dict[str, Tuple[str, str, str]] = {}

    def _load(self, source_lang: str) -> Tuple[MarianTokenizer, MarianMTModel]:
        if source_lang not in self.translation_models:
            raise KeyError(f"No translator configured for language '{source_lang}'.")
        if source_lang not in self.model_cache:
            model_name = self.translation_models[source_lang]
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name).to(self.device)
            model.eval()
            self.tokenizer_cache[source_lang] = tokenizer
            self.model_cache[source_lang] = model
        return self.tokenizer_cache[source_lang], self.model_cache[source_lang]

    @torch.inference_mode()
    def translate_batch(self, texts: Sequence[str], source_lang: str) -> List[str]:
        tokenizer, model = self._load(source_lang)
        outputs: List[str] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            generated = model.generate(**encoded, max_length=256, num_beams=4)
            outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        return [clean_text(text) for text in outputs]

    def normalize_text(self, text: str) -> Tuple[str, str, str]:
        text = clean_text(text)
        if not text:
            return "", "unknown", "empty"
        if text in self.text_cache:
            return self.text_cache[text]

        language = detect_language_safe(text)
        if language == "en":
            result = (text, language, "english_pass")
            self.text_cache[text] = result
            return result
        if language not in self.translation_models:
            result = (text, language, "translation_unavailable")
            self.text_cache[text] = result
            return result

        translated = self.translate_batch([text], language)[0]
        result = (translated, language, "translated")
        self.text_cache[text] = result
        return result

    def release(self) -> None:
        self.model_cache.clear()
        self.tokenizer_cache.clear()
        self.text_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def preprocess_frame(
    frame: pd.DataFrame,
    cfg: ExperimentConfig,
    logger: Any,
    normalizer: MultilingualNormalizer,
    dataset_name: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["text"] = result["text"].map(clean_text)
    result = result[result["text"].astype(bool)]
    result = result[~result["text"].str.lower().str.startswith(TEXT_PREFIX_BLACKLIST)]
    result = result.drop_duplicates(subset=["text"]).reset_index(drop=True)

    unique_texts = result["text"].drop_duplicates().tolist()
    detected_languages: Dict[str, str] = {}
    for text in tqdm(unique_texts, desc=f"[{dataset_name}] language detection"):
        detected_languages[text] = detect_language_safe(text)

    translated_texts: Dict[str, str] = {text: text for text in unique_texts}
    for lang in cfg.translation_models:
        lang_texts = [text for text in unique_texts if detected_languages[text] == lang]
        if not lang_texts:
            continue
        logger.info("[%s] translating %d rows from %s -> en", dataset_name, len(lang_texts), lang)
        translated = normalizer.translate_batch(lang_texts, lang)
        translated_texts.update(dict(zip(lang_texts, translated)))

    result["detected_language"] = result["text"].map(detected_languages)
    result["normalized_text"] = result["text"].map(lambda text: clean_text(translated_texts.get(text, text)))
    result["normalization_status"] = np.where(
        result["detected_language"].eq("en"),
        "english_pass",
        np.where(result["detected_language"].isin(cfg.translation_models.keys()), "translated", "translation_unavailable"),
    )
    result = result[result["normalized_text"].astype(bool)].reset_index(drop=True)
    return result


def build_balanced_d2_benign_raw(d1_raw: pd.DataFrame, d2_harmful_raw: pd.DataFrame, seed: int) -> pd.DataFrame:
    benign_pool = d1_raw[d1_raw["label"] == 0].copy()
    if len(benign_pool) < len(d2_harmful_raw):
        raise ValueError("D1 benign pool is smaller than D2 harmful pool; cannot balance external held-out set.")
    return benign_pool.sample(n=len(d2_harmful_raw), random_state=seed).copy()


def finalize_external_heldout(
    d1_clean: pd.DataFrame,
    d2_harmful_clean: pd.DataFrame,
    reserved_benign_ids: Sequence[str],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    benign_pool = d1_clean[d1_clean["label"] == 0].copy()
    preferred = benign_pool[benign_pool["sample_id"].isin(set(reserved_benign_ids))].copy()
    preferred = preferred.drop_duplicates(subset=["normalized_text"]).reset_index(drop=True)

    target_size = len(d2_harmful_clean)
    if len(preferred) >= target_size:
        d2_benign = preferred.sample(n=target_size, random_state=seed).copy()
    else:
        remaining = benign_pool[~benign_pool["normalized_text"].isin(set(preferred["normalized_text"]))].copy()
        top_up = remaining.sample(n=target_size - len(preferred), random_state=seed).copy()
        d2_benign = pd.concat([preferred, top_up], ignore_index=True)

    d2_benign["source"] = "D1_benign_for_D2"
    reserved_benign_texts = set(d2_benign["normalized_text"])

    d1_internal = d1_clean[~d1_clean["normalized_text"].isin(reserved_benign_texts)].copy().reset_index(drop=True)
    d2_external = (
        pd.concat([d2_harmful_clean, d2_benign], ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return d1_internal, d2_external, d2_benign


def stratified_split_d1(frame: pd.DataFrame, cfg: ExperimentConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        frame,
        test_size=(1.0 - cfg.train_size),
        stratify=frame["label"],
        random_state=cfg.seed,
    )
    relative_test_size = cfg.test_size / (cfg.val_size + cfg.test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["label"],
        random_state=cfg.seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
