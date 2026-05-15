#!/usr/bin/env python3
"""
sentra_guard_runner.py

Sentra-Guard End-to-End Pipeline Runner.

Orchestrates all 20 pipeline phases sequentially:
  Phase 1  — Environment Initialization
  Phase 2  — Dataset Loading
  Phase 3  — Data Cleaning and Normalization
  Phase 4  — Multilingual Translation
  Phase 5  — Dataset Preparation
  Phase 6  — Tokenization
  Phase 7  — Transformer Classifier Training
  Phase 8  — SBERT Embedding Generation
  Phase 9  — FAISS Retrieval Index Construction
  Phase 10 — Zero-Shot NLI Inference
  Phase 11 — Hybrid Risk Fusion
  Phase 12 — Threshold Optimization
  Phase 13 — Prediction Pipeline
  Phase 14 — Human-in-the-Loop Logic
  Phase 15 — Evaluation Metrics
  Phase 16 — ROC / PR / Heatmap Visualization
  Phase 17 — Ablation Analysis
  Phase 18 — Artifact Export
  Phase 19 — Final Inference Pipeline
  Phase 20 — End-to-End Runner Execution
"""

import importlib
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Phase 1 — Environment Initialization
# ─────────────────────────────────────────────

REQUIRED_PACKAGES = {
    "transformers": "transformers",
    "sentence_transformers": "sentence-transformers",
    "faiss": "faiss-cpu",
    "datasets": "datasets",
    "accelerate": "accelerate",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "tqdm": "tqdm",
    "langdetect": "langdetect",
    "sentencepiece": "sentencepiece",
    "sacremoses": "sacremoses",
}

missing_packages = []
for module_name, package_name in REQUIRED_PACKAGES.items():
    try:
        importlib.import_module(module_name)
    except ImportError:
        missing_packages.append(package_name)

if missing_packages:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-U", *missing_packages]
    )

import torch

from utils.config import build_config, build_artifact_paths
from utils.seed import set_global_seeds


def main() -> None:
    print("\n" + "=" * 60)
    print("  Sentra-Guard Pipeline Runner")
    print("=" * 60 + "\n")

    # ─────────────────────────────────────────────
    # Phase 1 — Configuration and Reproducibility
    # ─────────────────────────────────────────────
    print("[Phase 1] Initializing configuration and reproducibility...")
    config = build_config()
    artifact_paths = build_artifact_paths(config)
    checkpoint_dir = artifact_paths["checkpoint_dir"]
    set_global_seeds(config)
    print(f"  Device: {config.device}")
    print(f"  Random seed: {config.random_seed}")
    print(f"  Classifier: {config.classifier_model_name}")

    # ─────────────────────────────────────────────
    # Phase 2 — Dataset Loading
    # ─────────────────────────────────────────────
    print("\n[Phase 2] Loading JailBreakV-28K dataset...")
    from preprocessing.data_preparation import load_raw_dataset, profile_dataset
    raw_df = load_raw_dataset(config)
    profile_dataset(raw_df)

    # ─────────────────────────────────────────────
    # Phase 3, 4, 5 — Preparation, Translation, Splitting
    # ─────────────────────────────────────────────
    print("\n[Phase 3-5] Splitting and preparing dataset splits...")
    from preprocessing.data_preparation import (
        split_dataset,
        verify_dataset_splits,
        clean_training_dataframe,
    )
    (
        train_df, val_df, test_df,
        jbv_text_df, train_raw_df, val_raw_df, test_raw_df,
    ) = split_dataset(raw_df, config)
    verify_dataset_splits(
        jbv_text_df, train_raw_df, val_raw_df, test_raw_df,
        train_df, val_df, test_df,
    )

    print("\n[Phase 4] Applying multilingual normalization and translation...")
    from preprocessing.translation import (
        build_translator_manager,
        preprocess_dataframe,
    )
    translator_manager = build_translator_manager(config)
    train_df = preprocess_dataframe(train_df, translator_manager, config)
    val_df = preprocess_dataframe(val_df, translator_manager, config)
    test_df = preprocess_dataframe(test_df, translator_manager, config)

    print("\n[Phase 5] Cleaning training DataFrames...")
    train_df = clean_training_dataframe(train_df, "train_df")
    val_df = clean_training_dataframe(val_df, "val_df")
    test_df = clean_training_dataframe(test_df, "test_df")

    # ─────────────────────────────────────────────
    # Phase 6 — Tokenization
    # ─────────────────────────────────────────────
    print("\n[Phase 6] Building tokenizer and DataLoaders...")
    from models.classifier_model import build_classifier_tokenizer
    from preprocessing.tokenization import build_text_only_dataloader

    classifier_tokenizer = build_classifier_tokenizer(config)

    train_loader = build_text_only_dataloader(
        train_df, classifier_tokenizer, config, shuffle=True, batch_size=8
    )
    val_loader = build_text_only_dataloader(
        val_df, classifier_tokenizer, config, shuffle=False, batch_size=8
    )
    test_loader = build_text_only_dataloader(
        test_df, classifier_tokenizer, config, shuffle=False, batch_size=8
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # ─────────────────────────────────────────────
    # Phase 7 — Transformer Classifier Training
    # ─────────────────────────────────────────────
    print("\n[Phase 7] Training transformer classifier...")
    import torch.nn as nn
    from models.classifier_model import build_classifier_model
    from scripts.train import (
        compute_safe_class_weights,
        SafeCrossEntropyLoss,
        build_optimizer_and_scheduler,
        run_training_loop,
    )

    config.learning_rate = 1e-5
    classifier_model = build_classifier_model(config, classifier_tokenizer)
    class_weights = compute_safe_class_weights(
        train_df["label"].tolist(), config.device
    )
    criterion = SafeCrossEntropyLoss(class_weights)

    optimizer, scheduler, total_steps, warmup_steps = build_optimizer_and_scheduler(
        classifier_model, train_loader, config
    )
    classifier_model, training_history_df = run_training_loop(
        classifier_model, classifier_tokenizer,
        train_loader, val_loader,
        criterion, optimizer, scheduler,
        config, checkpoint_dir,
    )

    # ─────────────────────────────────────────────
    # Phase 8, 9 — Retrieval Embedding and FAISS Index
    # ─────────────────────────────────────────────
    print("\n[Phase 8] Loading retrieval encoder...")
    from models.retrieval_model import build_retrieval_model
    retrieval_model = build_retrieval_model(config)

    print("\n[Phase 9] Building FAISS retrieval index...")
    from scripts.retrieval import build_retrieval_corpus, build_faiss_index
    retrieval_corpus_df, retrieval_corpus_embeddings = build_retrieval_corpus(
        train_df, retrieval_model, config
    )
    faiss_index, retrieval_index_embeddings = build_faiss_index(
        retrieval_corpus_embeddings
    )

    # ─────────────────────────────────────────────
    # Phase 10 — Zero-Shot NLI Initialization
    # ─────────────────────────────────────────────
    print("\n[Phase 10] Loading zero-shot NLI classifier...")
    from models.zero_shot_model import build_zero_shot_classifier
    zero_shot_classifier = build_zero_shot_classifier(config)

    # ─────────────────────────────────────────────
    # Phase 11, 12 — Fusion and Threshold Optimization
    # ─────────────────────────────────────────────
    print("\n[Phase 11-12] Optimizing fusion weights and decision threshold...")
    from models.classifier_model import compute_classifier_scores
    from models.zero_shot_model import compute_zero_shot_scores
    from scripts.retrieval import compute_retrieval_scores
    from scripts.thresholding import optimize_threshold, optimize_fusion_weights
    from utils.helpers import _get_text_series

    val_classifier_scores = compute_classifier_scores(
        _get_text_series(val_df).tolist(), classifier_model, classifier_tokenizer, config
    )
    val_retrieval_scores = compute_retrieval_scores(
        _get_text_series(val_df).tolist(), retrieval_model, faiss_index, config
    )
    val_zero_shot_scores = compute_zero_shot_scores(
        _get_text_series(val_df).tolist(), zero_shot_classifier,
        batch_size=config.zero_shot_batch_size,
    )

    best_threshold, threshold_search_df = optimize_threshold(
        val_classifier_scores, val_retrieval_scores, val_zero_shot_scores,
        val_df["label"].tolist(), config,
    )
    best_weights, best_weight_threshold, weight_search_df = optimize_fusion_weights(
        val_classifier_scores, val_retrieval_scores, val_zero_shot_scores,
        val_df["label"].tolist(), config,
    )

    # ─────────────────────────────────────────────
    # Phase 13 — Prediction Pipeline
    # ─────────────────────────────────────────────
    print("\n[Phase 13] Scoring validation and test splits...")
    from scripts.inference import score_split
    val_scored = score_split(
        val_df, "validation",
        classifier_model, classifier_tokenizer,
        retrieval_model, faiss_index, zero_shot_classifier, config,
    )
    test_scored = score_split(
        test_df, "test",
        classifier_model, classifier_tokenizer,
        retrieval_model, faiss_index, zero_shot_classifier, config,
    )

    # ─────────────────────────────────────────────
    # Phase 14 — Human-in-the-Loop Memory Update
    # ─────────────────────────────────────────────
    print("\n[Phase 14] Initializing HITL adaptive memory updater...")
    from scripts.hitl import update_adversarial_memory
    hitl_result = update_adversarial_memory(
        new_harmful_prompts=[],
        retrieval_corpus_df=retrieval_corpus_df,
        retrieval_corpus_embeddings=retrieval_corpus_embeddings,
        retrieval_index_embeddings=retrieval_index_embeddings,
        faiss_index=faiss_index,
        retrieval_model=retrieval_model,
    )
    retrieval_corpus_df = hitl_result["retrieval_corpus_df"]
    retrieval_corpus_embeddings = hitl_result["retrieval_corpus_embeddings"]
    retrieval_index_embeddings = hitl_result["retrieval_index_embeddings"]
    faiss_index = hitl_result["faiss_index"]
    print(f"  HITL updater ready. Added: {hitl_result['added_count']}")

    # ─────────────────────────────────────────────
    # Phase 15, 17 — Evaluation and Ablation
    # ─────────────────────────────────────────────
    print("\n[Phase 15, 17] Running evaluation and ablation study...")
    from scripts.evaluate import run_full_evaluation
    final_metrics_df, baseline_results_df, ablation_df = run_full_evaluation(
        test_df, test_scored,
        classifier_model, classifier_tokenizer,
        retrieval_model, faiss_index, zero_shot_classifier, config,
    )
    sentra_guard_report = final_metrics_df.to_dict(orient="records")[0]

    # ─────────────────────────────────────────────
    # Phase 16 — Visualization
    # ─────────────────────────────────────────────
    print("\n[Phase 16] Generating visualizations...")
    from scripts.visualization import (
        plot_confusion_matrix,
        plot_training_and_evaluation_curves,
    )
    plot_confusion_matrix(sentra_guard_report["confusion_matrix"])
    plot_training_and_evaluation_curves(training_history_df, test_scored)

    # ─────────────────────────────────────────────
    # Phase 18 — Artifact Export
    # ─────────────────────────────────────────────
    print("\n[Phase 18] Exporting reproducibility artifacts...")
    from scripts.export_results import export_all_artifacts
    export_all_artifacts(
        test_scored, ablation_df, baseline_results_df,
        sentra_guard_report, training_history_df,
        config, checkpoint_dir,
    )

    # ─────────────────────────────────────────────
    # Phase 19 — Final Demo Inference
    # ─────────────────────────────────────────────
    print("\n[Phase 19] Running demo inference...")
    from scripts.inference import run_demo_inference
    run_demo_inference(
        classifier_model, classifier_tokenizer,
        retrieval_model, faiss_index, zero_shot_classifier, config,
    )

    print("\n" + "=" * 60)
    print("  Sentra-Guard Pipeline Complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
