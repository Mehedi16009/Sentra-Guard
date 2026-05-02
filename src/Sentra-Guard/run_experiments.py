from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from sentra_guard.config import ExperimentConfig, create_experiment_logger, set_global_determinism
    from sentra_guard.data import (
        MultilingualNormalizer,
        build_balanced_d2_benign_raw,
        finalize_external_heldout,
        load_d1_harmbench_frame,
        load_d2_jailbreakv_frame,
        preprocess_frame,
        stratified_split_d1,
    )
    from sentra_guard.evaluate import (
        build_confusion_dataframe,
        evaluate_scored_frame,
        finalize_predictions,
        run_ablation_study,
        run_threshold_sweep,
        run_weight_search,
    )
    from sentra_guard.inference import OptimizedZeroShotScorer, cleanup_memory, score_split
    from sentra_guard.retrieval import SemanticRetriever
    from sentra_guard.train import TransformerHarmClassifier
else:
    from .config import ExperimentConfig, create_experiment_logger, set_global_determinism
    from .data import (
        MultilingualNormalizer,
        build_balanced_d2_benign_raw,
        finalize_external_heldout,
        load_d1_harmbench_frame,
        load_d2_jailbreakv_frame,
        preprocess_frame,
        stratified_split_d1,
    )
    from .evaluate import (
        build_confusion_dataframe,
        evaluate_scored_frame,
        finalize_predictions,
        run_ablation_study,
        run_threshold_sweep,
        run_weight_search,
    )
    from .inference import OptimizedZeroShotScorer, cleanup_memory, score_split
    from .retrieval import SemanticRetriever
    from .train import TransformerHarmClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Sentra-Guard manuscript-faithful reproduction pipeline.")
    parser.add_argument("--dataset-source", default="huggingface", choices=["huggingface", "local"])
    parser.add_argument("--d1-dataset-name", default="YinkaiW/harmbench-dataset")
    parser.add_argument("--d2-dataset-name", default="JailbreakV-28K/JailBreakV-28k")
    parser.add_argument("--d2-dataset-config", default="JailBreakV_28K")
    parser.add_argument("--d1-path")
    parser.add_argument("--d2-path")
    parser.add_argument("--output-dir", default="./artifacts")
    parser.add_argument("--experiment-name", default="sentra_guard_reproduction")
    parser.add_argument("--classifier-model-name", default="distilbert-base-uncased")
    parser.add_argument("--sbert-model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--zero-shot-model-name", default="facebook/bart-large-mnli")
    parser.add_argument("--zero-shot-runtime-option", default="default", choices=["default", "fast"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--theta", type=float, default=0.50)
    parser.add_argument("--delta", type=float, default=0.08)
    parser.add_argument("--weight-grid-step", type=float, default=0.05)
    return parser.parse_args()


def encode_and_log(retriever: SemanticRetriever, frame: pd.DataFrame, batch_size: int, logger: Any, split_name: str) -> np.ndarray:
    logger.info("Encoding %s with SBERT | n=%d", split_name, len(frame))
    return retriever.encode(
        frame["normalized_text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
    )


def run_full_experiment(cfg: ExperimentConfig) -> Dict[str, pd.DataFrame]:
    set_global_determinism(seed=cfg.seed, deterministic=cfg.deterministic)
    logger = create_experiment_logger(cfg.output_dir, cfg.experiment_name)
    logger.info("Starting Sentra-Guard reproduction pipeline")
    cfg.save(cfg.output_root / "config.json")

    d1_raw = load_d1_harmbench_frame(cfg, logger)
    d2_harmful_raw = load_d2_jailbreakv_frame(cfg, logger)
    d2_benign_raw = build_balanced_d2_benign_raw(d1_raw, d2_harmful_raw, cfg.seed)
    reserved_benign_ids = list(d2_benign_raw["sample_id"])

    normalizer = MultilingualNormalizer(
        translation_models=cfg.translation_models,
        device=cfg.device,
        batch_size=cfg.translation_batch_size,
    )
    d1_clean = preprocess_frame(d1_raw, cfg, logger, normalizer, dataset_name="D1")
    d2_harmful_clean = preprocess_frame(d2_harmful_raw, cfg, logger, normalizer, dataset_name="D2_harmful")
    normalizer.release()
    cleanup_memory(logger, "translation complete")

    d1_internal, d2_external, d2_benign_clean = finalize_external_heldout(
        d1_clean=d1_clean,
        d2_harmful_clean=d2_harmful_clean,
        reserved_benign_ids=reserved_benign_ids,
        seed=cfg.seed,
    )

    train_df, val_df, test_df = stratified_split_d1(d1_internal, cfg)
    for name, frame in (("d1_clean", d1_clean), ("d1_internal", d1_internal), ("d1_train", train_df), ("d1_val", val_df), ("d1_test", test_df), ("d2_external", d2_external)):
        frame.to_csv(cfg.cache_dir / f"{name}.csv", index=False)
        logger.info("%s | n=%d | label_dist=%s", name, len(frame), frame["label"].value_counts().to_dict())

    classifier = TransformerHarmClassifier(cfg)
    training_history = classifier.fit(train_df, val_df, logger)
    training_history.to_csv(cfg.output_root / "training_logs.csv", index=False)

    retriever = SemanticRetriever(
        model_name=cfg.sbert_model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    train_embeddings = encode_and_log(retriever, train_df, cfg.embedding_batch_size, logger, "D1_train")
    val_embeddings = encode_and_log(retriever, val_df, cfg.embedding_batch_size, logger, "D1_val")
    test_embeddings = encode_and_log(retriever, test_df, cfg.embedding_batch_size, logger, "D1_test")
    d2_embeddings = encode_and_log(retriever, d2_external, cfg.embedding_batch_size, logger, "D2_heldout")

    np.save(cfg.retrieval_dir / "train_embeddings.npy", train_embeddings)
    knowledge_base = retriever.build_knowledge_base(
        embeddings=train_embeddings,
        texts=train_df["normalized_text"].tolist(),
        labels=train_df["label"].tolist(),
        languages=train_df["detected_language"].tolist(),
        sample_ids=train_df["sample_id"].tolist(),
        sources=train_df["source"].tolist(),
    )
    knowledge_base.save(cfg.retrieval_dir)

    zero_shot_scorer = OptimizedZeroShotScorer(cfg, logger)

    val_scored, val_profile = score_split(
        frame=val_df,
        embeddings=val_embeddings,
        split_name="D1_val",
        cfg=cfg,
        classifier=classifier,
        knowledge_base=knowledge_base,
        zero_shot_scorer=zero_shot_scorer,
        logger=logger,
    )
    cleanup_memory(logger, "validation complete")

    test_scored, test_profile = score_split(
        frame=test_df,
        embeddings=test_embeddings,
        split_name="D1_test",
        cfg=cfg,
        classifier=classifier,
        knowledge_base=knowledge_base,
        zero_shot_scorer=zero_shot_scorer,
        logger=logger,
    )
    cleanup_memory(logger, "test complete")

    d2_scored, d2_profile = score_split(
        frame=d2_external,
        embeddings=d2_embeddings,
        split_name="D2_heldout",
        cfg=cfg,
        classifier=classifier,
        knowledge_base=knowledge_base,
        zero_shot_scorer=zero_shot_scorer,
        logger=logger,
    )
    cleanup_memory(logger, "external held-out complete")

    pd.DataFrame([val_profile, test_profile, d2_profile]).to_csv(cfg.output_root / "runtime_profiles.csv", index=False)
    val_scored.to_csv(cfg.cache_dir / "val_scored_default.csv", index=False)
    test_scored.to_csv(cfg.cache_dir / "test_scored_default.csv", index=False)
    d2_scored.to_csv(cfg.cache_dir / "d2_scored_default.csv", index=False)

    ablation_df = run_ablation_study({"D1_test": test_scored, "D2_heldout": d2_scored}, cfg)
    ablation_df.to_csv(cfg.output_root / "ablation.csv", index=False)

    threshold_search_df = run_threshold_sweep(val_scored, cfg)
    threshold_search_df.to_csv(cfg.output_root / "threshold_search.csv", index=False)

    weight_search_df, best_weight_row = run_weight_search(val_scored, cfg)
    weight_search_df.to_csv(cfg.output_root / "weight_search.csv", index=False)
    selected_weights = (
        float(best_weight_row["w1"]),
        float(best_weight_row["w2"]),
        float(best_weight_row["w3"]),
    )
    selected_theta = float(best_weight_row["theta"])

    final_test_predictions = finalize_predictions(test_scored, "D1_test", selected_weights, selected_theta, cfg.delta)
    final_d2_predictions = finalize_predictions(d2_scored, "D2_heldout", selected_weights, selected_theta, cfg.delta)
    predictions_df = pd.concat([final_test_predictions, final_d2_predictions], ignore_index=True)
    predictions_df.to_csv(cfg.output_root / "predictions.csv", index=False)

    metrics_df = pd.DataFrame(
        [
            evaluate_scored_frame(final_test_predictions, "D1_test", "full_sentra_guard_optimized"),
            evaluate_scored_frame(final_d2_predictions, "D2_heldout", "full_sentra_guard_optimized"),
        ]
    )
    metrics_df["selected_w1"] = selected_weights[0]
    metrics_df["selected_w2"] = selected_weights[1]
    metrics_df["selected_w3"] = selected_weights[2]
    metrics_df["selected_theta"] = selected_theta
    metrics_df["selected_delta"] = cfg.delta
    metrics_df.to_csv(cfg.output_root / "metrics.csv", index=False)

    confusion_df = build_confusion_dataframe(
        {
            "D1_test": final_test_predictions,
            "D2_heldout": final_d2_predictions,
        }
    )
    confusion_df.to_csv(cfg.output_root / "confusion_matrix.csv", index=False)

    manifest = sorted(str(path.relative_to(cfg.output_root)) for path in cfg.output_root.rglob("*") if path.is_file())
    with open(cfg.output_root / "artifact_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    logger.info("Saved artifacts to %s", cfg.output_root)
    logger.info("Selected weights=%s | theta=%.2f", selected_weights, selected_theta)
    return {
        "metrics": metrics_df,
        "predictions": predictions_df,
        "threshold_search": threshold_search_df,
        "weight_search": weight_search_df,
        "ablation": ablation_df,
        "confusion_matrix": confusion_df,
        "training_logs": training_history,
    }


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        dataset_source=args.dataset_source,
        d1_dataset_name=args.d1_dataset_name,
        d2_dataset_name=args.d2_dataset_name,
        d2_dataset_config=args.d2_dataset_config,
        d1_path=args.d1_path,
        d2_path=args.d2_path,
        classifier_model_name=args.classifier_model_name,
        sbert_model_name=args.sbert_model_name,
        zero_shot_model_name=args.zero_shot_model_name,
        zero_shot_runtime_option=args.zero_shot_runtime_option,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        top_k=args.top_k,
        max_eval_samples=args.max_eval_samples,
        theta=args.theta,
        delta=args.delta,
        weight_grid_step=args.weight_grid_step,
    )
    run_full_experiment(cfg)


if __name__ == "__main__":
    main()
