# Sentra-Guard

Reproducibility-ready repository for the paper **Sentra-Guard: A Real-Time Multilingual Defense Against Adversarial LLM Prompts**.

This repository converts the Colab implementation into a modular Python package with deterministic execution, saved checkpoints, cached preprocessing outputs, and manuscript-faithful experiment scripts. The pipeline preserves the core equations used in the paper:

- `S_final = w1 * P_C + w2 * R_score + w3 * P_Z`
- `R_score = Σ(sim_i * y_i) / Σ(sim_i)`
- `HITL uncertainty = abs(S_final - theta) < delta`

## Repository Layout

```text
sentra-guard/
├── README.md
├── requirements.txt
├── environment.yml
├── reproduction.md
├── dataset_notes.md
├── .gitignore
│
├── figures/
│   ├── architecture.png
│   ├── pipeline_overview.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── threshold_sweep.png
│   └── confusion_matrix.png
│
├── artifacts/
│   └── .gitkeep
│
├── scripts/
│   ├── reproduce_huggingface.sh
│   ├── reproduce_local.sh
│   └── run_sanity_tests.sh
│
├── src/
│   └── sentra_guard/
│       ├── __init__.py
│       ├── __main__.py
│       ├── config.py
│       ├── data.py
│       ├── train.py
│       ├── retrieval.py
│       ├── fusion.py
│       ├── inference.py
│       ├── evaluate.py
│       └── run_experiments.py
│
└── tests/
    ├── conftest.py
    └── test_sanity.py
```

## Quick Start

Create the environment:

```bash
conda env create -f environment.yml
conda activate sentra-guard
```

Run the Hugging Face reproduction pipeline:

```bash
bash scripts/reproduce_huggingface.sh
```

Run the local fallback:

```bash
D1_PATH=/absolute/path/to/d1.csv \
D2_PATH=/absolute/path/to/d2.csv \
bash scripts/reproduce_local.sh
```

Run sanity checks:

```bash
bash scripts/run_sanity_tests.sh
```

## Main Outputs

All run artifacts are written under `artifacts/`:

- `config.json`
- `metrics.csv`
- `predictions.csv`
- `threshold_search.csv`
- `weight_search.csv`
- `ablation.csv`
- `confusion_matrix.csv`
- `runtime_profiles.csv`
- `training_logs.csv`
- `artifact_manifest.json`
- `cache/`
- `checkpoints/`
- `retrieval/`
- `logs/`

## Data Sources

Default Hugging Face datasets:

- `YinkaiW/harmbench-dataset`
- `JailbreakV-28K/JailBreakV-28k`

See [dataset_notes.md](./dataset_notes.md) for the normalization schema, label mapping, and the D2 held-out construction logic.

## Reproducibility Notes

- Global seed is fixed to `42`.
- Deterministic PyTorch and CUDA settings are enabled.
- Early stopping and checkpoint saving are enabled for the classifier.
- Preprocessed splits and scored outputs are saved to `artifacts/cache/`.
- The zero-shot branch supports both the default model and a faster runtime option while preserving the same NLI scoring logic.

The exact execution recipe is documented in [reproduction.md](./reproduction.md).
