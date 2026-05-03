# Sentra-Guard

Reproducibility-ready repository for the paper:

**Sentra-Guard: A Real-Time Multilingual Defense Against Adversarial LLM Prompts**

This repository provides the modular implementation of Sentra-Guard, converted from the original Google Colab experimental pipeline into a reproducibility-oriented Python package with deterministic execution, checkpoint persistence, cached preprocessing, and manuscript-faithful evaluation workflows.

The implementation preserves the core manuscript equations:

```text
S_final = w1 · P_C + w2 · R_score + w3 · P_Z

R_score = Σ(sim_i · y_i) / Σ(sim_i)

HITL uncertainty = |S_final − θ| < δ
```

---

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

---

## Installation

Create the reproducibility environment:

```bash
conda env create -f environment.yml
conda activate sentra-guard
```

Alternatively:

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Reproduce experiments using Hugging Face datasets

```bash
bash scripts/reproduce_huggingface.sh
```

### Reproduce experiments using local datasets

```bash
D1_PATH=/absolute/path/to/d1.csv \
D2_PATH=/absolute/path/to/d2.csv \
bash scripts/reproduce_local.sh
```

### Run sanity tests

```bash
bash scripts/run_sanity_tests.sh
```

---

## Main Outputs

All generated artifacts are stored under `artifacts/`.

```text
artifacts/
├── config.json
├── metrics.csv
├── predictions.csv
├── threshold_search.csv
├── weight_search.csv
├── ablation.csv
├── confusion_matrix.csv
├── runtime_profiles.csv
├── training_logs.csv
├── artifact_manifest.json
├── cache/
├── checkpoints/
├── retrieval/
└── logs/
```

---

## Data Sources

Default datasets are loaded from :contentReference[oaicite:1]{index=1}:

- `YinkaiW/harmbench-dataset`
- `JailbreakV-28K/JailBreakV-28k`

See `dataset_notes.md` for:

- schema normalization
- label mapping
- multilingual preprocessing
- external held-out benchmark construction

---

## Reproducibility Notes

- Global random seed is fixed to `42`
- Deterministic PyTorch and CUDA execution are enabled
- Classifier training uses checkpointing and early stopping
- Preprocessed data splits are cached under `artifacts/cache/`
- Retrieval indices and embeddings are persisted for reproducibility
- Zero-shot inference supports both default and optimized runtime models
- Full reproduction instructions are documented in `reproduction.md`

---

## Figures

Manuscript figures are stored under `figures/`, including:

- system architecture
- pipeline overview
- ROC curve
- precision-recall curve
- threshold optimization curve
- confusion matrix

---

## Citation

If you use this repository, please cite:

```bibtex
@article{hasan2025sentra,
  title={Sentra-Guard: A Multilingual Human-AI Framework for Real-Time Defense Against Adversarial LLM Jailbreaks},
  author={Hasan, Md Mehedi and Mehedi, Sk Tanzir and Rahman, Ziaur and Mostafiz, Rafid and Hossain, Md Abir},
  journal={arXiv preprint arXiv:2510.22628},
  year={2025}
}
```

---

## License

This repository is released under the LICENSE included in this repository.
