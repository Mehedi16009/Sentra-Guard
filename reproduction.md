# Reproduction Guide

This document describes the exact execution path used by the modular Sentra-Guard repository.

## 1. Environment Setup

### Conda

```bash
conda env create -f environment.yml
conda activate sentra-guard
```

### Pip

```bash
python -m pip install -r requirements.txt
```

## 2. Repository Root

Run all commands from the repository root:

```bash
cd sentra-guard
```

## 3. Hugging Face Reproduction

The default experiment uses:

- D1: `YinkaiW/harmbench-dataset`
- D2: `JailbreakV-28K/JailBreakV-28k`

Execute:

```bash
bash scripts/reproduce_huggingface.sh
```

Equivalent direct CLI:

```bash
PYTHONPATH=src python -m sentra_guard.run_experiments \
  --dataset-source huggingface \
  --d1-dataset-name YinkaiW/harmbench-dataset \
  --d2-dataset-name JailbreakV-28K/JailBreakV-28k \
  --d2-dataset-config JailBreakV_28K \
  --output-dir ./artifacts
```

## 4. Local Fallback

If you have local copies of the datasets:

```bash
D1_PATH=/absolute/path/to/d1.csv \
D2_PATH=/absolute/path/to/d2.csv \
bash scripts/reproduce_local.sh
```

Equivalent direct CLI:

```bash
PYTHONPATH=src python -m sentra_guard.run_experiments \
  --dataset-source local \
  --d1-path /absolute/path/to/d1.csv \
  --d2-path /absolute/path/to/d2.csv \
  --output-dir ./artifacts
```

## 5. Experimental Logic Preserved

The repository preserves the manuscript-level decision logic exactly:

- `S_final = w1 * P_C + w2 * R_score + w3 * P_Z`
- `R_score = Σ(sim_i * y_i) / Σ(sim_i)`
- `harmful if S_final >= theta`
- `HITL uncertainty if abs(S_final - theta) < delta`

Default experimental settings:

- seed: `42`
- train/validation/test split: `0.70 / 0.15 / 0.15`
- classifier: `distilbert-base-uncased`
- retriever: `sentence-transformers/all-MiniLM-L6-v2`
- zero-shot: `facebook/bart-large-mnli`
- top-k retrieval: `5`
- weights: `w1=0.50`, `w2=0.25`, `w3=0.25`
- threshold: `theta=0.50`
- HITL margin: `delta=0.08`

## 6. Produced Artifacts

After a successful run, `artifacts/` will contain:

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

## 7. Sanity Tests

Run lightweight, deterministic checks:

```bash
bash scripts/run_sanity_tests.sh
```

These tests do not download large models. They validate:

- fusion equation behavior
- HITL uncertainty mask
- retrieval score equation
- HarmBench label mapping
- D2 text-jailbreak extraction
- weight grid constraints

## 8. Notes for ACM-Style Artifact Review

- All outputs are written to a single `artifacts/` root for traceability.
- The saved `config.json` captures the full runtime configuration.
- Classifier checkpoints are saved under `artifacts/checkpoints/`.
- Retrieval artifacts are saved under `artifacts/retrieval/`.
- Cached processed splits are saved under `artifacts/cache/`.
- Deterministic seeds are configured centrally in `src/sentra_guard/config.py`.
