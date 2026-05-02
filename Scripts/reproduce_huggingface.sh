#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

D1_DATASET_NAME="${D1_DATASET_NAME:-YinkaiW/harmbench-dataset}"
D2_DATASET_NAME="${D2_DATASET_NAME:-JailbreakV-28K/JailBreakV-28k}"
D2_DATASET_CONFIG="${D2_DATASET_CONFIG:-JailBreakV_28K}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts}"
ZERO_SHOT_RUNTIME_OPTION="${ZERO_SHOT_RUNTIME_OPTION:-default}"

python -m sentra_guard.run_experiments \
  --dataset-source huggingface \
  --d1-dataset-name "${D1_DATASET_NAME}" \
  --d2-dataset-name "${D2_DATASET_NAME}" \
  --d2-dataset-config "${D2_DATASET_CONFIG}" \
  --zero-shot-runtime-option "${ZERO_SHOT_RUNTIME_OPTION}" \
  --output-dir "${OUTPUT_DIR}"
