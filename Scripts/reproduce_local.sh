#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

: "${D1_PATH:?Set D1_PATH to a local HarmBench-format file.}"
: "${D2_PATH:?Set D2_PATH to a local JailBreakV-format file.}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts}"
ZERO_SHOT_RUNTIME_OPTION="${ZERO_SHOT_RUNTIME_OPTION:-default}"

python -m sentra_guard.run_experiments \
  --dataset-source local \
  --d1-path "${D1_PATH}" \
  --d2-path "${D2_PATH}" \
  --zero-shot-runtime-option "${ZERO_SHOT_RUNTIME_OPTION}" \
  --output-dir "${OUTPUT_DIR}"
