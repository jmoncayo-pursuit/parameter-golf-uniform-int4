#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-uniform_int4_$(date +%Y%m%d_%H%M%S)}"
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
EXPECTED_GPU_NAME_SUBSTR="${EXPECTED_GPU_NAME_SUBSTR:-H100}"

echo "=== Uniform Int4@4.0 preflight ==="
echo "RUN_ID: ${RUN_ID}"
echo "DATA_PATH: ${DATA_PATH}"
echo "TOKENIZER_PATH: ${TOKENIZER_PATH}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "MAX_WALLCLOCK_SECONDS: ${MAX_WALLCLOCK_SECONDS}"
echo "=================================="

if [[ ! -f "train_gpt.py" ]]; then
  echo "ERROR: train_gpt.py not found in $(pwd)"
  exit 1
fi

if [[ ! -d "${DATA_PATH}" ]]; then
  echo "ERROR: DATA_PATH directory not found: ${DATA_PATH}"
  exit 1
fi

if [[ ! -f "${TOKENIZER_PATH}" ]]; then
  echo "ERROR: TOKENIZER_PATH file not found: ${TOKENIZER_PATH}"
  exit 1
fi

shopt -s nullglob
train_shards=( "${DATA_PATH}"/fineweb_train_*.bin )
val_shards=( "${DATA_PATH}"/fineweb_val_*.bin )
shopt -u nullglob

if (( ${#train_shards[@]} == 0 )); then
  echo "ERROR: No training shards found under ${DATA_PATH}"
  exit 1
fi

if (( ${#val_shards[@]} == 0 )); then
  echo "ERROR: No validation shards found under ${DATA_PATH}"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is not available"
  exit 1
fi

python3 -m py_compile train_gpt.py

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is not available"
  exit 1
fi

gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "${gpu_count}" != "${NPROC_PER_NODE}" ]]; then
  echo "ERROR: Expected ${NPROC_PER_NODE} visible GPUs, found ${gpu_count}"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv
  exit 1
fi

if [[ -n "${EXPECTED_GPU_NAME_SUBSTR}" ]]; then
  if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "${EXPECTED_GPU_NAME_SUBSTR}"; then
    echo "ERROR: GPU name does not include expected substring: ${EXPECTED_GPU_NAME_SUBSTR}"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    exit 1
  fi
fi

if [[ "${MAX_WALLCLOCK_SECONDS}" != "0" ]]; then
  echo "WARNING: MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS} so training may stop before ITERATIONS"
fi

echo "Python syntax: OK"
echo "Train shards: ${#train_shards[@]}"
echo "Val shards: ${#val_shards[@]}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo "Preflight: OK"
