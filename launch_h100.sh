#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-uniform_int4_$(date +%Y%m%d_%H%M%S)}"
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-0}"
COMPRESSOR="${COMPRESSOR:-lzma}"
EVAL_CACHE="${EVAL_CACHE:-1}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"

export RUN_ID DATA_PATH TOKENIZER_PATH MAX_WALLCLOCK_SECONDS NPROC_PER_NODE
export VAL_MAX_TOKENS COMPRESSOR EVAL_CACHE VAL_LOSS_EVERY

echo "=== Uniform Int4@4.0 (8xH100) ==="
echo "RUN_ID: ${RUN_ID}"
echo "DATA_PATH: ${DATA_PATH}"
echo "TOKENIZER_PATH: ${TOKENIZER_PATH}"
echo "MAX_WALLCLOCK_SECONDS: ${MAX_WALLCLOCK_SECONDS}"
echo "VAL_MAX_TOKENS: ${VAL_MAX_TOKENS}"
echo "COMPRESSOR: ${COMPRESSOR}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "================================="

bash ./preflight_h100.sh
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" train_gpt.py
