#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-uniform_int4_cache_lzma_baseline}"
EVAL_CACHE="${EVAL_CACHE:-1}"
COMPRESSOR="${COMPRESSOR:-lzma}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-0}"

export RUN_ID EVAL_CACHE COMPRESSOR VAL_LOSS_EVERY MAX_WALLCLOCK_SECONDS VAL_MAX_TOKENS

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" train_gpt.py
