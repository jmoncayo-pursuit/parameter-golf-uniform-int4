#!/bin/bash
set -euo pipefail
# H100 baseline for Uniform Int4@4.0 — QAT Int4 MLP + Int6 GPS attention; Bayesian cache eval path
# -----------------------------------------------------------------------------
# Traceability: Resolve current commit at runtime
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    export GIT_COMMIT=$(git rev-parse --short HEAD)
    echo "H100 Execution Start | Commit: $GIT_COMMIT"
else
    echo "H100 Execution Start | (Not in a Git repo)"
fi
# -----------------------------------------------------------------------------

# Hyperparameters can be overridden here or via env vars
export RUN_ID=${RUN_ID:-uniform_int4_cache_lzma_baseline}
export EVAL_CACHE=${EVAL_CACHE:-1}
export COMPRESSOR=${COMPRESSOR:-lzma}
export VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-0}
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-0}
export VAL_MAX_TOKENS=${VAL_MAX_TOKENS:-0}

# Execution via torchrun (Distributed Training + Evaluation)
torchrun --standalone --nproc_per_node=8 train_gpt.py
