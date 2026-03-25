#!/bin/bash
# H100 Baseline Execution (Int4 + Bayesian Cache)
# Commit: b735e22

# Hyperparameters can be overridden here or via env vars
export RUN_ID=${RUN_ID:-main_cache_lzma_baseline}
export EVAL_CACHE=${EVAL_CACHE:-1}
export COMPRESSOR=${COMPRESSOR:-lzma}
export VAL_LOSS_EVERY=${VAL_LOSS_EVERY:-0}
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-600}

# Execution via torchrun (Distributed Training + Evaluation)
torchrun --standalone --nproc_per_node=8 train_gpt.py
