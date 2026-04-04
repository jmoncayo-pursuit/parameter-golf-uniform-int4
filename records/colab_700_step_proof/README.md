# Colab 700-Step Proof Run (T4, Seed 42)

**Date:** 2026-04-04
**Hardware:** Google Colab T4 GPU (1×)
**Branch:** main

## Results

| Metric | Value |
|--------|-------|
| Steps | 700/700 |
| Final train_loss | 3.1889 |
| val_loss (pre-quant) | 3.0707 |
| val_bpb (pre-quant) | 1.8156 |
| Peak memory | 3834 MiB allocated, 5120 MiB reserved |
| Wall-clock | 24.8 minutes |
| Step avg | 2074.00 ms |
| SWA checkpoints averaged | 2 |

## Artifact Size (with zstd — **FAILED 16MB limit**)

| Component | Size |
|-----------|------|
| final_model.pt (raw) | 119,446,367 bytes |
| mixed_int6_int4_zstd | 16,232,814 bytes |
| train_gpt.py | 72,530 bytes |
| **Total** | **16,305,344 bytes** |
| **Verdict** | ❌ Over 16MB limit by 305,344 bytes |

## Root Cause & Fix

The artifact exceeded 16MB because:
1. Default compressor was `zstd` — now changed to `lzma` (10-20% better compression)
2. Magnitude pruning was 3% — now increased to 5% (more zeros = better compressibility)

These two fixes (committed after this run) should bring the artifact comfortably under 16MB.

## Status

- **Training pipeline:** ✅ Proven end-to-end
- **Artifact size:** ❌ Over limit (fix committed, re-run needed)
- **Loss curve:** ✅ Healthy downtrend (6.93 → 3.19)
