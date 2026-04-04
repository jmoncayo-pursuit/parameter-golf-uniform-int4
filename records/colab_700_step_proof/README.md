# Colab 700-Step Proof Run (T4, Seed 42) — LZMA + 5% Pruning

**Date:** 2026-04-04
**Hardware:** Google Colab T4 GPU (1×)
**Branch:** main (commit 84fc178+)

## Results — PASS ✅

| Metric | Value |
|--------|-------|
| Steps | 700/700 |
| Final train_loss | 3.1889 |
| val_loss (pre-quant) | 3.0707 |
| val_bpb (pre-quant) | 1.8156 |
| **Roundtrip val_loss** | **3.0443** |
| **Roundtrip val_bpb** | **1.8008** |
| Peak memory | 3834 MiB allocated, 5120 MiB reserved |
| Wall-clock | 25.2 minutes |
| Step avg | 2070.04 ms |
| SWA checkpoints | 2 |
| Exit code | **0** |

## Artifact Size — PASS ✅

| Component | Size |
|-----------|------|
| final_model.pt (raw) | 119,446,367 bytes |
| mixed_int6_int4_lzma | 15,859,108 bytes |
| train_gpt.py | 72,149 bytes |
| **Total submission** | **15,931,257 bytes** |
| **Under 16MB?** | **YES ✅ (69KB headroom)** |

## Comparison: Previous Run (zstd + 3% pruning) vs This Run

| Metric | Before (zstd) | After (lzma + 5%) |
|--------|--------------|-------------------|
| Compressed model | 16,232,814 | 15,859,108 |
| Total artifact | 16,305,344 ❌ | 15,931,257 ✅ |
| Savings | — | 374,087 bytes |
| 16MB check | FAIL | **PASS** |
| Roundtrip BPB | (crashed) | **1.8008** |
| Exit code | 1 | **0** |

## What Changed

1. Default compressor: `zstd` → `lzma` (better compression ratio)
2. Magnitude pruning: 3% → 5% (more zeros improve compressibility)
3. Clean `CHECKPOINT_ONLY` gate in `main()` (no more `sys.exit(0)` hacks)

## Status

This run proves the full end-to-end pipeline works:
- ✅ Training completes (700 steps, healthy loss curve)
- ✅ SWA applies
- ✅ Serialization + quantization + compression
- ✅ Artifact fits under 16MB with 69KB headroom
- ✅ Roundtrip eval completes with BPB = 1.8008
- ✅ `final_summary.json` + `final_summary.md` written
- ✅ Exit code 0
