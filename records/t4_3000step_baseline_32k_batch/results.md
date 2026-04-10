# T4 3,000-Step Baseline (32k Batch) — True Zero Control Run

**Date**: 2026-04-09
**Hardware**: Google Colab T4 GPU (14.56 GiB)
**Commit**: `787c21b` (train_gpt.py with all T4 stability patches)

## Configuration
| Setting | Value |
|---|---|
| train_batch_tokens | 32,768 |
| train_seq_len | 1,024 |
| iterations | 3,000 |
| warmup_steps | 20 |
| warmdown_iters | 500 |
| val_loss_every | 125 |
| val_max_tokens | 65,536 |
| val_batch_size | 32,768 |
| seed | 42 |
| model_params | 30,760,017 |
| NO_COMPILE | 1 |

## Final Results
| Metric | Value |
|---|---|
| **Final val_bpb** | **1.4799** |
| **Roundtrip val_bpb (quantized)** | **1.4530** |
| Final val_loss | 2.5029 |
| Roundtrip val_loss | 2.4549 |
| Total train time | 13,447s (3h 44m) |
| Step avg | 4,482ms |
| SWA checkpoints | 4 (started at step 2850) |

## Validation BPB Curve
| Step | val_bpb | Notes |
|------|---------|-------|
| 125 | 2.3638 | Early descent |
| 250 | 2.0403 | |
| 375 | 1.9078 | |
| 500 | 1.7783 | |
| 625 | 1.7536 | |
| 750 | 1.7214 | |
| 875 | 1.6979 | |
| 1000 | 1.6652 | |
| 1125 | 1.6596 | Plateau begins |
| 1250 | 1.6604 | |
| 1375 | 1.6905 | |
| 1500 | 1.6535 | |
| 1625 | 1.6360 | |
| 1750 | 1.6314 | |
| 1875 | 1.6165 | |
| 2000 | 1.6214 | |
| 2125 | 1.5997 | |
| 2250 | 1.5987 | |
| 2375 | 1.5911 | |
| 2500 | 1.5910 | Warmdown begins |
| 2625 | 1.5424 | Warmdown + QAT ramp |
| 2750 | 1.5157 | |
| 2875 | 1.4985 | SWA starts |
| 3000 | 1.4799 | Final |

## Key Observations
1. BPB plateaued ~1.62 from steps 1000-2500 due to 8x smaller batch (32k vs 262k)
2. Warmdown phase (2500-3000) provided massive final push: 1.59 → 1.48
3. SWA averaging (4 checkpoints from step 2850) improved roundtrip by 0.027 BPB
4. Quantized roundtrip (1.453) matches the original Kaggle 1.45 record
