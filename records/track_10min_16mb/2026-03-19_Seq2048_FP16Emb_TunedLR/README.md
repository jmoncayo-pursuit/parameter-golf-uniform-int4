# 10L Seq2048 + FP16 Emb + Int6 Mid + Sliding Window

## Summary

Seven changes stacked on the Naive Baseline:

1. **10 transformer layers** (from 9): Extra capacity from an additional transformer block.

2. **Sequence length 2048** (from 1024): Longer context per training example improves modeling quality despite fewer total steps (~9,100 vs ~12,100).

3. **FP16 tied embedding passthrough**: The tied embedding matrix serves dual duty as both input embedding and output projection head. INT8 quantization introduces ~0.007 BPB of noise. Keeping it in FP16 during export reduces the quant gap to ~0.002 BPB.

4. **Mixed int8/int6 quantization**: First 3 and last 3 layers use full int8. Middle 4 layers use int6 (step-4 rounding: 64 distinct values in int8 container). Same raw size but zlib compresses much better, fitting the extra layer under 16MB.

5. **MLP hidden dim 960** (from 1024): Trimmed to compensate for fp16 embedding and extra layer.

6. **Lower learning rates**: MATRIX_LR=0.032, SCALAR_LR=0.032, TIED_EMBED_LR=0.04 (from 0.04, 0.04, 0.05).

7. **Sliding window evaluation** (stride=64): Overlapping windows give every scored token ~2000 tokens of context instead of average ~1024. Pure eval-time improvement, ~0.021 BPB gain.

Warmdown set to 3600 (from 1200) to ensure LR decay fires under wallclock cap.

## Configuration

```bash
RUN_ID=v3_sliding_window \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Layout: `NUM_LAYERS=10 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_HIDDEN=960 TRAIN_SEQ_LEN=2048`
Tied embeddings with FP16 export passthrough. Int6 for blocks 3-6.

## Results

| Seed | Steps | val_bpb (standard) | val_bpb (sliding) | Artifact size |
|------|-------|--------------------|--------------------|---------------|
| 1337 | 9,373 | 1.1999 | 1.1787 | 15,693,340 |
| 42 | ~9,100 | 1.2008 | 1.1798 | ~15,690,000 |
| 3 | ~9,100 | 1.2006 | 1.1794 | 15,690,354 |

**Mean val_bpb (sliding): 1.1793** (std: 0.00057)
**Mean val_loss (sliding): 1.9912** (std: 0.00097)

Statistical significance vs SOTA (1.2244 BPB / 2.0727 val_loss):
- Improvement: 0.0815 nats (threshold: 0.005)
- t-statistic: -137.28, df=2, p << 0.01

Sliding window eval time: ~357s (within the 10-minute eval budget).
Hardware: 8xH100 80GB HBM3, PyTorch 2.8.0+cu128, ~65.9ms/step avg.

## Included Files

- `train_gpt.py` (modified training script)
- `train_seed1337.log` (seed 1337 training log)
- `train_seed42.log` (seed 42 training log)
- `train_seed3.log` (seed 3 training log)
- `submission.json` (leaderboard metadata)
