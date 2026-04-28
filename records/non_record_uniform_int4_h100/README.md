# Uniform Int4@4.0 Baseline (Non-Record Proof)

This is a non-record submission for the **Uniform Int4@4.0** architecture baseline. 
The purpose of this submission is to demonstrate a hybrid-precision approach that utilizes 6-bit GPS (General Purpose Slots/Attention) and a 4-bit MLP feedforward engine, fitting safely under the 16MB limit.

## Performance Metrics
- **Hardware:** 8x H100 SXM (RunPod)
- **Validation BPB:** 1.4420
- **Artifact Size:** 15.78 MB

## Architecture Innovations
- **Int6 GPS:** 6-bit quantization for key/query/value projections to maintain high sensitivity in the attention mechanism.
- **Int4 Engine:** Aggressive 4-bit compression for parameter-dense MLP blocks.
- **Dynamic Per-Block Scaling:** Adaptive normalization across 128-element weight blocks to handle outlier channels.

## Reproduction
Run the training script using:
```bash
RUN_ID=proof_run_uniform_int4 DATA_PATH=./data/datasets/fineweb10B_sp1024/ TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model VOCAB_SIZE=1024 torchrun --standalone --nproc_per_node=8 train_gpt.py
```
