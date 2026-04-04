# Uniform Int4@4.0 Non-Record Summary (700-step T4 Proof)

## Final Metrics
- **Steps**: 700 / 700
- **Train Loss**: 3.1889
- **Validation Loss (Pre-Quant)**: 3.0707
- **Validation BPB (Pre-Quant)**: 1.8156
- **Roundtrip Validation Loss**: 3.0443
- **Roundtrip Validation BPB**: 1.8008

## Submission Parameters
- **Compressor**: `lzma`
- **Magnitude Pruning**: 5%
- **Quantization strategy**: Mixed Int6/Int4 
  - Attention/Embeddings: Int6
  - MLP/States: Int4

## Artifact Compliance
- **Final Model (LZMA compressed)**: 15,859,108 bytes
- **Code (train_gpt.py)**: 72,149 bytes
- **TOTAL SUBMISSION SIZE**: **15,931,257 bytes**
- **Under 16MB?**: **YES ✅ (69KB headroom)**

## Phase Markers
- `phase:training_complete`
- `phase:serialization_start`
- `phase:quantization_start`
- `phase:roundtrip_eval_start`
- `phase:done`

## Notes
Proof of concept only (700 steps). Proves architecture and pipeline stability on standard T4 GPUs within the Gimlet standard.
