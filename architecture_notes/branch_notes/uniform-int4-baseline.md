# Uniform Int4@4.0

This file describes the canonical baseline stack for **Uniform Int4@4.0** in **`parameter-golf-uniform-int4`**.

## Positioning
- This standalone repo is the canonical home for the current Uniform Int4@4.0 baseline.
- This note focuses on the baseline as it is maintained here.
- Future improvements in this repo should be described relative to this baseline unless they materially change the experiment itself.

## Purpose
- Canonical standalone experiment home for the current Parameter Golf submission line.
- Holds the mixed-precision training/export pipeline we would actually want to reproduce on target hardware.
- Exists to answer "does the current candidate really run and score as expected on challenge hardware?"

## Verified Now
- `train_gpt.py` already contains the core candidate stack:
  - Muon with Newton-Schulz orthogonalized matrix updates for matrix-shaped parameters.
  - Orthogonal initialization for large non-zero `nn.Linear` weights, with extra projection scaling.
  - QAT clip ranges that map to the current mixed-precision design: Int4 MLP weights and Int6 attention/bigram weights.
  - `mixed_quantize_int6()` / `dequantize_mixed_int6()` export roundtrip and `final_model.mixed.ptz` artifact path.
  - SWA application plus 3% magnitude pruning before serialization.
  - `BayesianBackoffCache` inside `eval_val_sliding_cached()` for evaluation-side mixing.
- No local RunPod/H100 artifact or benchmark log is committed for this repo.

## Feasibility Summary
- **Muon is already integrated, not hypothetical.** The baseline already routes matrix-shaped transformer parameters through `Muon` and keeps scalar/embedding-style parameters on `AdamW`.
- **Muon + orthogonal init is a coherent pairing.** Muon orthogonalizes updates, while orthogonal init gives large linear weights well-conditioned starting geometry, so early steps are spent learning rather than correcting poor initial singular values.
- **The quantization path is already end-to-end.** This baseline is not waiting on a future quantization implementation; QAT clip ranges are attached to the actual layers, fake quantization is used during training, and the export path already serializes a compressed mixed-precision artifact.
- **The remaining uncertainty is execution evidence, not missing plumbing.** The primary unknown is whether the current coded stack reproduces under target wall-clock and memory constraints, not whether the branch has the necessary components.
- **Eval-side cache feasibility is rule-based, not officially blessed.** The current reading of the README is that using already-graded tokens for backward-looking mixing is acceptable, but there is no explicit organizer ruling attached to this branch note.

## Current Evidence
- No official leaderboard submission from this repo yet.
- No committed Runpod/H100 artifact, benchmark log, or `records/` entry for this stack yet.

## Not Yet Proven
- Whether the current cached sliding evaluation path fits comfortably inside the 10-minute evaluation budget on target hardware.
- Whether current README-compatible cache usage matches organizer intent in a stricter ruling.

## Risks / Open Questions
- **Runtime risk:** cached sliding evaluation may help BPB but still fail the wall-clock budget if not measured carefully.
- **Artifact risk:** final compressed size under the 16MB constraint is not yet confirmed on target hardware in a committed run.

## Next Concrete Step
Run and save one reproducible target-hardware baseline result for this repo, including artifact size, wall-clock, and `val_bpb`, under `records/` so claims are grounded in logs.
