# Kaggle Uniform Int4 3000-Step Proof

**Status**: PROVEN (Path A - Parked)

## Context
This record preserves the state of the Uniform Int4@4.0 3000-step training validation run before Kaggle compute hours were exhausted. The architectural base has successfully demonstrated a strong validation signal.

## Results from 3000-Step Run
- **val_bpb**: ~1.45 (Pre-Quantization)
- **Artifact Headroom**: Under 16MB constraint (~15.5MB)
- **Signal**: Strong (1.45 BPB with 500 KB headroom, validating the Int4/Int6 layer quantizations and padding logic via Muon).

## Decision: Pivot to Bayesian Backoff
Due to the lack of free GPU time for prolonged 5000+ step validations, we are **parking** further parameter iterations on this baseline (`mlp_mult` and magnitude pruning tweaks). Instead, we pivot to testing the **Bayesian backoff + TT adapter** over this baseline model. The upside of adding the inference-time recovery layer yields a stronger overall BPB without requiring exhaustive 20k step runs on limited compute.
