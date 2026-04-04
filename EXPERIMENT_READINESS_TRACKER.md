# Experiment Readiness Tracker

This tracker records the **canonical working location** and operational-readiness status for the experiment housed in this repository.

It is intentionally about **operational readiness**, not leaderboard quality.

Use [RUNPOD_PROOF_RUNBOOK.md](/Users/jmoncayopursuit.org/Desktop/parameter-golf-uniform-int4/RUNPOD_PROOF_RUNBOOK.md), [preflight_h100.sh](/Users/jmoncayopursuit.org/Desktop/parameter-golf-uniform-int4/preflight_h100.sh), and [launch_h100.sh](/Users/jmoncayopursuit.org/Desktop/parameter-golf-uniform-int4/launch_h100.sh) as the operational reference surface for this repo.

## Key

- **Full Repo** = standalone repository with its own history
- **Operational Gold Standard** = already at the Gimlet-Hetero level of readiness
- **Operational Scaffolding Applied, Proof Pending** = readiness scaffolding is in place, but a fresh end-to-end proof artifact still needs to be recorded
- **Partial** = some evidence exists, but readiness has not been audited end-to-end

## Tracker

| Experiment Name | Canonical Location | Type | Readiness Status | Notes |
|:---|:---|:---:|:---|:---|
| Uniform Int4@4.0 | [parameter-golf-uniform-int4](/Users/jmoncayopursuit.org/Desktop/parameter-golf-uniform-int4) | Full Repo | Colab Proof Complete | 700-step T4 proof: roundtrip BPB=1.8008, artifact=15.93MB (under 16MB ✅), exit code 0. Evidence saved under `records/colab_700_step_proof/`. H100 proof still pending. |

## How to use this

When working on this experiment:

1. open the conversation in [parameter-golf-uniform-int4](/Users/jmoncayopursuit.org/Desktop/parameter-golf-uniform-int4)
2. treat this repo as the canonical home
3. use Gimlet as the operational reference standard
4. focus first on proof evidence and end-to-end reproducibility before model-quality debate
