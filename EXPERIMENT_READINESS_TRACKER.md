# Experiment Readiness Tracker

This tracker records the **canonical working location** and operational-readiness status for the experiment housed in this repository.

It is intentionally about **operational readiness**, not leaderboard quality.

Use [RUNPOD_PROOF_RUNBOOK.md](/Users/jmoncayopursuit.org/Desktop/parameter-golf-uniform-int4/RUNPOD_PROOF_RUNBOOK.md), [preflight_h100.sh](/Users/jmoncayopursuit.org/Desktop/parameter-golf-uniform-int4/preflight_h100.sh), and [launch_h100.sh](/Users/jmoncayopursuit.org/Desktop/parameter-golf-uniform-int4/launch_h100.sh) as the operational reference surface for this repo.

## Key

- **Full Repo** = standalone repository with its own history
- **Branch Origin** = experiment originally lived inside a larger repo branch
- **Operational Gold Standard** = already at the Gimlet-Hetero level of readiness
- **Gimlet-Level Scaffolding Applied** = readiness scaffolding is in place, but a fresh end-to-end proof artifact still needs to be recorded
- **Partial** = some evidence exists, but readiness has not been audited end-to-end

## Tracker

| Experiment Name | Canonical Location | Type | Readiness Status | Notes |
|:---|:---|:---:|:---|:---|
| Uniform Int4@4.0 | [parameter-golf-uniform-int4](/Users/jmoncayopursuit.org/Desktop/parameter-golf-uniform-int4) | Full Repo | Gimlet-Level Scaffolding Applied | Canonical standalone home. Originated from `parameter-golf-qat-int4` -> `qat-int4-int6-gps-mlp`. Exact-step default, validation cap support, phase markers, summaries, preflight, launch script, and clean runbook are in place; fresh proof evidence still needs to be saved under `records/`. |

## Origin Reference

The implementation lineage for this repo is:

`parameter-golf-qat-int4` -> `qat-int4-int6-gps-mlp` -> [parameter-golf-uniform-int4](/Users/jmoncayopursuit.org/Desktop/parameter-golf-uniform-int4)

This repo is the canonical home for ongoing work on that experiment line.

## How to use this

When working on this experiment:

1. open the conversation in [parameter-golf-uniform-int4](/Users/jmoncayopursuit.org/Desktop/parameter-golf-uniform-int4)
2. treat this repo as the canonical home, not the old branch location
3. use Gimlet as the operational reference standard
4. focus first on proof evidence and end-to-end reproducibility before model-quality debate
