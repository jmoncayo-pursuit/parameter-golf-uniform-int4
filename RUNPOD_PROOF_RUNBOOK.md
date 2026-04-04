# RunPod Proof Runbook

This is the clean operational path for `Uniform Int4@4.0` in `parameter-golf-uniform-int4`.

## Cheap gate first

Run a 1xGPU proof before any larger paid run:

```bash
RUN_ID=uniform_int4_smoke \
NPROC_PER_NODE=1 \
EXPECTED_GPU_NAME_SUBSTR=H100 \
VAL_MAX_TOKENS=262144 \
ITERATIONS=200 \
MAX_WALLCLOCK_SECONDS=0 \
bash ./launch_h100.sh
```

This keeps exact-step behavior while capping validation work so we can prove the full export path cheaply.

## 8xH100 proof

When the 1xGPU smoke is clean, move to the normal multi-GPU proof:

```bash
RUN_ID=uniform_int4_proof \
NPROC_PER_NODE=8 \
EXPECTED_GPU_NAME_SUBSTR=H100 \
VAL_MAX_TOKENS=524288 \
MAX_WALLCLOCK_SECONDS=0 \
bash ./launch_h100.sh
```

## Expected outputs

The run should leave these files in the working directory:

- `final_model.pt`
- `final_model.mixed.ptz`
- `final_summary.json`
- `final_summary.md`

The trainer should also log these end-of-run phase markers:

- `phase:training_complete`
- `phase:serialization_start`
- `phase:quantization_start`
- `phase:roundtrip_eval_start`
- `phase:done`

## Save proof evidence

After a successful proof, save the notebook, logs, and summaries under `records/` so the result is not stranded in a transient RunPod session.
