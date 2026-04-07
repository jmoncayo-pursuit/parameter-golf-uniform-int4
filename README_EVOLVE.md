# Parameter Golf Evolution Engine (Phase 1.2) - General & Reusable

This is a generalized version of the "AlphaEvolve" engine. It is no longer tied to a specific project and can evolve any marked code snippet in any file.

## Architecture
- **`evolver.py`**: The general engine logic.
- **`config.yaml`**: Project-specific configuration (file paths, markers, test commands, environments).
- **`evolution_leaderboard.json`**: Historical log of all generations.

## Usage (Current Project: Uniform Int4@4.0)

### 1. Set Gemini API Key
```bash
export GEMINI_API_KEY="your-google-api-key"
```

### 2. Run Evolution for a Target
```bash
# Evolve the QAT ramp
python3 evolver.py --target qat_ramp --gen 1

# Evolve the pruning logic
python3 evolver.py --target pruning --gen 1
```

### 3. Switch Environments
You can define multiple environments in `config.yaml` (e.g., `local`, `kaggle`, `colab`).
```bash
# Use kaggle environment settings
export EVO_ENV_NAME="kaggle"
python3 evolver.py --target qat_ramp --gen 1
```

## How it Works
1. **Mutation Phase**: The engine Proposed 4 mutations and tests them for **600 steps** each (fast screen).
2. **Scoring Phase**: Metrics are extracted and scored using **high-priority BPB weighting** (BPB x 10,000).
3. **Validation Phase**: The engine runs the winner for a more reliable **1500-step validation** run.
4. **Merge Phase**: Only if validation passes, the change is merged into the code and committed to git.
5. **Safety**: A `target_file.bak` is created at the start of each generation.

## Making it Work for a New Project
To use this in any other project, just:
1. Copy `evolver.py` and `config.yaml`.
2. Edit `config.yaml` to point to your `target_file` and define your `test_cmd` and `result_file`.
3. Add markers to your code like:
   ```python
   # EVOLVE: my_target_START
   def my_function(): ...
   # EVOLVE: my_target_END
   ```
4. Run: `python3 evolver.py --target my_target --gen 1`

## Customizing Scoring
In `config.yaml`, the scoring logic is fully adjustable:
```yaml
scoring:
  bpb_weight: 10000
  size_weight_factor: 1000 # Score = BPB*10000 + Size/1000
```
This ensures the engine always prioritizes quality (lower BPB) over minor size changes, while still respecting the 16MB constraint.
