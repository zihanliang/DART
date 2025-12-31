# DART: Mitigating Harm Drift in Difference-Aware LLMs via Distill-Audit-Repair Training

This repository contains the official implementation of **DART** (**D**istill–**A**udit–**R**epair **T**raining), a framework for training difference-aware language models while mitigating harm drift.

## Overview

Large language models tuned for safety often default to identity-blindness, producing incorrect responses when demographic differences are factually or legally relevant. DART addresses this through a three-stage pipeline:

1. **Distill**: Train a student model on teacher-generated reasoning chains
2. **Audit**: Detect harm regressions by comparing model outputs against a baseline
3. **Repair**: Fine-tune on safer alternatives for regression cases with severity-weighted sampling

On eight benchmarks, DART improves Llama-3-8B-Instruct accuracy from 39.0% to 68.8% while reducing harm regressions by 72.6%.

## Repository Structure

```
├── model/
│   ├── config.py            # Configuration management
│   ├── datasets.py          # Dataset loading and normalization
│   ├── teacher_generate.py  # Teacher model generation (distill & repair)
│   ├── train_lora.py        # LoRA/QLoRA training
│   ├── inference.py         # Model inference (single & paired)
│   ├── audit.py             # Harm drift detection and regression analysis
│   └── evaluate.py          # Evaluation and reporting
├── configs/
│   └── exp_acl.json         # Experiment configuration
├── benchmark_suite/         # Primary benchmarks (D1-D4, N1-N4)
├── dataset/                 # External evaluation suites
└── runs/                    # Experiment outputs
    └── {EXP_ID}/
        ├── configs/         # Snapshots
        ├── datasets/        # Normalized datasets
        ├── teacher_outputs/ # Teacher generations
        ├── adapters/        # Trained LoRA adapters
        ├── predictions/     # Inference outputs
        ├── audit/           # Audit reports
        └── evaluation/      # Final evaluation results
```

## Setup

### Requirements

```bash
pip install torch transformers peft accelerate aiohttp tqdm scipy pandas
```

### API Configuration

**Teacher Model API** (for distillation and repair generation):

```bash
# Option 1: DeepSeek (default)
export DEEPSEEK_API_KEY="your-deepseek-api-key"

# Option 2: OpenAI
export OPENAI_API_KEY="your-openai-api-key"
```

**Hugging Face** (for base model access):

```bash
export HF_TOKEN="your-huggingface-token"
# Or login via CLI
huggingface-cli login
```

### Experiment ID

All pipeline stages use a shared `EXP_ID` to organize outputs under `runs/{EXP_ID}/`. Set this before running:

```bash
export EXP_ID="dart_experiment_001"
```

## Running the Full Pipeline

The DART pipeline consists of 9 sequential steps. Each step uses `--materialize` to create the run directory structure and `--override "run.exp_id=$EXP_ID"` to maintain consistent experiment tracking.

### Step 1: Initialize Experiment

```bash
python -m model.config --materialize \
  --config configs/exp_acl.json \
  --exp-id "$EXP_ID"
```

This creates the experiment directory structure and exports normalized datasets.

### Step 2: Generate Distillation Targets

```bash
python -m model.teacher_generate --materialize \
  --task distill \
  --override "run.exp_id=$EXP_ID" \
  --config configs/exp_acl.json
```

Generates teacher reasoning chains for training data. Outputs: `runs/{EXP_ID}/teacher_outputs/distill/outputs.jsonl`

### Step 3: Train DART (Distilled Model)

```bash
python -m model.train_lora --materialize \
  --stage mdac \
  --adapter-name DART \
  --override "run.exp_id=$EXP_ID" \
  --config configs/exp_acl.json
```

Trains LoRA adapter on distillation targets. Outputs: `runs/{EXP_ID}/adapters/DART/final/`

### Step 4: Paired Inference (M₀ vs DART)

```bash
python -m model.inference --materialize \
  --paired \
  --policy off \
  --model-a-id M0 \
  --model-b-id DART \
  --model-b-adapter "runs/${EXP_ID}/adapters/DART/final" \
  --run-name primary_test_off_M0_vs_DART \
  --override "run.exp_id=$EXP_ID" \
  --config configs/exp_acl.json
```

Generates paired outputs for baseline (M₀) and distilled model (DART).

### Step 5: Audit for Harm Regressions

```bash
python -m model.audit --materialize \
  --paired-input "runs/${EXP_ID}/predictions/primary_test_off_M0_vs_DART/outputs.jsonl" \
  --a-name M0 --b-name DART \
  --run-name audit_M0_vs_DART \
  --mild-delta 0.01 \
  --no-abs-floors \
  --use-llm-judge \
  --llm-judge-mode validate \
  --llm-judge-provider deepseek \
  --llm-judge-model deepseek-chat \
  --override "run.exp_id=$EXP_ID" \
  --config configs/exp_acl.json
```

Identifies harm regressions using dual evaluators (toxicity + hate classifiers) validated by LLM judge. Outputs: `runs/{EXP_ID}/audit/{run_name}/regression_pool.jsonl`

### Step 6: Generate Repair Targets

```bash
python -m model.teacher_generate --materialize \
  --task repair \
  --input "runs/${EXP_ID}/audit/audit_M0_vs_DART/regression_pool.jsonl" \
  --override "run.exp_id=$EXP_ID" \
  --config configs/exp_acl.json
```

Generates safer alternative reasoning for regression cases.

### Step 7: Train DART-H (Repaired Model)

```bash
python -m model.train_lora --materialize \
  --stage mdac_h \
  --recipe mix \
  --adapter-name DART_H \
  --init-adapter "runs/${EXP_ID}/adapters/DART/final" \
  --repair-oversample mild=1,moderate=2,severe=3,extreme=4 \
  --override "run.exp_id=$EXP_ID" \
  --epochs 3 \
  --config configs/exp_acl.json
```

Continues training from DART with severity-weighted repair samples mixed with original distillation data.

### Step 8: Final Paired Inference (M₀ vs DART-H)

```bash
python -m model.inference --materialize \
  --paired \
  --policy off \
  --model-a-id M0 \
  --model-b-id DART_H \
  --model-b-adapter "runs/${EXP_ID}/adapters/DART_H/final" \
  --run-name primary_test_off_M0_vs_DART_H \
  --override "run.exp_id=$EXP_ID" \
  --config configs/exp_acl.json
```

### Step 9: Final Evaluation

```bash
python -m model.evaluate --materialize \
  --pred "runs/${EXP_ID}/predictions/primary_test_off_M0_vs_DART_H/outputs.jsonl" \
  --run-name eval_main \
  --a-name M0 --b-name DART_H \
  --override "run.exp_id=$EXP_ID" \
  --config configs/exp_acl.json
```

Generates final evaluation metrics, significance tests, and paper-ready tables.

## Key Outputs

| Stage | Output Location | Description |
|-------|-----------------|-------------|
| Distill | `teacher_outputs/distill/outputs.jsonl` | Teacher reasoning chains |
| Train DART | `adapters/DART/final/` | Distilled LoRA adapter |
| Audit | `audit/{run_name}/regression_pool.jsonl` | Harm regression cases |
| Repair | `teacher_outputs/repair/outputs.jsonl` | Safe alternative reasoning |
| Train DART-H | `adapters/DART_H/final/` | Repaired LoRA adapter |
| Evaluate | `evaluation/{run_name}/summary.json` | Final metrics and tables |

## Configuration

Key parameters in `configs/exp_acl.json`:

```json
{
  "model": {
    "base_model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
    "torch_dtype": "bfloat16"
  },
  "teacher": {
    "provider": "deepseek",
    "model": "deepseek-chat",
    "max_concurrency": 16
  },
  "training": {
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 4
  },
  "lora": {
    "r": 16,
    "alpha": 32,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
  }
}
```

## License

This project is released under the MIT License.
