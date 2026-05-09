# Task-Aware Adaptation and Test-Time Scaling for Text-Rich Multimodal Reasoning

This repository houses the official implementation of a complete training, alignment, and evaluation framework for enhancing Multimodal Large Language Models (specifically Qwen-VL) on text-dense and structurally complex tasks (e.g., Document QA, Chart QA, and Scientific Figure Reasoning).

By integrating **Parameter-Efficient Adapters (TRA/DoRA)**, **Task-Stratified Reinforcement Learning (TS-GRPO)**, and **Test-Time Compute Scaling (Best-of-N)**, this project provides a robust pipeline to improve in-domain expert reasoning while preserving cross-domain generalization.

---

## 🌟 Core Technical Contributions

### 1. Text-Rich Adapter (TRA)
Located in `src/text_rich_mllm/adapters/text_rich_adapter.py`. 
Unlike standard LoRA, TRA is an intermediate adaptation module injected into the middle-to-late layers of the Transformer backbone. It integrates:
- **Task Embeddings**: Conditions the model on task-specific priors (e.g., `DocVQA` vs `ChartQA`).
- **Layout Embeddings**: (Optional) Utilizes OCR bounding boxes to provide explicit spatial and structural context.
- **Gated Residual Updates**: Learns to balance the injected task-specific signals against the pretrained VLM representations.

### 2. Task-Stratified GRPO (TS-GRPO)
Located in `src/text_rich_mllm/training/ts_grpo_trainer.py`.
Standard RLHF/GRPO struggles when applied jointly to highly heterogeneous datasets. We implemented a custom **Task-Stratified GRPO Trainer** that:
- Groups generated rollouts strictly by task.
- Computes the Advantage function (Reward scaling) within task-homogeneous batches, preventing reward-scale domination by easier tasks.
- Directly optimizes reasoning steps and answer formatting.

### 3. Best-of-N (BoN) Test-Time Scaling
Located in `scripts/inference_best_of_n.py` & `scripts/pipeline_stage_11_bon_eval.sh`.
Systematically investigates how allocating more inference compute improves multi-step reasoning accuracy. The pipeline supports:
- High-throughput batched sampling for $N \in \{1, 2, 4, 8\}$.
- Automatic extraction of scaling curves across datasets.
- Verification of test-time scaling performance on top of SFT (LoRA) vs. RL (TS-GRPO) checkpoints.

---

## 📁 Repository Architecture

```text
├── configs/                  # YAML configs for PEFT, TRA, Models, Data, and Training
├── data/                     # Raw, cached, and processed datasets (DocVQA, ChartQA, ScienceQA, MMMU)
├── docs/                     # Project proposals and academic documentation
├── outputs/                  # Checkpoints, predictions, metrics, and generated figures
├── scripts/                  # Shell pipelines (01-11) and CLI entry points
└── src/text_rich_mllm/       # Core Library
    ├── adapters/             # TRA module implementation (text_rich_adapter.py)
    ├── datasets/             # Unified schema loaders and dataset formatters
    ├── evaluation/           # Benchmark-aware metrics (ANLS, exact match, numeric eval)
    ├── models/               # Qwen-VL architecture wrapper and checkpoint loading
    ├── prompts/              # Structured prompt builders
    ├── training/             # Custom trainers (ts_grpo_trainer.py, hf_trainer.py, mixing.py)
    └── utils/                # Utilities for loss masking, collators, etc.
```

---

## 🚀 Execution Pipelines

The experimental workflow is highly reproducible, divided into distinct shell pipelines in the `scripts/` directory:

### Phase 1: Data & Baselines
- **`pipeline_stage_01_*`**: Downloads datasets and normalizes them into a unified JSONL schema (e.g., Doc10k, Chart15k).
- **`pipeline_stage_02/03`**: Prompt sanity checking and zero-shot baseline evaluation.

### Phase 2: Supervised Fine-Tuning (SFT)
- **`pipeline_stage_04_training.sh`**: Standard LoRA fine-tuning.
- **`pipeline_stage_08_dora_training.sh`**: Weight-Decomposed Low-Rank Adaptation (DoRA).
- **`pipeline_stage_09_tra_training.sh`**: Training with the custom Text-Rich Adapter (TRA).

### Phase 3: RL Alignment & Test-Time Scaling
- **`pipeline_stage_10_grpo.sh`**: Launches the TS-GRPO reinforcement learning phase on top of SFT checkpoints.
- **`pipeline_stage_11_bon_eval.sh`**: Evaluates checkpoints using Best-of-N inference, auto-generating numerical summaries and scaling curve plots.

---

## 💻 Quick Start

### Installation
Ensure you have a modern GPU with CUDA support.

```bash
conda env create -f environment.yml
conda activate nlp_final
pip install -e .

# Verify the environment and PyTorch/CUDA setup
python scripts/check_runtime.py
```

### Reproducing an Experiment
To run the Task-Stratified GRPO alignment and generate Best-of-N curves:

```bash
# 1. Run GRPO alignment on a pretrained/SFT checkpoint
bash scripts/pipeline_stage_10_grpo.sh \
    --checkpoint outputs/checkpoints/my_sft_model \
    --train-config configs/train/train_joint_grpo.yaml

# 2. Run BoN Scaling Curve Evaluation
bash scripts/pipeline_stage_11_bon_eval.sh \
    --checkpoint outputs/checkpoints/grpo_final \
    --samples data/processed/docvqa/validation.jsonl \
    --scaling-curve \
    --exp-name E10_grpo_bon_curve
```

### Plotting & Reports
Extract academic tables and visualizations from generated metrics:
```bash
python scripts/generate_report_figures.py
python scripts/export_tables_figures.py
```

---
*Developed for research in multi-modal text-rich reasoning.*
