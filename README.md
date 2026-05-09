# Task-Aware Adaptation and Validation-Time Oracle Analysis for Text-Rich Multimodal Reasoning

[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](NLP_final.pdf)

This repository houses the official implementation for the paper **"Task-Aware Adaptation and Validation-Time Oracle Analysis for Text-Rich Multimodal Reasoning"**. It provides a complete training, alignment, and evaluation framework for enhancing Multimodal Large Language Models (specifically Qwen3-VL-8B-Instruct) on text-dense and structurally complex tasks.

## 🌟 Core Technical Contributions

### 1. Progressive Supervised Adaptation (DoRA)
The adaptation starts with Weight-Decomposed Low-Rank Adaptation (DoRA) applied to `q_proj, k_proj, v_proj, o_proj` (rank 48, alpha 96) to establish a strong foundational model for text-rich tasks.

### 2. Text-Rich Adapter (TRA)
Located in `src/text_rich_mllm/adapters/text_rich_adapter.py`. 
An intermediate adaptation module injected after decoder layers 16, 20, 24, and 28 (hidden size 4096, bottleneck rank 64). It integrates task embeddings to explicitly condition the model on task-specific priors.

### 3. Task-Stratified GRPO (TS-GRPO)
Located in `src/text_rich_mllm/training/ts_grpo_trainer.py`.
To align predictions with target-task metrics, we use a Task-Stratified GRPO Trainer that groups generated rollouts strictly by task. It computes the Advantage function within task-homogeneous batches, ensuring stable optimization across different metrics (e.g., ANLS for DocVQA, Exact Match for ChartQA).

### 4. Validation-Time Oracle Best-of-N Analysis
Located in `scripts/inference_best_of_n.py`.
Rather than a deployable test-time verifier, this is an **oracle upper-bound analysis** evaluating the headroom of candidate diversity. It samples $N \in \{1, 2, 4, 8\}$ candidate answers and selects the best one using gold-reference task rewards on the validation set.

## 📊 Main Results
All models were trained and evaluated using a fixed random seed (`42`) to ensure reproducibility.

| Stage | DocVQA (ANLS) | ChartQA (EM) | In-Domain Avg | ScienceQA (Acc) | MMMU (Acc) | External Avg |
|-------|---------------|--------------|---------------|-----------------|------------|--------------|
| Qwen3-VL + prompt | 0.1218 | 0.0760 | 0.1019 | 0.6822 | 0.2900 | 0.4861 |
| LoRA SFT | 0.7096 | 0.6792 | 0.6964 | **0.7980** | **0.4411** | **0.6195** |
| DoRA SFT | 0.7389 | 0.7313 | 0.7356 | 0.7888 | 0.4300 | 0.6094 |
| DoRA SFT + TRA | 0.8100 | 0.7427 | 0.7808 | 0.7893 | 0.4256 | 0.6074 |
| DoRA SFT + TRA + GRPO | **0.9190** | **0.7682** | **0.8535** | 0.7871 | 0.4156 | 0.6013 |

**Oracle Best-of-N Analysis (N=8)** on the final model further improves DocVQA to **0.9339** and ChartQA to **0.7896**.

## 📁 Repository Architecture

```text
├── configs/                  # YAML configs (Seed 42 is fixed here)
├── data/                     # Data directory (DocVQA, ChartQA, ScienceQA, MMMU)
├── docs/                     # Project report and proposals
├── outputs/                  # Checkpoints, predictions, and metrics
├── scripts/                  # Shell pipelines (01-11) and CLI entry points
└── src/text_rich_mllm/       # Core Library (TRA, TS-GRPO, etc.)
```

## 🚀 Reproducibility Guide

### Hardware Requirements
The main experiments were validated on a single **NVIDIA RTX PRO 6000 GPU (96GB memory)**. Max recorded VRAM usage is ~61.8GB during GRPO training.

### Environment Setup
```bash
conda env create -f environment.yml
conda activate nlp_final
pip install -e .
python scripts/check_runtime.py
```

### End-to-End Execution
The reproducible workflow is divided into bash pipelines:
1. **Data Preparation**: `bash scripts/pipeline_stage_01_build_datasets_chart15k_doc10k.sh`
2. **DoRA SFT Training**: `bash scripts/pipeline_stage_08_dora_training.sh`
3. **TRA Training**: `bash scripts/pipeline_stage_09_tra_training.sh`
4. **TS-GRPO Alignment**: `bash scripts/pipeline_stage_10_grpo.sh`
5. **Best-of-N Oracle Analysis**: `bash scripts/pipeline_stage_11_bon_eval.sh` (Use `--scaling-curve` flag to reproduce Figure 4).

### Evaluation & Plotting
To reproduce the exact metrics and figures reported in the paper:
```bash
python scripts/generate_report_figures.py
python scripts/export_tables_figures.py
```
