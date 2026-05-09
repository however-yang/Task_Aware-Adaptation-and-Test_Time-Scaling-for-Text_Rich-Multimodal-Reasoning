# qwen_vl_text_rich_adapter.md

# 面向 Qwen-VL 的 Text-Rich Adapter 技术方案

## 项目代号

**TRA-QwenVL**  
(Text-Rich Adapter for Qwen-VL)

---

# 1. 设计目标

通用视觉语言模型（如 Qwen-VL）具备强大的开放域视觉理解与语言生成能力，但在 **text-rich multimodal tasks**（文档、图表、科学图示）中，仍存在以下瓶颈：

- OCR 文字读取不稳定
- 版面结构感知弱
- 图表数值关系建模不足
- 多任务迁移时表示冲突
- 输出答案格式不稳定

本方案目标：

> 在不重训 backbone 的前提下，通过轻量中间层适配模块，提升模型在文本密集视觉任务上的表示能力与泛化能力。

---

# 2. 总体思路

在 Qwen-VL Transformer 中后层插入 **Text-Rich Adapter（TRA）**，利用：

- task embedding（任务类型）
- layout embedding（区域结构）
- gated residual adapter（可控更新）

实现任务专用增强。

---

# 3. 模块结构

## 3.1 单层 TRA Block

输入：

- hidden states: `H ∈ R^(N×d)`
- task embedding: `t ∈ R^d`
- optional layout embedding: `L ∈ R^(N×d)`

输出：

- enhanced hidden states

---

## 公式

### Step 1: 条件融合

\[
Z = H + W_t t + L
\]

### Step 2: Bottleneck Adapter

\[
A(Z)=W_{up}(\sigma(W_{down}Z))
\]

其中：

- `W_down: d -> r`
- `W_up: r -> d`
- `r << d`

---

### Step 3: Gated Residual Update

\[
g = \sigma(W_g \cdot mean(H))
\]

\[
H' = H + g \cdot A(Z)
\]

---

## 3.2 模块特点

- 参数量小
- 插拔式设计
- 支持 task conditioning
- 支持 layout signal
- 对 backbone 权重改动最小

---

# 4. 插入位置（重点）

## 推荐位置：中后层插入

例如 Qwen-VL 共 L 层 Transformer：

插入：

- layer 16
- layer 20
- layer 24
- layer 28

若是 32 层模型：

```text
[1 ... 15] 原始层
[16] + TRA
[17 ... 19]
[20] + TRA
...
```

---

## 为什么不是前层？

前层主要负责：

- 基础视觉 token
- 低级语义

不适合过早 task specialization。

---

## 为什么不是最后层？

最后层更接近输出空间，过晚注入效果有限。

---

## 最佳经验

> 在中后层插入 3~4 个 adapter block 往往效果最好。

---

# 5. Layout Embedding（可选增强）

对于文档 / 图表任务，可利用 OCR boxes：

每个 region box:

```text
(x1, y1, x2, y2, area, reading_order)
```

编码为：

\[
L_i = MLP(box_i)
\]

再加到对应 token 上。

---

# 6. Task Embedding（推荐）

任务标签：

- DocVQA
- ChartQA
- ScienceQA
- MMMU

每类学习一个 embedding：

\[
t = E(task)
\]

作用：

帮助共享 backbone 做任务条件化。

---

# 7. 训练方式

## 推荐阶段训练

---

## Stage 1：LoRA Baseline

训练：

- q_proj
- k_proj
- v_proj
- o_proj

冻结 backbone。

---

## Stage 2：加入 TRA

训练：

- LoRA 参数
- TRA 参数
- task embedding

冻结 backbone 主体。

---

## Stage 3（可选）

解冻最后 2 层 LayerNorm / FFN 小学习率微调。

---

# 8. Loss 设计

## 主损失

标准自回归生成 loss：

\[
L_{ans}
\]

---

## 可选辅助损失

### Format Loss

约束输出结构：

```json
{"answer":"..."}
```

### Task Consistency Loss

不同任务 embedding 分离正则。

---

# 9. Ablation 设计（必须做）

---

## 9.1 核心对比

| Model | Adapter | Task Emb | Layout | Score |
|------|--------|----------|--------|------|
| LoRA baseline | ✗ | ✗ | ✗ | |
| TRA-small | ✓ | ✗ | ✗ | |
| TRA-task | ✓ | ✓ | ✗ | |
| TRA-full | ✓ | ✓ | ✓ | |

---

## 9.2 插入层位置

| Insert Layers | Score |
|--------------|------|
| early | |
| middle | |
| late | |
| mid+late | |

---

## 9.3 Adapter Rank

| Rank r | Params | Score |
|-------|-------|------|
| 16 | | |
| 32 | | |
| 64 | | |

---

## 9.4 单任务 vs 联合训练

| Train Set | DocVQA | ChartQA | OOD |
|----------|-------|--------|-----|
| Doc only | | | |
| Chart only | | | |
| Joint | | | |

---

# 10. 预期收益

相比 LoRA only：

- 更强任务区分能力
- 更强结构理解能力
- 更稳联合训练
- 更好的外部泛化
- 更清晰错误分析

---

# 11. 最终 Report 中 Method Section 写法

---

# 3 Method

## 3.1 Overview

We propose **Text-Rich Adapter (TRA)**, a lightweight intermediate adaptation module for open multimodal models. Instead of modifying the pretrained backbone extensively, TRA injects task-aware and layout-aware signals into middle-to-late Transformer layers.

---

## 3.2 Task-Aware Intermediate Adaptation

Given hidden states \(H_l\) at layer \(l\), we condition the adapter on task embedding \(t\):

\[
Z = H_l + W_t t
\]

This enables a shared backbone to specialize across heterogeneous text-rich tasks.

---

## 3.3 Layout-Aware Enhancement

For document and chart inputs, OCR region boxes are encoded into layout embeddings and added to token representations.

\[
Z = H_l + W_t t + L
\]

This provides explicit structural priors absent in generic VLM pretraining.

---

## 3.4 Gated Residual Adapter

We use a bottleneck adapter with gated residual injection:

\[
H_{l+1}=H_l+g_lA(Z)
\]

where \(g_l\) is a learnable gate controlling update magnitude.

---

## 3.5 Training Objective

We optimize autoregressive answer generation loss while freezing the original backbone parameters, updating only LoRA and TRA parameters.

---

# 12. Repo 落地建议

```text
src/
 ├── adapters/
 │   └── text_rich_adapter.py
 ├── models/
 │   └── qwen_with_tra.py
```

---

# 13. 最小可行版本（建议立即实现）

第一版只做：

- task embedding
- bottleneck adapter
- mid-layer insertion

先不做 layout。

跑通后再加 layout 分支。

---

# 14. 一句话总结

> TRA 不是重做 Qwen-VL，而是让通用模型拥有更强的 text-rich task specialization 能力。
