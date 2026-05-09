"""
Task-Stratified GRPO Trainer (TS-GRPO)
=======================================
实验 E8：在 SFT（E3/E5）checkpoint 基础上，用 Group Relative Policy Optimization
做 RL 对齐。核心创新是「任务分层」：每个 group 只包含同一 task 的样本，
确保 advantage 在 task-homogeneous 的 reward scale 内计算。

支持的 loss_type:
  - "grpo"  : 标准 GRPO（DeepSeekMath 2024）— group-normalized advantage + PPO-clip
  - "dapo"  : DAPO 变体 — 不对称 clip + 移除 KL 惩罚 + token-level 归一化
  - "dr_grpo": Dr.GRPO — 移除 std 归一化，解决 difficulty bias

参考文献：
  DeepSeekMath (2024)  — GRPO 算法原始提出
  DAPO (2025)          — Decoupled clip + dynamic sampling
  Dr.GRPO (2025)       — 移除 length/difficulty bias
  TRL GRPOTrainer      — HuggingFace 参考实现
  本项目创新：task-stratified group sampling（多任务 reward scale 对齐）
"""
from __future__ import annotations

import copy
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from text_rich_mllm.schemas import UnifiedSample

from text_rich_mllm.evaluation.evaluator import UnifiedEvaluator
from text_rich_mllm.evaluation.parsing import parse_prediction
from text_rich_mllm.models.generation_utils import (
    open_image_as_rgb,
    take_answer_tail_after_marker,
)
from text_rich_mllm.models.vision_prompt import ensure_image_placeholders_in_text
from text_rich_mllm.prompts import PromptBuilder
from text_rich_mllm.utils import get_logger

logger = get_logger("ts_grpo")


# ─────────────────────────────────────────────────────────────────────────────
# Task-Stratified Sampler
# ─────────────────────────────────────────────────────────────────────────────

class TaskStratifiedSampler:
    """
    按 task_id 分桶，每次随机选 1 个 task，再从该 task 内随机选 1 道题。
    保证同一个 GRPO group 内所有 G 个生成都来自同一 task_id。

    Attributes:
        buckets: {task_name: [UnifiedSample, ...]}
        task_names: 所有 task 名称列表（非空 bucket）
    """

    def __init__(
        self,
        samples: list["UnifiedSample"],
        task_names: list[str],
    ) -> None:
        self.buckets: dict[str, list["UnifiedSample"]] = defaultdict(list)
        for sample in samples:
            if sample.dataset_name in task_names:
                self.buckets[sample.dataset_name].append(sample)
        # 只保留非空 bucket
        self.task_names = [t for t in task_names if self.buckets.get(t)]
        if not self.task_names:
            raise ValueError("TaskStratifiedSampler: 所有 task 的 bucket 均为空，请检查 task_names 配置。")
        n_total = sum(len(v) for v in self.buckets.values())
        logger.info("TaskStratifiedSampler: %d tasks, %d samples total", len(self.task_names), n_total)
        for t in self.task_names:
            logger.info("  task=%s  samples=%d", t, len(self.buckets[t]))

    def sample_one(self) -> "UnifiedSample":
        """随机选 1 个 task，再从该 task 随机选 1 道题。"""
        task = random.choice(self.task_names)
        return random.choice(self.buckets[task])


# ─────────────────────────────────────────────────────────────────────────────
# 生成辅助：对同一道题采样 G 个答案
# ─────────────────────────────────────────────────────────────────────────────

def _sample_completions(
    model,
    processor,
    sample: "UnifiedSample",
    prompt: str,
    *,
    G: int,
    temperature: float,
    max_new_tokens: int,
) -> list[str]:
    """
    对同一张图 + 同一个 prompt，用 temperature 采样生成 G 个答案。
    返回 G 个字符串（经过 take_answer_tail_after_marker 后处理）。

    实现说明：
      - do_sample=True + temperature 采样（多样性）
      - 循环 G 次生成（num_return_sequences=1），避免 Qwen3-VL
        在 _expand_inputs_for_generation 中对视觉 token 扩展报错
    """
    image = open_image_as_rgb(sample.image_path)
    prompt_for_model = ensure_image_placeholders_in_text(processor, prompt, num_images=1)
    
    # ── 关键加速：在 Processor 层面组装 Batch，实现并行生成 ──
    texts = [prompt_for_model] * G
    images = [image] * G
    # 由于所有 text 完全一致，不需要 padding=True
    inputs = processor(images=images, text=texts, return_tensors="pt")
    
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,  # 每个 prompt 1 个，但总共有 G 个 prompt
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        
        # generated_ids 的 shape 将是 (G, input_len + new_tokens)
        decoded_texts = processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
        )
        
    completions = [take_answer_tail_after_marker(text.strip()) for text in decoded_texts]
    return completions


# ─────────────────────────────────────────────────────────────────────────────
# Reward 计算（直接复用 UnifiedEvaluator._score）
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rewards(
    evaluator: UnifiedEvaluator,
    sample: "UnifiedSample",
    completions: list[str],
) -> list[float]:
    """
    对 G 个 completion 分别计算 reward。
    reward = evaluator._score(sample, parsed_prediction)
    task-specific metric：
      docvqa / infographicvqa / textvqa → ANLS ∈ [0, 1]
      chartqa                           → chartqa_score ∈ {0, 1}
      scienceqa / mmmu（MCQ）           → accuracy ∈ {0, 1}
    """
    rewards = []
    for comp in completions:
        parsed = parse_prediction(comp, answer_type=sample.answer_type)
        r = evaluator._score(sample, parsed)
        rewards.append(float(r))
    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# Log-probability 计算（用于 GRPO loss）
# ─────────────────────────────────────────────────────────────────────────────

def _compute_log_probs(
    model: "nn.Module",
    processor: AutoProcessor,
    sample: UnifiedSample,
    prompt: str,
    completions: list[str],
    return_per_token: bool = False,
) -> tuple[list[torch.Tensor], list[torch.Tensor]] | list[torch.Tensor]:
    image = open_image_as_rgb(sample.image_path)
    prompt_for_model = ensure_image_placeholders_in_text(processor, prompt, num_images=1)
    device = next(model.parameters()).device

    log_probs_list = []
    per_token_list = []

    # 预计算 prompt_len
    prompt_inputs = processor(
        images=image, text=prompt_for_model,
        return_tensors="pt", padding=False, truncation=False,
    )
    prompt_len = prompt_inputs["input_ids"].shape[1]

    for comp in completions:
        full_text = f"{prompt_for_model} {comp}".strip()
        full_inputs = processor(
            images=image, text=full_text,
            return_tensors="pt", padding=False, truncation=False,
        )
        full_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in full_inputs.items()}

        input_ids = full_inputs["input_ids"]
        outputs = model(**{k: v for k, v in full_inputs.items() if k != "labels"})
        logits = outputs.logits

        shift_logits = logits[0, :-1, :]
        shift_ids = input_ids[0, 1:]
        answer_start = max(prompt_len - 1, 0)
        answer_logits = shift_logits[answer_start:]
        answer_ids = shift_ids[answer_start:]

        n_answer_tokens = answer_ids.shape[0]
        if n_answer_tokens == 0:
            log_probs_list.append(torch.tensor(0.0, device=device))
            per_token_list.append(torch.tensor([], device=device))
            continue

        log_prob_per_token = F.log_softmax(answer_logits, dim=-1)
        token_log_probs = log_prob_per_token[
            torch.arange(n_answer_tokens, device=device),
            answer_ids,
        ]

        per_token_list.append(token_log_probs.clone())
        log_probs_list.append(token_log_probs.mean())

    if return_per_token:
        return log_probs_list, per_token_list
    return log_probs_list


# ─────────────────────────────────────────────────────────────────────────────
# TS-GRPO 主训练循环
# ─────────────────────────────────────────────────────────────────────────────

class TSGRPOTrainer:
    """
    Task-Stratified GRPO Trainer（支持 GRPO / DAPO / Dr.GRPO 变体）。

    不使用 HuggingFace Trainer，采用手动训练循环，
    便于精确控制 group 采样和 task-stratified reward。

    关键特性：
      - task-stratified group sampling（多任务 reward scale 对齐）
      - PEFT adapter disable 实现 ref_model（零额外显存）
      - advantage collapse 自动跳过（DAPO dynamic sampling）
      - 可选 loss_type: grpo / dapo / dr_grpo
    """

    def __init__(
        self,
        model,
        processor,
        train_samples: list["UnifiedSample"],
        train_config: dict,
        eval_samples: list["UnifiedSample"] | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.eval_samples = eval_samples or []
        self.evaluator = UnifiedEvaluator()

        # ── 超参 ────────────────────────────────────────────────────────
        self.G = int(train_config.get("grpo_group_size", 4))
        self.beta = float(train_config.get("grpo_kl_coef", 0.01))
        self.clip_eps = float(train_config.get("grpo_clip_eps", 0.2))
        self.clip_eps_high = float(train_config.get("grpo_clip_eps_high", self.clip_eps))  # DAPO 不对称
        self.temperature = float(train_config.get("grpo_temperature", 0.8))
        self.max_new_tokens = int(train_config.get("grpo_max_new_tokens", 32))
        self.num_steps = int(train_config.get("grpo_num_steps", 500))
        self.eval_steps = int(train_config.get("grpo_eval_steps", 100))
        self.save_steps = int(train_config.get("grpo_save_steps", 100))
        self.lr = float(train_config.get("learning_rate", 5e-6))
        self.loss_type = train_config.get("grpo_loss_type", "grpo")  # grpo / dapo / dr_grpo
        from text_rich_mllm.utils.paths import resolve_training_output_dir
        self.output_dir = Path(resolve_training_output_dir(
            train_config.get("output_dir", "outputs/checkpoints/joint_ts_grpo")
        ))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_style = train_config.get("prompt_style", "structured")
        task_names = list(train_config.get("grpo_task_names", []))

        # ── 数据采样器 ──────────────────────────────────────────────────
        self.sampler = TaskStratifiedSampler(train_samples, task_names)
        self.prompt_builder = PromptBuilder(style=self.prompt_style)

        # ── Reference model ──────────────────────────────────────────────
        # 关键：使用 PEFT adapter disable 而非 deepcopy，零额外显存开销。
        # PeftModel 可以通过 disable_adapter() 暂时回退到 base model，
        # 这正是 TRL GRPOTrainer 的标准做法。
        # 对于 DAPO (loss_type="dapo")，beta=0 时不需要 ref_model。
        from peft import PeftModel
        self._is_peft = isinstance(model, PeftModel)
        if self._is_peft:
            logger.info("PeftModel detected: using adapter disable for ref_model (zero extra VRAM)")
            self.ref_model = None  # 不需要额外 copy
        else:
            logger.info("Building reference model (frozen copy for KL)...")
            self.ref_model = copy.deepcopy(model)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad_(False)

        # DAPO 模式下默认移除 KL 惩罚
        if self.loss_type == "dapo":
            if self.beta > 0:
                logger.info("DAPO mode: overriding beta=%.4f -> 0 (no KL penalty)", self.beta)
                self.beta = 0.0

        # ── 开启 Gradient Checkpointing（关键：防止长文本/大图 OOM）─────
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for RL loop.")

        # ── 优化器（只优化可训练参数：LoRA/DoRA + TRA）──────────────────
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        logger.info("Trainable parameters: %d tensors", len(trainable_params))
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.lr)

        # 记录 task name 到 task_id 的映射（为 TRA router 准备）
        self.task_name_to_id = {name: idx for idx, name in enumerate(task_names)}

        # ── 统计 ────────────────────────────────────────────────────────
        self.step_log: list[dict] = []

        logger.info(
            "Loss type: %s | clip_eps=[%.2f, %.2f] | beta=%.4f",
            self.loss_type, self.clip_eps, self.clip_eps_high, self.beta,
        )

    # ── 评测（在 val set 上跑 greedy decoding）──────────────────────────
    def _run_eval(self, step: int) -> dict[str, float]:
        if not self.eval_samples:
            return {}
        from text_rich_mllm.inference import generate_predictions

        # 评测时切回 eval 模式，结束后恢复 train 模式
        self.model.eval()
        gen_cfg = {"max_new_tokens": self.max_new_tokens, "do_sample": False}
        preds = generate_predictions(
            samples=self.eval_samples,
            model=self.model,
            processor=self.processor,
            prompt_style=self.prompt_style,
            generation_config=gen_cfg,
            limit=200,  # 快速评测，取前 200 条
        )
        self.model.train()
        # 评测用 eval_samples[:200]（与 limit 对齐）
        eval_subset = self.eval_samples[:200]
        from text_rich_mllm.analysis import tag_prediction_records
        from text_rich_mllm.evaluation import build_evaluation_report
        from text_rich_mllm.evaluation.console_summary import print_evaluation_report_summary
        records, summary = self.evaluator.evaluate(eval_subset, preds)
        tagged_records, error_counts = tag_prediction_records(records)
        summary["error_counts"] = error_counts
        report = build_evaluation_report(tagged_records, summary)
        print_evaluation_report_summary(
            report,
            title=f"GRPO EVAL @ step {step} (n={len(eval_subset)} subset)",
        )
        return summary

    # ── 单步 RL 更新 ─────────────────────────────────────────────────────
    def _step(self) -> dict[str, float]:
        sample = self.sampler.sample_one()
        prompt = self.prompt_builder.build(sample)

        # ── 激活 TRA Router (供所有 forward/generate 使用) ──────────────
        device = next(self.model.parameters()).device
        task_id = self.task_name_to_id.get(sample.dataset_name, 0)
        self.model._tra_task_ids = torch.tensor([task_id], dtype=torch.long, device=device)
        if hasattr(self, "ref_model") and self.ref_model is not None:
            self.ref_model._tra_task_ids = self.model._tra_task_ids

        # 1. 采样 G 个 completion（eval 模式，无梯度）
        self.model.eval()
        completions = _sample_completions(
            self.model,
            self.processor,
            sample,
            prompt,
            G=self.G,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )

        # 释放 generate 阶段的 KV cache
        torch.cuda.empty_cache()

        # 2. 计算 reward（task-specific，在同 task 内 scale 一致）
        rewards = _compute_rewards(self.evaluator, sample, completions)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)

        # 3. Advantage 计算
        mean_r = rewards_t.mean()
        std_r = rewards_t.std()

        # ── DAPO dynamic sampling / Advantage collapse 检测 ───────────
        if std_r < 1e-6:
            return {
                "loss": 0.0, "policy_loss": 0.0, "kl": 0.0,
                "mean_reward": mean_r.item(), "std_reward": 0.0,
                "task": sample.dataset_name, "skipped": True,
            }

        if self.loss_type == "dr_grpo":
            advantages = rewards_t - mean_r
        else:
            advantages = (rewards_t - mean_r) / (std_r + 1e-4)

        # 4. 计算 reference policy 的 per-token log_probs（π_ref，无梯度）
        #    使用 PEFT adapter disable（零额外 VRAM）
        ref_per_token = [None] * self.G
        if self.beta > 0:
            with torch.no_grad():
                if self._is_peft:
                    with self.model.disable_adapter():
                        self.model.eval()
                        _, ref_per_token = _compute_log_probs(
                            self.model, self.processor, sample, prompt, completions,
                            return_per_token=True,
                        )
                else:
                    _, ref_per_token = _compute_log_probs(
                        self.ref_model, self.processor, sample, prompt, completions,
                        return_per_token=True,
                    )

        # 5. Batched 梯度更新 (充分利用显存)
        self.model.train()
        self.optimizer.zero_grad()
        device = next(self.model.parameters()).device
        advantages = advantages.to(device).detach()

        # ── 激活 TRA Router ──────────────────────────────────────────
        task_id = self.task_name_to_id.get(sample.dataset_name, 0)
        self.model._tra_task_ids = torch.tensor([task_id], dtype=torch.long, device=device)

        image = open_image_as_rgb(sample.image_path)
        prompt_for_model = ensure_image_placeholders_in_text(self.processor, prompt, num_images=1)
        total_policy_loss = 0.0
        total_kl = 0.0

        # 预计算 prompt_len（所有 completion 共享同一 prompt）
        prompt_inputs = self.processor(
            images=image, text=prompt_for_model,
            return_tensors="pt", padding=False, truncation=False,
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        for i, comp in enumerate(completions):
            # ── 单个 completion 的 forward（有梯度）──────────────────────
            full_text = f"{prompt_for_model} {comp}".strip()
            full_inputs = self.processor(
                images=image, text=full_text,
                return_tensors="pt", padding=False, truncation=False,
            )
            full_inputs = {k: v.to(device) if hasattr(v, "to") else v
                           for k, v in full_inputs.items()}

            input_ids = full_inputs["input_ids"]
            outputs = self.model(**{k: v for k, v in full_inputs.items() if k != "labels"})
            logits = outputs.logits

            shift_logits = logits[0, :-1, :]
            shift_ids = input_ids[0, 1:]
            answer_start = max(prompt_len - 1, 0)
            answer_logits = shift_logits[answer_start:]
            answer_ids = shift_ids[answer_start:]

            n_ans = answer_ids.shape[0]
            if n_ans == 0:
                continue

            log_prob_per_token = F.log_softmax(answer_logits, dim=-1)
            token_log_probs = log_prob_per_token[
                torch.arange(n_ans, device=device), answer_ids
            ]
            cur_mean_log_prob = token_log_probs.mean()  # scalar, 有梯度

            # old_log_prob = 同一 forward 结果 detach（因为参数还没更新）
            old_lp = cur_mean_log_prob.detach()

            # ── PPO-clip（单个 completion）──────────────────────────────
            adv_i = advantages[i]
            ratio = torch.exp(cur_mean_log_prob - old_lp)  # 第 1 步 ratio ≈ 1
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps_high)
            surr = -torch.min(ratio * adv_i, clipped_ratio * adv_i)

            # ── KL（detached，不贡献梯度）─────────────────────────────
            kl_i = torch.tensor(0.0, device=device)
            if self.beta > 0 and ref_per_token[i] is not None:
                ref_tok = ref_per_token[i]
                cur_tok = token_log_probs.detach()
                min_len = min(ref_tok.shape[0], cur_tok.shape[0])
                diff = ref_tok[:min_len].detach() - cur_tok[:min_len]
                kl_i = (torch.exp(diff) - diff - 1.0).mean()

            # ── 累积 loss（除以 G 实现均值）────────────────────────────
            loss_i = (surr + self.beta * kl_i) / self.G
            loss_i.backward()

            total_policy_loss += surr.item() / self.G
            total_kl += kl_i.item() / self.G

        # 6. 梯度裁剪 + 优化器更新
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self.optimizer.step()

        return {
            "loss": total_policy_loss + self.beta * total_kl,
            "policy_loss": total_policy_loss,
            "kl": total_kl,
            "mean_reward": mean_r.item(),
            "std_reward": std_r.item(),
            "task": sample.dataset_name,
        }

    # ── 主训练循环 ───────────────────────────────────────────────────────
    def train(self, resume_step: int = 0) -> None:
        logger.info(
            "TS-GRPO training: G=%d, β=%.3f, ε=%.2f/%.2f, T=%.2f, steps=%d, lr=%.2e, loss_type=%s",
            self.G, self.beta, self.clip_eps, self.clip_eps_high,
            self.temperature, self.num_steps, self.lr, self.loss_type,
        )
        t0 = time.perf_counter()
        n_skipped = 0

        for step in range(resume_step + 1, self.num_steps + 1):
            step_t0 = time.perf_counter()
            stats = self._step()
            step_dt = time.perf_counter() - step_t0
            self.step_log.append({"step": step, "step_time": step_dt, **stats})

            if stats.get("skipped"):
                n_skipped += 1

            # ── 日志（每步都打印，含 ETA）────────────────────────────
            elapsed = time.perf_counter() - t0
            done_steps = step - resume_step
            remaining = self.num_steps - step
            eta_sec = (elapsed / done_steps) * remaining if done_steps > 0 else 0
            eta_min = eta_sec / 60
            skip_rate = n_skipped / done_steps * 100

            skipped_tag = " SKIP" if stats.get("skipped") else ""
            vram_mb = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0

            logger.info(
                "[%d/%d] %.1fs/step | loss=%.4f ploss=%.4f kl=%.4f | "
                "r=%.3f task=%s | skip=%.0f%% | ETA=%.1fmin | VRAM=%.0fMB%s",
                step, self.num_steps, step_dt,
                stats["loss"], stats["policy_loss"], stats["kl"],
                stats["mean_reward"], stats["task"],
                skip_rate, eta_min, vram_mb, skipped_tag,
            )

            # ── 评测 ────────────────────────────────────────────────────
            if step % self.eval_steps == 0:
                self._run_eval(step)

            # ── 保存 checkpoint ─────────────────────────────────────────
            if step % self.save_steps == 0:
                ckpt_dir = self.output_dir / f"checkpoint-{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                # 保存 LoRA / PEFT adapter（若模型是 PeftModel）
                if hasattr(self.model, "save_pretrained"):
                    self.model.save_pretrained(str(ckpt_dir))
                else:
                    torch.save(self.model.state_dict(), str(ckpt_dir / "model.pt"))
                
                # 额外保存 TRA 参数（如果存在）
                from text_rich_mllm.models.qwen_with_tra import save_tra_state
                save_tra_state(self.model, str(ckpt_dir / "tra_state.pt"))
                
                logger.info("Saved checkpoint: %s", ckpt_dir)

        # ── 最终保存 ─────────────────────────────────────────────────────
        final_dir = self.output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(final_dir))
        else:
            torch.save(self.model.state_dict(), str(final_dir / "model.pt"))
            
        # 额外保存 TRA 参数（如果存在）
        from text_rich_mllm.models.qwen_with_tra import save_tra_state
        save_tra_state(self.model, str(final_dir / "tra_state.pt"))

        if hasattr(self.processor, "save_pretrained"):
            self.processor.save_pretrained(str(final_dir))

        # ── 保存训练统计 ──────────────────────────────────────────────────
        import json
        stats_path = self.output_dir / "grpo_step_log.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(self.step_log, f, ensure_ascii=False, indent=2)
        logger.info("TS-GRPO training complete. Stats saved to %s", stats_path)

        # ── 训练结束：按 task 汇总平均 reward（论文表格内容）──────────────
        task_rewards: dict[str, list[float]] = defaultdict(list)
        task_losses: dict[str, list[float]] = defaultdict(list)
        for entry in self.step_log:
            task_rewards[entry["task"]].append(entry["mean_reward"])
            task_losses[entry["task"]].append(entry["loss"])
        line = "=" * 70
        print(f"\n{line}")
        print("  TS-GRPO TRAINING SUMMARY")
        print(f"  Total steps : {len(self.step_log)}")
        print(f"  Output dir  : {self.output_dir}")
        print(f"  Stats file  : {stats_path}")
        print(f"  {'Task':<20} {'Steps':>6}  {'mean_reward':>12}  {'std_reward':>12}  {'mean_loss':>10}")
        print(f"  {'-'*20} {'-'*6}  {'-'*12}  {'-'*12}  {'-'*10}")
        for task in sorted(task_rewards):
            rs = task_rewards[task]
            ls = task_losses[task]
            mean_r = sum(rs) / len(rs)
            std_r = (sum((x - mean_r) ** 2 for x in rs) / max(len(rs) - 1, 1)) ** 0.5
            mean_l = sum(ls) / len(ls)
            print(f"  {task:<20} {len(rs):>6}  {mean_r:>12.4f}  {std_r:>12.4f}  {mean_l:>10.4f}")
        all_r = [e["mean_reward"] for e in self.step_log]
        overall_mean = sum(all_r) / max(len(all_r), 1)
        print(f"  {'OVERALL':<20} {len(all_r):>6}  {overall_mean:>12.4f}")
        print(f"{line}\n")
