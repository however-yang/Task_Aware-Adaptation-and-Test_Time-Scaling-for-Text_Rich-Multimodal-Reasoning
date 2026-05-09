"""
generate_report_figures.py — Publication-quality figures for Final Report (Morandi Edition)

Produces separated figures with academic-grade aesthetics and a Morandi color palette:
  1. fig1: Two-panel grouped bar chart (In-Domain vs Cross-Domain)
  2. fig2a: GRPO training reward
  3. fig2b: GRPO batch skip rate
  4. fig2c: GRPO checkpoint evaluation (unified to 12k/16k/20k, legend top-right)
  5. fig3: BoN scaling curve (fixed text overlaps)
  6. fig4: Error taxonomy horizontal stacked bars (all numbers shown, Morandi palette)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ══════════════════════════════════════════════════════════════════════════
#  Global Style
# ══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Palatino"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 9.5,
    "legend.framealpha": 0.9,
    "figure.dpi": 200,
    "savefig.dpi": 300,
})

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── 莫兰迪色系 (Morandi Palette) ────────────────────────────────────────────────
# 降低饱和度、带灰调的优雅颜色
C = {
    "base_d":  "#C1B7DB",  
    "lora":    "#A1CCEB",  
    "dora":    "#BEDBAD",  
    "tra":     "#EEC0A0",  
    "grpo":    "#D0847D",  
    
    # 任务对应的配色
    "doc":     "#A1CCEB",
    "chart":   "#BEDBAD",
    "sci":     "#EEC0A0",
    "mmmu":    "#C1B7DB",  
    "overall": "#D0847D",
    
    # 错误分类配色
    "correct":       "#BEDBAD",
    "text_read":     "#A1CCEB",
    "chart_reason":  "#EEC0A0",
    "sci_reason":    "#C1B7DB",
    "output_mm":     "#D0847D",
}

def _style_ax(ax, grid_axis="y"):
    """Apply clean academic styling to an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    if grid_axis:
        ax.grid(axis=grid_axis, alpha=0.3, linestyle="--", linewidth=0.6, color="#B0AEAC")
    ax.tick_params(direction="out", length=4, width=0.6, colors="#444444")
    ax.xaxis.label.set_color("#333333")
    ax.yaxis.label.set_color("#333333")


# ══════════════════════════════════════════════════════════════════════════
#  Figure 1: Two-Panel — In-Domain (DocVQA, ChartQA) + Cross-Domain
# ══════════════════════════════════════════════════════════════════════════
def fig1_main_results():
    stages = ["Baseline", "E1\nLoRA", "E2\nDoRA", "E5\nTRA", "E8\nGRPO"]
    
    # In-domain scores
    doc_scores  = [0.1218, 0.7096, 0.7389, 0.8100, 0.9190]
    chart_scores = [0.0760, 0.6792, 0.7312, 0.7427, 0.7682]

    # Cross-domain scores
    sci_scores  = [0.6822, 0.7980, 0.7888, 0.7893, 0.7871]
    mmmu_scores = [0.2900, 0.4411, 0.4300, 0.4256, 0.4156]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8),
                                    gridspec_kw={"width_ratios": [1, 1], "wspace": 0.28})

    x = np.arange(len(stages))
    w = 0.32

    # ── Panel A: In-Domain ────────────────────────────────────
    bars_d = ax1.bar(x - w/2, doc_scores, w, color=C["doc"], edgecolor="#AAAAAA",
                     linewidth=0.8, label="DocVQA (ANLS)", zorder=3)
    bars_c = ax1.bar(x + w/2, chart_scores, w, color=C["chart"], edgecolor="#AAAAAA",
                     linewidth=0.8, label="ChartQA (Acc)", zorder=3)

    for bars, scores in [(bars_d, doc_scores), (bars_c, chart_scores)]:
        for bar, v in zip(bars, scores):
            offset = 0.015
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                     f"{v:.1%}" if v >= 0.10 else f"{v:.2%}",
                     ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                     color="#555555")

    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=10)
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax1.set_title("(a) In-Domain Benchmarks", pad=12)
    ax1.legend(loc="upper left", frameon=True, edgecolor="#E0E0E0")
    _style_ax(ax1)

    # ── Panel B: Cross-Domain ─────────────────────────────────
    bars_s = ax2.bar(x - w/2, sci_scores, w, color=C["sci"], edgecolor="#AAAAAA",
                     linewidth=0.8, label="ScienceQA (Acc)", zorder=3)
    bars_m = ax2.bar(x + w/2, mmmu_scores, w, color=C["mmmu"], edgecolor="#AAAAAA",
                     linewidth=0.8, label="MMMU (Acc)", zorder=3)

    for bars, scores in [(bars_s, sci_scores), (bars_m, mmmu_scores)]:
        for bar, v in zip(bars, scores):
            offset = 0.015
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                     f"{v:.1%}",
                     ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                     color="#555555")

    ax2.set_xticks(x)
    ax2.set_xticklabels(stages, fontsize=10)
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2.set_title("(b) Cross-Domain Generalization", pad=12)
    ax2.legend(loc="upper left", frameon=True, edgecolor="#E0E0E0")
    _style_ax(ax2)

    fig.suptitle("Performance Across Training Stages on Text-Rich Benchmarks",
                 fontsize=14, fontweight="bold", y=1.02)

    out = OUT / "fig1_main_results_bar.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] {out}")


# ══════════════════════════════════════════════════════════════════════════
#  Figure 2: GRPO Training Dynamics — Split into 3 files
# ══════════════════════════════════════════════════════════════════════════
def generate_grpo_figures():
    log_json = ROOT / "outputs" / "analysis" / "grpo_training_log_parsed.json"
    with open(log_json, encoding="utf-8") as f:
        raw = json.load(f)

    steps   = np.array([d["step"] for d in raw])
    rewards = np.array([d["reward"] for d in raw])
    skips   = np.array([d["skip"] for d in raw])
    tasks   = np.array([d["task"] for d in raw])

    def smooth(arr, w=200):
        if len(arr) < w: return arr
        return np.convolve(arr, np.ones(w)/w, mode="same")

    doc_mask = tasks == "docvqa"
    chart_mask = tasks == "chartqa"

    # --- Fig 2a: Reward ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    ax.plot(steps[doc_mask], smooth(rewards[doc_mask]), color=C["doc"], linewidth=2.5, label="DocVQA", zorder=5)
    ax.plot(steps[chart_mask], smooth(rewards[chart_mask]), color=C["chart"], linewidth=2.5, label="ChartQA", zorder=5)
    
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title("TS-GRPO Training Reward", pad=12)
    ax.set_xlim(0, 20000)
    ax.set_ylim(0.60, 1.05)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x!=0 else "0"))
    # 把图例放到左下方或中下方，因为刚开始曲线快速上升，右侧都有数据
    ax.legend(loc="lower left", frameon=True, edgecolor="#E0E0E0")
    _style_ax(ax)
    out = OUT / "fig2a_grpo_reward.png"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(OUT / "fig2_grpo_convergence.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2a] {out}")

    # --- Fig 2b: Skip Rate ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, skips, color=C["tra"], alpha=0.15, linewidth=0.5, rasterized=True)
    skip_s = smooth(skips.astype(float), w=500)
    ax.plot(steps, skip_s, color=C["tra"], linewidth=2.2, label="Batch Skip Rate", zorder=5)
    ax.fill_between(steps, skip_s, alpha=0.1, color=C["tra"])
    
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Skip Rate (%)")
    ax.set_title("TS-GRPO Batch Skip Rate", pad=12)
    ax.set_xlim(0, 20000)
    ax.set_ylim(30, 100)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x!=0 else "0"))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend(loc="lower right", frameon=True, edgecolor="#E0E0E0")
    _style_ax(ax)
    out = OUT / "fig2b_grpo_skip.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2b] {out}")

    # --- Fig 2c: Eval Score ---
    # 统一使用 12k, 16k, 20k 以保证图中各线节点一致
    unified_steps = [12000, 16000, 20000]
    eval_subset_doc = [0.9422, 0.9420, 0.9461]
    eval_full_doc   = [0.9170, 0.9206, 0.9190]
    eval_full_chart = [0.7625, 0.7641, 0.7682]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    ax.plot(unified_steps, eval_subset_doc, "o-", color=C["overall"], linewidth=2.5,
            markersize=8, label="DocVQA (200-subset)", zorder=5)
    ax.plot(unified_steps, eval_full_doc, "s--", color=C["doc"], linewidth=2.5,
            markersize=8, label="DocVQA (full val)", zorder=5)
    ax.plot(unified_steps, eval_full_chart, "^--", color=C["chart"], linewidth=2.5,
            markersize=8, label="ChartQA (full val)", zorder=5)

    # Annotate final values
    for xs, ys, color, off in [
        (unified_steps, eval_subset_doc, C["overall"], (0, 10)),
        (unified_steps, eval_full_doc, C["doc"], (0, -18)),
        (unified_steps, eval_full_chart, C["chart"], (0, 10)),
    ]:
        for xi, yi in zip(xs, ys):
            ax.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points", xytext=off,
                        ha="center", fontsize=9, fontweight="bold", color=color)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Eval Score")
    ax.set_title("TS-GRPO Checkpoint Evaluation", pad=12)
    # 调整x轴范围，留出空间
    ax.set_xlim(11000, 21500)
    # 图例放右上角，同时将Y轴上限调高，把折线图往下压
    ax.legend(loc="upper right", frameon=True, edgecolor="#E0E0E0")
    
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax.set_xticks([12000, 16000, 20000])
    
    # 动态调整y轴范围以适应图例 (缩小空白)
    ax.set_ylim(0.70, 1.05)
    _style_ax(ax)
    
    out = OUT / "fig2c_grpo_eval.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2c] {out}")


# ══════════════════════════════════════════════════════════════════════════
#  Figure 3: BoN Scaling Curve
# ══════════════════════════════════════════════════════════════════════════
def fig3_bon_scaling():
    n_vals = [1, 2, 4, 8]
    doc_scores   = [0.9220, 0.9257, 0.9315, 0.9339]
    chart_scores = [0.7656, 0.7714, 0.7844, 0.7896]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"wspace": 0.25})

    for ax, scores, ds, color, marker in [
        (ax1, doc_scores, "DocVQA (ANLS)", C["doc"], "o"),
        (ax2, chart_scores, "ChartQA (Acc)", C["chart"], "s"),
    ]:
        ax.fill_between(n_vals, [scores[0]]*4, scores, alpha=0.15, color=color)
        ax.plot(n_vals, scores, f"{marker}-", color=color, linewidth=2.5,
                markersize=9, markeredgecolor="white", markeredgewidth=1.2,
                label="E8 GRPO + BoN", zorder=5)
        ax.axhline(y=scores[0], color=color, linestyle=":", alpha=0.6, linewidth=1.5)

        # 调整标签位置，避免与线重叠
        for i, (xi, yi) in enumerate(zip(n_vals, scores)):
            xytext = (0, 12) if i != len(n_vals)-1 else (-15, 8)
            ax.annotate(f"{yi:.2%}", (xi, yi), textcoords="offset points",
                        xytext=xytext, ha="center" if i != len(n_vals)-1 else "right",
                        fontsize=9.5, fontweight="bold", color="#555555")

        gain = scores[-1] - scores[0]
        ax.annotate(f"$\\Delta$ = +{gain:.2%}", xy=(n_vals[-1], scores[-1]),
                    xytext=(n_vals[-1] - 0.5, scores[0] + (scores[-1]-scores[0])*0.4),
                    ha="right", fontsize=10, fontweight="bold", color=C["grpo"],
                    arrowprops=dict(arrowstyle="->", color=C["grpo"], lw=1.5, shrinkA=5, shrinkB=5))

        ax.set_xlabel("N (# candidates)")
        ax.set_ylabel("Score")
        ax.set_title(ds, fontweight="bold", pad=12)
        ax.set_xticks(n_vals)
        
        # 加大 Y 轴上限，防止文字被切割
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max + (y_max - y_min) * 0.15)
        
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.legend(loc="lower right", frameon=True, edgecolor="#E0E0E0")
        _style_ax(ax)

    fig.suptitle("Best-of-N Test-time Scaling (E10: GRPO Checkpoint)",
                 fontsize=14, fontweight="bold", y=1.05)

    out = OUT / "fig3_bon_scaling_curve.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] {out}")


# ══════════════════════════════════════════════════════════════════════════
#  Figure 4: Error Taxonomy
# ══════════════════════════════════════════════════════════════════════════
def fig4_error_taxonomy():
    cats = ["Correct", "Text Reading\nFailure", "Chart Reasoning\nFailure",
            "Sci. Figure\nFailure", "Output\nMismatch"]
    
    # 使用莫兰迪配色
    cat_colors = [C["correct"], C["text_read"], C["chart_reason"], C["sci_reason"], C["output_mm"]]

    bl_total = 74 + 146 + 2894 + 261 + 2426 + 1774 + 1345 + 631 + 3 + 8
    bl_vals = np.array([
        74 + 146 + 2894 + 261,  # correct
        2426,                    # text reading
        1774,                    # chart reasoning
        1345 + 631,              # sci figure
        3 + 8,                   # output mismatch
    ]) / bl_total * 100

    gr_total = 3581 + 3339 + 374 + 394 + 445 + 902 + 491 + 1 + 35
    gr_vals = np.array([
        3581 + 3339 + 374,  # correct
        394,                 # text reading
        445,                 # chart reasoning
        902 + 491,           # sci figure
        1 + 35,              # output mismatch
    ]) / gr_total * 100

    fig, ax = plt.subplots(figsize=(10, 4.8))

    y_pos = np.array([0, 1])
    labels = ["E8 GRPO\n(20k steps)", "Baseline\n(Structured)"]

    for ds_idx, (pcts, y) in enumerate([(gr_vals, 0), (bl_vals, 1)]):
        left = 0
        for i, (val, color) in enumerate(zip(pcts, cat_colors)):
            bar = ax.barh(y, val, left=left, height=0.55,
                          color=color, edgecolor="#AAAAAA", linewidth=1.2, zorder=3)
            
            # 为了确保所有数字都显示，这里优化了标签展示逻辑
            text_x = left + val/2
            text_color = "#333333"
            
            if val > 3.0: # 如果空间足够大，放在条形图内部
                ax.text(text_x, y, f"{val:.1f}%",
                        ha="center", va="center", fontsize=9, fontweight="bold", color=text_color)
            elif val > 0: # 如果空间很小，采用箭头标注引出
                ax.annotate(f"{val:.1f}%", xy=(text_x, y + 0.28), xytext=(text_x, y + 0.5),
                            ha="center", fontsize=8.5, fontweight="bold", color="#333333",
                            arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8))
            
            left += val

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_xlabel("Proportion (%)", fontsize=11)
    ax.set_xlim(0, 100.5)
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=c, edgecolor="white", label=l.replace('\\n', ' '))
                      for l, c in zip(cats, cat_colors)]
    ax.legend(handles=legend_patches, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=5, fontsize=9.5,
              frameon=False, handlelength=1.5, handleheight=1.0)

    ax.set_title("Error Taxonomy Distribution: Baseline vs. TS-GRPO",
                 fontsize=14, fontweight="bold", pad=15)
    _style_ax(ax, grid_axis=None)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    # 去除100以免与0.1%重叠
    ax.set_xticks([0, 25, 50, 75])
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

    for xv in [25, 50, 75]:
        ax.axvline(xv, color="#E8E8E8", linewidth=1.0, zorder=1)

    out = OUT / "fig4_error_taxonomy.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4] {out}")


# ══════════════════════════════════════════════════════════════════════════
#  Figure 5: Structured Prompt vs Direct Prompt Ablation
# ══════════════════════════════════════════════════════════════════════════
def fig5_prompt_ablation():
    datasets = ["DocVQA", "ChartQA", "ScienceQA", "MMMU"]
    direct   = [0.0609, 0.0755, 0.5243, 0.2122]
    struct_  = [0.1218, 0.0760, 0.6822, 0.2900]

    fig, ax = plt.subplots(figsize=(8, 4.8))

    x = np.arange(len(datasets))
    w = 0.34

    bars_d = ax.bar(x - w/2, direct,  w, color=C["mmmu"], edgecolor="#AAAAAA",
                    linewidth=0.8, label="Direct Prompt", zorder=3)
    bars_s = ax.bar(x + w/2, struct_, w, color=C["doc"],  edgecolor="#AAAAAA",
                    linewidth=0.8, label="Structured Prompt", zorder=3)

    for bars, vals in [(bars_d, direct), (bars_s, struct_)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{v:.1%}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#444444")

    # Δ gain annotations above each pair
    for i, (d, s) in enumerate(zip(direct, struct_)):
        delta = s - d
        if abs(delta) > 0.005:
            ax.annotate(f"Δ+{delta:.1%}", xy=(x[i], max(d, s) + 0.04),
                        ha="center", fontsize=8, color=C["grpo"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.90)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_title("Prompt Engineering Ablation: Direct vs. Structured Prompt (Zero-Shot Baseline)",
                 fontsize=12, fontweight="bold", pad=14)
    ax.legend(loc="upper left", frameon=True, edgecolor="#DDDDDD", fontsize=10)
    _style_ax(ax)

    out = OUT / "fig5_prompt_ablation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig5] {out}")


# ══════════════════════════════════════════════════════════════════════════
#  Figure 6: Answer-Type Breakdown across GRPO Checkpoints
# ══════════════════════════════════════════════════════════════════════════
def fig6_answer_type_breakdown():
    checkpoints  = ["12k", "16k", "20k"]
    overall      = [0.849886, 0.852618, 0.853499]
    numeric_acc  = [0.738739, 0.740125, 0.743590]
    opentext_acc = [0.903760, 0.907146, 0.906773]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    x = np.arange(len(checkpoints))
    w = 0.24

    bar_o  = ax.bar(x - w,   overall,      w, color=C["mmmu"],  edgecolor="#AAAAAA", linewidth=0.8, label="Overall",     zorder=3)
    bar_n  = ax.bar(x,       numeric_acc,  w, color=C["grpo"],  edgecolor="#AAAAAA", linewidth=0.8, label="Numeric Q",   zorder=3)
    bar_ot = ax.bar(x + w,   opentext_acc, w, color=C["chart"], edgecolor="#AAAAAA", linewidth=0.8, label="Open-Text Q", zorder=3)

    for bars, vals in [(bar_o, overall), (bar_n, numeric_acc), (bar_ot, opentext_acc)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{v:.1%}", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#444444")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Checkpoint\n{c}" for c in checkpoints], fontsize=10.5)
    ax.set_ylabel("Score")
    ax.set_ylim(0.65, 1.00)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_title("Answer-Type Performance Across GRPO Checkpoints\n(DocVQA + ChartQA Joint Eval)",
                 fontsize=12, fontweight="bold", pad=14)
    ax.legend(loc="lower right", frameon=True, edgecolor="#DDDDDD", fontsize=9.5)
    _style_ax(ax)

    # Annotate improvements
    for vals, bar_grp, color in [
        (numeric_acc, bar_n, C["grpo"]),
        (opentext_acc, bar_ot, C["chart"]),
    ]:
        gain = vals[-1] - vals[0]
        ax.annotate(f"Δ+{gain:.2%}", xy=(x[-1], vals[-1]),
                    xytext=(x[-1] + 0.35, vals[-1] - 0.01),
                    fontsize=8.5, color=color, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2, shrinkA=3, shrinkB=3))

    out = OUT / "fig6_answer_type_breakdown.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig6] {out}")


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Generating Publication-Quality Figures (Morandi Palette)")
    print("=" * 60)
    fig1_main_results()
    generate_grpo_figures()
    fig3_bon_scaling()
    fig4_error_taxonomy()
    fig5_prompt_ablation()
    fig6_answer_type_breakdown()
    print("=" * 60)
    print("  Done.")
    print("=" * 60)

