"""
可视化模块 v2：绘制实验2 Phase 1A-1C 的"层 x 准确率"曲线。

绘制 4 条线叠加：
  - 实验0 基线（GPT-2 原始数据）
  - Phase 1A（IFEval 4 选项修复）
  - Phase 1B（within-MMLU skill vs knowledge）
  - Phase 1C（within-MMLU 纯文本科目）

输出:
  - results/phase1_comparison.png  — 4 线叠加图
  - results/phase1_layer0_comparison.png  — Layer 0 准确率柱状图
  - results/phase1_go_nogo_summary.txt  — go/no-go 判定摘要

用法:
    python plot_phase1.py                                    # 使用默认路径
    python plot_phase1.py --exp0-run run_1783686552          # 指定实验0 run
"""

import argparse
import json
import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def find_run_with_probe_results(results_dir):
    """在目录下找到包含 probe_results.json 的最新 run。"""
    # 先检查直接在 results_dir 下
    probe_path = os.path.join(results_dir, "probe_results.json")
    if os.path.exists(probe_path):
        return results_dir

    # 检查子目录（run_* 格式）
    run_dirs = sorted(glob.glob(os.path.join(results_dir, "run_*")))
    for run_dir in reversed(run_dirs):
        if os.path.exists(os.path.join(run_dir, "probe_results.json")):
            return run_dir

    return None


def load_probe_results(results_dir):
    """从结果目录加载 probe_results.json。"""
    run_dir = find_run_with_probe_results(results_dir)
    if run_dir is None:
        return None

    probe_path = os.path.join(run_dir, "probe_results.json")
    with open(probe_path, "r") as f:
        return json.load(f)


def extract_layer_accuracies(probe_data):
    """提取每层 CV 准确率。"""
    if probe_data is None:
        return [], []

    num_layers = probe_data["num_layers"]
    layers = list(range(num_layers))
    cv_means = []

    for layer_idx in layers:
        lr = probe_data["layer_results"][str(layer_idx)]
        cv_means.append(lr["cv_mean_accuracy"])

    return layers, cv_means


def plot_phase1_comparison(exp0_data, phase1a_data, phase1b_data, phase1c_data, output_dir):
    """绘制4线叠加的层 x 准确率曲线。"""
    fig, ax = plt.subplots(figsize=(12, 7))

    datasets = [
        (exp0_data, "Exp0 Baseline (GPT-2)", "#9E9E9E", "--", "o"),
        (phase1a_data, "Phase 1A (Qwen2.5-3B, IFEval 4-opt)", "#2196F3", "-", "s"),
        (phase1b_data, "Phase 1B (Qwen2.5-3B, within-MMLU)", "#FF9800", "-", "^"),
        (phase1c_data, "Phase 1C (Qwen2.5-3B, text-only)", "#4CAF50", "-", "D"),
    ]

    for data, label, color, linestyle, marker in datasets:
        layers, cv_means = extract_layer_accuracies(data)
        if layers:
            ax.plot(layers, cv_means, linestyle=linestyle, marker=marker,
                    label=label, color=color, linewidth=2, markersize=6)

    # 关键阈值线
    ax.axhline(y=0.75, color="#4CAF50", linestyle=":", alpha=0.5, label="Go Threshold (75%)")
    ax.axhline(y=0.65, color="#F44336", linestyle=":", alpha=0.5, label="No-Go Threshold (65%)")
    ax.axhline(y=0.50, color="#9E9E9E", linestyle=":", alpha=0.3, label="Random (50%)")

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Probe CV Accuracy", fontsize=12)
    ax.set_title("Experiment 2: Format-Controlled Diagnosis\n"
                 "Layer x Accuracy (Skill vs Knowledge Probe)", fontsize=14)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "phase1_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Phase 1 comparison saved to {out_path}")


def plot_layer0_bar(exp0_data, phase1a_data, phase1b_data, phase1c_data, output_dir):
    """绘制 Layer 0 准确率柱状图。"""
    phases = ["Exp0\nBaseline", "Phase 1A\n(IFEval\n4-opt)", "Phase 1B\n(within\nMMLU)", "Phase 1C\n(text\nonly)"]
    layer0_accs = []

    for data in [exp0_data, phase1a_data, phase1b_data, phase1c_data]:
        if data is not None and "0" in data.get("layer_results", {}):
            acc = data["layer_results"]["0"]["cv_mean_accuracy"]
            layer0_accs.append(acc)
        else:
            layer0_accs.append(0)

    colors = ["#9E9E9E", "#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(range(len(phases)), layer0_accs, color=colors, edgecolor="black", width=0.6)

    for bar, acc in zip(bars, layer0_accs):
        ax.annotate(f"{acc:.2%}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(phases, fontsize=10)
    ax.set_ylabel("Layer 0 Probe Accuracy", fontsize=12)
    ax.set_title("Layer 0 Accuracy by Phase\n(Expecting progressive decrease as confounds are removed)", fontsize=13)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # 添加混淆源消除标注
    confound_labels = [
        "C1-C4 present",
        "C1 removed",
        "C2-C4 removed",
        "All removed",
    ]
    for i, label in enumerate(confound_labels):
        ax.annotate(label, xy=(i, 0.02), ha="center", fontsize=8, color="white", fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "phase1_layer0_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Layer 0 comparison saved to {out_path}")


def generate_go_nogo_summary(phase1c_data, output_dir):
    """生成 Phase 1C 的 go/no-go 判定摘要。"""
    if phase1c_data is None:
        print("[WARN] No Phase 1C data available for go/no-go assessment")
        return

    num_layers = phase1c_data["num_layers"]
    mid_high_start = num_layers // 2  # >50% depth

    layer_accuracies = {}
    for layer_idx in range(num_layers):
        acc = phase1c_data["layer_results"][str(layer_idx)]["cv_mean_accuracy"]
        layer_accuracies[layer_idx] = acc

    # 评估 go/no-go
    mid_high_max = max(acc for layer, acc in layer_accuracies.items() if layer >= mid_high_start)
    all_max = max(layer_accuracies.values())

    go_threshold = 0.75
    nogo_threshold = 0.65

    if mid_high_max > go_threshold:
        decision = "GO"
        reason = f"Mid-high layer accuracy {mid_high_max:.4f} > {go_threshold} — functional separation signal detected"
    elif all_max < nogo_threshold:
        decision = "NO-GO (H2 FALSIFIED)"
        reason = f"All layer accuracy {all_max:.4f} < {nogo_threshold} — skill/knowledge not linearly separable"
    else:
        decision = "AMBIGUOUS"
        reason = f"Low layers high ({all_max:.4f}), mid-high layers < {go_threshold} — residual confound suspected"

    summary_lines = [
        "=" * 60,
        "Phase 1C Go/No-Go Assessment",
        "=" * 60,
        f"Model: {phase1c_data.get('model', 'unknown')}",
        f"Number of layers: {num_layers}",
        f"Mid-high layer range: {mid_high_start}-{num_layers-1}",
        f"Best mid-high layer accuracy: {mid_high_max:.4f}",
        f"Best overall layer accuracy: {all_max:.4f}",
        "",
        f"DECISION: {decision}",
        f"REASON: {reason}",
        "",
        "Per-layer accuracies:",
    ]

    for layer_idx in range(num_layers):
        acc = layer_accuracies[layer_idx]
        marker = " <-- mid-high" if layer_idx >= mid_high_start else ""
        marker += " *" if acc > go_threshold else ""
        summary_lines.append(f"  Layer {layer_idx:2d}: {acc:.4f}{marker}")

    summary_lines.extend([
        "",
        "Thresholds:",
        f"  Go: mid-high layer accuracy > {go_threshold}",
        f"  No-Go: all layers < {nogo_threshold}",
        "=" * 60,
    ])

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    out_path = os.path.join(output_dir, "phase1_go_nogo_summary.txt")
    with open(out_path, "w") as f:
        f.write(summary_text)
    print(f"[Save] Go/No-Go summary saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 1 comparison plots")
    parser.add_argument("--exp0-run", type=str, default=None, help="Experiment 0 run ID")
    parser.add_argument("--results-dir", type=str, default=None, help="Base results directory")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_base = args.results_dir or os.path.join(base_dir, "results")

    # 加载实验0基线
    exp0_data = None
    if args.exp0_run:
        exp0_dir = os.path.join(results_base, args.exp0_run)
    else:
        # 查找最早的 run（通常是实验0）
        run_dirs = sorted(glob.glob(os.path.join(results_base, "run_*")))
        for run_dir in run_dirs:
            # 排除 phase1 子目录
            if "phase1" in run_dir:
                continue
            if os.path.exists(os.path.join(run_dir, "probe_results.json")):
                exp0_dir = run_dir
                break
        else:
            exp0_dir = None

    if exp0_dir:
        probe_path = os.path.join(exp0_dir, "probe_results.json")
        if os.path.exists(probe_path):
            with open(probe_path, "r") as f:
                exp0_data = json.load(f)
            print(f"[Load] Exp0 data from {exp0_dir}")

    # 加载 Phase 1A/B/C 数据
    phase1a_data = load_probe_results(os.path.join(results_base, "phase1a"))
    phase1b_data = load_probe_results(os.path.join(results_base, "phase1b"))
    phase1c_data = load_probe_results(os.path.join(results_base, "phase1c"))

    if phase1a_data:
        print(f"[Load] Phase 1A data loaded")
    if phase1b_data:
        print(f"[Load] Phase 1B data loaded")
    if phase1c_data:
        print(f"[Load] Phase 1C data loaded")

    # 确保输出目录存在
    os.makedirs(results_base, exist_ok=True)

    # 绘制图表
    plot_phase1_comparison(exp0_data, phase1a_data, phase1b_data, phase1c_data, results_base)
    plot_layer0_bar(exp0_data, phase1a_data, phase1b_data, phase1c_data, results_base)

    # 生成 go/no-go 摘要
    generate_go_nogo_summary(phase1c_data, results_base)

    print("\n[Done] All Phase 1 visualizations generated.")


if __name__ == "__main__":
    main()
