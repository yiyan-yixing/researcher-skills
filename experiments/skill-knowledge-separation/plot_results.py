"""
可视化模块：生成探针准确率曲线和消融损伤矩阵。

用法:
    python plot_results.py                       # 使用最新的 run
    python plot_results.py --run-id run_12345    # 指定 run
    python plot_results.py --results-dir results/run_12345

输出:
    results/{run_id}/probe_accuracy_curve.png
    results/{run_id}/ablation_heatmap.png
    results/{run_id}/damage_bar_chart.png
"""

import argparse
import json
import os
import glob

import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import yaml


def find_latest_run(results_dir):
    """找到最新的 run 目录。"""
    run_dirs = sorted(glob.glob(os.path.join(results_dir, "run_*")))
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {results_dir}")
    return os.path.basename(run_dirs[-1])


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def plot_probe_accuracy_curve(probe_results, output_path):
    """
    绘制每层探针分类准确率曲线。

    X 轴: 层数 (0 to N-1)
    Y 轴: 分类准确率
    包含: CV mean accuracy (with std error bars) + Test accuracy
    """
    num_layers = probe_results["num_layers"]
    layers = list(range(num_layers))

    cv_means = []
    cv_stds = []
    test_accs = []

    for layer_idx in layers:
        lr = probe_results["layer_results"][str(layer_idx)]
        cv_means.append(lr["cv_mean_accuracy"])
        cv_stds.append(lr["cv_std_accuracy"])
        test_accs.append(lr["test_accuracy"])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(layers, cv_means, "o-", label="CV Mean Accuracy", color="#2196F3", linewidth=2, markersize=6)
    ax.fill_between(layers,
                    [m - s for m, s in zip(cv_means, cv_stds)],
                    [m + s for m, s in zip(cv_means, cv_stds)],
                    alpha=0.2, color="#2196F3")
    ax.plot(layers, test_accs, "s--", label="Test Accuracy", color="#FF9800", linewidth=2, markersize=6)

    # 标记最佳层
    best_layer = probe_results["best_layer"]
    best_acc = cv_means[best_layer]
    ax.axvline(x=best_layer, color="#4CAF50", linestyle=":", alpha=0.7, label=f"Best Layer ({best_layer})")
    ax.scatter([best_layer], [best_acc], s=200, color="#4CAF50", zorder=5, edgecolors="black", linewidths=1.5)

    # 放弃阈值线
    ax.axhline(y=0.60, color="#F44336", linestyle="--", alpha=0.5, label="Abort Threshold (60%)")

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Classification Accuracy", fontsize=12)
    ax.set_title("Linear Probe Accuracy by Layer\n(Skill vs Knowledge Classification)", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Probe accuracy curve saved to {output_path}")


def plot_ablation_heatmap(ablation_results, output_path):
    """
    绘制消融损伤矩阵热力图（3 行 x 4 列）。

    行: 知识方向扰动, 技能方向扰动, 投影消融(对照)
    列: GSM8K, IFEval, MMLU, TriviaQA
    值: 准确率变化 (损伤 = 基线 - 消融后)
    """
    results = ablation_results["results"]

    # 从 predictions 中按 benchmark 计算准确率
    baselines = results["baseline"].get("predictions", [])
    knowledge_abl = results["knowledge_ablation"].get("predictions", [])
    skill_abl = results["skill_ablation"].get("predictions", [])
    proj_abl = results.get("projection_ablation", {}).get("predictions", [])

    benchmarks = ["gsm8k", "ifeval", "mmlu", "triviaqa"]

    def benchmark_accuracy(predictions, benchmark_name):
        preds = [p for p in predictions if p["benchmark"] == benchmark_name]
        if not preds:
            return 0.0
        return sum(1 for p in preds if p["correct"]) / len(preds)

    # 构建损伤矩阵
    # 行: [知识方向扰动, 技能方向扰动, 投影消融(对照)]
    # 列: [GSM8K, IFEval, MMLU, TriviaQA]
    num_rows = 3 if proj_abl else 2
    damage_matrix = np.zeros((num_rows, 4))

    for j, bm in enumerate(benchmarks):
        base_acc = benchmark_accuracy(baselines, bm)
        knowledge_acc = benchmark_accuracy(knowledge_abl, bm)
        skill_acc = benchmark_accuracy(skill_abl, bm)
        damage_matrix[0, j] = base_acc - knowledge_acc  # 知识方向扰动的损伤
        damage_matrix[1, j] = base_acc - skill_acc      # 技能方向扰动的损伤
        if proj_abl:
            proj_acc = benchmark_accuracy(proj_abl, bm)
            damage_matrix[2, j] = base_acc - proj_acc   # 投影消融的损伤

    fig, ax = plt.subplots(figsize=(10, 5 if proj_abl else 4))

    im = ax.imshow(damage_matrix, cmap="RdYlGn_r", aspect="auto", vmin=-0.3, vmax=0.3)

    ax.set_xticks(range(4))
    ax.set_xticklabels(["GSM8K\n(Skill)", "IFEval\n(Skill)", "MMLU\n(Knowledge)", "TriviaQA\n(Knowledge)"],
                       fontsize=10)

    row_labels = ["Knowledge\nPerturbation", "Skill\nPerturbation"]
    if proj_abl:
        row_labels.append("Projection\nAblation (Ctrl)")
    ax.set_yticks(range(num_rows))
    ax.set_yticklabels(row_labels, fontsize=10)

    # 在每个格子上标注数值
    for i in range(num_rows):
        for j in range(4):
            val = damage_matrix[i, j]
            text_color = "white" if abs(val) > 0.15 else "black"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=12, color=text_color, fontweight="bold")

    ax.set_title("Ablation Damage Matrix\n(Accuracy Drop per Benchmark)", fontsize=14)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy Drop", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Ablation heatmap saved to {output_path}")


def plot_damage_bar_chart(ablation_results, output_path):
    """
    绘制消融损伤柱状图：按 skill/knowledge 类别聚合，
    包含投影消融对照。
    """
    selectivity = ablation_results.get("selectivity", {})

    categories = ["Skill Tasks", "Knowledge Tasks"]
    x = np.arange(len(categories))
    width = 0.20

    baseline_accs = [
        selectivity.get("baseline_skill_acc", 0),
        selectivity.get("baseline_knowledge_acc", 0),
    ]
    knowledge_abl_accs = [
        selectivity.get("knowledge_ablation", {}).get("skill_acc", 0),
        selectivity.get("knowledge_ablation", {}).get("knowledge_acc", 0),
    ]
    skill_abl_accs = [
        selectivity.get("skill_ablation", {}).get("skill_acc", 0),
        selectivity.get("skill_ablation", {}).get("knowledge_acc", 0),
    ]
    proj_abl_accs = [
        selectivity.get("projection_ablation", {}).get("skill_acc", 0),
        selectivity.get("projection_ablation", {}).get("knowledge_acc", 0),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - 1.5 * width, baseline_accs, width, label="Baseline", color="#9E9E9E", edgecolor="black")
    bars2 = ax.bar(x - 0.5 * width, knowledge_abl_accs, width, label="Knowledge Perturbation", color="#2196F3", edgecolor="black")
    bars3 = ax.bar(x + 0.5 * width, skill_abl_accs, width, label="Skill Perturbation", color="#FF9800", edgecolor="black")
    bars4 = ax.bar(x + 1.5 * width, proj_abl_accs, width, label="Projection Ablation (Ctrl)", color="#F44336", edgecolor="black")

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy by Task Category under Different Ablation Conditions", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    # 在柱子上标注数值
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Damage bar chart saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate experiment visualizations")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--run-id", type=str, default=None, help="Run identifier")
    parser.add_argument("--results-dir", type=str, default=None, help="Direct path to results directory")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(args.config)

    # 确定结果目录
    if args.results_dir:
        results_dir = args.results_dir
    else:
        runs_base = os.path.join(base_dir, config["output"]["results_dir"])
        run_id = args.run_id or find_latest_run(runs_base)
        results_dir = os.path.join(runs_base, run_id)

    if not os.path.isdir(results_dir):
        print(f"[Error] Results directory not found: {results_dir}")
        return

    print(f"[Plot] Using results from: {results_dir}")

    # 1. 探针准确率曲线
    probe_results_path = os.path.join(results_dir, "probe_results.json")
    if os.path.exists(probe_results_path):
        with open(probe_results_path, "r") as f:
            probe_results = json.load(f)
        plot_probe_accuracy_curve(probe_results, os.path.join(results_dir, "probe_accuracy_curve.png"))
    else:
        print(f"[Skip] Probe results not found: {probe_results_path}")

    # 2. 消融热力图
    ablation_results_path = os.path.join(results_dir, "ablation_results.json")
    if os.path.exists(ablation_results_path):
        with open(ablation_results_path, "r") as f:
            ablation_results = json.load(f)
        plot_ablation_heatmap(ablation_results, os.path.join(results_dir, "ablation_heatmap.png"))
        plot_damage_bar_chart(ablation_results, os.path.join(results_dir, "damage_bar_chart.png"))
    else:
        print(f"[Skip] Ablation results not found: {ablation_results_path}")

    print("\n[Done] All plots generated.")


if __name__ == "__main__":
    main()
