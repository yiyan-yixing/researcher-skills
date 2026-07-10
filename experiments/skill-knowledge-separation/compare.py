"""
对比两个 run 的结果。

用法:
    python compare.py run_12345 run_67890
    python compare.py run_12345 run_67890 --metric probe
    python compare.py run_12345 run_67890 --metric ablation

输出:
  - 控制台对比表格
  - results/comparison_{run1}_vs_{run2}.json
"""

import argparse
import json
import os
import sys

import numpy as np


def load_run_results(results_dir, run_id):
    """加载一个 run 的所有结果。"""
    run_dir = os.path.join(results_dir, run_id)
    if not os.path.isdir(run_dir):
        print(f"[Error] Run directory not found: {run_dir}")
        return None

    data = {"run_id": run_id, "dir": run_dir}

    # 加载配置
    config_path = os.path.join(run_dir, "config.yaml")
    if os.path.exists(config_path):
        import yaml
        with open(config_path, "r") as f:
            data["config"] = yaml.safe_load(f)

    # 加载探针结果
    probe_path = os.path.join(run_dir, "probe_results.json")
    if os.path.exists(probe_path):
        with open(probe_path, "r") as f:
            data["probe"] = json.load(f)

    # 加载消融结果
    ablation_path = os.path.join(run_dir, "ablation_results.json")
    if os.path.exists(ablation_path):
        with open(ablation_path, "r") as f:
            data["ablation"] = json.load(f)

    # 加载摘要
    summary_path = os.path.join(run_dir, "run_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            data["summary"] = json.load(f)

    return data


def compare_probes(run1, run2):
    """对比两个 run 的探针结果。"""
    probe1 = run1.get("probe", {})
    probe2 = run2.get("probe", {})

    if not probe1 or not probe2:
        print("[Skip] One or both runs missing probe results")
        return

    num_layers1 = probe1.get("num_layers", 0)
    num_layers2 = probe2.get("num_layers", 0)

    model1 = run1.get("config", {}).get("model", {}).get("name", "unknown")
    model2 = run2.get("config", {}).get("model", {}).get("name", "unknown")

    print("\n" + "=" * 80)
    print("PROBE ACCURACY COMPARISON")
    print("=" * 80)
    print(f"{'Layer':<8} | {model1} CV Acc | {model2} CV Acc | Difference")
    print("-" * 80)

    max_layers = max(num_layers1, num_layers2)
    differences = []

    for layer in range(max_layers):
        acc1 = probe1.get("layer_results", {}).get(str(layer), {}).get("cv_mean_accuracy", None)
        acc2 = probe2.get("layer_results", {}).get(str(layer), {}).get("cv_mean_accuracy", None)

        if acc1 is not None and acc2 is not None:
            diff = acc2 - acc1
            differences.append(diff)
            marker = " *" if abs(diff) > 0.05 else "  "
            print(f"{layer:<8} | {acc1:.4f}      | {acc2:.4f}      | {diff:+.4f}{marker}")
        elif acc1 is not None:
            print(f"{layer:<8} | {acc1:.4f}      | N/A         | ---")
        elif acc2 is not None:
            print(f"{layer:<8} | N/A         | {acc2:.4f}      | ---")

    print("-" * 80)
    print(f"Best Layer: {probe1.get('best_layer', 'N/A')} vs {probe2.get('best_layer', 'N/A')}")
    print(f"Best Accuracy: {probe1.get('best_cv_accuracy', 0):.4f} vs {probe2.get('best_cv_accuracy', 0):.4f}")

    if differences:
        print(f"Mean Difference: {np.mean(differences):+.4f}")
        print(f"Max Difference:   {np.max(np.abs(differences)):+.4f} (layer {np.argmax(np.abs(differences))})")


def compare_ablations(run1, run2):
    """对比两个 run 的消融结果。"""
    abl1 = run1.get("ablation", {})
    abl2 = run2.get("ablation", {})

    if not abl1 or not abl2:
        print("[Skip] One or both runs missing ablation results")
        return

    model1 = run1.get("config", {}).get("model", {}).get("name", "unknown")
    model2 = run2.get("config", {}).get("model", {}).get("name", "unknown")

    print("\n" + "=" * 80)
    print("ABLATION COMPARISON")
    print("=" * 80)

    sel1 = abl1.get("selectivity", {})
    sel2 = abl2.get("selectivity", {})

    def fmt(val, width=15):
        """Format a value for the comparison table."""
        if isinstance(val, float):
            return f"{val:.4f}".ljust(width)
        return str(val).ljust(width)

    # 基线对比
    print(f"\n{'Metric':<35} | {model1:<20} | {model2:<20}")
    print("-" * 80)
    print(f"{'Baseline Skill Acc':<35} | {fmt(sel1.get('baseline_skill_acc', 0)):<20} | {fmt(sel2.get('baseline_skill_acc', 0))}")
    print(f"{'Baseline Knowledge Acc':<35} | {fmt(sel1.get('baseline_knowledge_acc', 0)):<20} | {fmt(sel2.get('baseline_knowledge_acc', 0))}")

    print(f"\n{'Knowledge Perturbation:':<35}")
    ka1 = sel1.get("knowledge_ablation", {})
    ka2 = sel2.get("knowledge_ablation", {})
    print(f"{'  Skill Damage':<35} | {fmt(ka1.get('skill_damage', 0)):<20} | {fmt(ka2.get('skill_damage', 0))}")
    print(f"{'  Knowledge Damage':<35} | {fmt(ka1.get('knowledge_damage', 0)):<20} | {fmt(ka2.get('knowledge_damage', 0))}")

    print(f"\n{'Skill Perturbation:':<35}")
    sa1 = sel1.get("skill_ablation", {})
    sa2 = sel2.get("skill_ablation", {})
    print(f"{'  Skill Damage':<35} | {fmt(sa1.get('skill_damage', 0)):<20} | {fmt(sa2.get('skill_damage', 0))}")
    print(f"{'  Knowledge Damage':<35} | {fmt(sa1.get('knowledge_damage', 0)):<20} | {fmt(sa2.get('knowledge_damage', 0))}")

    # 投影消融对照
    pa1 = sel1.get("projection_ablation", {})
    pa2 = sel2.get("projection_ablation", {})
    if pa1 or pa2:
        print(f"\n{'Projection Ablation (Control):':<35}")
        print(f"{'  Skill Damage':<35} | {fmt(pa1.get('skill_damage', 0)):<20} | {fmt(pa2.get('skill_damage', 0))}")
        print(f"{'  Knowledge Damage':<35} | {fmt(pa1.get('knowledge_damage', 0)):<20} | {fmt(pa2.get('knowledge_damage', 0))}")

    print(f"\n{'Selectivity Ratios:':<35}")
    print(f"{'  Knowledge-dir Selectivity':<35} | {sel1.get('knowledge_direction_selectivity', 0):.2f}:1".ljust(57) + f"| {sel2.get('knowledge_direction_selectivity', 0):.2f}:1")
    print(f"{'  Skill-dir Selectivity':<35} | {sel1.get('skill_direction_selectivity', 0):.2f}:1".ljust(57) + f"| {sel2.get('skill_direction_selectivity', 0):.2f}:1")

    # Abort status
    print(f"\n{'Abort Status:':<35}")
    print(f"{'  Exp0 Aborted':<35} | {str(abl1.get('exp0_aborted', 'N/A')):<20} | {str(abl2.get('exp0_aborted', 'N/A'))}")
    print(f"{'  Exp1 Aborted':<35} | {str(abl1.get('aborted', 'N/A')):<20} | {str(abl2.get('aborted', 'N/A'))}")


def main():
    parser = argparse.ArgumentParser(description="Compare two experiment runs")
    parser.add_argument("run1", type=str, help="First run ID")
    parser.add_argument("run2", type=str, help="Second run ID")
    parser.add_argument("--results-dir", type=str, default=None, help="Results base directory")
    parser.add_argument("--metric", type=str, choices=["probe", "ablation", "all"], default="all",
                        help="Which metric to compare")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 确定结果目录
    import yaml
    config_path = os.path.join(base_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    results_base = args.results_dir or os.path.join(base_dir, config["output"]["results_dir"])

    run1 = load_run_results(results_base, args.run1)
    run2 = load_run_results(results_base, args.run2)

    if run1 is None or run2 is None:
        sys.exit(1)

    if args.metric in ("probe", "all"):
        compare_probes(run1, run2)

    if args.metric in ("ablation", "all"):
        compare_ablations(run1, run2)

    # 保存对比结果
    comparison = {
        "run1": args.run1,
        "run2": args.run2,
        "model1": run1.get("config", {}).get("model", {}).get("name", "unknown"),
        "model2": run2.get("config", {}).get("model", {}).get("name", "unknown"),
    }

    if "probe" in run1 and "probe" in run2:
        comparison["probe_comparison"] = {
            "best_layer": [run1["probe"].get("best_layer"), run2["probe"].get("best_layer")],
            "best_accuracy": [run1["probe"].get("best_cv_accuracy"), run2["probe"].get("best_cv_accuracy")],
        }

    if "ablation" in run1 and "ablation" in run2:
        comparison["ablation_comparison"] = {
            "knowledge_selectivity": [
                run1["ablation"].get("selectivity", {}).get("knowledge_direction_selectivity"),
                run2["ablation"].get("selectivity", {}).get("knowledge_direction_selectivity"),
            ],
            "skill_selectivity": [
                run1["ablation"].get("selectivity", {}).get("skill_direction_selectivity"),
                run2["ablation"].get("selectivity", {}).get("skill_direction_selectivity"),
            ],
        }

    output_path = os.path.join(results_base, f"comparison_{args.run1}_vs_{args.run2}.json")
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[Save] Comparison saved to {output_path}")


if __name__ == "__main__":
    main()
