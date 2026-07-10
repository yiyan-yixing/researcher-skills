"""
实验主入口：一条命令运行实验0+1。

用法:
    python run_experiment.py --config config.yaml
    python run_experiment.py --config config.yaml --run-id my_run
    python run_experiment.py --config config.yaml --exp-only 0
"""

import argparse
import json
import os
import sys
import time

import yaml


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_data_preparation(config_path, base_dir):
    """运行数据准备。"""
    print("\n" + "=" * 70)
    print("STEP 1: Data Preparation")
    print("=" * 70)

    from data.prepare_data import (
        load_config as load_data_config,
        prepare_gsm8k, prepare_ifeval, prepare_mmlu, prepare_triviaqa,
    )
    import random

    config = load_data_config(config_path)
    cfg_data = config["data"]

    output_dir = os.path.join(base_dir, "data")
    os.makedirs(output_dir, exist_ok=True)

    all_samples = []
    all_samples.extend(prepare_gsm8k(cfg_data, output_dir))
    all_samples.extend(prepare_ifeval(cfg_data, output_dir))
    all_samples.extend(prepare_mmlu(cfg_data, output_dir))
    all_samples.extend(prepare_triviaqa(cfg_data, output_dir))

    # 保存合并数据
    combined_path = os.path.join(output_dir, "all_samples.json")
    num_skill = sum(1 for s in all_samples if s["label"] == 1)
    num_knowledge = sum(1 for s in all_samples if s["label"] == 0)
    with open(combined_path, "w") as f:
        json.dump({
            "samples": all_samples,
            "metadata": {
                "num_skill": num_skill,
                "num_knowledge": num_knowledge,
                "total": len(all_samples),
                "seed": cfg_data["seed"],
            }
        }, f, indent=2)
    print(f"[Data] Combined: {len(all_samples)} samples ({num_skill} skill, {num_knowledge} knowledge)")

    return combined_path


def run_hidden_state_extraction(config_path, run_id, base_dir, samples_file):
    """运行隐藏状态提取。"""
    print("\n" + "=" * 70)
    print("STEP 2: Hidden State Extraction")
    print("=" * 70)

    config = load_config(config_path)
    cfg_model = config["model"]
    cfg_data = config["data"]
    results_dir = os.path.join(base_dir, config["output"]["results_dir"], run_id)
    os.makedirs(results_dir, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = cfg_model.get("device", "cpu")

    # 加载模型
    print(f"[Model] Loading {cfg_model['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg_model["name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg_model["name"],
        torch_dtype=getattr(torch, cfg_model.get("torch_dtype", "float32")),
    )
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    num_layers = model.config.n_layer if hasattr(model.config, "n_layer") else model.config.num_hidden_layers
    hidden_size = model.config.n_embd if hasattr(model.config, "n_embd") else model.config.hidden_size

    # 加载样本
    with open(samples_file, "r") as f:
        data = json.load(f)
    samples = data["samples"]
    print(f"[Data] {len(samples)} samples loaded")

    # 提取隐藏状态
    from extract_hidden import extract_hidden_states
    start_time = time.time()
    results, num_layers = extract_hidden_states(model, tokenizer, samples, cfg_data, cfg_model)
    elapsed = time.time() - start_time
    print(f"[Extract] Done in {elapsed:.1f}s")

    # 保存
    output_path = os.path.join(results_dir, "hidden_states.json")
    output_data = {
        "model": cfg_model["name"],
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_samples": len(results),
        "extraction_time_s": round(elapsed, 1),
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f)
    print(f"[Save] {output_path}")

    # 元信息
    meta_path = os.path.join(results_dir, "hidden_states_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "model": cfg_model["name"],
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "num_samples": len(results),
            "sample_ids": [r["id"] for r in results],
            "sample_benchmarks": [r["benchmark"] for r in results],
            "sample_labels": [r["label"] for r in results],
        }, f, indent=2)

    return output_path


def run_probe_training(config_path, run_id, base_dir, hidden_states_path):
    """运行探针训练。"""
    print("\n" + "=" * 70)
    print("STEP 3: Linear Probe Training (Experiment 0)")
    print("=" * 70)

    from probe import load_hidden_states, run_all_probes, check_abort_criteria

    config = load_config(config_path)
    cfg_probe = config["probe"]
    cfg_abort = config.get("abort_criteria", {})

    results_dir = os.path.join(base_dir, config["output"]["results_dir"], run_id)

    hidden_data = load_hidden_states(hidden_states_path)
    layer_results, best_layer = run_all_probes(hidden_data, cfg_probe)

    # 保存结果
    probe_results_path = os.path.join(results_dir, "probe_results.json")
    probe_summary = {
        "model": hidden_data["model"],
        "num_layers": hidden_data["num_layers"],
        "hidden_size": hidden_data["hidden_size"],
        "best_layer": best_layer,
        "best_cv_accuracy": layer_results[str(best_layer)]["cv_mean_accuracy"],
        "layer_results": {k: {kk: vv for kk, vv in v.items() if kk != "weight_vector"}
                         for k, v in layer_results.items()},
    }
    with open(probe_results_path, "w") as f:
        json.dump(probe_summary, f, indent=2)

    # 保存权重
    probe_weights_path = os.path.join(results_dir, "probe_weights.json")
    weights_data = {
        "model": hidden_data["model"],
        "num_layers": hidden_data["num_layers"],
        "hidden_size": hidden_data["hidden_size"],
        "best_layer": best_layer,
        "layer_weights": {},
    }
    for layer_idx, result in layer_results.items():
        weights_data["layer_weights"][layer_idx] = {
            "weight_vector": result["weight_vector"],
            "intercept": result["intercept"],
            "scaler_mean": result["scaler_mean"],
            "scaler_scale": result["scaler_scale"],
        }
    with open(probe_weights_path, "w") as f:
        json.dump(weights_data, f)

    # 检查放弃条件
    should_abort = check_abort_criteria(layer_results, cfg_abort)
    if should_abort:
        abort_path = os.path.join(results_dir, "abort_exp0.json")
        with open(abort_path, "w") as f:
            json.dump({
                "aborted": True,
                "reason": "All layers probe accuracy < 60%",
                "best_accuracy": layer_results[str(best_layer)]["cv_mean_accuracy"],
                "threshold": cfg_abort.get("exp0_min_accuracy", 0.60),
            }, f, indent=2)

    return should_abort, probe_weights_path


def run_ablation(config_path, run_id, base_dir, samples_file, probe_weights_path,
                 hidden_states_path):
    """运行消融实验，直接调用 ablation.py 的 run_ablation_experiment。"""
    print("\n" + "=" * 70)
    print("STEP 4: Directional Ablation (Experiment 1)")
    print("=" * 70)

    from ablation import (
        load_probe_weights, evaluate_benchmark, PerturbationHook,
        ProjectionAblationHook, compute_perturbation_alpha,
        compute_selectivity,
    )

    config = load_config(config_path)
    cfg_model = config["model"]
    cfg_data = config["data"]
    cfg_ablation = config["ablation"]
    cfg_abort = config.get("abort_criteria", {})

    device = cfg_model.get("device", "cpu")
    results_dir = os.path.join(base_dir, config["output"]["results_dir"], run_id)

    # 确定消融层
    probe_results_path = os.path.join(results_dir, "probe_results.json")
    with open(probe_results_path, "r") as f:
        probe_results = json.load(f)

    if cfg_ablation.get("layer_selection", "best_probe") == "best_probe":
        layer_idx = probe_results["best_layer"]
    else:
        layer_idx = int(cfg_ablation["layer_selection"])

    print(f"[Ablation] Using layer {layer_idx}")

    # 加载探针权重
    weight_vector, scaler_mean, scaler_scale = load_probe_weights(probe_weights_path, layer_idx)

    # 转换到原始空间
    w_original = weight_vector / scaler_scale

    # 计算扰动幅度
    alpha = compute_perturbation_alpha(
        hidden_states_path, layer_idx, (weight_vector, scaler_mean, scaler_scale)
    )
    strength = cfg_ablation.get("strength", 1.0)
    alpha_scaled = alpha * strength

    # 加载样本
    with open(samples_file, "r") as f:
        samples_data = json.load(f)
    samples = samples_data["samples"]

    # 加载模型
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Model] Loading {cfg_model['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg_model["name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg_model["name"],
        torch_dtype=getattr(torch, cfg_model.get("torch_dtype", "float32")),
    )
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    w_torch = torch.tensor(w_original, dtype=torch.float32).to(device)

    ablation_results = {}

    # 基线评估
    print("[Ablation] Baseline evaluation...")
    baseline_metrics, baseline_preds = evaluate_benchmark(model, tokenizer, samples, cfg_data, cfg_model, device)
    ablation_results["baseline"] = {
        "metrics": baseline_metrics,
        "predictions": [
            {"id": p["id"], "benchmark": p["benchmark"], "category": p["category"], "correct": p["correct"]}
            for p in baseline_preds
        ],
    }
    print(f"  Baseline accuracy: {baseline_metrics['accuracy']:.4f}")

    # 知识方向扰动（沿 +w 添加扰动，将表示推向 skill 区域，远离 knowledge 区域）
    print("[Ablation] Knowledge-direction perturbation (push toward skill, away from knowledge)...")
    knowledge_hook = PerturbationHook(w_torch, alpha_scaled)
    hook_handle = model.transformer.h[layer_idx].register_forward_hook(knowledge_hook)
    knowledge_metrics, knowledge_preds = evaluate_benchmark(model, tokenizer, samples, cfg_data, cfg_model, device)
    hook_handle.remove()
    ablation_results["knowledge_ablation"] = {
        "metrics": knowledge_metrics,
        "predictions": [
            {"id": p["id"], "benchmark": p["benchmark"], "category": p["category"], "correct": p["correct"]}
            for p in knowledge_preds
        ],
    }
    print(f"  Knowledge perturbation accuracy: {knowledge_metrics['accuracy']:.4f}")

    # 技能方向扰动（沿 -w 添加扰动，将表示推向 knowledge 区域，远离 skill 区域）
    print("[Ablation] Skill-direction perturbation (push toward knowledge, away from skill)...")
    skill_hook = PerturbationHook(-w_torch, alpha_scaled)
    hook_handle = model.transformer.h[layer_idx].register_forward_hook(skill_hook)
    skill_metrics, skill_preds = evaluate_benchmark(model, tokenizer, samples, cfg_data, cfg_model, device)
    hook_handle.remove()
    ablation_results["skill_ablation"] = {
        "metrics": skill_metrics,
        "predictions": [
            {"id": p["id"], "benchmark": p["benchmark"], "category": p["category"], "correct": p["correct"]}
            for p in skill_preds
        ],
    }
    print(f"  Skill perturbation accuracy: {skill_metrics['accuracy']:.4f}")

    # 投影消融（对照：非选择性消融）
    print("[Ablation] Projection ablation (non-selective control)...")
    proj_hook = ProjectionAblationHook(w_torch, strength)
    hook_handle = model.transformer.h[layer_idx].register_forward_hook(proj_hook)
    proj_metrics, proj_preds = evaluate_benchmark(model, tokenizer, samples, cfg_data, cfg_model, device)
    hook_handle.remove()
    ablation_results["projection_ablation"] = {
        "metrics": proj_metrics,
        "predictions": [
            {"id": p["id"], "benchmark": p["benchmark"], "category": p["category"], "correct": p["correct"]}
            for p in proj_preds
        ],
    }
    print(f"  Projection ablation accuracy: {proj_metrics['accuracy']:.4f}")

    # 计算选择性
    selectivity = compute_selectivity(ablation_results)
    print("\n[Selectivity]")
    print(f"  Knowledge-direction selectivity: {selectivity['knowledge_direction_selectivity']:.2f}:1")
    print(f"  Skill-direction selectivity: {selectivity['skill_direction_selectivity']:.2f}:1")

    # 检查放弃条件
    min_selectivity = cfg_abort.get("exp1_min_selectivity", 2.0)
    max_sel = max(selectivity["knowledge_direction_selectivity"], selectivity["skill_direction_selectivity"])
    should_abort = max_sel < min_selectivity

    if should_abort:
        print(f"\n[ABORT] Selectivity ({max_sel:.2f}) < threshold ({min_selectivity})")
    else:
        print(f"\n[OK] Selectivity ({max_sel:.2f}) >= threshold ({min_selectivity})")

    # 保存
    output_path = os.path.join(results_dir, "ablation_results.json")
    output_data = {
        "ablation_layer": layer_idx,
        "ablation_method": "additive_perturbation",
        "selectivity": selectivity,
        "aborted": should_abort,
        "results": {
            "baseline": {"accuracy": baseline_metrics["accuracy"], "metrics": baseline_metrics},
            "knowledge_ablation": {"accuracy": knowledge_metrics["accuracy"], "metrics": knowledge_metrics},
            "skill_ablation": {"accuracy": skill_metrics["accuracy"], "metrics": skill_metrics},
            "projection_ablation": {"accuracy": proj_metrics["accuracy"], "metrics": proj_metrics},
        },
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"[Save] {output_path}")

    if should_abort:
        abort_path = os.path.join(results_dir, "abort_exp1.json")
        with open(abort_path, "w") as f:
            json.dump({
                "aborted": True,
                "reason": f"Selectivity ratio < {min_selectivity}:1",
                "max_selectivity": max_sel,
                "threshold": min_selectivity,
            }, f, indent=2)

    return should_abort


def main():
    parser = argparse.ArgumentParser(description="Run skill-knowledge separation experiment")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--run-id", type=str, default=None, help="Run identifier (auto-generated if omitted)")
    parser.add_argument("--exp-only", type=int, choices=[0, 1], default=None,
                        help="Run only experiment 0 or 1 (default: run both)")
    parser.add_argument("--skip-data-prep", action="store_true", help="Skip data preparation if already done")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = load_config(args.config)

    # 确定 run_id
    run_id = args.run_id or f"run_{int(time.time())}"
    results_dir = os.path.join(base_dir, config["output"]["results_dir"], run_id)
    os.makedirs(results_dir, exist_ok=True)

    # 保存运行配置
    config_copy_path = os.path.join(results_dir, "config.yaml")
    with open(config_copy_path, "w") as f:
        yaml.dump(config, f)

    print("=" * 70)
    print("LLM Skill/Knowledge Separation Experiment")
    print(f"Run ID: {run_id}")
    print(f"Model: {config['model']['name']}")
    print(f"Results dir: {results_dir}")
    print("=" * 70)

    overall_start = time.time()

    # --- Step 1: 数据准备 ---
    samples_file = os.path.join(base_dir, "data", "all_samples.json")
    if args.skip_data_prep and os.path.exists(samples_file):
        print(f"[Data] Skipping preparation, using existing: {samples_file}")
    else:
        # 添加 data 目录到 sys.path 以便导入
        sys.path.insert(0, os.path.join(base_dir, "data"))
        sys.path.insert(0, base_dir)
        samples_file = run_data_preparation(args.config, base_dir)

    # --- Step 2: 隐藏状态提取 ---
    # 添加 base_dir 到 sys.path
    sys.path.insert(0, base_dir)

    hidden_states_path = os.path.join(results_dir, "hidden_states.json")
    if config["experiment"].get("skip_if_exists") and os.path.exists(hidden_states_path):
        print(f"[Extract] Skipping, using existing: {hidden_states_path}")
    else:
        hidden_states_path = run_hidden_state_extraction(args.config, run_id, base_dir, samples_file)

    # --- Step 3: 实验0 - 探针训练 ---
    if args.exp_only is not None and args.exp_only != 0:
        print("\n[Skip] Experiment 0 (exp_only=1)")
        should_abort_exp0 = False
    else:
        should_abort_exp0, probe_weights_path = run_probe_training(
            args.config, run_id, base_dir, hidden_states_path
        )

    # --- Step 4: 实验1 - 消融 ---
    if should_abort_exp0:
        print("\n[SKIP] Experiment 1 aborted because Experiment 0 failed abort criteria.")
        print("The hypothesis that skill and knowledge have separable representations is not supported.")
    elif args.exp_only is not None and args.exp_only != 1:
        print("\n[Skip] Experiment 1 (exp_only=0)")
    else:
        run_ablation(
            args.config, run_id, base_dir, samples_file, probe_weights_path,
            hidden_states_path
        )

    overall_elapsed = time.time() - overall_start
    print("\n" + "=" * 70)
    print(f"Experiment complete. Total time: {overall_elapsed:.1f}s")
    print(f"Results saved to: {results_dir}")
    print("=" * 70)

    # 写入运行摘要
    summary_path = os.path.join(results_dir, "run_summary.json")
    summary = {
        "run_id": run_id,
        "model": config["model"]["name"],
        "total_elapsed_s": round(overall_elapsed, 1),
        "exp0_aborted": should_abort_exp0 if 'should_abort_exp0' in dir() else None,
    }
    # 尝试加载各阶段结果
    for filename in ["probe_results.json", "ablation_results.json", "abort_exp0.json", "abort_exp1.json"]:
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                summary[filename.replace(".json", "")] = json.load(f)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
