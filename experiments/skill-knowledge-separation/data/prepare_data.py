"""
数据准备模块：下载并采样 mini-benchmark 数据集。

输出：
  - data/gsm8k_mini.json
  - data/ifeval_mini.json
  - data/mmlu_mini.json
  - data/triviaqa_mini.json
  - data/all_samples.json  (合并所有)
"""

import argparse
import json
import os
import random
import sys

import yaml


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_gsm8k(cfg_data, output_dir):
    """下载并采样 GSM8K 数据集。"""
    from datasets import load_dataset

    print("[GSM8K] Loading dataset...")
    ds_cfg = cfg_data["gsm8k"]
    dataset = load_dataset(ds_cfg["dataset_name"], ds_cfg["subset"], split=ds_cfg["split"])

    samples = []
    indices = list(range(len(dataset)))
    random.seed(cfg_data["seed"])
    selected = random.sample(indices, min(cfg_data["num_samples"], len(indices)))

    for i in selected:
        item = dataset[i]
        # GSM8K: question + answer
        input_text = item["question"]
        answer = item["answer"]
        samples.append({
            "id": f"gsm8k_{i}",
            "benchmark": "gsm8k",
            "category": "skill",
            "label": 1,
            "input_text": input_text,
            "answer": answer,
            "choices": None,
        })

    out_path = os.path.join(output_dir, "gsm8k_mini.json")
    with open(out_path, "w") as f:
        json.dump({"samples": samples, "metadata": {"benchmark": "gsm8k", "count": len(samples)}}, f, indent=2)
    print(f"[GSM8K] Saved {len(samples)} samples to {out_path}")
    return samples


def prepare_ifeval(cfg_data, output_dir):
    """下载并采样 IFEval 数据集。"""
    from datasets import load_dataset

    print("[IFEval] Loading dataset...")
    ds_cfg = cfg_data["ifeval"]
    dataset = load_dataset(ds_cfg["dataset_name"], split=ds_cfg["split"])

    samples = []
    indices = list(range(len(dataset)))
    random.seed(cfg_data["seed"])
    selected = random.sample(indices, min(cfg_data["num_samples"], len(indices)))

    for i in selected:
        item = dataset[i]
        # IFEval: prompt 字段包含指令
        input_text = item["prompt"]
        # IFEval 的约束信息用于后续评估
        constraint_id = item.get("instruction_id", "unknown")
        samples.append({
            "id": f"ifeval_{i}",
            "benchmark": "ifeval",
            "category": "skill",
            "label": 1,
            "input_text": input_text,
            "answer": "",  # IFEval 是开放式生成
            "choices": None,
            "instruction_id": constraint_id,
        })

    out_path = os.path.join(output_dir, "ifeval_mini.json")
    with open(out_path, "w") as f:
        json.dump({"samples": samples, "metadata": {"benchmark": "ifeval", "count": len(samples)}}, f, indent=2)
    print(f"[IFEval] Saved {len(samples)} samples to {out_path}")
    return samples


def prepare_mmlu(cfg_data, output_dir):
    """下载并采样 MMLU 数据集。"""
    from datasets import load_dataset

    print("[MMLU] Loading dataset...")
    ds_cfg = cfg_data["mmlu"]
    dataset = load_dataset(ds_cfg["dataset_name"], ds_cfg["subset"], split=ds_cfg["split"])

    samples = []
    indices = list(range(len(dataset)))
    random.seed(cfg_data["seed"])
    selected = random.sample(indices, min(cfg_data["num_samples"], len(indices)))

    for i in selected:
        item = dataset[i]
        # MMLU: question + 4 choices + answer
        question = item["question"]
        choices = [item["choices"][j] for j in range(4)]
        answer_idx = item["answer"]
        answer_letter = "ABCD"[answer_idx]

        # 构造输入文本：问题 + 选项
        input_text = f"{question}\n"
        for j, c in enumerate(choices):
            input_text += f"{'ABCD'[j]}. {c}\n"
        input_text += "Answer:"

        samples.append({
            "id": f"mmlu_{i}",
            "benchmark": "mmlu",
            "category": "knowledge",
            "label": 0,
            "input_text": input_text,
            "answer": answer_letter,
            "choices": choices,
        })

    out_path = os.path.join(output_dir, "mmlu_mini.json")
    with open(out_path, "w") as f:
        json.dump({"samples": samples, "metadata": {"benchmark": "mmlu", "count": len(samples)}}, f, indent=2)
    print(f"[MMLU] Saved {len(samples)} samples to {out_path}")
    return samples


def prepare_triviaqa(cfg_data, output_dir):
    """下载并采样 TriviaQA 数据集。"""
    from datasets import load_dataset

    print("[TriviaQA] Loading dataset...")
    ds_cfg = cfg_data["triviaqa"]
    dataset = load_dataset(ds_cfg["dataset_name"], ds_cfg["subset"], split=ds_cfg["split"])

    samples = []
    indices = list(range(len(dataset)))
    random.seed(cfg_data["seed"])
    selected = random.sample(indices, min(cfg_data["num_samples"], len(indices)))

    for i in selected:
        item = dataset[i]
        # TriviaQA: question + answer
        question = item["question"]
        # 取第一个别名作为答案
        answer_value = item["answer"]["value"] if isinstance(item["answer"], dict) else str(item["answer"])

        input_text = f"Question: {question}\nAnswer:"

        samples.append({
            "id": f"triviaqa_{i}",
            "benchmark": "triviaqa",
            "category": "knowledge",
            "label": 0,
            "input_text": input_text,
            "answer": answer_value,
            "choices": None,
        })

    out_path = os.path.join(output_dir, "triviaqa_mini.json")
    with open(out_path, "w") as f:
        json.dump({"samples": samples, "metadata": {"benchmark": "triviaqa", "count": len(samples)}}, f, indent=2)
    print(f"[TriviaQA] Saved {len(samples)} samples to {out_path}")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare mini-benchmark datasets")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    cfg_data = config["data"]

    output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Preparing mini-benchmark datasets")
    print("=" * 60)

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
    print(f"\n[Combined] Saved {len(all_samples)} samples ({num_skill} skill, {num_knowledge} knowledge) to {combined_path}")
    print("=" * 60)
    print("Data preparation complete.")


if __name__ == "__main__":
    main()
