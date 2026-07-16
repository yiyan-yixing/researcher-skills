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
    """下载并采样 GSM8K 数据集，转为选择题格式以控制格式混淆。

    将开放数学题转为 A/B/C/D 选择题：
      - 正确答案 + 3 个干扰项（正确答案±随机偏移）
      - 输入格式与 MMLU 一致：问题 + 4 选项 + "Answer:"
    """
    from datasets import load_dataset
    import re

    print("[GSM8K] Loading dataset (multiple-choice format)...")
    ds_cfg = cfg_data["gsm8k"]
    dataset = load_dataset(ds_cfg["dataset_name"], ds_cfg["subset"], split=ds_cfg["split"])

    samples = []
    indices = list(range(len(dataset)))
    random.seed(cfg_data["seed"])
    selected = random.sample(indices, min(cfg_data["num_samples"], len(indices)))

    for i in selected:
        item = dataset[i]
        question = item["question"]
        # 提取正确答案数字：GSM8K 格式 "....#### 18"
        answer_str = item["answer"]
        try:
            correct_num = float(answer_str.split("####")[-1].strip().replace(",", ""))
        except (ValueError, IndexError):
            continue

        # 生成 3 个干扰项（合理范围内的整数）
        rng = random.Random(i + cfg_data["seed"])  # 确定性干扰项
        distractors = set()
        attempts = 0
        while len(distractors) < 3 and attempts < 50:
            offset = rng.choice([-10, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 10])
            d = int(correct_num) + offset
            if d != int(correct_num) and d >= 0:
                distractors.add(d)
            attempts += 1
        # 如果干扰项不够，用大偏移补
        while len(distractors) < 3:
            offset = rng.randint(1, 20) * rng.choice([-1, 1])
            d = int(correct_num) + offset
            if d != int(correct_num) and d >= 0:
                distractors.add(d)
            else:
                distractors.add(int(correct_num) + 100 + len(distractors))

        distractors = list(distractors)[:3]

        # 构造选项：正确答案随机放在 A/B/C/D 中
        all_options = [int(correct_num)] + distractors
        rng2 = random.Random(i + cfg_data["seed"] + 1000)
        rng2.shuffle(all_options)
        correct_idx = all_options.index(int(correct_num))
        correct_letter = "ABCD"[correct_idx]

        # 构造与 MMLU 一致的输入格式
        input_text = f"{question}\n"
        for j, opt in enumerate(all_options):
            input_text += f"{'ABCD'[j]}. {opt}\n"
        input_text += "Answer:"

        samples.append({
            "id": f"gsm8k_{i}",
            "benchmark": "gsm8k",
            "category": "skill",
            "label": 1,
            "input_text": input_text,
            "answer": correct_letter,
            "choices": [str(o) for o in all_options],
        })

    out_path = os.path.join(output_dir, "gsm8k_mini.json")
    with open(out_path, "w") as f:
        json.dump({"samples": samples, "metadata": {"benchmark": "gsm8k", "count": len(samples), "format": "multiple_choice"}}, f, indent=2)
    print(f"[GSM8K] Saved {len(samples)} samples (multiple-choice format) to {out_path}")
    return samples


def prepare_ifeval(cfg_data, output_dir):
    """下载并采样 IFEval 数据集，转为判断题格式以控制格式混淆。

    将开放式指令遵循转为二选一判断题：
      - 输入格式与 MMLU 一致：问题 + 2 选项 + "Answer:"
      - A = Yes (指令被遵循), B = No (指令未被遵循)
    """
    from datasets import load_dataset

    print("[IFEval] Loading dataset (yes/no format)...")
    ds_cfg = cfg_data["ifeval"]
    dataset = load_dataset(ds_cfg["dataset_name"], split=ds_cfg["split"])

    samples = []
    indices = list(range(len(dataset)))
    random.seed(cfg_data["seed"])
    selected = random.sample(indices, min(cfg_data["num_samples"], len(indices)))

    for i in selected:
        item = dataset[i]
        # IFEval: prompt 字段包含指令
        instruction = item["prompt"]
        constraint_id = item.get("instruction_id", "unknown")

        # 转为判断题格式：给定一段文本，判断是否遵循了指令
        # 格式与 MMLU 一致
        input_text = (
            f"Does the following instruction require a specific format constraint?\n\n"
            f"Instruction: {instruction}\n\n"
            f"A. Yes\n"
            f"B. No\n"
            f"Answer:"
        )

        # 所有 IFEval 样本都有格式约束，所以答案是 A
        samples.append({
            "id": f"ifeval_{i}",
            "benchmark": "ifeval",
            "category": "skill",
            "label": 1,
            "input_text": input_text,
            "answer": "A",
            "choices": ["Yes", "No"],
            "instruction_id": constraint_id,
        })

    out_path = os.path.join(output_dir, "ifeval_mini.json")
    with open(out_path, "w") as f:
        json.dump({"samples": samples, "metadata": {"benchmark": "ifeval", "count": len(samples), "format": "yes_no_choice"}}, f, indent=2)
    print(f"[IFEval] Saved {len(samples)} samples (yes/no format) to {out_path}")
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


def prepare_triviaqa(cfg_data, output_dir, max_retries=3, timeout=120):
    """下载并采样 TriviaQA 数据集，转为选择题格式以控制格式混淆。

    将开放问答转为 A/B/C/D 选择题：
      - 正确答案 + 3 个干扰项（从其他样本的答案中随机抽取）
      - 输入格式与 MMLU 一致：问题 + 4 选项 + "Answer:"
    """
    from datasets import load_dataset

    ds_cfg = cfg_data["triviaqa"]
    dataset = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[TriviaQA] Loading dataset (attempt {attempt}/{max_retries}, multiple-choice format)...")
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(timeout))
            dataset = load_dataset(
                ds_cfg["dataset_name"], ds_cfg["subset"], split=ds_cfg["split"],
                trust_remote_code=True,
            )
            break
        except Exception as e:
            print(f"[TriviaQA] Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                import time
                wait = attempt * 5
                print(f"[TriviaQA] Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"[TriviaQA] All {max_retries} attempts failed. Skipping TriviaQA.")
                return []

    if dataset is None:
        return []

    # 先收集所有答案作为干扰项池
    all_answers = []
    for item in dataset:
        ans = item["answer"]["value"] if isinstance(item["answer"], dict) else str(item["answer"])
        all_answers.append(ans)

    samples = []
    indices = list(range(len(dataset)))
    random.seed(cfg_data["seed"])
    selected = random.sample(indices, min(cfg_data["num_samples"], len(indices)))

    for i in selected:
        item = dataset[i]
        question = item["question"]
        correct_answer = item["answer"]["value"] if isinstance(item["answer"], dict) else str(item["answer"])

        # 从其他样本的答案中抽取 3 个干扰项
        rng = random.Random(i + cfg_data["seed"])
        distractors = set()
        attempts = 0
        while len(distractors) < 3 and attempts < 50:
            d = rng.choice(all_answers)
            if d != correct_answer and d not in distractors and len(d) < 50:
                distractors.add(d)
            attempts += 1
        # 如果不够，补占位
        while len(distractors) < 3:
            distractors.add(f"Unknown option {len(distractors)+1}")

        distractors = list(distractors)[:3]

        # 构造选项
        all_options = [correct_answer] + distractors
        rng2 = random.Random(i + cfg_data["seed"] + 2000)
        rng2.shuffle(all_options)
        correct_idx = all_options.index(correct_answer)
        correct_letter = "ABCD"[correct_idx]

        # 构造与 MMLU 一致的输入格式
        input_text = f"{question}\n"
        for j, opt in enumerate(all_options):
            input_text += f"{'ABCD'[j]}. {opt}\n"
        input_text += "Answer:"

        samples.append({
            "id": f"triviaqa_{i}",
            "benchmark": "triviaqa",
            "category": "knowledge",
            "label": 0,
            "input_text": input_text,
            "answer": correct_letter,
            "choices": all_options,
        })

    out_path = os.path.join(output_dir, "triviaqa_mini.json")
    with open(out_path, "w") as f:
        json.dump({"samples": samples, "metadata": {"benchmark": "triviaqa", "count": len(samples), "format": "multiple_choice"}}, f, indent=2)
    print(f"[TriviaQA] Saved {len(samples)} samples (multiple-choice format) to {out_path}")
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
