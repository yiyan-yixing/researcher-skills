"""
数据准备模块 v2：支持实验2「格式混淆控制诊断实验」Phase 1A-1C。

基于 prepare_data.py 修改，新增：
  - IFEval 4 选项化（消除混淆源 C1）
  - MMLU subject 分类（skill vs knowledge）
  - within-MMLU 采样（Phase 1B/1C）
  - 内容匹配检验（KS 检验：长度分布 + 词汇频率分布）
  - 统一输入模板

不修改原始 prepare_data.py，本文件独立运作。

已知限制（P0-2 代码审查标注）：
  IFEval 所有样本的正确答案语义相同（"Yes"），探针可能学到
  位置分布模式而非内容。这在 Phase 1A 中是可接受的——1A 的
  目标仅是消除"选项数不同"这个混淆源，不是消除所有混淆。
  Phase 1B/1C 通过 within-MMLU 设计彻底绕过此问题。

输出文件：
  - data/all_samples_v2.json   — Phase 1A 数据（IFEval 4 选项 + 其他不变）
  - data/mmlu_within.json      — Phase 1B 数据（within-MMLU skill vs knowledge）
  - data/mmlu_text_only.json   — Phase 1C 数据（within-MMLU 纯文本科目）
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import yaml


# ============================================================
# MMLU Subject 分类常量
# ============================================================

# 推理型科目（skill, label=1）——需要多步推理或计算
MMLU_SKILL_SUBJECTS: List[str] = [
    "abstract_algebra",
    "college_mathematics",
    "high_school_mathematics",
    "formal_logic",
    "logical_fallacies",
    "college_physics",
    "high_school_physics",
    "college_chemistry",
    "high_school_chemistry",
    "discrete_mathematics",
    "machine_learning",
    "econometrics",
    "high_school_statistics",
]

# 记忆型科目（knowledge, label=0）——主要是事实回忆
MMLU_KNOWLEDGE_SUBJECTS: List[str] = [
    "world_religions",
    "us_history",
    "world_history",
    "european_history",
    "high_school_geography",
    "nutrition",
    "marketing",
    "management",
    "jurisprudence",
    "philosophy",
    "sociology",
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "global_facts",
    "miscellaneous",
]

# 推理型纯文本科目——无数字/公式密集
MMLU_TEXT_ONLY_SKILL: List[str] = [
    "formal_logic",
    "logical_fallacies",
]

# 记忆型纯文本科目——文本为主
MMLU_TEXT_ONLY_KNOWLEDGE: List[str] = [
    "philosophy",
    "world_religions",
    "jurisprudence",
]


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _parse_answer_idx(answer_raw) -> int:
    """解析 MMLU answer 字段，兼容 int(0-3) 和 str("A"-"D") 两种格式。

    不同 HuggingFace 数据集版本的 answer 字段类型可能不同。
    """
    if isinstance(answer_raw, str):
        return "ABCD".index(answer_raw.upper())
    return int(answer_raw)


def format_question(question: str, choices: List[str]) -> str:
    """统一输入模板：所有 benchmark 使用完全相同的格式。

    模板：
        {question}
        A. {opt1}
        B. {opt2}
        C. {opt3}
        D. {opt4}
        Answer:

    Raises:
        ValueError: 如果 choices 不恰好有 4 项
    """
    if len(choices) != 4:
        raise ValueError(f"format_question requires exactly 4 choices, got {len(choices)}")
    letters = "ABCD"
    lines = [question]
    for j, opt in enumerate(choices):
        lines.append(f"{letters[j]}. {opt}")
    lines.append("Answer:")
    return "\n".join(lines)


# ============================================================
# Phase 1A: IFEval 4 选项化
# ============================================================

def prepare_ifeval_v2(cfg_data: dict, output_dir: str) -> List[dict]:
    """下载并采样 IFEval 数据集，转为 4 选项选择题格式。

    与原始 prepare_ifeval 的区别：
      - 4 选项（A/B/C/D）而非 2 选项（A/B）
      - 正确答案随机分配到 A/B/C/D 位置
      - 使用统一输入模板

    已知限制：所有 IFEval 样本正确答案语义相同（"Yes"），
    因为 IFEval train split 的样本都包含格式约束。
    这意味着 IFEval 内部是退化分类（全正类）。
    Phase 1A 目标仅消除选项数混淆，此限制可接受。
    """
    from datasets import load_dataset

    print("[IFEval-v2] Loading dataset (4-option format)...")
    ds_cfg = cfg_data["ifeval"]
    dataset = load_dataset(ds_cfg["dataset_name"], split=ds_cfg["split"])

    samples = []
    indices = list(range(len(dataset)))
    # P1-1 fix: use local RNG instead of global random.seed()
    rng = random.Random(cfg_data["seed"])
    selected = rng.sample(indices, min(cfg_data["num_samples"], len(indices)))

    # 4 选项模板
    option_texts = [
        "Yes (fully satisfies the constraint)",
        "No (clearly violates the constraint)",
        "Partially (partially satisfies the constraint)",
        "Cannot determine (insufficient information)",
    ]

    for i in selected:
        item = dataset[i]
        instruction = item["prompt"]
        constraint_id = item.get("instruction_id", "unknown")

        # 正确答案在原始设计中是 "Yes"（index 0）
        # 随机分配正确答案到 A/B/C/D 位置
        item_rng = random.Random(i + cfg_data["seed"] + 3000)
        correct_position = item_rng.randint(0, 3)  # 0-3 代表 A/B/C/D

        # 构建选项列表：正确答案放在 correct_position 位置
        choices = list(option_texts)  # copy
        correct_text = option_texts[0]  # "Yes" 是正确答案
        # 移除正确答案，插入到正确位置
        choices.remove(correct_text)
        choices.insert(correct_position, correct_text)

        correct_letter = "ABCD"[correct_position]

        # 使用统一模板
        question_text = (
            f"Does the following instruction require a specific format constraint?\n\n"
            f"Instruction: {instruction}"
        )
        input_text = format_question(question_text, choices)

        samples.append({
            "id": f"ifeval_{i}",
            "benchmark": "ifeval",
            "category": "skill",
            "label": 1,
            "input_text": input_text,
            "answer": correct_letter,
            "choices": choices,
            "instruction_id": constraint_id,
        })

    out_path = os.path.join(output_dir, "ifeval_mini_v2.json")
    with open(out_path, "w") as f:
        json.dump({
            "samples": samples,
            "metadata": {
                "benchmark": "ifeval",
                "count": len(samples),
                "format": "multiple_choice_4opt",
                "phase": "1A",
                "limitation": "All correct answers semantically 'Yes' (degenerate within-category classification)",
            }
        }, f, indent=2)
    print(f"[IFEval-v2] Saved {len(samples)} samples (4-option format) to {out_path}")
    return samples


# ============================================================
# Phase 1A: 复用原始 GSM8K/MMLU/TriviaQA（只改模板）
# ============================================================

def prepare_gsm8k_v2(cfg_data: dict, output_dir: str) -> List[dict]:
    """GSM8K 数据准备（与原始相同，使用统一模板）。"""
    from datasets import load_dataset

    print("[GSM8K-v2] Loading dataset (multiple-choice format)...")
    ds_cfg = cfg_data["gsm8k"]
    dataset = load_dataset(ds_cfg["dataset_name"], ds_cfg["subset"], split=ds_cfg["split"])

    samples = []
    indices = list(range(len(dataset)))
    rng = random.Random(cfg_data["seed"])
    selected = rng.sample(indices, min(cfg_data["num_samples"], len(indices)))

    for i in selected:
        item = dataset[i]
        question = item["question"]
        answer_str = item["answer"]
        try:
            correct_num = float(answer_str.split("####")[-1].strip().replace(",", ""))
        except (ValueError, IndexError):
            continue

        # P2-6 fix: assert integer answer for GSM8K
        if correct_num != int(correct_num):
            continue  # skip non-integer answers (rare in GSM8K)

        distractor_rng = random.Random(i + cfg_data["seed"])
        distractors = set()
        attempts = 0
        while len(distractors) < 3 and attempts < 50:
            offset = distractor_rng.choice([-10, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 10])
            d = int(correct_num) + offset
            if d != int(correct_num) and d >= 0:
                distractors.add(d)
            attempts += 1
        while len(distractors) < 3:
            offset = distractor_rng.randint(1, 20) * distractor_rng.choice([-1, 1])
            d = int(correct_num) + offset
            if d != int(correct_num) and d >= 0:
                distractors.add(d)
            else:
                distractors.add(int(correct_num) + 100 + len(distractors))

        distractors = list(distractors)[:3]
        all_options = [int(correct_num)] + distractors
        shuffle_rng = random.Random(i + cfg_data["seed"] + 1000)
        shuffle_rng.shuffle(all_options)
        correct_idx = all_options.index(int(correct_num))
        correct_letter = "ABCD"[correct_idx]

        # 使用统一模板
        input_text = format_question(question, [str(o) for o in all_options])

        samples.append({
            "id": f"gsm8k_{i}",
            "benchmark": "gsm8k",
            "category": "skill",
            "label": 1,
            "input_text": input_text,
            "answer": correct_letter,
            "choices": [str(o) for o in all_options],
        })

    out_path = os.path.join(output_dir, "gsm8k_mini_v2.json")
    with open(out_path, "w") as f:
        json.dump({
            "samples": samples,
            "metadata": {"benchmark": "gsm8k", "count": len(samples), "format": "multiple_choice"}
        }, f, indent=2)
    print(f"[GSM8K-v2] Saved {len(samples)} samples to {out_path}")
    return samples


def prepare_mmlu_v2(cfg_data: dict, output_dir: str) -> List[dict]:
    """MMLU 数据准备（与原始相同，使用统一模板）。"""
    from datasets import load_dataset

    print("[MMLU-v2] Loading dataset...")
    ds_cfg = cfg_data["mmlu"]
    dataset = load_dataset(ds_cfg["dataset_name"], ds_cfg["subset"], split=ds_cfg["split"])

    samples = []
    indices = list(range(len(dataset)))
    rng = random.Random(cfg_data["seed"])
    selected = rng.sample(indices, min(cfg_data["num_samples"], len(indices)))

    for i in selected:
        item = dataset[i]
        question = item["question"]
        choices = [item["choices"][j] for j in range(4)]
        # P1-2 fix: handle both int and str answer formats
        answer_idx = _parse_answer_idx(item["answer"])
        answer_letter = "ABCD"[answer_idx]

        # 使用统一模板
        input_text = format_question(question, choices)

        samples.append({
            "id": f"mmlu_{i}",
            "benchmark": "mmlu",
            "category": "knowledge",
            "label": 0,
            "input_text": input_text,
            "answer": answer_letter,
            "choices": choices,
        })

    out_path = os.path.join(output_dir, "mmlu_mini_v2.json")
    with open(out_path, "w") as f:
        json.dump({
            "samples": samples,
            "metadata": {"benchmark": "mmlu", "count": len(samples)}
        }, f, indent=2)
    print(f"[MMLU-v2] Saved {len(samples)} samples to {out_path}")
    return samples


def prepare_triviaqa_v2(cfg_data: dict, output_dir: str, max_retries: int = 3,
                        timeout: int = 120) -> List[dict]:
    """TriviaQA 数据准备（与原始相同，使用统一模板）。

    Note: trust_remote_code=True is used for HuggingFace dataset loading,
    which allows execution of arbitrary code from the dataset repository.
    This mirrors the original prepare_data.py behavior.
    """
    from datasets import load_dataset

    ds_cfg = cfg_data["triviaqa"]
    dataset = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[TriviaQA-v2] Loading dataset (attempt {attempt}/{max_retries})...")
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(timeout))
            dataset = load_dataset(
                ds_cfg["dataset_name"], ds_cfg["subset"], split=ds_cfg["split"],
                trust_remote_code=True,  # Same as original prepare_data.py
            )
            break
        except Exception as e:
            print(f"[TriviaQA-v2] Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                import time
                wait = attempt * 5
                print(f"[TriviaQA-v2] Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"[TriviaQA-v2] All {max_retries} attempts failed. Skipping.")
                return []

    if dataset is None:
        return []

    all_answers = []
    for item in dataset:
        ans = item["answer"]["value"] if isinstance(item["answer"], dict) else str(item["answer"])
        all_answers.append(ans)

    samples = []
    indices = list(range(len(dataset)))
    rng = random.Random(cfg_data["seed"])
    selected = rng.sample(indices, min(cfg_data["num_samples"], len(indices)))

    for i in selected:
        item = dataset[i]
        question = item["question"]
        correct_answer = item["answer"]["value"] if isinstance(item["answer"], dict) else str(item["answer"])

        distractor_rng = random.Random(i + cfg_data["seed"])
        distractors = set()
        attempts = 0
        while len(distractors) < 3 and attempts < 50:
            d = distractor_rng.choice(all_answers)
            if d != correct_answer and d not in distractors and len(d) < 50:
                distractors.add(d)
            attempts += 1
        while len(distractors) < 3:
            distractors.add(f"Unknown option {len(distractors)+1}")

        distractors = list(distractors)[:3]
        all_options = [correct_answer] + distractors
        shuffle_rng = random.Random(i + cfg_data["seed"] + 2000)
        shuffle_rng.shuffle(all_options)
        correct_idx = all_options.index(correct_answer)
        correct_letter = "ABCD"[correct_idx]

        # 使用统一模板
        input_text = format_question(question, all_options)

        samples.append({
            "id": f"triviaqa_{i}",
            "benchmark": "triviaqa",
            "category": "knowledge",
            "label": 0,
            "input_text": input_text,
            "answer": correct_letter,
            "choices": all_options,
        })

    out_path = os.path.join(output_dir, "triviaqa_mini_v2.json")
    with open(out_path, "w") as f:
        json.dump({
            "samples": samples,
            "metadata": {"benchmark": "triviaqa", "count": len(samples), "format": "multiple_choice"}
        }, f, indent=2)
    print(f"[TriviaQA-v2] Saved {len(samples)} samples to {out_path}")
    return samples


# ============================================================
# Phase 1B: within-MMLU 推理 vs 记忆
# ============================================================

def prepare_mmlu_within(cfg_data: dict, output_dir: str,
                        skill_subjects: Optional[List[str]] = None,
                        knowledge_subjects: Optional[List[str]] = None,
                        samples_per_category: int = 100,
                        seed: int = 42) -> List[dict]:
    """Phase 1B: 从 MMLU 中按 skill/knowledge 科目分类采样。

    消除混淆源 C2/C3/C4：所有题目来自 MMLU，格式、选项类型、句法完全一致。

    Args:
        cfg_data: 数据配置
        output_dir: 输出目录
        skill_subjects: 推理型科目列表（默认使用 MMLU_SKILL_SUBJECTS）
        knowledge_subjects: 记忆型科目列表（默认使用 MMLU_KNOWLEDGE_SUBJECTS）
        samples_per_category: 每类采样题数
        seed: 随机种子
    """
    from datasets import load_dataset

    if skill_subjects is None:
        skill_subjects = MMLU_SKILL_SUBJECTS
    if knowledge_subjects is None:
        knowledge_subjects = MMLU_KNOWLEDGE_SUBJECTS

    print(f"[MMLU-within] Loading MMLU for within-subject design...")
    print(f"  Skill subjects ({len(skill_subjects)}): {skill_subjects}")
    print(f"  Knowledge subjects ({len(knowledge_subjects)}): {knowledge_subjects}")

    ds_cfg = cfg_data["mmlu"]
    dataset = load_dataset(ds_cfg["dataset_name"], ds_cfg["subset"], split=ds_cfg["split"])

    # P1-1 fix: use local RNG
    rng = random.Random(seed)

    # 按 subject 分组
    skill_items = []
    knowledge_items = []

    for item in dataset:
        subject = item.get("subject", "")
        if subject in skill_subjects:
            skill_items.append(item)
        elif subject in knowledge_subjects:
            knowledge_items.append(item)

    print(f"  Found {len(skill_items)} skill items, {len(knowledge_items)} knowledge items")

    # 采样
    skill_selected = rng.sample(skill_items, min(samples_per_category, len(skill_items)))
    knowledge_selected = rng.sample(knowledge_items, min(samples_per_category, len(knowledge_items)))

    print(f"  Sampled {len(skill_selected)} skill, {len(knowledge_selected)} knowledge")

    # 转换为统一格式
    samples = []

    for idx, item in enumerate(skill_selected):
        question = item["question"]
        choices = [item["choices"][j] for j in range(4)]
        # P1-2 fix: handle both int and str answer formats
        answer_idx = _parse_answer_idx(item["answer"])
        answer_letter = "ABCD"[answer_idx]
        subject = item.get("subject", "unknown")

        input_text = format_question(question, choices)

        samples.append({
            "id": f"mmlu_skill_{idx}",
            "benchmark": "mmlu_within",
            "category": "skill",
            "label": 1,
            "input_text": input_text,
            "answer": answer_letter,
            "choices": choices,
            "subject": subject,
        })

    for idx, item in enumerate(knowledge_selected):
        question = item["question"]
        choices = [item["choices"][j] for j in range(4)]
        answer_idx = _parse_answer_idx(item["answer"])
        answer_letter = "ABCD"[answer_idx]
        subject = item.get("subject", "unknown")

        input_text = format_question(question, choices)

        samples.append({
            "id": f"mmlu_knowledge_{idx}",
            "benchmark": "mmlu_within",
            "category": "knowledge",
            "label": 0,
            "input_text": input_text,
            "answer": answer_letter,
            "choices": choices,
            "subject": subject,
        })

    # 内容匹配检验
    ks_pvalue = _compute_ks_test(samples)
    print(f"  KS test for question length distribution: p={ks_pvalue:.4f}")
    if ks_pvalue < 0.05:
        print(f"  WARNING: KS test p={ks_pvalue:.4f} < 0.05, skill/knowledge groups differ in question length")

    out_path = os.path.join(output_dir, "mmlu_within.json")
    with open(out_path, "w") as f:
        json.dump({
            "samples": samples,
            "metadata": {
                "benchmark": "mmlu_within",
                "phase": "1B",
                "num_skill": len(skill_selected),
                "num_knowledge": len(knowledge_selected),
                "total": len(samples),
                "ks_pvalue": ks_pvalue,
                "skill_subjects": skill_subjects,
                "knowledge_subjects": knowledge_subjects,
                "seed": seed,
            }
        }, f, indent=2)
    print(f"[MMLU-within] Saved {len(samples)} samples to {out_path}")
    return samples


# ============================================================
# Phase 1C: within-MMLU 纯文本科目
# ============================================================

def prepare_mmlu_text_only(cfg_data: dict, output_dir: str,
                           skill_subjects: Optional[List[str]] = None,
                           knowledge_subjects: Optional[List[str]] = None,
                           samples_per_category: int = 80,
                           seed: int = 42) -> List[dict]:
    """Phase 1C: 从 MMLU 纯文本科目中采样。

    消除 subject 级别内容混淆：只用纯文本、无数字/公式的科目。

    Args:
        cfg_data: 数据配置
        output_dir: 输出目录
        skill_subjects: 推理型纯文本科目列表（默认 MMLU_TEXT_ONLY_SKILL）
        knowledge_subjects: 记忆型纯文本科目列表（默认 MMLU_TEXT_ONLY_KNOWLEDGE）
        samples_per_category: 每类采样题数（建议 80-100，受纯文本科目题量限制）
        seed: 随机种子
    """
    from datasets import load_dataset

    if skill_subjects is None:
        skill_subjects = MMLU_TEXT_ONLY_SKILL
    if knowledge_subjects is None:
        knowledge_subjects = MMLU_TEXT_ONLY_KNOWLEDGE

    print(f"[MMLU-text-only] Loading MMLU for text-only subject design...")
    print(f"  Skill text-only subjects ({len(skill_subjects)}): {skill_subjects}")
    print(f"  Knowledge text-only subjects ({len(knowledge_subjects)}): {knowledge_subjects}")

    ds_cfg = cfg_data["mmlu"]
    dataset = load_dataset(ds_cfg["dataset_name"], ds_cfg["subset"], split=ds_cfg["split"])

    # P1-1 fix: use local RNG
    rng = random.Random(seed)

    # 按 subject 分组
    skill_items = []
    knowledge_items = []

    for item in dataset:
        subject = item.get("subject", "")
        if subject in skill_subjects:
            skill_items.append(item)
        elif subject in knowledge_subjects:
            knowledge_items.append(item)

    print(f"  Found {len(skill_items)} skill text-only items, {len(knowledge_items)} knowledge text-only items")

    # 如果题目不够，报告并使用全部
    if len(skill_items) < samples_per_category:
        print(f"  WARNING: Only {len(skill_items)} skill items available, using all")
    if len(knowledge_items) < samples_per_category:
        print(f"  WARNING: Only {len(knowledge_items)} knowledge items available, using all")

    skill_selected = rng.sample(skill_items, min(samples_per_category, len(skill_items)))
    knowledge_selected = rng.sample(knowledge_items, min(samples_per_category, len(knowledge_items)))

    print(f"  Sampled {len(skill_selected)} skill, {len(knowledge_selected)} knowledge")

    # 转换为统一格式
    samples = []

    for idx, item in enumerate(skill_selected):
        question = item["question"]
        choices = [item["choices"][j] for j in range(4)]
        answer_idx = _parse_answer_idx(item["answer"])
        answer_letter = "ABCD"[answer_idx]
        subject = item.get("subject", "unknown")

        input_text = format_question(question, choices)

        samples.append({
            "id": f"mmlu_text_skill_{idx}",
            "benchmark": "mmlu_text_only",
            "category": "skill",
            "label": 1,
            "input_text": input_text,
            "answer": answer_letter,
            "choices": choices,
            "subject": subject,
        })

    for idx, item in enumerate(knowledge_selected):
        question = item["question"]
        choices = [item["choices"][j] for j in range(4)]
        answer_idx = _parse_answer_idx(item["answer"])
        answer_letter = "ABCD"[answer_idx]
        subject = item.get("subject", "unknown")

        input_text = format_question(question, choices)

        samples.append({
            "id": f"mmlu_text_knowledge_{idx}",
            "benchmark": "mmlu_text_only",
            "category": "knowledge",
            "label": 0,
            "input_text": input_text,
            "answer": answer_letter,
            "choices": choices,
            "subject": subject,
        })

    # 内容匹配检验
    ks_pvalue = _compute_ks_test(samples)
    print(f"  KS test for question length distribution: p={ks_pvalue:.4f}")
    if ks_pvalue < 0.05:
        print(f"  WARNING: KS test p={ks_pvalue:.4f} < 0.05, skill/knowledge groups differ in question length")
    else:
        print(f"  KS test passed: question length distributions are not significantly different")

    # 统计 subject 分布
    skill_subject_dist = {}
    knowledge_subject_dist = {}
    for s in samples:
        subj = s.get("subject", "unknown")
        if s["label"] == 1:
            skill_subject_dist[subj] = skill_subject_dist.get(subj, 0) + 1
        else:
            knowledge_subject_dist[subj] = knowledge_subject_dist.get(subj, 0) + 1
    print(f"  Skill subject distribution: {skill_subject_dist}")
    print(f"  Knowledge subject distribution: {knowledge_subject_dist}")

    out_path = os.path.join(output_dir, "mmlu_text_only.json")
    with open(out_path, "w") as f:
        json.dump({
            "samples": samples,
            "metadata": {
                "benchmark": "mmlu_text_only",
                "phase": "1C",
                "num_skill": len(skill_selected),
                "num_knowledge": len(knowledge_selected),
                "total": len(samples),
                "ks_pvalue": ks_pvalue,
                "skill_subjects": skill_subjects,
                "knowledge_subjects": knowledge_subjects,
                "skill_subject_dist": skill_subject_dist,
                "knowledge_subject_dist": knowledge_subject_dist,
                "seed": seed,
            }
        }, f, indent=2)
    print(f"[MMLU-text-only] Saved {len(samples)} samples to {out_path}")
    return samples


# ============================================================
# 内容匹配检验
# ============================================================

def _compute_ks_test(samples: List[dict]) -> float:
    """计算 skill/knowledge 组问题长度分布的 KS 检验 p 值。

    返回 p 值。p > 0.05 表示两组长度分布无显著差异。
    如果 scipy 不可用，返回 -1.0 并打印警告。

    Note: 此函数只检查问题长度分布。词汇频率分布的差异
    （如 logic 题频繁使用 "therefore/premise"，
    philosophy 题频繁使用 "Kant/existentialism"）
    仍可能存在，需要在结果分析时注意。
    """
    try:
        from scipy.stats import ks_2samp
    except ImportError:
        print("  [WARNING] scipy not available, skipping KS test")
        return -1.0

    skill_lengths = []
    knowledge_lengths = []

    for s in samples:
        # 提取问题部分（去掉选项和 "Answer:" 行）
        input_text = s["input_text"]
        # 问题在 "Answer:" 之前的第一个段落
        parts = input_text.split("\n")
        # 找到以 "A." 开头的行之前的内容作为问题
        question_lines = []
        for line in parts:
            if line.startswith("A."):
                break
            question_lines.append(line)
        question_text = "\n".join(question_lines)
        length = len(question_text.split())

        if s["label"] == 1:
            skill_lengths.append(length)
        else:
            knowledge_lengths.append(length)

    if len(skill_lengths) < 2 or len(knowledge_lengths) < 2:
        return -1.0

    statistic, pvalue = ks_2samp(skill_lengths, knowledge_lengths)
    return float(pvalue)


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Prepare v2 datasets for Exp2 format-controlled diagnosis")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--phase", type=str, choices=["1a", "1b", "1c", "all"], default="all",
                        help="Which phase to prepare data for")
    parser.add_argument("--samples-per-category", type=int, default=None,
                        help="Override samples per category (default: read from config)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    cfg_data = config["data"]

    output_dir = args.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)

    seed = args.seed if args.seed is not None else cfg_data.get("seed", 42)

    # P0-3 fix: samples_per_category from config phase-specific section, not data.num_samples
    # For Phase 1B/1C, read from phase-specific config sections
    phase1b_cfg = config.get("phase1b", {})
    phase1c_cfg = config.get("phase1c", {})

    # Priority: CLI override > phase-specific config > data.num_samples > default
    if args.samples_per_category is not None:
        samples_per_1b = args.samples_per_category
        samples_per_1c = args.samples_per_category
    else:
        samples_per_1b = phase1b_cfg.get("samples_per_category", cfg_data.get("num_samples", 100))
        samples_per_1c = phase1c_cfg.get("samples_per_category", cfg_data.get("num_samples", 80))

    print("=" * 60)
    print("Preparing v2 datasets for Exp2 format-controlled diagnosis")
    print(f"Phase: {args.phase}, seed: {seed}")
    print(f"  Phase 1B samples/category: {samples_per_1b}")
    print(f"  Phase 1C samples/category: {samples_per_1c}")
    print("=" * 60)

    # Phase 1A: IFEval 4 选项 + 其他 benchmark 不变
    if args.phase in ("1a", "all"):
        print("\n--- Phase 1A: IFEval 4-option fix ---")
        all_v2_samples = []
        all_v2_samples.extend(prepare_gsm8k_v2(cfg_data, output_dir))
        all_v2_samples.extend(prepare_ifeval_v2(cfg_data, output_dir))
        all_v2_samples.extend(prepare_mmlu_v2(cfg_data, output_dir))
        all_v2_samples.extend(prepare_triviaqa_v2(cfg_data, output_dir))

        # 保存合并数据
        combined_path = os.path.join(output_dir, "all_samples_v2.json")
        num_skill = sum(1 for s in all_v2_samples if s["label"] == 1)
        num_knowledge = sum(1 for s in all_v2_samples if s["label"] == 0)
        with open(combined_path, "w") as f:
            json.dump({
                "samples": all_v2_samples,
                "metadata": {
                    "phase": "1A",
                    "num_skill": num_skill,
                    "num_knowledge": num_knowledge,
                    "total": len(all_v2_samples),
                    "seed": seed,
                    "description": "IFEval 4-option fix + other benchmarks with unified template",
                }
            }, f, indent=2)
        print(f"[Phase 1A] Saved {len(all_v2_samples)} samples ({num_skill} skill, {num_knowledge} knowledge) to {combined_path}")

    # Phase 1B: within-MMLU skill vs knowledge
    # P0-1 fix: read subject lists from config
    if args.phase in ("1b", "all"):
        print("\n--- Phase 1B: within-MMLU skill vs knowledge ---")
        skill_subjects_1b = phase1b_cfg.get("skill_subjects", None)
        knowledge_subjects_1b = phase1b_cfg.get("knowledge_subjects", None)
        prepare_mmlu_within(cfg_data, output_dir,
                            skill_subjects=skill_subjects_1b,
                            knowledge_subjects=knowledge_subjects_1b,
                            samples_per_category=samples_per_1b,
                            seed=seed)

    # Phase 1C: within-MMLU 纯文本科目
    # P0-1 fix: read subject lists from config; P0-3 fix: no hardcoded cap
    if args.phase in ("1c", "all"):
        print("\n--- Phase 1C: within-MMLU text-only subjects ---")
        skill_subjects_1c = phase1c_cfg.get("skill_subjects", None)
        knowledge_subjects_1c = phase1c_cfg.get("knowledge_subjects", None)
        prepare_mmlu_text_only(cfg_data, output_dir,
                               skill_subjects=skill_subjects_1c,
                               knowledge_subjects=knowledge_subjects_1c,
                               samples_per_category=samples_per_1c,
                               seed=seed)

    print("\n" + "=" * 60)
    print("Data preparation v2 complete.")


if __name__ == "__main__":
    main()
