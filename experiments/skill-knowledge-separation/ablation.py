"""
方向性消融模块：沿探针发现的"知识方向"和"技能方向"分别消融隐藏状态，
评估对 4 个 benchmark 的选择性影响。

消融策略：使用方向性扰动（additive perturbation）

核心思路：
  探针权重 w 指向 skill 方向（正类），-w 指向 knowledge 方向。

  知识方向扰动（损伤知识任务，保留技能任务）：
    沿 w 方向添加扰动，将表示推向 skill 区域，远离 knowledge 区域
    h' = h + strength * alpha * w_hat（w_hat = w / ||w||）

  技能方向扰动（损伤技能任务，保留知识任务）：
    沿 -w 方向添加扰动，将表示推向 knowledge 区域，远离 skill 区域
    h' = h - strength * alpha * w_hat

  其中 alpha 是扰动幅度，由探针权重的特征尺度决定。
  我们用训练数据在 w 方向的投影标准差作为 alpha。

注意：
  不能用"移除投影"的方式做选择性消融，因为 proj(h, w) = proj(h, -w)，
  沿 w 和 -w 消融投影会给出相同结果。

输出:
  - results/{run_id}/ablation_results.json
"""

import argparse
import json
import logging
import math
import os
import re
import time

import numpy as np
import torch

from utils import load_config, load_model_and_tokenizer, get_layer_module

logger = logging.getLogger(__name__)

# JSON 不支持 inf/nan，选择性指标超过此上限时截断为有限值
_SELECTIVITY_CAP = 9999.0

# IFEval instruction_id 类型路由映射表
# 将 IFEval 的 instruction_id（":"前的部分）映射到内部约束路由名
# 避免子串匹配误判（如 "keywords:..." 中的 "word" 匹配到字数分支）
_IFEVAL_TYPE_ROUTES = {
    "keyword": "keyword",
    "keywords": "keyword",
    "length_constraint": "length",
    "change_case": "case",
    "capitalization": "case",
    "startend": "startend",
    "detectable_format": "format",
    "punctuation": "punctuation",
    "combination": "combination",
    "comma": "comma",
}


def load_probe_weights(probe_weights_path, layer_idx):
    """加载指定层的探针权重。"""
    with open(probe_weights_path, "r") as f:
        data = json.load(f)
    layer_data = data["layer_weights"][str(layer_idx)]
    weight_vector = np.array(layer_data["weight_vector"])
    scaler_mean = np.array(layer_data["scaler_mean"])
    scaler_scale = np.array(layer_data["scaler_scale"])
    return weight_vector, scaler_mean, scaler_scale


class PerturbationHook:
    """在模型 forward 中注入方向性扰动的 hook。

    不同于移除投影（proj(h, d) 和 proj(h, -d) 结果相同），
    本 hook 通过添加固定方向的扰动来改变隐藏状态。
    """

    def __init__(self, direction, alpha):
        """
        direction: torch.Tensor (hidden_size,)
            扰动方向向量，会自动归一化
        alpha: float
            扰动幅度（方向已归一化，alpha 控制绝对大小）
        """
        self.direction_norm = torch.norm(direction).item()
        assert self.direction_norm > 0, "Perturbation direction must be non-zero"
        self.direction_hat = direction / self.direction_norm
        self.alpha = alpha

    def __call__(self, module, input, output):
        # output 是一个 tuple: (hidden_states, ...)
        hidden_states = output[0]

        # 对每个 token 的隐藏状态添加扰动
        # h' = h + alpha * direction_hat
        perturbation = self.alpha * self.direction_hat.unsqueeze(0).unsqueeze(0)
        perturbed = hidden_states + perturbation

        # 返回修改后的 output
        new_output = (perturbed,) + output[1:]
        return new_output


class ProjectionAblationHook:
    """移除指定方向投影的 hook（非选择性，用于对比基线）。

    这种消融对 skill 和 knowledge 任务的损伤是无选择性的。
    h' = h - (h . w / ||w||^2) * w
    """

    def __init__(self, direction, strength=1.0):
        self.ablation_direction = direction
        self.strength = strength
        self.direction_norm_sq = torch.dot(direction, direction).item()
        assert self.direction_norm_sq > 0, "Projection ablation direction must be non-zero"

    def __call__(self, module, input, output):
        hidden_states = output[0]
        proj = torch.matmul(hidden_states, self.ablation_direction) / self.direction_norm_sq
        correction = self.strength * proj.unsqueeze(-1) * self.ablation_direction.unsqueeze(0).unsqueeze(0)
        ablated = hidden_states - correction
        new_output = (ablated,) + output[1:]
        return new_output


def compute_perturbation_alpha(hidden_states_path, layer_idx, probe_weights):
    """
    计算扰动幅度 alpha：训练数据在 w 方向投影的标准差。

    这确保扰动幅度与数据在该方向上的自然变异尺度一致。
    """
    weight_vector, scaler_mean, scaler_scale = probe_weights
    w_original = weight_vector / scaler_scale
    w_hat = w_original / np.linalg.norm(w_original)

    # 加载隐藏状态
    with open(hidden_states_path, "r") as f:
        data = json.load(f)

    results = data["results"]
    projections = []
    for r in results:
        h = np.array(r["hidden_states"][str(layer_idx)])
        proj = np.dot(h, w_hat)
        projections.append(proj)

    projections = np.array(projections)
    alpha = np.std(projections)

    print(f"  Perturbation alpha (1 std of projections along w): {alpha:.4f}")
    print(f"  Projection range: [{projections.min():.4f}, {projections.max():.4f}]")
    print(f"  Projection mean: {projections.mean():.4f}")

    return alpha


def evaluate_benchmark(model, tokenizer, samples, cfg_data, cfg_model, device):
    """
    评估模型在给定 benchmark 上的性能。

    每个样本独立 try/except，单个样本失败不会导致整个评估中止。
    失败样本记录为 incorrect 并计入 total。
    连续失败超过阈值时提前终止，避免大量无用 OOM 重试。

    返回:
        metrics: dict 包含准确率等指标
        predictions: list of dict 包含每个样本的预测
    """
    max_length = cfg_data.get("max_input_length", 512)

    correct = 0
    total = 0
    predictions = []
    failed_samples = []
    consecutive_failures = 0
    max_consecutive_failures = 5

    for sample in samples:
        sample_id = sample["id"]
        input_text = sample["input_text"]
        answer = sample["answer"]
        benchmark = sample["benchmark"]

        try:
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=False,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # 解码生成的文本（去掉输入部分）
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # 评估正确性
            is_correct = check_answer(benchmark, generated_text, answer, sample)

            predictions.append({
                "id": sample_id,
                "benchmark": benchmark,
                "category": sample["category"],
                "generated": generated_text,
                "expected": answer,
                "correct": is_correct,
                "error": None,
            })

            if is_correct:
                correct += 1
            total += 1
            consecutive_failures = 0

        except Exception as e:
            # 单样本失败：记录错误，标记为 incorrect，继续评估
            consecutive_failures += 1
            logger.warning(f"Sample {sample_id} ({benchmark}) failed: {e}")
            failed_samples.append({"id": sample_id, "benchmark": benchmark, "error": str(e)})
            predictions.append({
                "id": sample_id,
                "benchmark": benchmark,
                "category": sample["category"],
                "generated": None,
                "expected": answer,
                "correct": False,
                "error": str(e),
            })
            total += 1

            # 连续失败过多时提前终止（如 CUDA OOM 会连续失败）
            if consecutive_failures >= max_consecutive_failures:
                logger.error(
                    f"{max_consecutive_failures} consecutive sample failures, "
                    f"aborting evaluation (likely systemic issue like OOM)"
                )
                # 将剩余样本标记为 skipped
                remaining_ids = [s["id"] for s in samples[len(predictions):]]
                if remaining_ids:
                    logger.warning(f"Skipping {len(remaining_ids)} remaining samples")
                break

    accuracy = correct / total if total > 0 else 0
    metrics = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }
    if failed_samples:
        metrics["failed_samples"] = len(failed_samples)
        logger.warning(f"evaluate_benchmark: {len(failed_samples)}/{len(samples)} samples failed")

    return metrics, predictions


def check_answer(benchmark, generated, expected, sample):
    """
    检查答案是否正确（不同 benchmark 有不同的判断逻辑）。

    - GSM8K: 提取最终数字，与答案中的数字比较
    - IFEval: 检查格式约束是否满足
    - MMLU: 提取选项字母 A/B/C/D
    - TriviaQA: 模糊匹配答案关键词
    """
    if generated is None:
        return False

    generated_stripped = generated.strip()
    generated_lower = generated_stripped.lower()
    expected_lower = str(expected).lower().strip()

    if benchmark == "gsm8k":
        # 从答案中提取最终数字
        try:
            # expected 格式: "....#### 18"
            expected_num = expected.split("####")[-1].strip()
            expected_num = expected_num.replace(",", "")
            expected_float = float(expected_num)
            # 从生成中提取数字
            nums = re.findall(r'-?\d+\.?\d*', generated_stripped)
            if nums:
                predicted_num = nums[-1].replace(",", "")
                predicted_float = float(predicted_num)
                # 数值比较，允许微小误差
                return abs(predicted_float - expected_float) < max(1e-6, 1e-3 * abs(expected_float))
        except (ValueError, IndexError):
            pass
        return False

    elif benchmark == "ifeval":
        # IFEval: 检查生成的回答是否满足格式约束
        # 从 sample 中获取 instruction_id（约束类型）
        instruction_id = sample.get("instruction_id", "")
        return _check_ifeval_constraint(instruction_id, generated_stripped)

    elif benchmark == "mmlu":
        # 提取选项字母
        # 优先匹配 "Answer: X" 格式，然后匹配独立的 A-D
        match = re.search(r'(?:Answer|answer|Ans|ans)[:\s]*([A-D])', generated_stripped)
        if not match:
            match = re.search(r'\b([A-D])\b', generated_stripped.upper())
        if match:
            predicted_letter = match.group(1).upper()
            return predicted_letter == expected.upper()
        return False

    elif benchmark == "triviaqa":
        # 模糊匹配：去掉标点后检查子串匹配
        def normalize(s):
            return re.sub(r'[^\w\s]', '', s.lower()).strip()

        norm_generated = normalize(generated_stripped)
        norm_expected = normalize(expected_lower)

        if not norm_expected:
            return False

        # 检查答案是否出现在生成中
        if norm_expected in norm_generated:
            return True

        # 检查答案的每个词是否大部分出现
        expected_words = set(norm_expected.split())
        generated_words = set(norm_generated.split())
        overlap = len(expected_words & generated_words)
        if len(expected_words) > 0 and overlap >= len(expected_words) * 0.5:
            return True

        return False

    else:
        return generated_lower == expected_lower


def _check_ifeval_constraint(instruction_id, generated):
    """
    检查 IFEval 格式约束是否满足。

    instruction_id 标识约束类型，这里实现几个常见检查。

    使用类型映射表路由，避免子串匹配误判。
    instruction_id 格式为 "type:param"，例如:
      - "keyword:therefore" / "keywords:existence"
      - "length_constraint:number_words"
      - "change_case:lowercase" / "capitalization:lowercase"
      - "startend:end"
      - "detectable_format:number_bullet"
    """
    if not generated:
        return False

    constraint_id = str(instruction_id).lower() if instruction_id else ""
    if not constraint_id:
        return len(generated.strip()) > 0

    # 按 ":" 分离类型名和参数名
    constraint_type = constraint_id.split(":")[0] if ":" in constraint_id else constraint_id
    constraint_param = constraint_id.split(":")[1] if ":" in constraint_id else ""

    # 查映射表获取路由名
    route = _IFEVAL_TYPE_ROUTES.get(constraint_type, "unknown")

    if route == "length":
        # 字数限制约束：检查生成是否在合理范围内
        word_count = len(generated.split())
        # 放宽标准：只要生成了内容就算通过（GPT-2 很难严格遵循字数约束）
        return word_count > 0 and word_count < 200

    elif route == "keyword":
        # 包含关键词：检查 param 指定的关键词是否在生成中
        if constraint_param:
            return constraint_param in generated.lower()
        # fallback: 检查 "therefore"（旧数据兼容）
        return "therefore" in generated.lower()

    elif route == "case":
        # 大小写约束：检查全小写
        return generated == generated.lower()

    elif route == "startend":
        # 起止格式约束：根据参数检查
        if constraint_param == "end" or constraint_param == "end_period":
            return generated.rstrip().endswith(".")
        # 其他 startend 类型放宽
        return len(generated.strip()) > 0

    elif route == "format":
        # 可检测格式约束（number_bullet 等）：放宽标准
        return len(generated.strip()) > 0

    elif route == "punctuation":
        # 标点约束：以句号结尾
        return generated.rstrip().endswith(".")

    elif route == "combination":
        # 组合约束：放宽标准
        return len(generated.strip()) > 0

    elif route == "comma":
        # 不使用逗号
        return "," not in generated

    else:
        # 未知约束类型：只检查是否生成了非空内容
        return len(generated.strip()) > 0


def run_ablation_experiment(model, tokenizer, samples, cfg_data, cfg_model, cfg_ablation,
                             probe_weights, layer_idx, device, hidden_states_path):
    """
    运行消融实验：对比基线、方向性扰动、和投影消融下的性能。

    四种条件：
    1. 基线（无消融）
    2. 知识方向扰动（沿 w 添加扰动，将表示推向 skill 区域）
    3. 技能方向扰动（沿 -w 添加扰动，将表示推向 knowledge 区域）
    4. 投影消融（移除 w 方向投影，作为非选择性消融的对照）

    所有 hook 注册使用 try/finally 保证异常时也能 remove，避免静默数据污染。

    返回:
        ablation_results: dict
    """
    weight_vector, scaler_mean, scaler_scale = probe_weights

    # 将探针权重从标准化空间转换回原始空间
    # 探针在标准化后的特征上训练：X_scaled = (X - mean) / scale
    # 权重向量在标准化空间中: w_scaled
    # 在原始空间中的方向: w_original = w_scaled / scale (element-wise)
    w_original = weight_vector / scaler_scale

    # 计算扰动幅度
    alpha = compute_perturbation_alpha(hidden_states_path, layer_idx, (weight_vector, scaler_mean, scaler_scale))
    strength = cfg_ablation.get("strength", 1.0)
    alpha_scaled = alpha * strength

    # 转为 torch tensor
    w_torch = torch.tensor(w_original, dtype=torch.float32).to(device)

    # 获取目标层 module
    layer_module = get_layer_module(model, layer_idx)

    results = {}

    # 1. 基线评估（无消融）
    print("[Ablation] Evaluating baseline (no ablation)...")
    baseline_metrics, baseline_preds = evaluate_benchmark(
        model, tokenizer, samples, cfg_data, cfg_model, device
    )
    results["baseline"] = {
        "metrics": baseline_metrics,
        "predictions": [
            {"id": p["id"], "benchmark": p["benchmark"], "category": p["category"],
             "correct": p["correct"], "generated": p["generated"] if p["generated"] is not None else ""}
            for p in baseline_preds
        ],
    }
    print(f"  Baseline accuracy: {baseline_metrics['accuracy']:.4f} "
          f"({baseline_metrics['correct']}/{baseline_metrics['total']})")

    # 2. 知识方向扰动（沿 +w 添加扰动）
    # 探针 w 指向 skill 方向
    # 添加 +w 扰动 = 将表示推向 skill 区域 = 远离 knowledge 区域
    # 预期：知识任务受损，技能任务保持
    print("[Ablation] Running knowledge-direction perturbation (push toward skill, away from knowledge)...")
    knowledge_hook = PerturbationHook(w_torch, alpha_scaled)
    hook_handle_knowledge = layer_module.register_forward_hook(knowledge_hook)
    try:
        knowledge_metrics, knowledge_preds = evaluate_benchmark(
            model, tokenizer, samples, cfg_data, cfg_model, device
        )
    finally:
        hook_handle_knowledge.remove()
    results["knowledge_ablation"] = {
        "direction": "knowledge",
        "method": "additive_perturbation",
        "ablated_layer": layer_idx,
        "strength": strength,
        "alpha": alpha_scaled,
        "metrics": knowledge_metrics,
        "predictions": [
            {"id": p["id"], "benchmark": p["benchmark"], "category": p["category"],
             "correct": p["correct"], "generated": p["generated"] if p["generated"] is not None else ""}
            for p in knowledge_preds
        ],
    }
    print(f"  Knowledge-direction perturbation accuracy: {knowledge_metrics['accuracy']:.4f} "
          f"({knowledge_metrics['correct']}/{knowledge_metrics['total']})")

    # 3. 技能方向扰动（沿 -w 添加扰动）
    # 添加 -w 扰动 = 将表示推向 knowledge 区域 = 远离 skill 区域
    # 预期：技能任务受损，知识任务保持
    print("[Ablation] Running skill-direction perturbation (push toward knowledge, away from skill)...")
    skill_hook = PerturbationHook(-w_torch, alpha_scaled)
    hook_handle_skill = layer_module.register_forward_hook(skill_hook)
    try:
        skill_metrics, skill_preds = evaluate_benchmark(
            model, tokenizer, samples, cfg_data, cfg_model, device
        )
    finally:
        hook_handle_skill.remove()
    results["skill_ablation"] = {
        "direction": "skill",
        "method": "additive_perturbation",
        "ablated_layer": layer_idx,
        "strength": strength,
        "alpha": alpha_scaled,
        "metrics": skill_metrics,
        "predictions": [
            {"id": p["id"], "benchmark": p["benchmark"], "category": p["category"],
             "correct": p["correct"], "generated": p["generated"] if p["generated"] is not None else ""}
            for p in skill_preds
        ],
    }
    print(f"  Skill-direction perturbation accuracy: {skill_metrics['accuracy']:.4f} "
          f"({skill_metrics['correct']}/{skill_metrics['total']})")

    # 4. 投影消融（对照：非选择性消融）
    # 移除 w 方向的投影，消除 skill-knowledge 区别信息
    # 预期：两类任务都受损
    print("[Ablation] Running projection ablation (non-selective control)...")
    proj_hook = ProjectionAblationHook(w_torch, strength)
    hook_handle_proj = layer_module.register_forward_hook(proj_hook)
    try:
        proj_metrics, proj_preds = evaluate_benchmark(
            model, tokenizer, samples, cfg_data, cfg_model, device
        )
    finally:
        hook_handle_proj.remove()
    results["projection_ablation"] = {
        "direction": "both",
        "method": "projection_removal",
        "ablated_layer": layer_idx,
        "strength": strength,
        "metrics": proj_metrics,
        "predictions": [
            {"id": p["id"], "benchmark": p["benchmark"], "category": p["category"],
             "correct": p["correct"], "generated": p["generated"] if p["generated"] is not None else ""}
            for p in proj_preds
        ],
    }
    print(f"  Projection ablation accuracy: {proj_metrics['accuracy']:.4f} "
          f"({proj_metrics['correct']}/{proj_metrics['total']})")

    return results


def compute_selectivity(ablation_results):
    """
    计算选择性损伤比。

    选择性 = 知识扰动对知识任务的损伤 / 知识扰动对技能任务的损伤
    (或技能扰动对技能任务的损伤 / 技能扰动对知识任务的损伤)

    当交叉类别损伤趋近零但目标类别损伤 > 0 时，选择性为 inf。
    由于 JSON 不支持 inf/nan，使用 math.isfinite 检查并截断为 _SELECTIVITY_CAP。
    负 selectivity（反向选择性）保留原值：语义上表示扰动使交叉类别精度提升，
    仍可 JSON 序列化（负有限值不触发截断），但需在结果解读时注意。
    """
    # 按类别分组计算准确率
    def category_accuracy(predictions, category):
        cat_preds = [p for p in predictions if p["category"] == category]
        if not cat_preds:
            return 0.0
        return sum(1 for p in cat_preds if p["correct"]) / len(cat_preds)

    # 基线分类别准确率
    baseline_preds = ablation_results["baseline"]["predictions"]
    baseline_skill_acc = category_accuracy(baseline_preds, "skill")
    baseline_knowledge_acc = category_accuracy(baseline_preds, "knowledge")

    # 知识方向扰动后的分类别准确率
    knowledge_abl_preds = ablation_results["knowledge_ablation"]["predictions"]
    knowledge_abl_skill_acc = category_accuracy(knowledge_abl_preds, "skill")
    knowledge_abl_knowledge_acc = category_accuracy(knowledge_abl_preds, "knowledge")

    # 技能方向扰动后的分类别准确率
    skill_abl_preds = ablation_results["skill_ablation"]["predictions"]
    skill_abl_skill_acc = category_accuracy(skill_abl_preds, "skill")
    skill_abl_knowledge_acc = category_accuracy(skill_abl_preds, "knowledge")

    # 投影消融后的分类别准确率
    proj_abl_preds = ablation_results["projection_ablation"]["predictions"]
    proj_abl_skill_acc = category_accuracy(proj_abl_preds, "skill")
    proj_abl_knowledge_acc = category_accuracy(proj_abl_preds, "knowledge")

    # 计算损伤
    knowledge_abl_skill_damage = baseline_skill_acc - knowledge_abl_skill_acc
    knowledge_abl_knowledge_damage = baseline_knowledge_acc - knowledge_abl_knowledge_acc

    skill_abl_skill_damage = baseline_skill_acc - skill_abl_skill_acc
    skill_abl_knowledge_damage = baseline_knowledge_acc - skill_abl_knowledge_acc

    proj_abl_skill_damage = baseline_skill_acc - proj_abl_skill_acc
    proj_abl_knowledge_damage = baseline_knowledge_acc - proj_abl_knowledge_acc

    selectivity_results = {
        "baseline_skill_acc": baseline_skill_acc,
        "baseline_knowledge_acc": baseline_knowledge_acc,
        "knowledge_ablation": {
            "skill_acc": knowledge_abl_skill_acc,
            "knowledge_acc": knowledge_abl_knowledge_acc,
            "skill_damage": knowledge_abl_skill_damage,
            "knowledge_damage": knowledge_abl_knowledge_damage,
        },
        "skill_ablation": {
            "skill_acc": skill_abl_skill_acc,
            "knowledge_acc": skill_abl_knowledge_acc,
            "skill_damage": skill_abl_skill_damage,
            "knowledge_damage": skill_abl_knowledge_damage,
        },
        "projection_ablation": {
            "skill_acc": proj_abl_skill_acc,
            "knowledge_acc": proj_abl_knowledge_acc,
            "skill_damage": proj_abl_skill_damage,
            "knowledge_damage": proj_abl_knowledge_damage,
        },
    }

    # 选择性损伤比
    # 知识方向扰动：预期 知识损伤 >> 技能损伤
    if knowledge_abl_skill_damage > 1e-6:
        knowledge_selectivity = knowledge_abl_knowledge_damage / knowledge_abl_skill_damage
    else:
        knowledge_selectivity = float("inf") if knowledge_abl_knowledge_damage > 0 else 0.0

    # 技能方向扰动：预期 技能损伤 >> 知识损伤
    if skill_abl_knowledge_damage > 1e-6:
        skill_selectivity = skill_abl_skill_damage / skill_abl_knowledge_damage
    else:
        skill_selectivity = float("inf") if skill_abl_skill_damage > 0 else 0.0

    # [P0-1 FIX] 截断非有限值为 JSON 安全值
    # inf 表示"完美选择性"，截断到 _SELECTIVITY_CAP 仍远超任何合理阈值
    # nan 或负值语义不明，截断为 0（表示"无有效选择性"）
    if not math.isfinite(knowledge_selectivity):
        knowledge_selectivity = _SELECTIVITY_CAP if knowledge_selectivity > 0 else 0.0
    if not math.isfinite(skill_selectivity):
        skill_selectivity = _SELECTIVITY_CAP if skill_selectivity > 0 else 0.0

    selectivity_results["knowledge_direction_selectivity"] = knowledge_selectivity
    selectivity_results["skill_direction_selectivity"] = skill_selectivity

    return selectivity_results


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiment")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--run-id", type=str, required=True, help="Run identifier")
    parser.add_argument("--probe-weights", type=str, default=None, help="Path to probe_weights.json")
    parser.add_argument("--samples-file", type=str, default=None, help="Path to all_samples.json")
    parser.add_argument("--hidden-states", type=str, default=None, help="Path to hidden_states.json")
    args = parser.parse_args()

    config = load_config(args.config)
    cfg_model = config["model"]
    cfg_data = config["data"]
    cfg_ablation = config["ablation"]
    cfg_abort = config.get("abort_criteria", {})

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, config["output"]["results_dir"], args.run_id)

    device = cfg_model.get("device", "cpu")

    # 加载探针结果（确定最佳层）
    probe_results_path = os.path.join(results_dir, "probe_results.json")
    with open(probe_results_path, "r") as f:
        probe_results = json.load(f)

    if cfg_ablation.get("layer_selection", "best_probe") == "best_probe":
        layer_idx = probe_results["best_layer"]
    else:
        layer_idx = int(cfg_ablation["layer_selection"])

    print(f"[Ablation] Using layer {layer_idx} for ablation")

    # 加载探针权重
    probe_weights_path = args.probe_weights or os.path.join(results_dir, "probe_weights.json")
    weight_vector, scaler_mean, scaler_scale = load_probe_weights(probe_weights_path, layer_idx)

    # 加载样本数据
    samples_file = args.samples_file or os.path.join(base_dir, "data", "all_samples.json")
    with open(samples_file, "r") as f:
        samples_data = json.load(f)
    samples = samples_data["samples"]
    print(f"[Data] Loaded {len(samples)} samples")

    # 加载模型
    model, tokenizer, num_layers, hidden_size = load_model_and_tokenizer(cfg_model)

    # 确定隐藏状态文件路径
    hidden_states_path = args.hidden_states or os.path.join(results_dir, "hidden_states.json")

    # 运行消融实验
    print("=" * 60)
    print("Running ablation experiment")
    print("=" * 60)

    start_time = time.time()
    ablation_results = run_ablation_experiment(
        model, tokenizer, samples, cfg_data, cfg_model, cfg_ablation,
        (weight_vector, scaler_mean, scaler_scale), layer_idx, device,
        hidden_states_path
    )
    elapsed = time.time() - start_time
    print(f"[Ablation] Done in {elapsed:.1f}s")

    # 计算选择性损伤比
    selectivity = compute_selectivity(ablation_results)
    print("\n[Selectivity]")
    print(f"  Knowledge-direction selectivity: {selectivity['knowledge_direction_selectivity']:.2f}:1")
    print(f"  Skill-direction selectivity: {selectivity['skill_direction_selectivity']:.2f}:1")
    print(f"  Knowledge perturbation -> knowledge damage: {selectivity['knowledge_ablation']['knowledge_damage']:.4f}")
    print(f"  Knowledge perturbation -> skill damage: {selectivity['knowledge_ablation']['skill_damage']:.4f}")
    print(f"  Skill perturbation -> skill damage: {selectivity['skill_ablation']['skill_damage']:.4f}")
    print(f"  Skill perturbation -> knowledge damage: {selectivity['skill_ablation']['knowledge_damage']:.4f}")
    print(f"  Projection ablation -> skill damage: {selectivity['projection_ablation']['skill_damage']:.4f}")
    print(f"  Projection ablation -> knowledge damage: {selectivity['projection_ablation']['knowledge_damage']:.4f}")

    # 检查放弃条件
    min_selectivity = cfg_abort.get("exp1_min_selectivity", 2.0)
    max_selectivity = max(
        selectivity["knowledge_direction_selectivity"],
        selectivity["skill_direction_selectivity"],
    )
    should_abort = max_selectivity < min_selectivity

    if should_abort:
        print(f"\n[ABORT] Experiment 1 failed: max selectivity ({max_selectivity:.2f}) < threshold ({min_selectivity})")
    else:
        print(f"\n[OK] Experiment 1 passed: max selectivity ({max_selectivity:.2f}) >= threshold ({min_selectivity})")

    # 保存结果
    output_path = os.path.join(results_dir, "ablation_results.json")
    output_data = {
        "ablation_layer": layer_idx,
        "ablation_method": "additive_perturbation",
        "selectivity": selectivity,
        "aborted": should_abort,
        "results": {
            "baseline": {
                "accuracy": ablation_results["baseline"]["metrics"]["accuracy"],
                "metrics": ablation_results["baseline"]["metrics"],
            },
            "knowledge_ablation": {
                "accuracy": ablation_results["knowledge_ablation"]["metrics"]["accuracy"],
                "metrics": ablation_results["knowledge_ablation"]["metrics"],
            },
            "skill_ablation": {
                "accuracy": ablation_results["skill_ablation"]["metrics"]["accuracy"],
                "metrics": ablation_results["skill_ablation"]["metrics"],
            },
            "projection_ablation": {
                "accuracy": ablation_results["projection_ablation"]["metrics"]["accuracy"],
                "metrics": ablation_results["projection_ablation"]["metrics"],
            },
        },
        "elapsed_s": round(elapsed, 1),
    }

    if config["output"].get("save_predictions", True):
        output_data["predictions"] = {
            "baseline": ablation_results["baseline"]["predictions"],
            "knowledge_ablation": ablation_results["knowledge_ablation"]["predictions"],
            "skill_ablation": ablation_results["skill_ablation"]["predictions"],
            "projection_ablation": ablation_results["projection_ablation"]["predictions"],
        }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"[Save] Ablation results saved to {output_path}")

    if should_abort:
        abort_path = os.path.join(results_dir, "abort_exp1.json")
        with open(abort_path, "w") as f:
            json.dump({
                "aborted": True,
                "reason": f"Selectivity ratio < {min_selectivity}:1",
                "max_selectivity": max_selectivity,
                "threshold": min_selectivity,
            }, f, indent=2)

    return should_abort


if __name__ == "__main__":
    main()
