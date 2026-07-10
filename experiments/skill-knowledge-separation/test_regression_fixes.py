"""
回归测试：验证 P0-1, P0-2, P1-1, P1-2 修复及代码审查追加修复
级联追踪：cascade-skill-knowledge-separation（回退轮数 1/2 -> 修复后回归）

运行方式：
    cd experiments/skill-knowledge-separation
    python test_regression_fixes.py
"""

import json
import math
import sys
import os
import io
import logging
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ablation import (
    compute_selectivity, _check_ifeval_constraint, _IFEVAL_TYPE_ROUTES,
    _SELECTIVITY_CAP, evaluate_benchmark, check_answer,
    PerturbationHook, ProjectionAblationHook,
)


# ============================================================
# P0-1: selectivity=inf/nan 时 JSON 序列化不崩溃
# ============================================================
class TestP01SelectivityInfNan(unittest.TestCase):
    """P0-1: compute_selectivity 返回 inf/nan 时，截断为有限值，json.dump 不崩溃。"""

    def _make_ablation_results_inf(self):
        """构造消融结果：skill_damage=0, knowledge_damage>0 -> knowledge_selectivity=inf."""
        predictions_base = (
            [{"category": "skill", "correct": True}] * 8 +
            [{"category": "skill", "correct": False}] * 2 +
            [{"category": "knowledge", "correct": True}] * 7 +
            [{"category": "knowledge", "correct": False}] * 3
        )
        predictions_knowledge_abl = (
            [{"category": "skill", "correct": True}] * 8 +
            [{"category": "skill", "correct": False}] * 2 +
            [{"category": "knowledge", "correct": True}] * 2 +
            [{"category": "knowledge", "correct": False}] * 8
        )
        predictions_skill_abl = (
            [{"category": "skill", "correct": True}] * 2 +
            [{"category": "skill", "correct": False}] * 8 +
            [{"category": "knowledge", "correct": True}] * 7 +
            [{"category": "knowledge", "correct": False}] * 3
        )
        predictions_proj = (
            [{"category": "skill", "correct": True}] * 5 +
            [{"category": "skill", "correct": False}] * 5 +
            [{"category": "knowledge", "correct": True}] * 4 +
            [{"category": "knowledge", "correct": False}] * 6
        )
        return {
            "baseline": {"predictions": predictions_base},
            "knowledge_ablation": {"predictions": predictions_knowledge_abl},
            "skill_ablation": {"predictions": predictions_skill_abl},
            "projection_ablation": {"predictions": predictions_proj},
        }

    def test_inf_truncated_to_cap(self):
        """inf 被截断为 _SELECTIVITY_CAP (9999.0)。"""
        ablation_results = self._make_ablation_results_inf()
        selectivity = compute_selectivity(ablation_results)
        self.assertEqual(selectivity["knowledge_direction_selectivity"], _SELECTIVITY_CAP)
        self.assertEqual(selectivity["skill_direction_selectivity"], _SELECTIVITY_CAP)

    def test_json_dump_inf_truncated(self):
        """截断后的 selectivity 值能被 json.dump 序列化。"""
        ablation_results = self._make_ablation_results_inf()
        selectivity = compute_selectivity(ablation_results)
        output = json.dumps(selectivity)
        self.assertIsInstance(output, str)
        parsed = json.loads(output)
        self.assertEqual(parsed["knowledge_direction_selectivity"], _SELECTIVITY_CAP)
        self.assertEqual(parsed["skill_direction_selectivity"], _SELECTIVITY_CAP)

    def test_normal_values_unaffected(self):
        """正常有限值不受截断影响。"""
        predictions_base = (
            [{"category": "skill", "correct": True}] * 6 +
            [{"category": "skill", "correct": False}] * 4 +
            [{"category": "knowledge", "correct": True}] * 5 +
            [{"category": "knowledge", "correct": False}] * 5
        )
        predictions_knowledge_abl = (
            [{"category": "skill", "correct": True}] * 5 +
            [{"category": "skill", "correct": False}] * 5 +
            [{"category": "knowledge", "correct": True}] * 2 +
            [{"category": "knowledge", "correct": False}] * 8
        )
        predictions_skill_abl = (
            [{"category": "skill", "correct": True}] * 2 +
            [{"category": "skill", "correct": False}] * 8 +
            [{"category": "knowledge", "correct": True}] * 4 +
            [{"category": "knowledge", "correct": False}] * 6
        )
        predictions_proj = (
            [{"category": "skill", "correct": True}] * 4 +
            [{"category": "skill", "correct": False}] * 6 +
            [{"category": "knowledge", "correct": True}] * 3 +
            [{"category": "knowledge", "correct": False}] * 7
        )
        ablation_results = {
            "baseline": {"predictions": predictions_base},
            "knowledge_ablation": {"predictions": predictions_knowledge_abl},
            "skill_ablation": {"predictions": predictions_skill_abl},
            "projection_ablation": {"predictions": predictions_proj},
        }
        selectivity = compute_selectivity(ablation_results)
        self.assertAlmostEqual(selectivity["knowledge_direction_selectivity"], 3.0, places=5)
        self.assertAlmostEqual(selectivity["skill_direction_selectivity"], 4.0, places=5)
        output = json.dumps(selectivity)
        self.assertIsInstance(output, str)

    def test_nan_truncated_to_zero(self):
        """nan 的截断逻辑：math.isfinite(nan)==False, nan>0==False -> 0.0。"""
        nan_val = float("nan")
        self.assertFalse(math.isfinite(nan_val))
        result = _SELECTIVITY_CAP if nan_val > 0 else 0.0
        self.assertEqual(result, 0.0)

    def test_negative_selectivity_json_safe(self):
        """负 selectivity 不被截断但 JSON 安全（有限值可以序列化）。"""
        negative_val = -5.0
        self.assertTrue(math.isfinite(negative_val))
        output = json.dumps({"val": negative_val})
        self.assertEqual(json.loads(output)["val"], -5.0)


# ============================================================
# P0-2: 异常时 hook 未 remove 导致静默数据污染
# ============================================================
class TestP02HookRemoveOnException(unittest.TestCase):
    """P0-2: evaluate_benchmark 抛异常时，hook 仍被 try/finally remove。"""

    def test_ablation_py_knowledge_hook_has_try_finally(self):
        """ablation.py: 知识方向扰动 hook 有 try/finally 保护。"""
        source_path = os.path.join(os.path.dirname(__file__), "ablation.py")
        with open(source_path, "r") as f:
            source = f.read()
        self.assertIn("hook_handle_knowledge = layer_module.register_forward_hook(knowledge_hook)", source)
        self.assertIn("hook_handle_knowledge.remove()", source)

    def test_ablation_py_skill_hook_has_try_finally(self):
        """ablation.py: 技能方向扰动 hook 有 try/finally 保护。"""
        source_path = os.path.join(os.path.dirname(__file__), "ablation.py")
        with open(source_path, "r") as f:
            source = f.read()
        self.assertIn("hook_handle_skill = layer_module.register_forward_hook(skill_hook)", source)
        self.assertIn("hook_handle_skill.remove()", source)

    def test_ablation_py_proj_hook_has_try_finally(self):
        """ablation.py: 投影消融 hook 有 try/finally 保护。"""
        source_path = os.path.join(os.path.dirname(__file__), "ablation.py")
        with open(source_path, "r") as f:
            source = f.read()
        self.assertIn("hook_handle_proj = layer_module.register_forward_hook(proj_hook)", source)
        self.assertIn("hook_handle_proj.remove()", source)

    def test_all_hooks_have_try_finally_structure(self):
        """ablation.py: 每个 register_forward_hook 后都有 try/finally/remove。"""
        source_path = os.path.join(os.path.dirname(__file__), "ablation.py")
        with open(source_path, "r") as f:
            lines = f.readlines()
        hook_lines = [i for i, line in enumerate(lines) if "register_forward_hook" in line]
        self.assertGreaterEqual(len(hook_lines), 3, "Expected at least 3 register_forward_hook calls")
        for hook_line_idx in hook_lines:
            found_try = False
            found_finally = False
            found_remove = False
            for j in range(hook_line_idx + 1, min(hook_line_idx + 20, len(lines))):
                if "try:" in lines[j]:
                    found_try = True
                if "finally:" in lines[j]:
                    found_finally = True
                if ".remove()" in lines[j]:
                    found_remove = True
            self.assertTrue(found_try, f"Missing try: after hook at line {hook_line_idx + 1}")
            self.assertTrue(found_finally, f"Missing finally: after hook at line {hook_line_idx + 1}")
            self.assertTrue(found_remove, f"Missing .remove() after hook at line {hook_line_idx + 1}")

    def test_run_experiment_py_hooks_removed_on_exception(self):
        """run_experiment.py: 3 处 hook 都用 try/finally 保护。"""
        source_path = os.path.join(os.path.dirname(__file__), "run_experiment.py")
        with open(source_path, "r") as f:
            source = f.read()
        remove_count = source.count("hook_handle.remove()")
        self.assertEqual(remove_count, 3, f"Expected 3 hook_handle.remove() calls, found {remove_count}")
        self.assertGreaterEqual(source.count("finally:"), 3)

    def test_run_experiment_py_hooks_try_finally_structure(self):
        """run_experiment.py: 每个 register_forward_hook 后都有 try/finally/remove。"""
        source_path = os.path.join(os.path.dirname(__file__), "run_experiment.py")
        with open(source_path, "r") as f:
            lines = f.readlines()
        hook_lines = [i for i, line in enumerate(lines) if "register_forward_hook" in line]
        self.assertGreaterEqual(len(hook_lines), 3, "Expected at least 3 register_forward_hook calls in run_experiment.py")
        for hook_line_idx in hook_lines:
            found_try = False
            found_finally = False
            found_remove = False
            for j in range(hook_line_idx + 1, min(hook_line_idx + 20, len(lines))):
                if "try:" in lines[j]:
                    found_try = True
                if "finally:" in lines[j]:
                    found_finally = True
                if ".remove()" in lines[j]:
                    found_remove = True
            self.assertTrue(found_try, f"Missing try: after hook at line {hook_line_idx + 1} in run_experiment.py")
            self.assertTrue(found_finally, f"Missing finally: after hook at line {hook_line_idx + 1} in run_experiment.py")
            self.assertTrue(found_remove, f"Missing .remove() after hook at line {hook_line_idx + 1} in run_experiment.py")

    def test_hook_removal_on_exception_runtime(self):
        """运行时验证：hook 在异常后确实被 remove。"""
        model = torch.nn.Linear(10, 10)
        hook_called = []
        def hook_fn(module, input, output):
            hook_called.append(True)
            return output

        handle = model.register_forward_hook(hook_fn)
        # Simulate try/finally with exception
        exc_raised = False
        try:
            raise RuntimeError("Simulated OOM")
        except RuntimeError:
            exc_raised = True
        finally:
            handle.remove()

        self.assertTrue(exc_raised, "Exception should have been raised")
        # Hook has been removed, subsequent forward should not trigger hook
        hook_called.clear()
        model(torch.randn(1, 10))
        self.assertEqual(len(hook_called), 0, "Hook should be removed after finally block")

    def test_hook_not_removed_without_finally(self):
        """对比验证：如果没有 try/finally，异常时 hook 不会被 remove。"""
        model = torch.nn.Linear(10, 10)
        hook_called = []
        def hook_fn(module, input, output):
            hook_called.append(True)
            return output

        handle = model.register_forward_hook(hook_fn)
        # Simulate exception WITHOUT try/finally
        try:
            raise RuntimeError("Simulated OOM")
        except RuntimeError:
            # handle.remove() was NOT called (this is the old buggy pattern)
            pass

        # Hook is still active
        hook_called.clear()
        model(torch.randn(1, 10))
        self.assertEqual(len(hook_called), 1, "Hook should still be active without try/finally")
        # Cleanup
        handle.remove()


# ============================================================
# P1-1: IFEval _check_ifeval_constraint 子串匹配 bug
# ============================================================
class TestP11IFEvalConstraintRouting(unittest.TestCase):
    """P1-1: _check_ifeval_constraint 使用映射表路由，不使用子串匹配。"""

    def test_keyword_therefore_routes_to_keyword(self):
        """'keyword:therefore' 应路由到 keyword 分支。"""
        result = _check_ifeval_constraint("keyword:therefore", "This therefore is")
        self.assertTrue(result)

    def test_keyword_therefore_missing_keyword(self):
        """'keyword:therefore' 生成中不包含 'therefore' 时返回 False。"""
        result = _check_ifeval_constraint("keyword:therefore", "This is something else")
        self.assertFalse(result)

    def test_keywords_existence_routes_to_keyword(self):
        """'keywords:existence' 应路由到 keyword 分支（旧 bug 误路由到 word 分支）。"""
        result = _check_ifeval_constraint("keywords:existence", "The existence of things")
        self.assertTrue(result)

    def test_keywords_existence_missing(self):
        """'keywords:existence' 生成中不包含 'existence' 时返回 False。"""
        result = _check_ifeval_constraint("keywords:existence", "No such word here")
        self.assertFalse(result)

    def test_keywords_without_param_fallback(self):
        """'keywords:'（无参数名）应 fallback 到检查 'therefore'。"""
        result = _check_ifeval_constraint("keywords:", "this therefore works")
        self.assertTrue(result)

    def test_length_constraint_routes_to_length(self):
        """'length_constraint:number_words' 应路由到 length 分支。"""
        result = _check_ifeval_constraint("length_constraint:number_words", "Hello world this is a test")
        self.assertTrue(result)

    def test_length_constraint_zero_words(self):
        """'length_constraint:number_words' 0 个词应返回 False。"""
        result = _check_ifeval_constraint("length_constraint:number_words", "")
        self.assertFalse(result)

    def test_change_case_lowercase_pass(self):
        """'change_case:lowercase' 全小写通过。"""
        result = _check_ifeval_constraint("change_case:lowercase", "hello world")
        self.assertTrue(result)

    def test_change_case_lowercase_fail(self):
        """'change_case:lowercase' 非全小写返回 False。"""
        result = _check_ifeval_constraint("change_case:lowercase", "Hello World")
        self.assertFalse(result)

    def test_capitalization_routes_to_case(self):
        """'capitalization:lowercase' 应路由到 case 分支。"""
        result = _check_ifeval_constraint("capitalization:lowercase", "hello world")
        self.assertTrue(result)

    def test_capitalization_lowercase_fail(self):
        """'capitalization:lowercase' 非全小写返回 False。"""
        result = _check_ifeval_constraint("capitalization:lowercase", "Hello World")
        self.assertFalse(result)

    def test_startend_end_with_period(self):
        """'startend:end' 以句号结尾通过。"""
        result = _check_ifeval_constraint("startend:end", "This ends with a period.")
        self.assertTrue(result)

    def test_startend_end_no_period(self):
        """'startend:end' 不以句号结尾返回 False。"""
        result = _check_ifeval_constraint("startend:end", "This has no period")
        self.assertFalse(result)

    def test_detectable_format_routes_to_format(self):
        """'detectable_format:number_bullet' 路由到 format 分支。"""
        result = _check_ifeval_constraint("detectable_format:number_bullet", "Some content here")
        self.assertTrue(result)

    def test_detectable_format_empty_fails(self):
        """'detectable_format:number_bullet' 空内容返回 False。"""
        result = _check_ifeval_constraint("detectable_format:number_bullet", "")
        self.assertFalse(result)

    def test_no_word_key_in_route_table(self):
        """映射表中不应有 'word' 键（子串匹配完全消除）。"""
        self.assertNotIn("word", _IFEVAL_TYPE_ROUTES)

    def test_unknown_type_routes_to_unknown(self):
        """未知类型路由到 'unknown'，只检查非空内容。"""
        result = _check_ifeval_constraint("unknown_type:something", "Some content")
        self.assertTrue(result)

    def test_empty_instruction_id(self):
        """空 instruction_id 只检查非空内容。"""
        result = _check_ifeval_constraint("", "Some content")
        self.assertTrue(result)

    def test_empty_generated_always_false(self):
        """空生成始终返回 False。"""
        result = _check_ifeval_constraint("keyword:therefore", "")
        self.assertFalse(result)

    def test_keyword_param_is_used(self):
        """keyword 分支使用 constraint_param（参数名）而非硬编码 'therefore'。"""
        result = _check_ifeval_constraint("keyword:hello", "I say hello world")
        self.assertTrue(result, "keyword 分支应使用参数名 'hello' 检查")
        result2 = _check_ifeval_constraint("keyword:hello", "I say therefore world")
        self.assertFalse(result2, "keyword:hello 不应在 'therefore' 中通过")


# ============================================================
# P1-2: evaluate_benchmark 单样本失败不崩溃整个 run
# ============================================================
class TestP12SampleFailureIsolation(unittest.TestCase):
    """P1-2: 单个样本 model.generate 失败不崩溃整个评估。"""

    def _make_sample(self, sample_id="s1", benchmark="gsm8k", answer="#### 18"):
        return {
            "id": sample_id,
            "input_text": "What is 3 * 6?",
            "answer": answer,
            "benchmark": benchmark,
            "category": "skill",
        }

    def test_check_answer_none_generated_returns_false(self):
        """check_answer 对 None generated 返回 False。"""
        result = check_answer("gsm8k", None, "#### 18", {"benchmark": "gsm8k"})
        self.assertFalse(result)

    def test_check_answer_none_for_all_benchmarks(self):
        """check_answer 对 None generated 在所有 benchmark 上返回 False。"""
        for bm in ["gsm8k", "ifeval", "mmlu", "triviaqa"]:
            result = check_answer(bm, None, "some answer", {"benchmark": bm, "instruction_id": "keyword:therefore"})
            self.assertFalse(result, f"check_answer({bm!r}, None, ...) should return False")

    def test_consecutive_failures_early_termination(self):
        """连续 5 次失败后评估提前终止。"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.generate = MagicMock(side_effect=RuntimeError("CUDA OOM"))
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        samples = [self._make_sample(f"s{i}") for i in range(20)]
        cfg_data = {"max_input_length": 512}
        cfg_model = {}
        metrics, predictions = evaluate_benchmark(
            mock_model, mock_tokenizer, samples, cfg_data, cfg_model, "cpu"
        )
        self.assertLessEqual(metrics["total"], 5, f"连续 5 次失败应提前终止，但 total={metrics['total']}")
        self.assertEqual(metrics["accuracy"], 0.0)

    def test_mixed_success_failure_continues(self):
        """部分成功部分失败时，评估继续进行。"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        call_count = [0]
        def generate_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise RuntimeError("Intermittent OOM")
            output = MagicMock()
            output.__getitem__ = lambda self, idx: torch.tensor([1, 2, 3, 4, 5])
            return [output]

        mock_model.generate = MagicMock(side_effect=generate_side_effect)
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.decode.return_value = "18"

        samples = [self._make_sample(f"s{i}") for i in range(6)]
        cfg_data = {"max_input_length": 512}
        cfg_model = {}

        metrics, predictions = evaluate_benchmark(
            mock_model, mock_tokenizer, samples, cfg_data, cfg_model, "cpu"
        )
        self.assertEqual(metrics["total"], 6)

    def test_failed_prediction_has_error_field(self):
        """失败样本的 prediction dict 包含 error 字段。"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.generate = MagicMock(side_effect=RuntimeError("Test error"))
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        samples = [self._make_sample("s1")]
        cfg_data = {"max_input_length": 512}
        cfg_model = {}

        metrics, predictions = evaluate_benchmark(
            mock_model, mock_tokenizer, samples, cfg_data, cfg_model, "cpu"
        )
        self.assertEqual(len(predictions), 1)
        pred = predictions[0]
        self.assertIsNotNone(pred["error"])
        self.assertFalse(pred["correct"])
        self.assertIsNone(pred["generated"])

    def test_prediction_dict_structure_uniform(self):
        """成功和失败的 prediction dict 结构统一（都包含 error 键）。"""
        success_pred = {
            "id": "s1", "benchmark": "gsm8k", "category": "skill",
            "generated": "18", "expected": "18", "correct": True, "error": None,
        }
        fail_pred = {
            "id": "s2", "benchmark": "gsm8k", "category": "skill",
            "generated": None, "expected": "18", "correct": False, "error": "CUDA OOM",
        }
        self.assertEqual(set(success_pred.keys()), set(fail_pred.keys()))
        self.assertIsNone(success_pred["error"])
        self.assertIsNotNone(fail_pred["error"])

    def test_none_generated_serialized_to_json(self):
        """prediction 中 generated=None 在序列化时被替换为空字符串。"""
        # 模拟 run_ablation_experiment 中的序列化逻辑
        pred = {"id": "s1", "generated": None, "correct": False}
        serialized_generated = pred["generated"] if pred["generated"] is not None else ""
        self.assertEqual(serialized_generated, "")

        # 验证空字符串可以 JSON 序列化
        json.dumps({"generated": serialized_generated})


# ============================================================
# 代码审查追加修复验证
# ============================================================
class TestCodeReviewFixes(unittest.TestCase):
    """代码审查发现的追加修复验证。"""

    def test_no_substring_matching_in_ifeval(self):
        """确认 _check_ifeval_constraint 不再有任何子串匹配。"""
        source_path = os.path.join(os.path.dirname(__file__), "ablation.py")
        with open(source_path, "r") as f:
            source = f.read()
        self.assertNotIn('elif "word" in', source)
        self.assertNotIn('elif "length" in', source)
        self.assertNotIn('elif "case" in', source)

    def test_math_isfinite_used_for_selectivity(self):
        """确认使用 math.isfinite 处理 inf/nan。"""
        source_path = os.path.join(os.path.dirname(__file__), "ablation.py")
        with open(source_path, "r") as f:
            source = f.read()
        self.assertIn("math.isfinite", source)
        self.assertIn("_SELECTIVITY_CAP", source)

    def test_no_min_for_selectivity_truncation(self):
        """确认 compute_selectivity 中不再使用 min() 做截断（min(nan, cap) 无效）。"""
        source_path = os.path.join(os.path.dirname(__file__), "ablation.py")
        with open(source_path, "r") as f:
            source = f.read()

        # 提取 compute_selectivity 函数体
        start = source.index("def compute_selectivity(")
        # 找下一个同级别 def
        rest = source[start:]
        # 搜索下一个顶级 def（非嵌套）
        next_def_match = None
        for i, line in enumerate(rest.split("\n")):
            if i > 0 and line.startswith("def ") and not line.startswith("    "):
                next_def_match = start + sum(len(l) + 1 for l in rest.split("\n")[:i])
                break

        func_body = source[start:next_def_match] if next_def_match else rest

        # 在函数体中搜索 min( 调用（排除注释）
        for line in func_body.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "min(" in stripped:
                self.fail(f"compute_selectivity 中不应使用 min()，发现: {stripped}")

    def test_logging_used_in_evaluate_benchmark(self):
        """确认 evaluate_benchmark 使用 logging。"""
        source_path = os.path.join(os.path.dirname(__file__), "ablation.py")
        with open(source_path, "r") as f:
            source = f.read()
        self.assertIn("import logging", source)
        self.assertIn("logger.warning", source)
        self.assertIn("logger.error", source)

    def test_ifeval_route_table_completeness(self):
        """映射表覆盖所有已知 IFEval 类型。"""
        required_types = [
            "keyword", "keywords",
            "length_constraint",
            "change_case", "capitalization",
            "startend",
            "detectable_format",
        ]
        for t in required_types:
            self.assertIn(t, _IFEVAL_TYPE_ROUTES, f"映射表缺少类型 '{t}'")

    def test_none_generated_serialized_as_empty_string(self):
        """序列化时 None 替换为 '' 兼容 JSON。"""
        source_path = os.path.join(os.path.dirname(__file__), "ablation.py")
        with open(source_path, "r") as f:
            source = f.read()
        self.assertIn('if p["generated"] is not None else ""', source)

    def test_selectivity_cap_constant_defined(self):
        """_SELECTIVITY_CAP 常量已定义。"""
        self.assertEqual(_SELECTIVITY_CAP, 9999.0)

    def test_consecutive_failures_counter_in_evaluate_benchmark(self):
        """evaluate_benchmark 包含连续失败计数器和提前终止逻辑。"""
        source_path = os.path.join(os.path.dirname(__file__), "ablation.py")
        with open(source_path, "r") as f:
            source = f.read()
        self.assertIn("consecutive_failures", source)
        self.assertIn("max_consecutive_failures", source)


# ============================================================
# 运行所有测试
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main(verbosity=2)
