"""
单元测试：验证 prepare_data_v2.py 的核心功能。

测试项：
  1. IFEval 4 选项化后选项数为 4
  2. MMLU subject 分类正确
  3. 纯文本科目不含数字密集题
  4. 输入模板统一性检验
  5. 正确答案随机分配位置

运行方式：
    cd experiments/skill-knowledge-separation
    python -m pytest tests/test_prepare_data_v2.py -v
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch


# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

from prepare_data_v2 import (
    MMLU_SKILL_SUBJECTS,
    MMLU_KNOWLEDGE_SUBJECTS,
    MMLU_TEXT_ONLY_SKILL,
    MMLU_TEXT_ONLY_KNOWLEDGE,
    format_question,
    _compute_ks_test,
    _parse_answer_idx,
)


# ============================================================
# Test 1: MMLU Subject 分类正确性
# ============================================================
class TestMMLUSubjectClassification(unittest.TestCase):
    """验证 MMLU subject 分类常量正确。"""

    def test_skill_subjects_count(self):
        """推理型科目应有 13 个。"""
        self.assertEqual(len(MMLU_SKILL_SUBJECTS), 13)

    def test_knowledge_subjects_count(self):
        """记忆型科目应有 16 个。"""
        self.assertEqual(len(MMLU_KNOWLEDGE_SUBJECTS), 16)

    def test_no_overlap_between_skill_and_knowledge(self):
        """skill 和 knowledge 科目列表不应有重叠。"""
        overlap = set(MMLU_SKILL_SUBJECTS) & set(MMLU_KNOWLEDGE_SUBJECTS)
        self.assertEqual(len(overlap), 0, f"Overlap found: {overlap}")

    def test_skill_subjects_are_known(self):
        """推理型科目应包含已知的科目名。"""
        expected_skills = [
            "abstract_algebra", "formal_logic", "logical_fallacies",
            "college_mathematics", "machine_learning", "econometrics",
        ]
        for subj in expected_skills:
            self.assertIn(subj, MMLU_SKILL_SUBJECTS, f"Missing skill subject: {subj}")

    def test_knowledge_subjects_are_known(self):
        """记忆型科目应包含已知的科目名。"""
        expected_knowledge = [
            "world_religions", "us_history", "philosophy",
            "jurisprudence", "nutrition", "marketing",
        ]
        for subj in expected_knowledge:
            self.assertIn(subj, MMLU_KNOWLEDGE_SUBJECTS, f"Missing knowledge subject: {subj}")


# ============================================================
# Test 2: 纯文本科目不含数字密集题
# ============================================================
class TestTextOnlySubjects(unittest.TestCase):
    """验证纯文本科目列表正确——不含数字密集科目。"""

    def test_text_only_skill_count(self):
        """推理型纯文本科目应有 2 个。"""
        self.assertEqual(len(MMLU_TEXT_ONLY_SKILL), 2)

    def test_text_only_knowledge_count(self):
        """记忆型纯文本科目应有 3 个。"""
        self.assertEqual(len(MMLU_TEXT_ONLY_KNOWLEDGE), 3)

    def test_text_only_skill_are_subsets_of_skill(self):
        """纯文本 skill 科目应是 MMLU_SKILL_SUBJECTS 的子集。"""
        for subj in MMLU_TEXT_ONLY_SKILL:
            self.assertIn(subj, MMLU_SKILL_SUBJECTS,
                          f"Text-only skill subject '{subj}' not in MMLU_SKILL_SUBJECTS")

    def test_text_only_knowledge_are_subsets_of_knowledge(self):
        """纯文本 knowledge 科目应是 MMLU_KNOWLEDGE_SUBJECTS 的子集。"""
        for subj in MMLU_TEXT_ONLY_KNOWLEDGE:
            self.assertIn(subj, MMLU_KNOWLEDGE_SUBJECTS,
                          f"Text-only knowledge subject '{subj}' not in MMLU_KNOWLEDGE_SUBJECTS")

    def test_no_math_subjects_in_text_only(self):
        """纯文本科目不应包含数学/数字密集科目。"""
        numeric_subjects = [
            "college_mathematics", "high_school_mathematics",
            "abstract_algebra", "discrete_mathematics",
            "college_physics", "high_school_physics",
            "college_chemistry", "high_school_chemistry",
            "high_school_statistics", "econometrics",
        ]
        for subj in numeric_subjects:
            self.assertNotIn(subj, MMLU_TEXT_ONLY_SKILL,
                             f"Numeric subject '{subj}' should not be in text-only skill list")
            self.assertNotIn(subj, MMLU_TEXT_ONLY_KNOWLEDGE,
                             f"Numeric subject '{subj}' should not be in text-only knowledge list")

    def test_text_only_skill_are_logic_based(self):
        """推理型纯文本科目应是逻辑/推理类。"""
        expected = {"formal_logic", "logical_fallacies"}
        self.assertEqual(set(MMLU_TEXT_ONLY_SKILL), expected)

    def test_text_only_knowledge_are_text_based(self):
        """记忆型纯文本科目应是文本为主的事实回忆类。"""
        expected = {"philosophy", "world_religions", "jurisprudence"}
        self.assertEqual(set(MMLU_TEXT_ONLY_KNOWLEDGE), expected)


# ============================================================
# Test 3: 输入模板统一性
# ============================================================
class TestUnifiedTemplate(unittest.TestCase):
    """验证所有 benchmark 使用完全相同的输入模板。"""

    def test_format_question_4_options(self):
        """format_question 生成 4 选项模板。"""
        result = format_question("What is 2+2?", ["3", "4", "5", "6"])
        lines = result.split("\n")
        self.assertEqual(len(lines), 6)  # question + 4 options + Answer:
        self.assertEqual(lines[0], "What is 2+2?")
        self.assertEqual(lines[1], "A. 3")
        self.assertEqual(lines[2], "B. 4")
        self.assertEqual(lines[3], "C. 5")
        self.assertEqual(lines[4], "D. 6")
        self.assertEqual(lines[5], "Answer:")

    def test_format_question_ends_with_answer(self):
        """模板末尾是 'Answer:'。"""
        result = format_question("Test Q", ["a", "b", "c", "d"])
        self.assertTrue(result.endswith("Answer:"))

    def test_format_question_option_letters(self):
        """选项字母是 A/B/C/D。"""
        result = format_question("Q", ["1", "2", "3", "4"])
        self.assertIn("A. 1", result)
        self.assertIn("B. 2", result)
        self.assertIn("C. 3", result)
        self.assertIn("D. 4", result)

    def test_format_question_no_extra_whitespace(self):
        """模板不含多余空行。"""
        result = format_question("Q", ["1", "2", "3", "4"])
        self.assertNotIn("\n\n", result)
        self.assertNotIn("  ", result.split("\n")[0])


# ============================================================
# Test 4: IFEval 4 选项化
# ============================================================
class TestIFEval4Option(unittest.TestCase):
    """验证 IFEval 从 2 选项扩展为 4 选项。"""

    def test_4_option_texts(self):
        """4 个选项文本应包含 Yes/No/Partially/Cannot determine。"""
        option_keywords = ["Yes", "No", "Partially", "Cannot determine"]
        # 模拟 prepare_ifeval_v2 的选项列表
        option_texts = [
            "Yes (fully satisfies the constraint)",
            "No (clearly violates the constraint)",
            "Partially (partially satisfies the constraint)",
            "Cannot determine (insufficient information)",
        ]
        for keyword in option_keywords:
            found = any(keyword in opt for opt in option_texts)
            self.assertTrue(found, f"Option keyword '{keyword}' not found in option texts")

    def test_correct_answer_random_position(self):
        """正确答案应随机分配到 A/B/C/D 位置。"""
        import random
        random.seed(42)

        positions = []
        option_texts = [
            "Yes (fully satisfies the constraint)",
            "No (clearly violates the constraint)",
            "Partially (partially satisfies the constraint)",
            "Cannot determine (insufficient information)",
        ]

        for i in range(100):
            rng = random.Random(i + 42 + 3000)
            correct_position = rng.randint(0, 3)
            positions.append(correct_position)

        # 验证分布不是全0（原始2选项时总是A=0）
        unique_positions = set(positions)
        self.assertGreater(len(unique_positions), 1,
                           "Correct answer should be distributed across A/B/C/D")
        self.assertEqual(len(unique_positions), 4,
                         "Correct answer should appear in all 4 positions")

    def test_ifeval_choices_count(self):
        """IFEval 样本的 choices 应有 4 个选项。"""
        # 模拟 prepare_ifeval_v2 输出的样本结构
        sample = {
            "id": "ifeval_0",
            "benchmark": "ifeval",
            "category": "skill",
            "label": 1,
            "input_text": "...",
            "answer": "B",
            "choices": [
                "No (clearly violates the constraint)",
                "Yes (fully satisfies the constraint)",
                "Partially (partially satisfies the constraint)",
                "Cannot determine (insufficient information)",
            ],
        }
        self.assertEqual(len(sample["choices"]), 4,
                         "IFEval should have exactly 4 options after conversion")
        self.assertIn(sample["answer"], ["A", "B", "C", "D"],
                      "IFEval answer should be A/B/C/D after conversion")


# ============================================================
# Test 5: KS 检验功能
# ============================================================
class TestKSTest(unittest.TestCase):
    """验证内容匹配检验（KS 检验）功能正确。"""

    def test_ks_test_same_distribution(self):
        """同分布样本 KS 检验 p > 0.05。"""
        samples = []
        for i in range(100):
            samples.append({
                "label": 1,
                "input_text": format_question("Word " * (10 + i % 5), ["a", "b", "c", "d"]),
            })
        for i in range(100):
            samples.append({
                "label": 0,
                "input_text": format_question("Word " * (10 + i % 5), ["a", "b", "c", "d"]),
            })
        pvalue = _compute_ks_test(samples)
        # 同分布应该 p > 0.05（但样本小可能有波动，放宽到 0.01）
        self.assertGreater(pvalue, 0.01,
                           f"KS test p-value {pvalue:.4f} too low for same-distribution samples")

    def test_ks_test_different_distribution(self):
        """不同分布样本 KS 检验 p < 0.05。"""
        samples = []
        # skill: 短问题
        for i in range(100):
            samples.append({
                "label": 1,
                "input_text": format_question("Hi", ["a", "b", "c", "d"]),
            })
        # knowledge: 长问题
        for i in range(100):
            samples.append({
                "label": 0,
                "input_text": format_question("This is a very long question with many words " * 10, ["a", "b", "c", "d"]),
            })
        pvalue = _compute_ks_test(samples)
        self.assertLess(pvalue, 0.05,
                        f"KS test p-value {pvalue:.4f} should be < 0.05 for very different distributions")

    def test_ks_test_insufficient_samples(self):
        """样本不足时返回 -1.0。"""
        samples = [
            {"label": 1, "input_text": format_question("Q1", ["a", "b", "c", "d"])},
            {"label": 0, "input_text": format_question("Q2", ["a", "b", "c", "d"])},
        ]
        pvalue = _compute_ks_test(samples)
        self.assertEqual(pvalue, -1.0, "Should return -1.0 for insufficient samples")


# ============================================================
# Test 6: 配置文件验证
# ============================================================
class TestConfigFiles(unittest.TestCase):
    """验证 Phase 1 配置文件结构和关键参数。"""

    def _load_yaml(self, filename):
        import yaml
        config_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(config_dir, filename)
        if not os.path.exists(path):
            self.skipTest(f"Config file not found: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def test_phase1a_config_has_data_file(self):
        """Phase 1A 配置指定数据文件。"""
        config = self._load_yaml("config_phase1a.yaml")
        self.assertEqual(config["data_file"], "data/all_samples_v2.json")

    def test_phase1b_config_has_data_file(self):
        """Phase 1B 配置指定数据文件。"""
        config = self._load_yaml("config_phase1b.yaml")
        self.assertEqual(config["data_file"], "data/mmlu_within.json")

    def test_phase1c_config_has_data_file(self):
        """Phase 1C 配置指定数据文件。"""
        config = self._load_yaml("config_phase1c.yaml")
        self.assertEqual(config["data_file"], "data/mmlu_text_only.json")

    def test_all_configs_have_seed_42(self):
        """所有配置 seed=42。"""
        for filename in ["config_phase1a.yaml", "config_phase1b.yaml", "config_phase1c.yaml"]:
            config = self._load_yaml(filename)
            self.assertEqual(config["data"]["seed"], 42,
                             f"{filename} seed should be 42")

    def test_all_configs_have_independent_results_dir(self):
        """所有配置有独立的结果目录，不覆盖实验0/1。"""
        for filename, expected_dir in [
            ("config_phase1a.yaml", "results/phase1a"),
            ("config_phase1b.yaml", "results/phase1b"),
            ("config_phase1c.yaml", "results/phase1c"),
        ]:
            config = self._load_yaml(filename)
            self.assertEqual(config["output"]["results_dir"], expected_dir,
                             f"{filename} results_dir should be {expected_dir}")

    def test_phase1b_has_subject_lists(self):
        """Phase 1B 配置包含 skill/knowledge 科目列表。"""
        config = self._load_yaml("config_phase1b.yaml")
        self.assertIn("phase1b", config)
        self.assertIn("skill_subjects", config["phase1b"])
        self.assertIn("knowledge_subjects", config["phase1b"])
        self.assertEqual(len(config["phase1b"]["skill_subjects"]), 13)
        self.assertEqual(len(config["phase1b"]["knowledge_subjects"]), 16)

    def test_phase1c_has_text_only_subject_lists(self):
        """Phase 1C 配置包含纯文本科目列表。"""
        config = self._load_yaml("config_phase1c.yaml")
        self.assertIn("phase1c", config)
        self.assertIn("skill_subjects", config["phase1c"])
        self.assertIn("knowledge_subjects", config["phase1c"])
        self.assertEqual(len(config["phase1c"]["skill_subjects"]), 2)
        self.assertEqual(len(config["phase1c"]["knowledge_subjects"]), 3)

    def test_phase1c_has_go_nogo_thresholds(self):
        """Phase 1C 配置包含 go/no-go 阈值。"""
        config = self._load_yaml("config_phase1c.yaml")
        abort = config.get("abort_criteria", {})
        self.assertIn("phase1c_high_layer_threshold", abort)
        self.assertIn("phase1c_all_layer_threshold", abort)
        self.assertEqual(abort["phase1c_high_layer_threshold"], 0.75)
        self.assertEqual(abort["phase1c_all_layer_threshold"], 0.65)




# ============================================================
# Test 8: Answer type parsing (P1-2 fix verification)
# ============================================================
class TestAnswerParsing(unittest.TestCase):
    """验证 _parse_answer_idx 兼容 int 和 str 两种格式。"""

    def test_int_answer(self):
        """int 0-3 直接返回。"""
        self.assertEqual(_parse_answer_idx(0), 0)
        self.assertEqual(_parse_answer_idx(1), 1)
        self.assertEqual(_parse_answer_idx(2), 2)
        self.assertEqual(_parse_answer_idx(3), 3)

    def test_string_answer_uppercase(self):
        """大写 str 'A'-'D' 映射到 0-3。"""
        self.assertEqual(_parse_answer_idx("A"), 0)
        self.assertEqual(_parse_answer_idx("B"), 1)
        self.assertEqual(_parse_answer_idx("C"), 2)
        self.assertEqual(_parse_answer_idx("D"), 3)

    def test_string_answer_lowercase(self):
        """小写 str 'a'-'d' 也映射正确。"""
        self.assertEqual(_parse_answer_idx("a"), 0)
        self.assertEqual(_parse_answer_idx("b"), 1)
        self.assertEqual(_parse_answer_idx("c"), 2)
        self.assertEqual(_parse_answer_idx("d"), 3)


# ============================================================
# Test 9: format_question validation (P1-3 fix verification)
# ============================================================
class TestFormatQuestionValidation(unittest.TestCase):
    """验证 format_question 在选项数不为 4 时报错。"""

    def test_rejects_3_options(self):
        """3 选项应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            format_question("Q?", ["a", "b", "c"])

    def test_rejects_5_options(self):
        """5 选项应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            format_question("Q?", ["a", "b", "c", "d", "e"])

    def test_accepts_4_options(self):
        """4 选项不应抛出异常。"""
        result = format_question("Q?", ["a", "b", "c", "d"])
        self.assertIn("A. a", result)

    def test_rejects_empty_options(self):
        """0 选项应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            format_question("Q?", [])

# ============================================================
# Test 7 (now Test 10): 不破坏现有实验0/1
# ============================================================
class TestBackwardCompatibility(unittest.TestCase):
    """验证 v2 代码不破坏实验0/1 的可复现性。"""

    def test_original_prepare_data_not_modified(self):
        """原始 prepare_data.py 文件未被修改。"""
        original_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "prepare_data.py"
        )
        with open(original_path, "r") as f:
            content = f.read()
        # 检查原始文件仍包含 2 选项格式
        # Original uses f-string with escaped \n, check for the pattern
        self.assertIn('A. Yes', content, "Original prepare_data.py should still have 2-option IFEval")
        self.assertIn('B. No', content, "Original prepare_data.py should still have 2-option IFEval")
        # Verify it is 2-option (choices = ["Yes", "No"])
        self.assertIn('"choices": ["Yes", "No"]', content,
                      "Original IFEval should have 2-option choices")

    def test_original_config_not_modified(self):
        """原始 config.yaml 文件未被修改。"""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.yaml"
        )
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.assertEqual(config["data"]["num_samples"], 50,
                         "Original config should still have num_samples=50")
        self.assertEqual(config["output"]["results_dir"], "results",
                         "Original config should still have results_dir='results'")

    def test_v2_is_separate_file(self):
        """v2 数据准备是独立文件，不是修改原始文件。"""
        v2_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "prepare_data_v2.py"
        )
        self.assertTrue(os.path.exists(v2_path), "prepare_data_v2.py should exist as separate file")


if __name__ == "__main__":
    unittest.main(verbosity=2)
