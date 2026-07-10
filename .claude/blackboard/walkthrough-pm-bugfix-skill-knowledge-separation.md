# PM 实现走查记录（Bug 修复回归）
级联追踪：cascade-skill-knowledge-separation
走查轮次：Bug 修复后回归走查
走查结果：通过

## 走查对象
- 实现记录：.claude/blackboard/impl-skill-knowledge-separation-bugfix.md
- QA 回归报告：.claude/blackboard/qa-report-skill-knowledge-separation-r2.md
- 变更文件：ablation.py, run_experiment.py

## 检查项 1：每个 bug 是否都有对应修复

| Bug ID | 修复方案 | 代码位置 | 判定 |
|--------|---------|---------|------|
| P0-1 | math.isfinite() + _SELECTIVITY_CAP | ablation.py:45, 648-651 | PASS |
| P0-2 | try/finally 包裹 6 处 hook | ablation.py:480-486,508-514,535-542 + run_experiment.py:271-275,288-292,305-309 | PASS |
| P1-1 | _IFEVAL_TYPE_ROUTES 映射表 | ablation.py:50-61, 374 | PASS |
| P1-2 | 每样本 try/except + consecutive_failures | ablation.py:179,188-254,245-253,278-279 | PASS |

## 检查项 2：无需求漂移

所有额外改动（isfinite 替代 min、keywords 复数映射、consecutive_failures 计数器、prediction dict 统一）均为已报告 bug 的合理延伸，无新功能需求。

## 检查项 3：无阻断级新风险

- P3-new: startend 路由内部 "end" in constraint_param 子串匹配（QA 已发现，不阻断）
- docstring: compute_selectivity 负值描述与代码不一致（代码正确，docstring 误导，极低风险）

## 检查项 4：QA Go 判定可信

48/48 用例通过，覆盖完整，运行时验证 + 静态分析双重确认，判定逻辑自洽。

## 附带建议（不阻断发布）

1. P3: startend 路由改为 constraint_param == "end" 精确匹配
2. docstring: 更新 compute_selectivity 中负 selectivity 的描述
