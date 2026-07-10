# @devops 部署记录 — skill-knowledge-separation bugfix
级联追踪：cascade-skill-knowledge-separation
日期：2026-07-10
判定：**Go**

## 发布检查清单

| # | 检查项 | 结果 | 说明 |
|---|--------|------|------|
| 1 | Python AST 语法解析 | PASS | ablation.py + run_experiment.py 均解析通过 |
| 2 | 无遗留调试代码 | PASS | 无 breakpoint/pdb/ipdb；print 为 CLI 进度输出（非调试残留） |
| 3 | 回归测试 48/48 | PASS | test_regression_fixes.py 全部通过，0.011s |
| 4 | Git diff 变更范围 | NOTE | bugfix 仅涉及 ablation.py + run_experiment.py；另有 3 文件(compare/extract_hidden/probe)含实现阶段格式化改动未提交，不属于本次 bugfix |
| 5 | 回滚方案 | READY | `git checkout HEAD -- experiments/skill-knowledge-separation/ablation.py experiments/skill-knowledge-separation/run_experiment.py` 可恢复到 bugfix 前状态 |

## 修复内容确认

| Bug ID | 修复方案 | 验证状态 |
|--------|---------|---------|
| P0-1 | math.isfinite + _SELECTIVITY_CAP 截断 inf/nan | 5 用例 PASS |
| P0-2 | 6 处 hook 全部 try/finally 保护 | 8 用例 PASS |
| P1-1 | _IFEVAL_TYPE_ROUTES 映射表替代子串匹配 | 17 用例 PASS |
| P1-2 | 单样本 try/except + consecutive_failures + None 防御 | 8 用例 PASS |
| 代码审查 | isfinite/min/logging/route/None 序列化/consecutive_failures | 10 用例 PASS |

## 已知不阻断项

- P3-new: startend 路由内部 "end" in constraint_param 子串匹配（QA 已发现，影响极低）
- docstring: compute_selectivity 负值描述与代码不一致（代码正确）

## 回滚方案

```bash
# 恢复 bugfix 前状态（回退两个变更文件）
cd /Users/zhanglei/yiyan-yixing/researcher-skills
git checkout HEAD -- experiments/skill-knowledge-separation/ablation.py experiments/skill-knowledge-separation/run_experiment.py
```

## 部署状态

- 变更文件已就绪，QA Go + PM 走查通过
- 无 CI/CD 流水线（本地实验项目，无自动部署目标）
- 变更可随时提交到 git

## 交接

- 变更文件：ablation.py, run_experiment.py
- 测试文件：test_regression_fixes.py
- QA 报告：.claude/blackboard/qa-report-skill-knowledge-separation-r2.md
- PM 走查：.claude/blackboard/walkthrough-pm-bugfix-skill-knowledge-separation.md
