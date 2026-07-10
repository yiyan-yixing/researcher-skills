# @dev 实现记录：Bug 修复（QA No-Go 回退轮 1/2）

级联追踪：cascade-skill-knowledge-separation
任务来源：@qa No-Go 判定
完成时间：2026-07-10

## 修复摘要

修复 QA 报告的 4 个 P0/P1 bug 及代码审查发现的额外问题。

## 变更文件列表

1. `experiments/skill-knowledge-separation/ablation.py` -- 核心修复文件
2. `experiments/skill-knowledge-separation/run_experiment.py` -- hook 泄漏修复

## 修复详情

### P0-1: selectivity=inf 时 JSON 序列化崩溃
- compute_selectivity() 返回前用 `math.isfinite()` 检查非有限值
- inf 截断为 `_SELECTIVITY_CAP` (9999.0)
- nan 或负值截断为 0.0
- 常量 `_SELECTIVITY_CAP = 9999.0` 定义在模块级别

### P0-2: 异常时 hook 未 remove 导致静默数据污染
- ablation.py run_ablation_experiment() 3 处 hook 改为 try/finally
- run_experiment.py run_ablation() 3 处 hook 改为 try/finally
- 共 6 处修改，确保异常时 hook 必定 remove

### P1-1: IFEval _check_ifeval_constraint 子串匹配 bug
- 用 `_IFEVAL_TYPE_ROUTES` 类型映射表完全替代子串匹配
- 10 个已知 IFEval 类型映射到路由名
- 按 ":" 分离类型名和参数名
- keyword 分支使用 constraint_param 检查指定关键词
- 消除了所有 `elif "word" in` / `elif "length" in` 子串匹配

### P1-2: evaluate_benchmark 单样本失败导致整个 run 崩溃
- 每个样本独立 try/except，失败不中断评估
- 连续 5 次失败后提前终止（防 OOM 连锁）
- prediction dict 结构统一（error 键，成功为 None）
- generated=None 区分失败和空输出
- check_answer 对 None 返回 False
- 序列化时 None 替换为 "" 兼容 JSON

## 代码审查结果

- 审查者发现 P1-2（keywords 复数误路由）和 P1-3（split 过度截断），已在第二轮修复中用映射表方案一并解决
- P2-1（min(nan, cap) 无效）已改用 math.isfinite
- P2-3（连续 OOM 无退出）已添加 consecutive_failures 计数器
- P3 级问题记录但不阻断

## QA 回归测试结果

- 判定: Go
- 48/48 测试通过
- 测试脚本: experiments/skill-knowledge-separation/test_regression_fixes.py
- 新发现 1 个 P3 问题（startend 内部子串匹配），不阻断

## QA 报告路径

- `.claude/blackboard/qa-report-skill-knowledge-separation-r2.md`
