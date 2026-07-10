# @qa -> @dev 交接
级联追踪：cascade-skill-knowledge-separation
任务来源：@pm(走查通过) -> @qa(质量把关)
本阶段判定：No-Go
交接物路径：.claude/blackboard/qa-report-skill-knowledge-separation.md
下游输入要求：修复 P0 bug 后，QA 重新验证

## No-Go 原因

2 个 P0 bug 需修复：

### P0-1: selectivity=inf 时 JSON 序列化崩溃
- 位置: ablation.py:521-528, ablation.py:662, run_experiment.py:341
- 场景: compute_selectivity 返回 float("inf")，json.dump() 抛 ValueError
- 影响: 所有计算完成后结果无法保存
- 修复建议: 序列化前将 inf 转为有限值或用自定义 JSON encoder

### P0-2: 异常时 hook 未 remove 导致静默数据污染
- 位置: run_experiment.py:268-300, ablation.py:376-432
- 场景: evaluate_benchmark 抛异常时 hook_handle.remove() 不执行
- 影响: 后续所有推理被静默注入扰动
- 修复建议: 用 try/finally 包裹 hook 注册/移除

## 额外建议修复（不阻断但本轮应修）

P1-1: IFEval _check_ifeval_constraint 子串匹配 bug — "keyword:therefore" 中 "word" 匹配到错误分支
P1-2: evaluate_benchmark 单样本失败导致整个 run 崩溃 — 加 try/except

## 回退轮数

当前：1/2（最多 2 轮回退，第 3 轮 No-Go 上报用户）
