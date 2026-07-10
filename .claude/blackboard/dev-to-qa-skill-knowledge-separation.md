# @dev -> @qa 交接：Bug 修复回归测试

级联追踪：cascade-skill-knowledge-separation
任务来源：@qa No-Go 判定（回退轮数 1/2）
任务摘要：修复 QA 报告的 P0-1, P0-2, P1-1, P1-2 四个 bug，并修复代码审查发现的遗留问题

## 修复清单

### P0-1: selectivity=inf 时 JSON 序列化崩溃
- 文件: ablation.py compute_selectivity()
- 修复: 使用 `math.isfinite()` 检查非有限值（inf/nan），inf 截断为 `_SELECTIVITY_CAP` (9999.0)，nan 或负值截断为 0.0
- 原方案用 `min()` 不够健壮（`min(nan, cap)` 返回 nan），改用 `math.isfinite()` 彻底处理
- 覆盖: ablation.py 和 run_experiment.py 共享同一个 compute_selectivity()，两处 json.dump 均受益

### P0-2: 异常时 hook 未 remove 导致静默数据污染
- 文件: ablation.py run_ablation_experiment() + run_experiment.py run_ablation()
- 修复: 所有 6 处 hook 注册改为 try/finally 模式
  ```python
  hook_handle = layer_module.register_forward_hook(hook)
  try:
      metrics, preds = evaluate_benchmark(...)
  finally:
      hook_handle.remove()
  ```

### P1-1: IFEval _check_ifeval_constraint 子串匹配 bug
- 文件: ablation.py _check_ifeval_constraint()
- 修复: 用类型映射表 `_IFEVAL_TYPE_ROUTES` 替代所有子串匹配逻辑
  - 按 ":" 分离类型名和参数名（如 "keyword:therefore" -> type="keyword", param="therefore"）
  - 映射表统一处理单复数（"keyword"/"keywords" -> "keyword"）和复合类型（"change_case"/"capitalization" -> "case"）
  - 完全消除了子串匹配（不再有 `elif "word" in constraint_id`）
  - keyword 分支还使用了参数名（constraint_param）来检查指定关键词，而非硬编码 "therefore"

### P1-2: evaluate_benchmark 单样本失败导致整个 run 崩溃
- 文件: ablation.py evaluate_benchmark()
- 修复: 对每个样本加 try/except
  - 失败样本: 标记 incorrect，记录 error 字段，计入 total
  - 新增连续失败计数器，超过 5 次连续失败时提前终止（防止 OOM 连锁重试）
  - 新增 `import logging`，用 logger.warning/error 输出失败信息
  - prediction dict 结构统一：成功和失败路径都包含 `"error"` 键（成功为 None）
  - 失败样本的 `generated` 字段使用 None（语义上与空输出区分）
  - check_answer 增加 `if generated is None: return False` 防御
  - 序列化时 None 替换为 "" 兼容 JSON

## 代码审查修复

审查发现的额外问题一并修复：
- P2-1: `min(nan, cap)` 对 nan 无效 -> 改用 `math.isfinite()` 彻底处理
- P1-2(review): "keywords:..." 复数形式误路由 -> 映射表统一 keyword/keywords
- P1-3(review): `split("_")[0]` 过度截断 -> 映射表保留完整类型名
- P2-2(review): 残留子串匹配 `elif "word" in` -> 完全消除
- P2-3(review): 连续 OOM 无提前退出 -> 新增 consecutive_failures 计数器
- P3-3(review): prediction dict 结构不一致 -> 统一 error 键
- P3-4(review): generated="" 无法区分失败和空输出 -> 改用 None

## 变更文件列表

1. `/Users/zhanglei/yiyan-yixing/researcher-skills/experiments/skill-knowledge-separation/ablation.py`
2. `/Users/zhanglei/yiyan-yixing/researcher-skills/experiments/skill-knowledge-separation/run_experiment.py`

## 回归测试请求

请针对以下场景验证：
1. knowledge_selectivity=inf 时 json.dump 不崩溃（P0-1）
2. skill_selectivity=float('nan') 时 json.dump 不崩溃（P0-1 额外修复）
3. evaluate_benchmark 抛异常后 hook 被正确 remove（P0-2）
4. instruction_id="keyword:therefore" 走 keyword 分支（P1-1）
5. instruction_id="keywords:existence" 走 keyword 分支而非 word 分支（P1-1 审查追加）
6. instruction_id="change_case:lowercase" 走 case 分支（P1-1 审查追加）
7. instruction_id="capitalization:lowercase" 走 case 分支（P1-1 审查追加）
8. instruction_id="startend:end" 走 startend 分支（P1-1 审查追加）
9. 单个样本 model.generate 失败不崩溃整个评估（P1-2）
10. 连续 5+ 样本失败后评估提前终止（P1-2 审查追加）
