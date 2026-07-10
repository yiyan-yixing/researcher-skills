# @qa 回归测试报告（第 2 轮）
级联追踪：cascade-skill-knowledge-separation
回退轮数：2/2（修复后回归）
判定：Go

## 修复验证结果

### P0-1: selectivity=inf/nan 时 JSON 序列化崩溃 -- FIXED

| 用例 | 预期 | 实际 | 结果 |
|------|------|------|------|
| skill_damage=0, knowledge_damage>0 -> inf 截断为 9999.0 | 9999.0 | 9999.0 | PASS |
| 截断后 json.dump 不抛 ValueError | 无异常 | 无异常 | PASS |
| 正常有限值不受截断影响 | 3.0, 4.0 | 3.0, 4.0 | PASS |
| nan 截断为 0.0 | 0.0 | 0.0 | PASS |
| 负值可以 JSON 序列化（不被截断，但也安全） | json 安全 | json 安全 | PASS |

验证方法：构造 ablation_results 使 compute_selectivity 产出 inf，验证截断为 _SELECTIVITY_CAP (9999.0) 且 json.dumps 不崩溃。

### P0-2: 异常时 hook 未 remove 导致静默数据污染 -- FIXED

| 用例 | 预期 | 实际 | 结果 |
|------|------|------|------|
| ablation.py 知识方向扰动 hook 有 try/finally/remove | 存在 | 存在 | PASS |
| ablation.py 技能方向扰动 hook 有 try/finally/remove | 存在 | 存在 | PASS |
| ablation.py 投影消融 hook 有 try/finally/remove | 存在 | 存在 | PASS |
| ablation.py 所有 3 处 hook 后都有 try/finally/remove 结构 | 3/3 | 3/3 | PASS |
| run_experiment.py 3 处 hook_handle.remove() | 3 | 3 | PASS |
| run_experiment.py 所有 3 处 hook 后有 try/finally | 3/3 | 3/3 | PASS |
| 运行时验证：异常后 hook 确实被 remove | hook 不再触发 | hook 不再触发 | PASS |
| 对比验证：无 try/finally 时 hook 泄漏 | hook 仍触发 | hook 仍触发 | PASS |

验证方法：源码静态分析（检查 register_forward_hook 后必须有 try/finally/remove）+ 运行时验证（torch.nn.Linear 模拟 hook 泄漏场景）。

### P1-1: IFEval _check_ifeval_constraint 子串匹配 bug -- FIXED

| 用例 | 预期 | 实际 | 结果 |
|------|------|------|------|
| "keyword:therefore" -> keyword 分支，检测 therefore | True | True | PASS |
| "keyword:therefore" 生成中无 therefore | False | False | PASS |
| "keywords:existence" -> keyword 分支（旧 bug 误路由到 word） | True | True | PASS |
| "keywords:existence" 生成中无 existence | False | False | PASS |
| "keywords:" fallback 到 therefore | True | True | PASS |
| "length_constraint:number_words" -> length 分支 | True | True | PASS |
| "length_constraint:number_words" 0 个词 | False | False | PASS |
| "change_case:lowercase" -> case 分支，全小写 | True | True | PASS |
| "change_case:lowercase" 非全小写 | False | False | PASS |
| "capitalization:lowercase" -> case 分支 | True | True | PASS |
| "capitalization:lowercase" 非全小写 | False | False | PASS |
| "startend:end" -> startend 分支，以句号结尾 | True | True | PASS |
| "startend:end" 不以句号结尾 | False | False | PASS |
| "detectable_format:number_bullet" -> format 分支 | True | True | PASS |
| "detectable_format:number_bullet" 空内容 | False | False | PASS |
| 映射表中无 "word" 键 | 无 | 无 | PASS |
| "keyword:hello" 使用参数名 hello（非硬编码 therefore） | True | True | PASS |
| "keyword:hello" 不包含 hello | False | False | PASS |
| 空 instruction_id | True | True | PASS |
| 空生成 | False | False | PASS |

### P1-2: evaluate_benchmark 单样本失败导致整个 run 崩溃 -- FIXED

| 用例 | 预期 | 实际 | 结果 |
|------|------|------|------|
| check_answer(None generated) 返回 False | False | False | PASS |
| check_answer(None generated) 对所有 4 个 benchmark | False | False | PASS |
| 连续 5 次失败后提前终止 | total <= 5 | total = 5 | PASS |
| 混合成功/失败时评估继续 | total = 6 | total = 6 | PASS |
| 失败样本有 error 字段 | error 非空 | error 非空 | PASS |
| 失败样本 generated=None | None | None | PASS |
| 失败样本 correct=False | False | False | PASS |
| prediction dict 结构统一（成功和失败都有 error 键） | 键集相同 | 键集相同 | PASS |
| generated=None 序列化为 "" 兼容 JSON | "" | "" | PASS |

## 代码审查追加修复验证

| 用例 | 预期 | 实际 | 结果 |
|------|------|------|------|
| 无残留子串匹配 ("word" in, "length" in, "case" in) | 不存在 | 不存在 | PASS |
| 使用 math.isfinite 而非 min() 做 inf/nan 截断 | 存在 | 存在 | PASS |
| compute_selectivity 中不使用 min() | 无 min() | 无 min() | PASS |
| evaluate_benchmark 使用 logging | import + warning + error | 均存在 | PASS |
| _IFEVAL_TYPE_ROUTES 覆盖所有已知类型 | 6+ 个 | 10 个 | PASS |
| generated=None 序列化逻辑存在 | 存在 | 存在 | PASS |
| _SELECTIVITY_CAP 常量 | 9999.0 | 9999.0 | PASS |
| consecutive_failures 计数器存在 | 存在 | 存在 | PASS |

## 新发现的问题（不阻断发布）

### P3-new: startend 路由内仍使用子串匹配 constraint_param
位置: ablation.py:395
场景: `if "end" in constraint_param:` -- 当 constraint_param 为 "startend" 时，"end" 子串匹配为 True
影响: 极低 -- IFEval 数据集中 constraint_param 不太可能是 "startend"，且即使误路由，也只是错误地检查"以句号结尾"，而非崩溃
建议: 改用 constraint_param == "end" 精确匹配

## 测试执行摘要

- 总用例数: 48
- 通过: 48
- 失败: 0
- 错误: 0
- 执行时间: 0.01s

## 判定：Go

上一轮 No-Go 的 4 个 P0/P1 bug 全部修复并通过回归验证：
- P0-1 (inf JSON 崩溃): math.isfinite 截断逻辑正确，json.dump 安全
- P0-2 (hook 泄漏): 6 处 hook 全部使用 try/finally，运行时验证通过
- P1-1 (子串匹配): 映射表路由完全替代子串匹配，keyword/keywords 复数正确路由
- P1-2 (单样本崩溃): try/except 隔离 + 连续失败终止 + None 防御 + 统一 dict 结构

代码审查追加修复也全部验证通过。仅发现 1 个 P3 级问题（startend 内部子串匹配），不阻断发布。
