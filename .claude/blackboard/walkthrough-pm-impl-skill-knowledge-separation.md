# PM 实现走查记录
级联追踪：cascade-skill-knowledge-separation
走查轮次：1/2
走查结果：通过（附文档修复建议）

## 走查对象
实现记录：.claude/blackboard/dev-impl-skill-knowledge-separation.md
代码目录：experiments/skill-knowledge-separation/

## 验收标准覆盖

| # | 要求 | 位置 | 判定 |
|---|------|------|------|
| 1 | GPT-2 (124M) | config.yaml + utils.py | PASS |
| 2 | 4 mini-benchmark 各100题 | config.yaml + prepare_data.py | PASS |
| 3 | skill/knowledge 分类 | prepare_data.py | PASS |
| 4 | 每层隐藏状态提取 | extract_hidden.py | PASS |
| 5 | logistic regression 探针 | probe.py Pipeline | PASS |
| 6 | 准确率曲线 | plot_results.py | PASS |
| 7 | 实验0放弃 <60% | probe.py check_abort_criteria | PASS |
| 8 | 知识/技能方向消融 | ablation.py PerturbationHook | PASS |
| 9 | 消融后4个benchmark评估 | ablation.py evaluate_benchmark | PASS |
| 10 | 实验1放弃 <2:1 | ablation.py compute_selectivity | PASS |

## 工具化要求

| # | 要求 | 实现 | 判定 |
|---|------|------|------|
| 1 | 一条命令启动 | python run_experiment.py / bash run.sh | PASS |
| 2 | 一条命令可视化 | python plot_results.py | PASS |
| 3 | 完全复现 | config.yaml + config副本 + set_seed | PASS |
| 4 | 对比两个run | python compare.py | PASS |

## 需求漂移检查

- 消融方法从投影移除改为加性扰动：技术修正（P0 bug fix），非漂移
- ProjectionAblationHook 对照：超出设计但增加严谨性
- 无偷偷加需求

## 发现问题

| 级别 | 问题 | 建议 |
|------|------|------|
| P1 | README 消融公式与实际代码不一致 | 更新 README 中的消融公式为加性扰动法 |
| P2 | README IFEval 评估描述过于简化 | 更新为"基于 instruction_id 的约束类型检查" |

## 走查判定：通过

核心逻辑正确，文档问题不影响实验有效性。
