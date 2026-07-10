# @qa 质量把关报告
级联追踪：cascade-skill-knowledge-separation
任务来源：@pm(走查通过) -> @qa
判定：No-Go（P0 bug 需修复）
回退轮数：1/2

## 验收标准逐项测试

| # | 要求 | 判定 | 说明 |
|---|------|------|------|
| 1 | GPT-2 (124M) 加载 | PASS | utils.py load_model_and_tokenizer 正确；get_layer_module 兼容 GPT-2/Llama |
| 2 | 4 mini-benchmark 各100题 | PASS | prepare_data.py 采样逻辑正确；min(num_samples, len(indices)) 防超界 |
| 3 | skill/knowledge 分类 | PASS | GSM8K/IFEval=skill(label=1), MMLU/TriviaQA=knowledge(label=0) |
| 4 | 每层隐藏状态提取 | PASS | extract_hidden.py 提取 layer_idx+1（跳过embedding），最后token |
| 5 | logistic regression 探针 | PASS | Pipeline(StandardScaler+LR) 每fold独立fit，无数据泄露 |
| 6 | 准确率曲线 | PASS | plot_results.py 正确绘制 CV mean+std, test accuracy, 最佳层, 放弃阈值 |
| 7 | 实验0放弃 <60% | PASS | check_abort_criteria 逻辑正确，边界 case 0.60 通过 |
| 8 | 知识/技能方向消融 | PASS | PerturbationHook 方向语义正确：+w推skill/远knowledge, -w推knowledge/远skill |
| 9 | 消融后4个benchmark评估 | PASS | evaluate_benchmark + check_answer 四种benchmark检查逻辑完整 |
| 10 | 实验1放弃 <2:1 | PASS | compute_selectivity 比值计算正确，1e-6防除零 |

## 工具化要求

| # | 要求 | 判定 | 说明 |
|---|------|------|------|
| 1 | 一条命令启动 | PASS | python run_experiment.py / bash run.sh |
| 2 | 一条命令可视化 | PASS | python plot_results.py --run-id |
| 3 | 完全复现 | PASS | config.yaml + config副本 + set_seed（GPU 缺 cudnn 标志，见 P1-6） |
| 4 | 对比两个run | PASS | python compare.py run1 run2 |

## 核心逻辑验证（QA 自行数学验证）

| 验证项 | 结果 |
|--------|------|
| 权重空间转换 w_original = w_scaled / scale | 数学正确。推导：X_scaled @ w_scaled + b = 0 => X @ (w_scaled/scale) + (b - mean@(w_scaled/scale)) = 0 |
| PerturbationHook 方向语义 | 正确。+w 推向 skill 区域（远离 knowledge），-w 推向 knowledge 区域（远离 skill） |
| ProjectionAblationHook 公式 | 正确。h' = h - strength * ((h.w)/(w.w)) * w，strength=1 时完全移除投影 |
| 选择性损伤比计算 | 正确。knowledge_selectivity = knowledge_damage / skill_damage |
| 放弃条件边界 | 正确。0.60 >= 0.60 通过（不放弃），2.0 < 2.0 为 False（不放弃） |

## 发现的 Bug 列表

### P0（阻断发布）

**P0-1: selectivity=inf 时 JSON 序列化崩溃导致数据丢失**
位置: ablation.py:521-528, ablation.py:662, run_experiment.py:341
场景: 当 cross-category damage 趋近零但 target-category damage > 0 时，compute_selectivity 返回 float("inf")。json.dump() 不支持 inf，抛出 ValueError。
影响: 所有昂贵的计算完成后，结果无法保存，数据全部丢失。
这是最讽刺的 bug：实验越成功（选择性越高），越可能崩溃。
修复: 在序列化前将 inf 转为有限值（如 9999.0）或使用自定义 JSON encoder。

**P0-2: 异常时 hook 未 remove 导致后续评估静默污染**
位置: run_experiment.py:268-270, 282-284, 298-300; ablation.py:376-380, 401-406, 428-432
场景: evaluate_benchmark() 在 hook 激活期间抛出异常（OOM、CUDA 错误等），hook_handle.remove() 不执行。
影响: 后续所有模型推理都被静默注入扰动，结果完全无效且无任何指示。
修复: 用 try/finally 包裹 hook 注册/移除。

### P1（高优先级）

**P1-1: IFEval 约束检查子串匹配 bug — "word" 匹配 "keyword"**
位置: ablation.py:292 _check_ifeval_constraint
场景: instruction_id="keyword:therefore" 包含子串 "word"，匹配到 length/word/50 分支（0 < word_count < 200），而非 keyword 分支（检查 "therefore" 是否在生成中）。
影响: keyword 类约束被误用为长度约束（极宽松），IFEval 准确率被人为抬高。
注意: 此 bug 同时影响基线和消融评估，对选择性比值影响较小，但绝对准确率不可信。
修复: 使用 instruction_id 前缀精确匹配（如 "keyword:"）代替子串匹配。

**P1-2: 单样本 generate 失败导致整个评估崩溃**
位置: ablation.py:166-172 evaluate_benchmark
场景: model.generate() 对单个样本失败（空输入、CUDA 错误等），无 try/except。
影响: 一个样本失败 = 整个 run 失败，已完成的所有评估结果丢失。
修复: 每个样本 try/except，失败记录为 correct=False。

**P1-3: run_experiment.py 与 ablation.py 消融逻辑重复实现**
位置: run_experiment.py:246-312 vs ablation.py:319-448
场景: 两个文件各自实现消融流程，保存格式不一致（run_experiment.py 省略 "generated" 字段）。
影响: 一处 bug 修复不会自动同步到另一处，两份代码将悄然分化。
修复: run_experiment.py 调用 ablation.run_ablation_experiment()，删除内联重复。

**P1-4: torch_dtype 不匹配导致 hook 中隐藏状态类型错误**
位置: ablation.py:76-82 PerturbationHook, run_experiment.py:246
场景: config.yaml 设置 float16 时，hook 用 float32 构建扰动，加到 float16 隐藏状态上。
影响: 类型不匹配可能抛异常或精度丢失。
修复: perturbation.to(dtype=hidden_states.dtype)

**P1-5: set_seed 缺少 CUDA 确定性标志**
位置: utils.py:16-23
场景: GPU 运行时，cuDNN 非确定性算法导致同 seed 不同结果。
影响: GPU 上不可复现。当前默认 CPU 配置不受影响。
修复: 添加 torch.backends.cudnn.deterministic = True 和 torch.backends.cudnn.benchmark = False

**P1-6: compute_perturbation_alpha 使用全部数据（含 test）**
位置: ablation.py:107-135
场景: alpha 从 ALL samples 的投影标准差计算，但 probe 只在 train 上训练。
影响: 轻微数据泄露（alpha 只是缩放标量，非方向），对实验结论影响小。
修复: 仅从 train split 计算 alpha。

**P1-7: TriviaQA 单字答案子串误匹配**
位置: ablation.py:264
场景: 答案 "yes"、"sun" 等单字作为子串匹配到不相关文本（"yesterday" 包含 "yes"）。
影响: TriviaQA 准确率被人为抬高。
修复: 对短答案使用词边界匹配 \b{answer}\b。

### P2（中等优先级）

| ID | 问题 | 位置 |
|----|------|------|
| P2-1 | assert 用于运行时验证（python -O 时剥离） | ablation.py:67,95 |
| P2-2 | 隐藏状态文件无格式验证 | probe.py:26-29 |
| P2-3 | find_latest_run 只匹配 run_* 前缀 | plot_results.py:29 |
| P2-4 | scaler_scale 零方差特征产生任意方向分量 | ablation.py:114,339 |
| P2-5 | skip_if_exists 不验证模型一致性 | run_experiment.py:412 |
| P2-6 | load_config 函数在多个文件重复定义 | probe.py, plot_results.py, prepare_data.py |
| P2-7 | 负 selectivity（消融改善性能）未显式处理 | ablation.py:486-529 |
| P2-8 | requirements.txt 未固定版本上限 | requirements.txt |
| P2-9 | MMLU 假设恰好 4 个选项且 answer_idx < 4 | prepare_data.py:115-117 |

### P3（低优先级）

| ID | 问题 | 位置 |
|----|------|------|
| P3-1 | 无配置验证 | 全局 |
| P3-2 | README 消融公式与代码不一致（PM 已知） | README.md:145 |
| P3-3 | README IFEval 描述过于简化（PM 已知） | README.md:150 |
| P3-4 | 无日志框架（全部 print） | 全局 |
| P3-5 | 部分结果失败时不清理 | run_experiment.py |
| P3-6 | run.sh 中绘图失败导致整体退出码非零 | run.sh |

## 判定理由

核心实验逻辑经过数学验证是正确的：
- 权重空间转换公式正确
- PerturbationHook 方向语义正确
- Probe Pipeline 防数据泄露正确
- 放弃条件边界处理正确
- 模型架构适配正确

但 P0-1（selectivity=inf 时 JSON 崩溃）是真实的数据丢失风险：
- 此 bug 在实验最成功时（选择性最高）最可能触发
- 修复只需 2-3 行代码
- 不修复则无法信任实验结果能完整保存

P0-2（hook 泄漏）是静默数据污染风险：
- 异常场景下后续所有评估都被污染且无指示
- 修复只需加 try/finally

因此判定 No-Go，需修复 P0 bug 后重新验证。

## 测试用例产出

子任务设计 75 个测试用例（正常33 + 边界24 + 异常18），覆盖全部 10 项验收标准和 4 项工具化要求。详见子任务产出。

## 建议修复优先级

1. P0-1: JSON inf 崩溃 — 2 行修复，立即
2. P0-2: hook 泄漏 — 加 try/finally，立即
3. P1-1: IFEval 子串匹配 — 改用精确匹配，本轮
4. P1-2: 单样本错误处理 — 加 try/except，本轮
5. 其余 P1/P2 可排期
