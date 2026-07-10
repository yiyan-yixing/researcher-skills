# @dev -> @pm 交接
级联追踪：cascade-skill-knowledge-separation
任务来源：@researcher
任务摘要：实现 LLM 技能/知识表示分离验证实验代码

## 本阶段产出

### 实现摘要
完整实现了实验0（探针可行性）和实验1（选择性损伤验证）的代码，支持一条命令运行全部实验。
经过代码审查（subagent）发现并修复了 P0 级 bug（消融方向符号不变性）和多个 P1/P2 级问题。

### 变更文件列表
- `experiments/skill-knowledge-separation/config.yaml` — 所有超参和配置
- `experiments/skill-knowledge-separation/run_experiment.py` — 主入口：一条命令跑实验0+1
- `experiments/skill-knowledge-separation/extract_hidden.py` — 提取每层隐藏状态
- `experiments/skill-knowledge-separation/probe.py` — 线性探针训练+评估（Pipeline 防数据泄露）
- `experiments/skill-knowledge-separation/ablation.py` — 方向性消融实验（加性扰动法）
- `experiments/skill-knowledge-separation/plot_results.py` — 可视化：探针准确率曲线 + 消融热力图
- `experiments/skill-knowledge-separation/compare.py` — 对比两个 run 的结果
- `experiments/skill-knowledge-separation/utils.py` — 共享工具（模型加载/架构适配/种子设定）
- `experiments/skill-knowledge-separation/run.sh` — 一键运行脚本
- `experiments/skill-knowledge-separation/data/prepare_data.py` — 下载+采样 mini-benchmark 数据
- `experiments/skill-knowledge-separation/data/README.md` — 数据说明
- `experiments/skill-knowledge-separation/README.md` — 实验说明
- `experiments/skill-knowledge-separation/requirements.txt` — 依赖声明
- `experiments/skill-knowledge-separation/.gitignore` — Git 忽略规则
- `experiments/skill-knowledge-separation/results/.gitkeep` — 结果目录占位

### 工具化要求满足情况
1. 一条命令启动实验 -> `python run_experiment.py --config config.yaml` 或 `bash run.sh`
2. 一条命令可视化 -> `python plot_results.py --run-id <id>`
3. 完全复现 -> config.yaml 包含所有超参，每次运行保存配置副本
4. 对比两个 run -> `python compare.py run1 run2`

### 代码审查发现与修复
| 级别 | 问题 | 修复 |
|------|------|------|
| P0 | 消融方向符号不变性：proj(h,-w)=proj(h,w)，知识/技能消融结果完全相同 | 改用加性扰动法(PerturbationHook)，并添加投影消融对照 |
| P1 | StandardScaler 数据泄露：在 split 前对全数据 fit | 改用 Pipeline(Scaler+LR)，每 fold 内独立 fit |
| P1 | IFEval 评估只检查非空 | 实现基于 instruction_id 的约束检查 |
| P1 | --exp-only 1 导致 NameError | 初始化变量+检查 probe_weights 文件是否存在 |
| P1 | GPT-2 硬编码 model.transformer.h | 抽象为 get_layer_module()，兼容 GPT-2/Llama |
| P2 | 无 torch/numpy 随机种子 | 在 run_experiment.py 入口调用 set_seed() |
| P2 | 模型加载两次 | 复用模型对象，只在首次加载 |
| P2 | GSM8K 字符串比较 | 改为浮点数值比较 |
| P2 | compare.py 格式崩溃 | 修正默认值类型 |

### 技术要点
- GPT-2 12层 x 768维，提取最后一个token的每层隐藏状态
- LogisticRegression + Pipeline(StandardScaler+LR) + 5-fold CV
- 探针权重从标准化空间转换回原始空间进行消融
- 消融使用加性扰动法（PerturbationHook），非投影移除法
- 扰动幅度 alpha = 训练数据在 w 方向投影的标准差
- 通过 register_forward_hook 注入扰动/消融操作
- 实验0放弃条件：所有层探针准确率 < 60%
- 实验1放弃条件：选择性损伤比 < 2:1

交接物路径：.claude/blackboard/dev-impl-skill-knowledge-separation.md
下游输入要求：代码 + 变更文件 + 自审报告 + 验收标准（PM 走查实现）
