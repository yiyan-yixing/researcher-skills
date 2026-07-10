# LLM 技能/知识表示分离验证实验

验证假设：LLM 内部对"技能"（推理/格式控制）和"知识"（事实记忆）使用不同的表示方向。

## 实验概览

| 实验 | 目的 | 方法 | 放弃条件 |
|------|------|------|----------|
| 实验0 | 探针可行性 | 每层隐藏状态训练线性探针区分技能/知识 | 所有层准确率 < 60% |
| 实验1 | 选择性损伤验证 | 沿知识/技能方向消融，评估选择性影响 | 选择性比 < 2:1 |
| 实验2 | Scale Up | 在 Llama/Mistral-7B 上重复 0+1（预留） | — |

## 快速开始

### 安装依赖

```bash
pip install torch transformers datasets scikit-learn matplotlib pyyaml numpy
```

### 一条命令运行实验

```bash
# 方式1：Python
python run_experiment.py --config config.yaml

# 方式2：Shell
bash run.sh

# 指定 run ID
python run_experiment.py --config config.yaml --run-id my_experiment

# 只运行实验0
python run_experiment.py --config config.yaml --exp-only 0
```

### 一条命令可视化

```bash
python plot_results.py --run-id my_experiment
```

### 对比两个 run

```bash
python compare.py run_12345 run_67890
```

## 代码结构

```
.
├── config.yaml              # 所有超参和配置
├── run_experiment.py         # 主入口：一条命令跑实验0+1
├── extract_hidden.py        # 提取每层隐藏状态
├── probe.py                 # 线性探针训练+评估
├── ablation.py              # 方向性消融实验
├── plot_results.py          # 可视化：探针准确率曲线 + 消融热力图
├── compare.py               # 对比两个 run 的结果
├── run.sh                   # 一键运行脚本
├── data/
│   ├── prepare_data.py      # 下载+采样 mini-benchmark 数据
│   └── README.md             # 数据说明
├── results/                  # 运行结果自动存这里
│   └── .gitkeep
└── README.md
```

## 实验流程

### Step 1: 数据准备

下载 4 个 mini-benchmark（每个 100 题）：

| Benchmark | 类型 | 说明 |
|-----------|------|------|
| GSM8K-mini | 技能 | 数学推理 |
| IFEval-mini | 技能 | 指令遵循 |
| MMLU-mini | 知识 | 世界知识 |
| TriviaQA-mini | 知识 | 问答 |

### Step 2: 隐藏状态提取

对每个样本用 GPT-2 (124M) 推理，提取 12 层 × 768 维隐藏状态（最后一个 token）。

### Step 3: 实验0 - 探针训练

对每层训练 logistic regression 二分类器（skill vs knowledge），5-fold CV。

预期结果：中间层探针准确率最高（信息最集中），浅层和深层较低。

### Step 4: 实验1 - 消融验证

在探针表现最好的层：
- **知识方向消融**：从隐藏状态中移除知识方向分量 → 预期知识任务损伤 > 15%, 技能任务损伤 < 5%
- **技能方向消融**：从隐藏状态中移除技能方向分量 → 预期技能任务损伤 > 15%, 知识任务损伤 < 5%

## 配置说明

所有超参在 `config.yaml` 中，包括：

- `model.name`: 模型名称（默认 gpt2，可切换到 Llama-7B）
- `data.num_samples`: 每个 benchmark 采样数量
- `probe.cv_folds`: 交叉验证折数
- `ablation.strength`: 消融强度
- `abort_criteria`: 放弃条件阈值

## 结果文件

每次运行的结果保存在 `results/{run_id}/` 下：

```
results/{run_id}/
├── config.yaml                  # 运行时的配置副本
├── hidden_states.json           # 提取的隐藏状态
├── hidden_states_meta.json      # 元信息
├── probe_results.json           # 探针结果
├── probe_weights.json           # 探针权重（用于消融）
├── ablation_results.json        # 消融结果
├── probe_accuracy_curve.png     # 探针准确率曲线
├── ablation_heatmap.png         # 消融热力图
├── damage_bar_chart.png         # 损伤柱状图
├── run_summary.json             # 运行摘要
└── abort_*.json                 # 放弃记录（如果触发放弃条件）
```

## 复现性

- `config.yaml` 包含所有超参
- 每次运行的配置副本保存在 `results/{run_id}/config.yaml`
- 随机种子在 config 中指定（默认 42）
- `compare.py` 可对比不同配置的结果

## 技术细节

### 探针训练
- 对每层最后一个 token 的隐藏状态训练 LogisticRegression
- 特征标准化：StandardScaler
- 5-fold StratifiedKFold CV + 20% hold-out test
- 权重向量 w 指向 skill 类别（正类）

### 消融操作
- 探针权重在标准化空间训练：`w_scaled`
- 转换回原始空间：`w_original = w_scaled / scale`
- 消融公式：`h' = h - strength * (h . w / ||w||^2) * w`
- 通过 PyTorch 的 `register_forward_hook` 在推理时注入

### 答案评估
- GSM8K：提取最终数字比较
- IFEval：检查是否生成非空回复
- MMLU：提取选项字母（A/B/C/D）
- TriviaQA：子串模糊匹配
