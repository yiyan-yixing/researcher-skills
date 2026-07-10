# @dev -> @pm 交接
级联追踪：cascade-skill-knowledge-separation
任务来源：@researcher
任务摘要：实现 LLM 技能/知识表示分离验证实验代码

## 本阶段产出

### 实现摘要
完整实现了实验0（探针可行性）和实验1（选择性损伤验证）的代码，支持一条命令运行全部实验。

### 变更文件列表
- `experiments/skill-knowledge-separation/config.yaml` — 所有超参和配置
- `experiments/skill-knowledge-separation/run_experiment.py` — 主入口：一条命令跑实验0+1
- `experiments/skill-knowledge-separation/extract_hidden.py` — 提取每层隐藏状态
- `experiments/skill-knowledge-separation/probe.py` — 线性探针训练+评估
- `experiments/skill-knowledge-separation/ablation.py` — 方向性消融实验
- `experiments/skill-knowledge-separation/plot_results.py` — 可视化：探针准确率曲线 + 消融热力图
- `experiments/skill-knowledge-separation/compare.py` — 对比两个 run 的结果
- `experiments/skill-knowledge-separation/run.sh` — 一键运行脚本
- `experiments/skill-knowledge-separation/data/prepare_data.py` — 下载+采样 mini-benchmark 数据
- `experiments/skill-knowledge-separation/data/README.md` — 数据说明
- `experiments/skill-knowledge-separation/README.md` — 实验说明
- `experiments/skill-knowledge-separation/requirements.txt` — 依赖声明
- `experiments/skill-knowledge-separation/results/.gitkeep` — 结果目录占位

### 工具化要求满足情况
1. 一条命令启动实验 -> `python run_experiment.py --config config.yaml` 或 `bash run.sh`
2. 一条命令可视化 -> `python plot_results.py --run-id <id>`
3. 完全复现 -> config.yaml 包含所有超参，每次运行保存配置副本
4. 对比两个 run -> `python compare.py run1 run2`

### 技术要点
- GPT-2 12层 x 768维，提取最后一个token的每层隐藏状态
- LogisticRegression + 5-fold CV + StandardScaler
- 探针权重从标准化空间转换回原始空间进行消融
- 通过 register_forward_hook 注入消融操作
- 实验0放弃条件：所有层探针准确率 < 60%
- 实验1放弃条件：选择性损伤比 < 2:1

交接物路径：.claude/blackboard/dev-impl-skill-knowledge-separation.md
下游输入要求：代码 + 变更文件 + 自审报告 + 验收标准（PM 走查实现）
