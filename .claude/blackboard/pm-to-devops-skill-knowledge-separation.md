# @pm -> @devops 交接
级联追踪：cascade-skill-knowledge-separation
任务来源：@qa Go 判定 -> @pm 回归走查通过
任务摘要：LLM 技能/知识表示分离实验 - Bug 修复后回归走查通过，可部署
本阶段产出：PM 走查通过，4 个 P0/P1 bug 修复验证完成，QA 48/48 用例通过
交接物路径：.claude/blackboard/walkthrough-pm-bugfix-skill-knowledge-separation.md
下游输入要求：部署实验代码到目标环境，确保可复现运行

## 部署要点
- 变更文件：experiments/skill-knowledge-separation/ablation.py, run_experiment.py
- 运行方式：python run_experiment.py --config config.yaml 或 bash run.sh
- 可视化：python plot_results.py --run-id <run_id>
- 无新增依赖，无需额外安装
