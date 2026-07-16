# 项目记忆入口

> 此文件加载 core 记忆到每个 session，研究员 Agent 共享。

@.claude/memory/core/tech-stack.md
@.claude/memory/core/architecture.md
@.claude/memory/core/project-context.md

## 记忆使用规则

- **core/** — 每个 session 自动加载，只放最关键信息（≤200 行总计）
- **archival/** — Agent 需要时用 Read 读取，不放 core 避免膨胀
- **recall/** — 历史会话摘要，按需检索

## Agent 协作规则

- 所有 Agent 产出写入 `.claude/blackboard/`
- 架构决策写入 `.claude/memory/core/architecture.md` 并归档到 `archival/decisions/`
- 经验教训写入 `.claude/memory/archival/lessons/`
- 研究日志写入 `.claude/blackboard/research-log.md`

## 技能路由

当用户的请求匹配已有技能时，通过 Skill 工具调用：

- 选择研究方向 / 选题评估 → `research-problem-pick`
- 读论文 / 技术报告 → `research-literature-deep-read`
- 信息源升级 / 打破茧房 → `research-input-diversify`
- 记录实验 / 假设日志 → `research-log`
- 实验设计 / 缩小验证 → `research-experiment-shrink`
- 分析失败 / 错误模式 → `research-failure-autopsy`
- 品味训练 / 预测校准 → `research-taste-train`
- 公开写作 / 知识蒸馏 → `research-public-distill`
- 基线调优 / 消融实验 → `research-baseline-tune`
- 跨领域探索 → `research-cross-field-explore`

## Session 启动规则

- 每个新 session 的首次对话，@researcher 自动执行：
  1. 读取 `.claude/blackboard/research-log.md`，回顾最近 5 条日志
  2. 读取 `.claude/memory/core/project-context.md`，确认当前研究方向
  3. 输出：🔬 当前方向 {X} | 📝 最近日志 {N} 条 | 💡 说「选个问题」「读这篇论文」「训练品味」开始
