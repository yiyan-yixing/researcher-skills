#!/bin/bash
set -e

# ============================================================
# researcher-skills init.sh
# 研究员技能包交互式初始化
# ============================================================

TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 支持环境变量预填（非交互模式）
# RESEARCHER_NAME=张三 RESEARCH_FIELD=NLP bash init.sh

echo "🔬 研究员技能包 — 初始化"
echo "=============================="
echo ""

# ── 交互式问答 ──

ask() {
  local prompt="$1"
  local default="$2"
  local var="$3"

  if [ -n "${!var}" ]; then
    echo "$prompt: ${!var} (环境变量)"
    return
  fi

  printf "%s [%s]: " "$prompt" "$default"
  read -r answer
  eval "$var=\"${answer:-$default}\""
}

ask "研究者姓名" "研究员" RESEARCHER_NAME
ask "研究领域" "AI/ML" RESEARCH_FIELD
ask "核心研究方向" "待定" CORE_DIRECTION
ask "研究风格 (探索型/工程型/理论型)" "探索型" RESEARCH_STYLE
ask "当前研究重点" "待定" CURRENT_FOCUS
ask "核心假设（如果有）" "待定" CORE_HYPOTHESES
ask "明确不做的事（逗号分隔）" "追潮流,读二手总结" NOT_DOING
ask "当前瓶颈" "待定" BOTTLENECK
ask "经典文献来源" "Semantic Scholar + 经典教材" CLASSIC_SOURCE
ask "跨领域来源" "相关领域的综述论文" CROSS_FIELD_SOURCE
ask "核心论文来源" "arXiv + 顶会 proceedings" CORE_PAPER_SOURCE
ask "社区来源" "Twitter/X + Reddit + 群聊" COMMUNITY_SOURCE
ask "实验框架" "PyTorch + custom harness" EXPERIMENT_FRAMEWORK
ask "日志工具" "Markdown + git" LOG_TOOL
ask "可视化工具" "matplotlib + wandb" VIZ_TOOL
ask "写作/发布工具" "Markdown + 博客" WRITING_TOOL
ask "文献管理" "Zotero" REF_TOOL
ask "编程语言" "Python" PROG_LANGUAGE

echo ""
echo "🧠 生成记忆文件..."

# ── 写入 project-context.md ──

NOT_DOING_1=$(echo "$NOT_DOING" | cut -d',' -f1 | xargs)
NOT_DOING_2=$(echo "$NOT_DOING" | cut -d',' -f2 | xargs)

cat > "$TARGET_DIR/memory/core/project-context.md" << CTX
# 项目上下文

> 此文件由 init.sh 自动生成，每 session 自动加载。

## 研究者信息

- **研究者**: $RESEARCHER_NAME
- **研究领域**: $RESEARCH_FIELD
- **核心方向**: $CORE_DIRECTION
- **研究风格**: $RESEARCH_STYLE

## 当前研究重点

$CURRENT_FOCUS

## 核心假设

$CORE_HYPOTHESES

## 季度 OKR

| O | KR | 目标值 | 当前值 |
|---|-----|--------|--------|
| O1: 产出可证伪假设 | KR1: 每周假设数 | ≥ 3 | 0 |
| O1: 产出可证伪假设 | KR2: 品味命中率 | ≥ 55% | 50% |
| O2: 深度阅读 | KR1: 论文数/周 | ≥ 5 | 0 |
| O2: 深度阅读 | KR2: 旧文献占比 | ≥ 20% | 0 |
| O3: 公共蒸馏 | KR1: 月产出 | ≥ 1 | 0 |

## 明确不做

- $NOT_DOING_1
- $NOT_DOING_2

## 当前瓶颈

- $BOTTLENECK

## 信息源配置

| 类型 | 来源 | 频率 | 状态 |
|------|------|------|------|
| 旧文献 | $CLASSIC_SOURCE | 每周 | 🟡 待验证 |
| 跨领域 | $CROSS_FIELD_SOURCE | 每周 | 🟡 待验证 |
| 核心论文 | $CORE_PAPER_SOURCE | 每天 | 🟡 待验证 |
| 社区 | $COMMUNITY_SOURCE | 每天 | 🟡 待验证 |

## 品味训练追踪

| 日期 | 预测 | 实际 | 校准信号 |
|------|------|------|----------|
| — | — | — | — |
CTX

# ── 写入 tech-stack.md ──

cat > "$TARGET_DIR/memory/core/tech-stack.md" << TECH
# 技术栈

> 此文件由 init.sh 自动生成，每 session 自动加载。

## 当前工具

| 类别 | 选择 | 理由 |
|------|------|------|
| 实验框架 | $EXPERIMENT_FRAMEWORK | 默认选择 |
| 日志工具 | $LOG_TOOL | 默认选择 |
| 可视化 | $VIZ_TOOL | 默认选择 |
| 写作/发布 | $WRITING_TOOL | 默认选择 |
| 文献管理 | $REF_TOOL | 默认选择 |
| 编程语言 | $PROG_LANGUAGE | 默认选择 |

## 选型原则

1. **一条命令启动实验** — 启动成本 > 0 会杀死探索欲
2. **一条命令可视化** — 看图比看表快 10 倍
3. **可复现** — 每个 experiment 从 config 完整复现
4. **比较秒级** — 对比两个 run 不应超过 30 秒

## 决策记录

| 日期 | 选择 | 备选 | 选择理由 |
|------|------|------|----------|
| $(date +%Y-%m-%d) | $EXPERIMENT_FRAMEWORK | — | 默认 |
TECH

# ── 写入 architecture.md ──

cat > "$TARGET_DIR/memory/core/architecture.md" << ARCH
# 研究架构

> 此文件由 init.sh 自动生成，每 session 自动加载。

## 研究循环

\`\`\`
┌─────────────────────────────────────────────────────────┐
│                     研究循环                              │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐           │
│  │ 选问题    │───→│ 设计实验  │───→│ 执行实验  │           │
│  │ (pick)   │    │ (shrink) │    │ (run)    │           │
│  └──────────┘    └──────────┘    └─────┬────┘           │
│       ↑                               │                  │
│       │  ┌──────────┐    ┌──────────┐ │                  │
│       └──│ 更新信念  │←──│ 盯输出   │←┘                  │
│          │ (log)    │    │ (stare)  │                    │
│          └──────────┘    └──────────┘                    │
│               ↑                                          │
│          ┌────┴────┐                                    │
│          │ 跨域输入 │                                    │
│          │ (wander) │                                    │
│          └─────────┘                                    │
│                                                          │
│  贯穿始终：                                               │
│  • 品味训练 (taste-train) — 预测→校准→预测               │
│  • 公共蒸馏 (public-distill) — 写清楚=真贡献             │
│  • 基线调优 (baseline-tune) — 调到痛处，消融找核心        │
└─────────────────────────────────────────────────────────┘
\`\`\`

## 研究日志格式

每条记录包含 5 个必填字段：

\`\`\`
## [{DATE}] {HYPOTHESIS_TITLE}

- **假设**: 如果 X，那么 Y
- **设置**: 实验配置一句话
- **预期**: 运行前写下的预测
- **结果**: 实际发生了什么
- **更新信念**: 基于结果，我现在相信什么
\`\`\`

## 八条研究铁律

1. **自己选问题** — 不吸收别人的问题，选择你真正想存在的结果并反推
2. **升级输入** — 读旧文献，跨领域借镜，读原文不看摘要
3. **写下一切** — 日志对抗自我欺骗，公开写作是最强凭证
4. **收紧循环** — 实验速度=发现自己错了的速度，工程是头等研究活动
5. **盯输出** — 拉百个失败，分类，攻最大堆
6. **有目的地漫游** — 先跑一次性版本，调基线到痛处，消融找核心
7. **找到你的人** — 开放门，慷慨复利，半成品公之于众
8. **长线复利** — 知识和生产像利息一样复利，尽早开始

## ADR 模板

\`\`\`markdown
# ADR-{ID}: {TITLE}

- **状态**: 提议 / 已接受 / 已废弃
- **日期**: {DATE}
- **背景**: 为什么要做这个决策
- **决策**: 具体选择了什么
- **备选**: 考虑过但没选的方案
- **后果**: 这个决策带来的利弊
\`\`\`
ARCH

# ── 重置 blackboard ──

cat > "$TARGET_DIR/blackboard/current-sprint.md" << SPRINT
# 当前 Sprint

## 目标

—
SPRINT

cat > "$TARGET_DIR/blackboard/open-questions.md" << OQ
# 待解问题

| # | 问题 | 优先级 | 状态 | 提出日期 |
|---|------|--------|------|----------|
| — | — | — | — | — |
OQ

cat > "$TARGET_DIR/blackboard/challenges.md" << CH
# 质疑记录

| C# | 质疑方 | 被质疑方 | 内容 | 严重度 | 结果 |
|----|--------|----------|------|--------|------|
| — | — | — | — | — | — |
CH

cat > "$TARGET_DIR/blackboard/decisions-log.md" << DL
# 决策日志

| # | 日期 | 决策 | 理由 | 影响 |
|---|------|------|------|------|
| — | — | — | — | — |
DL

cat > "$TARGET_DIR/blackboard/research-log.md" << RL
# 研究日志

> 格式：假设 → 设置 → 预期 → 结果 → 更新信念
> 这是研究员最重要的文件——对抗自我欺骗的第一道防线。

---

| # | 日期 | 假设 | 预期 | 结果 | 信念更新 |
|---|------|------|------|------|----------|
| — | — | — | — | — | — |
RL

# ── 清空 archival ──

cat > "$TARGET_DIR/memory/archival/decisions/decisions.md" << AD
# 决策记录

| ID | 日期 | 决策 | 理由 | 状态 |
|----|------|------|------|------|
| — | — | — | — | — |
AD

cat > "$TARGET_DIR/memory/archival/lessons/lessons.md" << AL
# 经验教训

| ID | 日期 | 教训 | 触发事件 | 如何避免 |
|----|------|------|----------|----------|
| — | — | — | — | — |
AL

echo ""
echo "✅ 研究员技能包初始化完成！"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🔬 研究者: $RESEARCHER_NAME"
echo "  🎯 领域: $RESEARCH_FIELD"
echo "  🧭 核心方向: $CORE_DIRECTION"
echo "  🎨 研究风格: $RESEARCH_STYLE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  📋 技能: 10 个"
echo "  🤖 Agent: 1 个 (@researcher)"
echo "  🧠 记忆文件: 3 core + 2 archival"
echo "  📝 Blackboard: 5 个"
echo ""
echo "  下一步："
echo "  1. 开始用 @researcher 选择你的第一个研究问题"
echo "  2. 说「读这篇论文」启动深度阅读"
echo "  3. 说「训练品味」开始预测训练"
echo ""
