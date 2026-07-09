# Researcher Skills / 研究员技能包

通用研究员 Agent 技能包——基于 "How to Be Good at Research" 提炼的 8 大核心原则，拆解为 **1 个角色 + 10 个可训练技能**。

## 核心理念

研究的真正技能是一堆小技能的栈，几乎所有技能都可以刻意训练。本技能包把这些小技能显式化、流程化、可执行化。

## 角色

| 角色 | 调用 | 职责 |
|------|------|------|
| 研究员 Researcher | `@researcher` | 问题选择、深度阅读、实验设计、品味训练、知识蒸馏、跨域探索 |

## 技能一览

| 技能 | 来源 | 触发场景 | 时间盒 |
|------|------|----------|--------|
| `research-problem-pick` | Pick your own problems | 选择研究方向/选题/评估问题价值 | 45min |
| `research-literature-deep-read` | Upgrade your inputs | 读论文/综述/技术报告 | 30min |
| `research-input-diversify` | Upgrade your inputs | 信息源升级/打破信息茧房 | 30min |
| `research-log` | Write everything down | 记录实验/假设/信念更新 | 10min |
| `research-experiment-shrink` | Tighten the loop | 实验设计/缩小到便宜/先验证 | 20min |
| `research-failure-autopsy` | Stare at the outputs | 分析失败案例/错误模式 | 30min |
| `research-taste-train` | Pick your own problems + Upgrade | 品味训练/预测结果/校准判断 | 15min |
| `research-public-distill` | Write everything down | 公开写作/知识蒸馏/解释复杂概念 | 60min |
| `research-baseline-tune` | Wander on purpose | 基线调优/消融实验/找核心组件 | 45min |
| `research-cross-field-explore` | Wander on purpose | 跨领域探索/寻找迁移机会 | 45min |

## 安装

```bash
# 安装到目标项目
bash install.sh /path/to/project

# 安装并跳过初始化（后续手动 init）
bash install.sh /path/to/project --skip-init

# 已安装后，重新运行初始化
bash /path/to/project/.claude/init.sh

# 非交互模式（环境变量预填）
RESEARCHER_NAME=张三 RESEARCH_FIELD=NLP bash init.sh
```

安装后，目标项目会多出 `.claude/` 目录：

```
your-project/.claude/
├── agents/researcher.md       # @researcher 角色
├── skills/                    # 10 个研究技能
├── memory/core/               # 自动加载的记忆
├── blackboard/                # 共享状态（研究日志/决策/问题）
├── evals/                     # 评估框架
└── CLAUDE.md                  # 记忆入口
```

### 初始化配置

运行 `init.sh` 会交互式收集你的研究信息：

| 问答项 | 示例 | 用途 |
|--------|------|------|
| 研究者姓名 | 张三 | 写入 project-context |
| 研究领域 | AI/ML | 确定搜索范围 |
| 核心研究方向 | LLM 可解释性 | 聚焦选题 |
| 研究风格 | 探索型/工程型/理论型 | 影响技能推荐 |
| 明确不做的事 | 追潮流,读二手总结 | 反模式红线 |
| 实验框架 | PyTorch + custom harness | 实验技能参考 |
| 可视化工具 | matplotlib + wandb | 产出格式参考 |

初始化会自动生成：
- `memory/core/project-context.md` — 研究画像 + OKR + 品味追踪表
- `memory/core/tech-stack.md` — 工具选型 + 选型原则
- `memory/core/architecture.md` — 研究循环图 + 八条铁律 + ADR 模板
- `blackboard/` 下 5 个文件 — 研究日志/决策/问题/sprint/质疑

### 合并安装到已有项目

如果你的项目已有 `.claude/` 目录（已装其他技能包），可以安全合并安装：

```bash
# Agent
cp researcher-skills/agents/researcher.md /your-project/.claude/agents/

# 技能（10 个）
cp -r researcher-skills/skills/research-* /your-project/.claude/skills/

# Blackboard（不覆盖已有）
cp researcher-skills/blackboard/research-log.md /your-project/.claude/blackboard/
mkdir -p /your-project/.claude/blackboard/literature-notes
mkdir -p /your-project/.claude/blackboard/distill-drafts

# 记忆模板（不覆盖已有 core 文件）
for f in researcher-skills/memory/core/*.template; do
  base=$(basename "$f" .template)
  target="/your-project/.claude/memory/core/$base"
  [ -f "$target" ] || cp "$f" "$target"
done
```

然后在 `CLAUDE.md` 的技能路由中添加：

```markdown
- 研究选题 / 方向评估 → @researcher 或 research-problem-pick
- 深度阅读论文 → @researcher 或 research-literature-deep-read
- 实验设计 / 快速验证 → @researcher 或 research-experiment-shrink
- 失败分析 / Bad case → @researcher 或 research-failure-autopsy
- 品味训练 / 预测校准 → @researcher 或 research-taste-train
- 知识蒸馏 / 公开写作 → @researcher 或 research-public-distill
- 跨领域探索 → @researcher 或 research-cross-field-explore
- 研究日志 / 信念更新 → research-log
```

## 使用方式

### 方式 A：通过 @researcher 角色调用

直接在 Claude Code 对话中说：

```
@researcher 帮我评估一下这个研究方向值不值得做
@researcher 我要读这篇论文 https://arxiv.org/abs/xxxx.xxxxx
@researcher 训练一下品味
```

@researcher 会自动判断该用哪个技能，并按技能定义的步骤执行。

### 方式 B：直接触发技能关键词

每个技能都有触发词，说出即可激活：

| 你说 | 触发技能 | 时间盒 |
|------|----------|--------|
| "选个问题" / "评估方向" / "这个问题值得做吗" | `research-problem-pick` | 45min |
| "读这篇论文" / "帮我理解这篇文章" | `research-literature-deep-read` | 30min |
| "升级信息源" / "打破茧房" | `research-input-diversify` | 30min |
| "记个日志" / "记录实验" / "更新信念" | `research-log` | 10min |
| "设计实验" / "怎么快速验证" / "最小实验" | `research-experiment-shrink` | 20min |
| "分析失败" / "看看 bad case" / "错误模式" | `research-failure-autopsy` | 30min |
| "训练品味" / "预测一下" / "校准判断" | `research-taste-train` | 15min |
| "写篇文章" / "蒸馏知识" / "公开分享" | `research-public-distill` | 60min |
| "基线调优" / "消融实验" | `research-baseline-tune` | 45min |
| "探索其他领域" / "跨领域迁移" | `research-cross-field-explore` | 45min |

## 典型使用场景

### 场景 1：开始一个新研究方向

```
你：@researcher 我想研究 LLM 的长上下文推理，帮我评估这个方向

@researcher 会执行：
1. 委托子 Agent 搜索该领域开放问题
2. 对每个候选做 Hamming 审查（5 维度打分）
3. Schulman 反推——从想要的结果倒推实验路径
4. 输出问题候选池 → blackboard/problem-candidates.md
5. 写入研究日志 → blackboard/research-log.md
```

### 场景 2：读一篇新论文

```
你：@researcher 读这篇论文 https://arxiv.org/abs/2407.xxxxx

@researcher 会执行：
1. 先让你写下预测（品味训练！）
2. 委托子 Agent 提取全文（含附录）
3. 反直觉阅读顺序：Limitations → Appendix → Method → Results → Intro
4. 预测校准——对比你的猜测和实际
5. 笔记写入 blackboard/literature-notes/
```

### 场景 3：设计实验

```
你：@researcher 帮我设计实验验证这个假设：增加中间层 attention 会提升推理链长度

@researcher 会执行：
1. 先写预期：假设成立/错误分别应该看到什么
2. Shannon 缩小：单 batch → 最小模型 → 最少步数
3. Karpathy 单 batch 检查：能过拟合吗？
4. 工具化检查：一条命令启动？一条命令可视化？
5. 输出实验设计 → blackboard/experiment-design.md
6. 如需实现 → 级联到 @dev
```

### 场景 4：分析失败案例

```
你：@researcher 分析这次实验为什么效果不好，日志在 outputs/failed/

@researcher 会执行：
1. 拉出 ≥ 20 个失败案例
2. 委托子 Agent 批量分类
3. 找到最大堆——这才是你要解决的问题
4. 从最大堆挑 1 个最奇怪的 transcript 深挖
5. 回到数据——大多数 ML bug 住在数据里
```

### 场景 5：日常品味训练

```
你：训练品味

@researcher 会执行：
1. 选训练目标（论文预测/实验预测/重要性预测）
2. 写下预测 + 置信度
3. 执行/阅读
4. 校准：偏差方向？系统性乐观还是悲观？
5. 更新 project-context.md 的品味追踪表
```

## 固定仪式

| 仪式 | 频率 | 触发方式 |
|------|------|----------|
| 📖 论文深读 | 每天 | `@researcher 读这篇论文` |
| 🎯 品味预测 | 每天 | `训练品味` |
| 📝 研究日志 | 每天 | `记个日志` |
| 🔍 失败尸检 | 每次实验后 | `分析失败` |
| 📡 信息源审计 | 每周 | `升级信息源` |
| 📊 品味命中率 | 每月 | `校准判断` |
| ✍️ 公共蒸馏 | 每月 | `写篇文章` |
| 🧭 跨域探索 | 每季度 | `探索其他领域` |

## 核心工作流：研究循环

```
选问题 ──→ 设计实验 ──→ 执行实验 ──→ 盯输出 ──→ 更新信念 ──→ 再选问题
  ↑                                                         │
  └────────── 跨域输入 + 品味训练 + 公共蒸馏 ─────────────────┘
```

贯穿始终：
- **品味训练** — 预测→校准→预测
- **公共蒸馏** — 写清楚 = 真贡献
- **基线调优** — 调到痛处，消融找核心
- **研究日志** — 对抗自我欺骗的第一道防线

## 八大原则 → 技能映射

| 原则 | 技能 | 核心动作 |
|------|------|----------|
| **Pick your own problems** | `problem-pick` + `taste-train` | 不吸收问题，Hamming 审查 + Schulman 反推 |
| **Upgrade your inputs** | `literature-deep-read` + `input-diversify` | 读原文不看摘要，跨领域借镜 |
| **Write everything down** | `research-log` + `public-distill` | 日志对抗自我欺骗，公开写作即凭证 |
| **Tighten the loop** | `experiment-shrink` | Shannon 缩小到平凡 + Karpathy 单 batch |
| **Stare at the outputs** | `failure-autopsy` | 拉失败案例，分类，攻最大堆 |
| **Wander on purpose** | `baseline-tune` + `cross-field-explore` | 调基线到痛处，跨子领域交学费 |
| **Find your people** | `public-distill` + `cross-field-explore` | 公开半成品，慷慨复利 |
| **The long game** | 整个体系的复利设计 | 日志+品味校准+记忆系统 |

## 注意事项

| 🚫 不要 | ✅ 要 |
|----------|--------|
| 跳过"预期"字段——写不出 = 还没想清楚 | 每条假设都要可证伪 |
| 读摘要代替原文——附录是埋尸体的地方 | 委托子 Agent 做搜索，@researcher 只做判断 |
| 追潮流——千人竞赛你起步晚+算力少 | 工程是头等研究活动 |
| 忽视失败——Darwin 规则：记忆删除不方便的证据 | 记录负面结果和正面结果一样详细 |

## 文章来源

技能设计基于 "How to Be Good at Research" 一文的 8 大核心原则（中译本见 `docs/`）：

1. **Pick your own problems** — 不吸收别人的问题，选择自己真正关心的结果并反推
2. **Upgrade your inputs** — 读旧文献，跨领域借镜，读原文不看摘要
3. **Write everything down** — 日志对抗自我欺骗，公开写作是最强凭证
4. **Tighten the loop** — 实验速度=发现自己错了的速度，工程是头等研究活动
5. **Stare at the outputs** — 拉百个失败，分类，攻最大堆
6. **Wander on purpose** — 有目的地漫游，先跑一次性版本，调基线到痛处
7. **Find your people** — 开放门政策，慷慨复利，半成品公之于众
8. **The long game** — 知识和生产像利息一样复利，尽早开始

## 引用

- Hamming, "You and Your Research" (1986) — 重要问题哲学
- Schulman, "ML Research" — 反推式选题
- Olah & Carter, "Research Debt" — 知识蒸馏即贡献
- Karpathy, "Recipe for Training Neural Networks" — 缩小实验 + 盯数据
- Graham, "Writing = Thinking" — 写作找漏洞
- Feynman — 首先不要骗自己
- Sutton, "The Bitter Lesson" (2019) — 1000 字预测整个领域
