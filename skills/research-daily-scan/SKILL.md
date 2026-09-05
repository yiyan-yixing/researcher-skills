---
name: research-daily-scan
description: 每日前沿扫描，覆盖 AI/ML、量化、产品工具、跨领域4个维度
trigger: "扫描" / "今日前沿" / "daily scan"
timebox: 30min
output: daily-scan/YYYY-MM-DD.md
---

# 每日前沿扫描

## 执行步骤

### 1. 信息源扫描（15min）

用 WebSearch 并行扫描以下维度：

**A. AI/ML 前沿**（5min）
- arXiv 热门论文（cs.AI, cs.LG, cs.CL）
- Twitter/X AI 圈热门讨论
- 关键词：`LLM`, `agent`, `reasoning`, `multimodal`, `training`

**B. 量化研究**（5min）
- SSRN 新论文
- 量化社区新动向
- 关键词：`alpha`, `factor`, `pairs trading`, `market microstructure`

**C. 产品与工具**（3min）
- GitHub Trending（AI/ML 方向）
- Product Hunt 新发布
- 关键词：`AI tool`, `developer tool`, `automation`

**D. 跨领域**（2min）
- 有趣的跨界发现
- 生物/物理/经济学 → AI 的启发
- 关键词：`nature machine intelligence`, `science ai`

### 2. 筛选与评级（10min）

对每条发现，评级：
- 🔥 **Must-read**：直接相关 + 高影响力（≤3条）
- ⭐ **Worth-reading**：间接相关或有启发（≤5条）
- 📌 **Bookmark**：未来可能有用（≤5条）
- 🗑️ **Skip**：噪音

### 3. 产出报告（5min）

写入 `daily-scan/YYYY-MM-DD.md`，格式：

```markdown
---
date: YYYY-MM-DD
type: daily-scan
---

# 🔬 前沿扫描 YYYY-MM-DD

## 🔥 Must-read

### 1. [标题](URL)
- **来源**：arXiv / GitHub / ...
- **一句话**：核心发现
- **为什么重要**：对咱们的影响
- **行动项**：读全文 / 跑实验 / 写选题

## ⭐ Worth-reading
...

## 📌 Bookmark
...

## 📊 扫描统计
- 总扫描：N 条
- 🔥 Must-read：N 条
- ⭐ Worth-reading：N 条
- 📌 Bookmark：N 条
- 🗑️ Skip：N 条
```

## 与研究日志的联动

扫描中发现的 🔥 级别发现，应同步记录到 `research-log.md`：
- 如果发现支持/反驳已有假设 → 更新信念
- 如果发现新假设 → 记录假设+预期

## 与其他业务线的联动

- **biz/quant/**：量化相关发现 → 摘要发送到 quant blackboard
- **biz/content/**：有公众吸引力的发现 → 标记为选题候选
- **biz/model/**：训练/架构相关发现 → 摘要发送到 model blackboard
- **workshop/**：新工具发现 → 标记为待评估
