---
name: "research-arxiv-publish"
description: "从研究发现到arXiv论文的完整流水线——LaTeX生成→编译→打包→CDP自动提交。当用户说'发论文''投稿arXiv''写paper''arXiv提交'时触发。"
when_to_use: "研究成果成熟到可以公开时触发。频次：每月0-2篇，时间盒：120min"
allowed-tools:
  - Read
  - Write
  - Agent
  - Bash
disable-model-invocation: true
version: "1.0.0"
---

你是研究员，正在执行 arXiv 论文发布——把研究发现变成可公开的学术论文。

> arXiv 是预印本服务器，不是同行评审期刊。速度 > 完美。先发 v1，再改 v2。

## 准备

1. 确认要发布的课题编号（如 P001）
2. 读取 `.claude/blackboard/research-log.md` — 确认哪些假设有足够证据支撑
3. 读取 `findings/` 下对应的研究报告 — 作为论文素材

## 执行步骤

### Step 1: 论文可行性评估（5min）

不是所有研究都适合发论文。检查以下条件：

| 条件 | 要求 | 当前状态 |
|------|------|---------|
| **有可证伪的假设** | 至少1个H通过验证 | ✅/❌ |
| **有对照实验** | baseline vs proposed | ✅/❌ |
| **有量化结果** | 具体数字，非定性描述 | ✅/❌ |
| **有创新点** | 大厂做不深的方向 | ✅/❌ |
| **可复现** | 代码+数据可公开 | ✅/❌ |

**Go/No-Go**：5项中≥4项✅ → 继续。<4项 → 先补实验，不急着写。

### Step 2: 论文结构设计（10min）

arXiv 论文标准结构（参考 NeurIPS/ICML 格式）：

```latex
\title{...}
\abstract{...}           % 250词以内，说清楚：问题+方法+结果+意义

\section{Introduction}    % 1页：问题+动机+贡献列表
\section{Related Work}     % 0.5-1页：定位差异
\section{Method}           % 核心：2-3页
\section{Experiments}      % 2-3页：setup + results + ablation
\section{Analysis}         % 1页：失败案例+局限性（诚实=可信）
\section{Conclusion}       % 0.5页

\section{Limitations}      % 必须有——最诚实的段落
\section*{Acknowledgments}
```

**一人公司论文的特殊注意事项**：
- 单作者论文在 arXiv 很正常（Andrew Ng、Yann LeCun 也经常单作者）
- 诚实标注：哪些用了 AI 辅助，哪些是纯人工
- 领域特化 = 优势，不需要假装做通用研究

### Step 3: LaTeX 生成（30min）

```bash
# 创建论文目录
mkdir -p biz/research/papers/{PAPER_ID}
cd biz/research/papers/{PAPER_ID}
```

生成文件结构：

```
{PAPER_ID}/
├── main.tex           # 主文件
├── references.bib     # 参考文献
├── figures/           # 图表
│   ├── architecture.pdf
│   └── results.png
├── sections/          # 分章节（可选）
│   ├── intro.tex
│   ├── method.tex
│   ├── experiments.tex
│   └── analysis.tex
└── build.sh           # 编译脚本
```

**main.tex 模板**：

```latex
\documentclass{article}
\usepackage[final]{neurips_2025}  % 或 icml2025, acl2025

\title{From Black-Box to Domain Expert: \\Selective Capability Distillation for 7B Models}

\author{
  Lei Zhang \\
  YiYan YiXing \\
  \texttt{zhanglei@yiyan-yixing.com}
}

\begin{document}

\maketitle

\begin{abstract}
{ABSTRACT_FROM_RESEARCH_LOG}
\end{abstract}

\input{sections/intro}
\input{sections/related}
\input{sections/method}
\input{sections/experiments}
\input{sections/analysis}
\input{sections/conclusion}

\bibliography{references}
\bibliographystyle{plainnat}

\section*{Limitations}
{LIMITATIONS_FROM_RESEARCH_LOG}

\section*{AI Assistance Disclosure}
This paper was written with AI assistance (Claude, Anthropic) for drafting and editing.
All experimental designs, analyses, and conclusions are the author's own.

\end{document}
```

**关键规则**：
- 中文论文用 `\usepackage{ctex}` + XeLaTeX
- 英文论文用 pdfLaTeX
- 图表用 PDF 矢量格式
- 文件总大小 < 10MB

### Step 4: 编译验证（10min）

```bash
#!/bin/bash
# build.sh — 编译论文

# 英文论文
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# 中文论文
# xelatex main.tex
# bibtex main
# xelatex main.tex
# xelatex main.tex

echo "Build complete: main.pdf"
```

检查清单：
- [ ] PDF 生成无错误
- [ ] 所有图表正确显示
- [ ] 参考文献完整
- [ ] 页数合理（arXiv 建议 ≤ 10 页正文）
- [ ] 文件总大小 < 10MB

### Step 5: arXiv 提交包打包（5min）

```bash
# 创建提交包
cd biz/research/papers/{PAPER_ID}
zip -r submission.zip \
  main.tex \
  references.bib \
  neurips_2025.sty \
  figures/ \
  sections/

# 验证包内容
unzip -l submission.zip
```

### Step 6: CDP 自动提交 arXiv（30min）

> 需要：已登录的 arXiv 账号（Cookie 保持登录态）

```python
# 使用现有 browser_cdp_base.py 基础设施
# 工作目录：biz/content/publish/

class ArxivBrowserPublisher(BrowserCDPPublisher):
    """arXiv 论文自动提交发布器。"""

    PLATFORM = "arXiv"
    BASE_URL = "https://arxiv.org"

    def publish(self, paper_dir: str, subject: str = "cs.AI",
                license: str = "cc-by") -> dict:
        """提交论文到 arXiv。

        流程：
        1. 打开 arXiv 提交页面
        2. 选择学科分类
        3. 上传 ZIP 文件
        4. 等待编译
        5. 预览 PDF
        6. 填写元数据
        7. 确认提交

        Args:
            paper_dir: 论文目录路径。
            subject: arXiv 学科分类（如 cs.AI, cs.LG, q-fin.ST）。
            license: 许可证类型（cc-by, cc-by-sa, cc-by-nc-sa）。

        Returns:
            {"arxiv_id": "2507.XXXXX", "status": "submitted"}
        """
        ...

    def _select_subject(self, page, subject: str) -> bool:
        """选择学科分类。"""
        # arXiv 提交页面的学科选择器
        ...

    def _upload_submission(self, page, zip_path: str) -> bool:
        """上传论文 ZIP 文件。"""
        # file chooser 上传
        ...

    def _wait_for_compilation(self, page, timeout: int = 300) -> bool:
        """等待 arXiv 编译完成。"""
        # 编译通常需要 1-5 分钟
        ...

    def _preview_and_verify(self, page) -> bool:
        """预览编译后的 PDF。"""
        # 确认 PDF 正确
        ...

    def _fill_metadata(self, page, title: str, abstract: str,
                       authors: str, license: str) -> bool:
        """填写元数据。"""
        ...

    def _confirm_submission(self, page) -> str:
        """确认提交，返回 arXiv ID。"""
        ...
```

**arXiv 提交页面的关键 DOM 元素**：

| 步骤 | 选择器 | 操作 |
|------|--------|------|
| 新建提交 | `a[href="/submit/"]` | 点击 |
| 学科选择 | `select[name="subject"]` | 选择 |
| 文件上传 | `input[type="file"]` | set_input_files |
| 编译等待 | `.compile-status` | 等待完成 |
| PDF预览 | `iframe.pdf-preview` | 检查 |
| 元数据填写 | `input[name="title"]` 等 | 填写 |
| 确认提交 | `button[type="submit"]` | 点击 |

**重要提醒**：
- arXiv 提交后**无法修改PDF**，只能通过 "replace" 创建新版本
- 首次提交后通常有 **0-24小时审核期**
- 需要 arXiv 账号 + 可能需要 endorsement
- 建议先用 dry_run 模式走完流程

### Step 7: 发布后记录（5min）

1. 更新 `.claude/blackboard/research-log.md`，记录论文发布
2. 更新对应 P 编号的状态
3. 写入 `findings/` 下的论文元数据

## 产出

| 文件 | 路径 | 说明 |
|------|------|------|
| 论文源码 | `biz/research/papers/{PAPER_ID}/` | LaTeX + 图表 + 编译脚本 |
| 编译PDF | `biz/research/papers/{PAPER_ID}/main.pdf` | 最终提交版本 |
| 提交包 | `biz/research/papers/{PAPER_ID}/submission.zip` | arXiv 上传包 |
| 发布器代码 | `biz/content/publish/arxiv_browser_publisher.py` | CDP 自动提交 |

## arXiv 学科分类参考

| 领域 | 分类代码 | 说明 |
|------|---------|------|
| AI | cs.AI | 人工智能 |
| ML | cs.LG | 机器学习 |
| CL | cs.CL | 计算语言学 |
| 量化金融 | q-fin.ST | 统计交易 |
| 计算金融 | q-fin.CP | 计算方法 |
| 信息检索 | cs.IR | 检索与RAG |

## 时间盒

| 步骤 | 预计耗时 |
|------|---------|
| 可行性评估 | 5min |
| 结构设计 | 10min |
| LaTeX 生成 | 30min |
| 编译验证 | 10min |
| 打包 | 5min |
| CDP 提交 | 30min |
| 记录 | 5min |
| **合计** | **~95min** |

## 关键指标

- 论文从研究到提交 ≤ 2 天
- Go/No-Go 检查 100% 执行
- LaTeX 编译零错误
- Limitations 章节 100% 覆盖
- AI 辅助声明 100% 诚实标注

## 反模式

- ❌ **等完美再发** — arXiv 是预印本，v1 不完美正常，v2 改
- ❌ **跳过 Limitations** — 诚实 = 可信，隐藏缺陷 = 学术负债
- ❌ **不标注 AI 辅助** — 2025 起这是学术诚信要求
- ❌ **单次提交不验证** — 必须预览 arXiv 编译的 PDF
- ❌ **选错学科分类** — 影响可见度和引用
- ❌ **不保存源码** — replace 版本需要完整源码

## 与其他技能的联动

- `research-public-distill` → 博客版蒸馏 → arXiv 正式论文版
- `research-literature-deep-read` → Related Work 部分的素材
- `research-experiment-shrink` → Experiments 部分的实验设计
- `research-failure-autopsy` → Limitations 部分的素材
- `research-log` → 论文核心论点的证据来源
