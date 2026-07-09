#!/bin/bash
set -e

# ============================================================
# researcher-skills install.sh
# 研究员技能包一键安装脚本
# ============================================================

TARGET_DIR="${1:-.}"
SKIP_INIT=false
FORCE_INIT=false

for arg in "$@"; do
  case "$arg" in
    --skip-init) SKIP_INIT=true ;;
    --init) FORCE_INIT=true ;;
  esac
done

# 定位源仓库
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/README.md" ] && [ -d "$SCRIPT_DIR/agents" ]; then
  CLONE_DIR="$SCRIPT_DIR"
  echo "✅ 使用本地源: $CLONE_DIR"
else
  REPO_URL="https://github.com/yiyan-yixing/researcher-skills.git"
  CLONE_DIR=$(mktemp -d)
  echo "📦 克隆远程仓库: $REPO_URL"
  git clone --depth 1 "$REPO_URL" "$CLONE_DIR"
  CLEANUP=true
fi

# 创建目标目录结构
mkdir -p "$TARGET_DIR/.claude/skills"
mkdir -p "$TARGET_DIR/.claude/agents"
mkdir -p "$TARGET_DIR/.claude/memory/core"
mkdir -p "$TARGET_DIR/.claude/memory/archival/decisions"
mkdir -p "$TARGET_DIR/.claude/memory/archival/lessons"
mkdir -p "$TARGET_DIR/.claude/memory/recall"
mkdir -p "$TARGET_DIR/.claude/blackboard"
mkdir -p "$TARGET_DIR/.claude/evals"

# Step 1: 安装技能
echo "📋 安装技能..."
cp -r "$CLONE_DIR/skills/"* "$TARGET_DIR/.claude/skills/" 2>/dev/null || true

# Step 2: 安装 Agent
echo "🤖 安装 Agent..."
cp "$CLONE_DIR/agents/"*.md "$TARGET_DIR/.claude/agents/" 2>/dev/null || true

# Step 3: 安装记忆系统
echo "🧠 安装记忆系统..."
for f in "$CLONE_DIR/memory/core/"*.template; do
  [ -f "$f" ] || continue
  base=$(basename "$f" .template)
  cp "$f" "$TARGET_DIR/.claude/memory/core/$base"
done
[ -f "$CLONE_DIR/memory/core/architecture.md.template" ] && cp "$CLONE_DIR/memory/core/architecture.md.template" "$TARGET_DIR/.claude/memory/core/architecture.md"
[ -f "$CLONE_DIR/memory/core/tech-stack.md.template" ] && cp "$CLONE_DIR/memory/core/tech-stack.md.template" "$TARGET_DIR/.claude/memory/core/tech-stack.md"
[ -f "$CLONE_DIR/memory/core/project-context.md.template" ] && cp "$CLONE_DIR/memory/core/project-context.md.template" "$TARGET_DIR/.claude/memory/core/project-context.md"
cp -r "$CLONE_DIR/memory/archival/"* "$TARGET_DIR/.claude/memory/archival/" 2>/dev/null || true
touch "$TARGET_DIR/.claude/memory/recall/.gitkeep"

# Step 4: 安装 Blackboard
echo "📝 安装 Blackboard..."
cp "$CLONE_DIR/blackboard/"*.md "$TARGET_DIR/.claude/blackboard/" 2>/dev/null || true

# Step 5: 安装评估框架
echo "📊 安装评估框架..."
cp -r "$CLONE_DIR/evals/"* "$TARGET_DIR/.claude/evals/" 2>/dev/null || true

# Step 6: 安装 CLAUDE.md
echo "📄 安装 CLAUDE.md..."
cp "$CLONE_DIR/CLAUDE.md.template" "$TARGET_DIR/.claude/CLAUDE.md"

# Step 7: 安装 init.sh
echo "🔧 安装初始化脚本..."
cp "$CLONE_DIR/init.sh" "$TARGET_DIR/.claude/init.sh"
chmod +x "$TARGET_DIR/.claude/init.sh"

# 清理临时克隆
if [ "${CLEANUP:-false}" = true ]; then
  rm -rf "$CLONE_DIR"
fi

echo ""
echo "✅ 研究员技能包安装完成！"
echo ""

# 运行初始化
if [ "$FORCE_INIT" = true ] || [ "$SKIP_INIT" = false ]; then
  echo "🚀 运行初始化..."
  bash "$TARGET_DIR/.claude/init.sh"
else
  echo "⏩ 跳过初始化。稍后运行: bash $TARGET_DIR/.claude/init.sh"
fi
