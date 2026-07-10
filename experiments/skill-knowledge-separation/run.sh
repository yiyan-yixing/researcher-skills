#!/bin/bash
# =============================================================================
# 一键运行实验
# 用法: bash run.sh [RUN_ID]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${CONFIG:-config.yaml}"
RUN_ID="${1:-run_$(date +%s)}"

echo "============================================================"
echo "LLM Skill/Knowledge Separation Experiment"
echo "Config: $CONFIG"
echo "Run ID: $RUN_ID"
echo "============================================================"

python run_experiment.py --config "$CONFIG" --run-id "$RUN_ID"

echo ""
echo "Generating visualizations..."
python plot_results.py --config "$CONFIG" --run-id "$RUN_ID"

echo ""
echo "============================================================"
echo "Experiment complete!"
echo "Run ID: $RUN_ID"
echo "Results: results/$RUN_ID/"
echo "============================================================"
