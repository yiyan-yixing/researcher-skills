#!/bin/bash
# =============================================================================
# 实验2 Phase 1A-C 一键运行脚本
# =============================================================================
# 用法:
#   bash run_phase1.sh                     # 默认使用 qwen2.5-3b
#   bash run_phase1.sh --model gpt2        # 使用 GPT-2 对照
#   bash run_phase1.sh --phase 1a          # 只跑 Phase 1A
#   bash run_phase1.sh --skip-data-prep    # 跳过数据准备（已准备好时）
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认参数
MODEL="Qwen/Qwen2.5-3B"
PHASE="all"
SKIP_DATA_PREP=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --skip-data-prep)
            SKIP_DATA_PREP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash run_phase1.sh [--model MODEL] [--phase {1a|1b|1c|all}] [--skip-data-prep]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "Experiment 2: Format-Controlled Diagnosis - Phase 1"
echo "Model: $MODEL"
echo "Phase: $PHASE"
echo "Working dir: $SCRIPT_DIR"
echo "============================================================"

# 创建结果子目录
mkdir -p results/phase1a results/phase1b results/phase1c

# --- Step 0: 数据准备 ---
if [ "$SKIP_DATA_PREP" = false ]; then
    echo ""
    echo "=== Step 0: Data Preparation ==="
    if [ "$PHASE" = "all" ]; then
        python data/prepare_data_v2.py --config config_phase1a.yaml --phase 1a
        python data/prepare_data_v2.py --config config_phase1b.yaml --phase 1b
        python data/prepare_data_v2.py --config config_phase1c.yaml --phase 1c
    else
        if [ "$PHASE" = "1a" ]; then
            CONFIG="config_phase1a.yaml"
        elif [ "$PHASE" = "1b" ]; then
            CONFIG="config_phase1b.yaml"
        else
            CONFIG="config_phase1c.yaml"
        fi
        python data/prepare_data_v2.py --config "$CONFIG" --phase "$PHASE"
    fi
else
    echo "=== Skipping data preparation ==="
fi

# 运行单个 phase 的函数
run_phase() {
    local phase_name=$1
    local config=$2
    local data_file=$3
    local results_subdir=$4

    echo ""
    echo "=== Phase ${phase_name} ==="
    echo "Config: $config"
    echo "Data: $data_file"
    echo "Results: $results_subdir"

    local run_id="phase_${phase_name}_$(date +%s)"

    # Step 1: 隐藏状态提取
    echo "--- Extracting hidden states ---"
    python extract_hidden.py \
        --config "$config" \
        --run-id "$run_id" \
        --samples-file "$data_file"

    # Step 2: 探针训练
    echo "--- Training probes ---"
    python probe.py \
        --config "$config" \
        --run-id "$run_id"

    echo "--- Phase ${phase_name} complete ---"
}

# --- Phase 1A ---
if [ "$PHASE" = "all" ] || [ "$PHASE" = "1a" ]; then
    run_phase "1a" "config_phase1a.yaml" "data/all_samples_v2.json" "results/phase1a"
fi

# --- Phase 1B ---
if [ "$PHASE" = "all" ] || [ "$PHASE" = "1b" ]; then
    run_phase "1b" "config_phase1b.yaml" "data/mmlu_within.json" "results/phase1b"
fi

# --- Phase 1C ---
if [ "$PHASE" = "all" ] || [ "$PHASE" = "1c" ]; then
    run_phase "1c" "config_phase1c.yaml" "data/mmlu_text_only.json" "results/phase1c"
fi

# --- 可视化 ---
echo ""
echo "=== Generating visualizations ==="
python plot_phase1.py 2>/dev/null || echo "[WARN] plot_phase1.py failed (check matplotlib installation)"

echo ""
echo "============================================================"
echo "All Phase 1 experiments complete."
echo "Results saved to:"
echo "  results/phase1a/  — IFEval 4-option fix"
echo "  results/phase1b/  — within-MMLU skill vs knowledge"
echo "  results/phase1c/  — within-MMLU text-only (go/no-go)"
echo ""
echo "Next: Check Phase 1C results for go/no-go decision."
echo "  If any mid-high layer (>50% depth) accuracy > 75%: GO -> Phase 2"
echo "  If all layers < 65%: NO-GO -> H2 falsified"
echo "============================================================"
