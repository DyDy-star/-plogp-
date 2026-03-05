#!/bin/bash
#
# 一键运行: 合并 FSDP 检查点 + Pass@K 评估
#
# Usage:
#   bash run_eval.sh              # 合并 + 评估 (默认 GPU 2)
#   bash run_eval.sh merge        # 仅合并
#   bash run_eval.sh eval         # 仅评估
#   DEVICE=cuda:3 bash run_eval.sh  # 指定 GPU
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"
echo "✓ 项目根目录: $PROJECT_ROOT"

# ---- 配置 ----
CKPT_BASE="/data/user5/TTRL begin/verl/checkpoints/TTRL-verl/AIME-TTT-Qwen2.5-Math-1.5B/0303"

STD_ACTOR="${CKPT_BASE}/GT-baseline-Len@3k-grpo-151743/global_step_240/actor"
SR_ACTOR="${CKPT_BASE}/GT-plogpPos-Len@3k-grpo-064004/global_step_240/actor"

STD_HF="${STD_ACTOR}/huggingface_merged"
SR_HF="${SR_ACTOR}/huggingface_merged"

DEVICE=${DEVICE:-"cuda:2"}
N_SAMPLES=${N_SAMPLES:-32}
GPU_MEMORY=${GPU_MEMORY:-0.85}
OUTPUT_DIR="${SCRIPT_DIR}/results_pass_k"

MODE=${1:-"all"}  # all | merge | eval

# =========================================
# 合并函数
# =========================================
merge_one() {
    local src="$1"
    local dst="$2"
    local name="$3"

    if [ -d "$dst" ] && ls "$dst"/*.safetensors 1>/dev/null 2>&1; then
        echo "✅ [$name] 已合并，跳过"
        return 0
    fi

    echo ""
    echo "─────────────────────────────────"
    echo "  合并: $name"
    echo "  Source: $src"
    echo "  Target: $dst"
    echo "─────────────────────────────────"
    mkdir -p "$dst"

    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$src" \
        --target_dir "$dst"

    echo "✅ [$name] 合并完成"
    ls -lh "$dst"/*.safetensors 2>/dev/null || ls -lh "$dst"/*.bin 2>/dev/null
}

# =========================================
# Step 1: 合并 FSDP 切片
# =========================================
if [ "$MODE" = "all" ] || [ "$MODE" = "merge" ]; then
    echo ""
    echo "================================================"
    echo "  Step 1: 合并 FSDP 检查点"
    echo "================================================"

    merge_one "$STD_ACTOR" "$STD_HF" "Standard GRPO (Baseline)"
    merge_one "$SR_ACTOR"  "$SR_HF"  "SR-GRPO (SurprisalRedistribution)"

    echo ""
    echo "✅ 合并完成"
fi

# =========================================
# Step 2: 评估 + 绘图
# =========================================
if [ "$MODE" = "all" ] || [ "$MODE" = "eval" ]; then
    echo ""
    echo "================================================"
    echo "  Step 2: Pass@K 评估"
    echo "================================================"
    echo "  Standard GRPO: $STD_HF"
    echo "  SR-GRPO:       $SR_HF"
    echo "  Device:        $DEVICE"
    echo "  N_samples:     $N_SAMPLES"
    echo ""

    python "${SCRIPT_DIR}/eval_pass_k.py" \
        --model_std "$STD_HF" \
        --model_sr  "$SR_HF" \
        --device    "$DEVICE" \
        --n_samples "$N_SAMPLES" \
        --gpu_memory "$GPU_MEMORY" \
        --output_dir "$OUTPUT_DIR"

    echo ""
    echo "================================================"
    echo "✅ 评估完成！结果: $OUTPUT_DIR"
    echo "================================================"
fi
