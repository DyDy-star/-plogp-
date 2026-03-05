#!/bin/bash
#
# 1) 将两个 FSDP 检查点合并为 HuggingFace 格式
# 2) 运行 logit 分布对比分析
#
set -e

# =========================================
# 确保使用正确的代码目录
# =========================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"
echo "✓ 项目根目录: $PROJECT_ROOT"

# =========================================
# 检查点路径
# =========================================
CKPT_BASE="/data/user5/TTRL begin/verl/checkpoints/TTRL-verl/AIME-TTT-Qwen2.5-Math-1.5B/0303"

STD_CKPT_DIR="${CKPT_BASE}/GT-baseline-Len@3k-grpo-151743/global_step_240/actor"
SR_CKPT_DIR="${CKPT_BASE}/GT-plogpPos-Len@3k-grpo-064004/global_step_240/actor"

STD_HF_DIR="${STD_CKPT_DIR}/huggingface_merged"
SR_HF_DIR="${SR_CKPT_DIR}/huggingface_merged"

BASE_MODEL="/data/user5/models/Qwen2.5-Math-1.5B"
DATA_PATH="/data/user5/TTRL/verl/data/AIME-TTT/test.parquet"
OUTPUT_DIR="${SCRIPT_DIR}/results"

# =========================================
# 可选参数
# =========================================
MAX_SAMPLES=${MAX_SAMPLES:-5}
MODE=${MODE:-"both"}
DEVICE=${DEVICE:-"cuda:0"}

# =========================================
# Step 1: 合并 FSDP 检查点
# =========================================
merge_checkpoint() {
    local source_dir="$1"
    local target_dir="$2"
    local name="$3"

    if [ -d "$target_dir" ] && ls "$target_dir"/*.safetensors 1>/dev/null 2>&1; then
        echo "✅ [$name] 已存在合并后的模型，跳过: $target_dir"
        return 0
    fi

    echo ""
    echo "========================================="
    echo "合并 FSDP 检查点: $name"
    echo "========================================="
    echo "  Source: $source_dir"
    echo "  Target: $target_dir"

    mkdir -p "$target_dir"

    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "$source_dir" \
        --target_dir "$target_dir"

    echo "✅ [$name] 合并完成"
    echo "  文件:"
    ls -lh "$target_dir"/*.safetensors 2>/dev/null || ls -lh "$target_dir"/*.bin 2>/dev/null || echo "  (no model weights found!)"
    echo ""
}

echo ""
echo "================================================"
echo "  Step 1/2: 合并 FSDP 检查点为 HuggingFace 格式"
echo "================================================"
echo ""

merge_checkpoint "$STD_CKPT_DIR" "$STD_HF_DIR" "Standard GRPO (Baseline)"
merge_checkpoint "$SR_CKPT_DIR"  "$SR_HF_DIR"  "SR-GRPO (SurprisalRedistribution)"

# =========================================
# Step 2: 运行 logit 分布对比
# =========================================
echo ""
echo "================================================"
echo "  Step 2/2: 运行 Logit 分布对比分析"
echo "================================================"
echo ""
echo "  Standard GRPO:  $STD_HF_DIR"
echo "  SR-GRPO:        $SR_HF_DIR"
echo "  Base Model:     $BASE_MODEL"
echo "  Data:           $DATA_PATH"
echo "  Output:         $OUTPUT_DIR"
echo "  Mode:           $MODE"
echo "  Max Samples:    $MAX_SAMPLES"
echo "  Device:         $DEVICE"
echo ""

python "${SCRIPT_DIR}/compare_logits.py" \
    --model_std  "$STD_HF_DIR" \
    --model_sr   "$SR_HF_DIR" \
    --model_base "$BASE_MODEL" \
    --data_path  "$DATA_PATH" \
    --mode       "$MODE" \
    --max_samples "$MAX_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --device     "$DEVICE"

echo ""
echo "================================================"
echo "✅ 全部完成！结果保存在: $OUTPUT_DIR"
echo "================================================"
ls -lh "$OUTPUT_DIR"
