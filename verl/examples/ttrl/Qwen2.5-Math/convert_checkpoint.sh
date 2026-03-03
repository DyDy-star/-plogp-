#!/bin/bash
#
# 将FSDP检查点转换为HuggingFace格式
#

set -e

# 源检查点路径（FSDP格式）
SOURCE_DIR="/data/user5/TTRL/verl/checkpoints/TTRL-verl/AIME-TTT-Llama-3.2-3B-Oat-Zero/0119/TTRL-Len@3k-grpo-170428/global_step_240/actor"

# 目标路径（HuggingFace格式）
TARGET_DIR="/data/user5/TTRL/verl/checkpoints/TTRL-verl/AIME-TTT-Llama-3.2-3B-Oat-Zero/0119/TTRL-Len@3k-grpo-170428/global_step_240/actor/huggingface_merged"

echo "========================================="
echo "Converting FSDP checkpoint to HuggingFace format"
echo "========================================="
echo ""
echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_DIR"
echo ""

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

# 检查是否已经存在转换后的模型
if [ -d "$TARGET_DIR" ] && [ -f "$TARGET_DIR/config.json" ]; then
    echo "Converted model already exists at: $TARGET_DIR"
    echo "Checking for model weights..."
    
    if ls "$TARGET_DIR"/*.safetensors 1> /dev/null 2>&1 || ls "$TARGET_DIR"/*.bin 1> /dev/null 2>&1; then
        echo "✅ Model weights found! Skipping conversion."
        exit 0
    else
        echo "⚠️  Config found but no model weights. Re-converting..."
        rm -rf "$TARGET_DIR"
    fi
fi

# 创建目标目录
mkdir -p "$TARGET_DIR"

echo "Starting conversion..."
echo ""

# 切换到 TTRL 项目目录
cd "/data/user5/TTRL/verl"

# 运行模型合并器
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$SOURCE_DIR" \
    --target_dir "$TARGET_DIR"

echo ""
echo "========================================="
echo "✅ Conversion completed!"
echo "========================================="
echo ""
echo "Merged model location: $TARGET_DIR"
echo ""

# 列出生成的文件
echo "Generated files:"
ls -lh "$TARGET_DIR"
echo ""

