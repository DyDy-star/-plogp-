#!/bin/bash
#
# AIME数据集评估脚本（完整词表熵计算）
#
# 功能：
# 1. 使用vLLM进行高效推理生成
# 2. 使用PyTorch模型计算完整词表的熵（与训练时一致）
# 3. 集成CoVo的process_thoughts函数进行推理步骤划分
# 4. 将详细结果（包括步骤划分和熵信息）保存到JSON文件
#
# 关键改进：
# - 熵计算基于完整词表（~100k维），与训练时的奖励函数一致
# - 使用与训练相同的 entropy_from_logits 函数
# - 输出包含 entropy_reward（前7步平均熵 - 后8步平均熵）
#
# 使用方法：
#   bash aime_eval_with_full_entropy.sh [MODEL_PATH] [OUTPUT_DIR]
#

set -e  # 遇到错误时退出

# ========================================
# 配置参数
# ========================================

# 模型路径（可以通过命令行参数覆盖）
# 使用Llama-3.2-3B-Oat-Zero模型
MODEL_PATH=${1:-"/data/user5/models/Llama-3.2-3B-Oat-Zero"}

# 输出目录（可以通过命令行参数覆盖）
OUTPUT_DIR=${2:-"./eval_results_aime_full_entropy"}

# 测试数据路径
TEST_DATA_PATH="/data/user5/TTRL begin/verl/data/AIME-TTT/test.parquet"

# 评估参数
N_SAMPLES=16        # 每个问题生成的响应数量
TEMPERATURE=1       # 采样温度
TOP_P=0.95          # Top-p采样
MAX_TOKENS=3072     # 最大生成token数
BATCH_SIZE=2        # 批次大小（较小以节省内存，因为需要加载两个模型）
TENSOR_PARALLEL=1   # 张量并行大小（vLLM）
DEVICE="cuda"       # PyTorch模型设备

# 限制评估样本数（用于快速测试，设为空则评估所有样本）
MAX_SAMPLES=""      # 空字符串表示评估所有样本

# 自然步骤划分（不强制15步，保留原始推理步骤结构）
NATURAL_STEPS=true  # true=自然步数, false=强制15步

# ========================================
# 环境变量设置
# ========================================

# CUDA设备设置
export CUDA_VISIBLE_DEVICES=3

# vLLM配置（完全禁用V1版本）
export VLLM_USE_V1=0
unset VLLM_ATTENTION_BACKEND
# 强制禁用V1引擎
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "========================================="
echo "AIME评估 - 完整词表熵分析"
echo "========================================="
echo ""
echo "关键特性："
echo "  ✅ 使用完整词表（~100k维）计算熵"
echo "  ✅ 与训练时的奖励函数计算方式一致"
echo "  ✅ 输出 entropy_reward = 前7步平均熵 - 后8步平均熵"
echo ""
echo "配置信息："
echo "  模型路径: $MODEL_PATH"
echo "  测试数据: $TEST_DATA_PATH"
echo "  输出目录: $OUTPUT_DIR"
echo "  采样数量: $N_SAMPLES"
echo "  温度: $TEMPERATURE"
echo "  Top-p: $TOP_P"
echo "  批次大小: $BATCH_SIZE"
echo "  限制样本数: ${MAX_SAMPLES:-所有样本}"
echo "  步骤划分: $([ "$NATURAL_STEPS" = true ] && echo '自然步数（不限15步）' || echo '强制15步')"
echo ""

# ========================================
# 检查模型路径是否存在
# ========================================

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型路径不存在: $MODEL_PATH"
    echo ""
    echo "如果您要使用FSDP检查点，请先运行转换脚本："
    echo "  bash convert_checkpoint.sh"
    echo ""
    exit 1
fi

# 检查是否是有效的HuggingFace模型（必须有config.json和模型权重）
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "❌ 错误: 未找到config.json文件: $MODEL_PATH/config.json"
    echo ""
    echo "如果您要使用FSDP检查点，请先运行转换脚本："
    echo "  bash convert_checkpoint.sh"
    echo ""
    exit 1
fi

# 检查是否有模型权重文件
if ! ls "$MODEL_PATH"/*.safetensors 1> /dev/null 2>&1 && ! ls "$MODEL_PATH"/*.bin 1> /dev/null 2>&1; then
    echo "❌ 错误: 未找到模型权重文件（.safetensors 或 .bin）"
    echo "   模型路径: $MODEL_PATH"
    echo ""
    echo "如果您要使用FSDP检查点，请先运行转换脚本："
    echo "  bash convert_checkpoint.sh"
    echo ""
    exit 1
fi

echo "✅ 模型检查通过"
echo ""

# ========================================
# 创建输出目录
# ========================================

mkdir -p "$OUTPUT_DIR"

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${OUTPUT_DIR}/aime_eval_full_entropy_${TIMESTAMP}.json"
LOG_FILE="${OUTPUT_DIR}/aime_eval_full_entropy_${TIMESTAMP}.log"

echo "输出文件: $OUTPUT_FILE"
echo "日志文件: $LOG_FILE"
echo ""

# ========================================
# 运行评估脚本
# ========================================

echo "开始评估..."
echo ""

# 构建命令
CMD="python eval_amc_vllm_with_full_entropy.py \
    --model_path \"$MODEL_PATH\" \
    --test_data_path \"$TEST_DATA_PATH\" \
    --output_file \"$OUTPUT_FILE\" \
    --n_samples $N_SAMPLES \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --max_tokens $MAX_TOKENS \
    --batch_size $BATCH_SIZE \
    --tensor_parallel_size $TENSOR_PARALLEL \
    --device $DEVICE"

# 如果设置了MAX_SAMPLES，添加该参数
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

# 如果启用自然步骤划分
if [ "$NATURAL_STEPS" = true ]; then
    CMD="$CMD --natural_steps"
fi

# 执行命令并记录日志
echo "执行命令："
echo "$CMD"
echo ""

eval "$CMD" 2>&1 | tee "$LOG_FILE"

# ========================================
# 评估完成
# ========================================

echo ""
echo "========================================="
echo "评估完成！"
echo "========================================="
echo ""
echo "结果文件: $OUTPUT_FILE"
echo "日志文件: $LOG_FILE"
echo ""

# 显示一些统计信息（如果jq可用）
if command -v jq &> /dev/null; then
    echo "快速统计："
    echo "  Overall Pass@1: $(jq '.["overall_pass@1"]' "$OUTPUT_FILE")"
    echo "  测试样本数: $(jq '.n_test_samples' "$OUTPUT_FILE")"
    echo "  总响应数: $(jq '.total_responses' "$OUTPUT_FILE")"
    echo "  正确响应数: $(jq '.total_correct_responses' "$OUTPUT_FILE")"
    echo "  熵类型: $(jq '.entropy_type' "$OUTPUT_FILE")"
    echo ""
    
    # 显示第一个样本的熵奖励信息
    echo "第一个样本的熵统计："
    jq '.results[0].responses[0].entropy_analysis.overall_stats | {
        n_steps,
        early_avg_entropy,
        late_avg_entropy,
        entropy_reward,
        overall_mean_entropy
    }' "$OUTPUT_FILE"
    echo ""
else
    echo "提示: 安装 jq 可以查看更多统计信息"
    echo "      sudo apt-get install jq"
    echo ""
fi

echo "========================================="
echo "使用说明："
echo "========================================="
echo ""
echo "JSON文件结构："
echo "  - overall_pass@1: 整体Pass@1分数"
echo "  - entropy_type: \"full_vocab\" (完整词表)"
echo "  - results[]: 每个测试样本的结果"
echo "    - responses[]: 每个响应的详细信息"
echo "      - entropy_analysis: 步骤划分和熵分析"
echo "        - overall_stats:"
echo "          - entropy_reward: 前12步平均熵 - 后3步平均熵"
echo "          - early_avg_entropy: 前12个步骤的平均熵"
echo "          - late_avg_entropy: 后3个步骤的平均熵"
echo "          - ib_summary: 信息瓶颈摘要"
echo "            - mean_js_divergence: 全局平均JS散度"
echo "            - total_js_path_length: JS散度路径总长"
echo "            - exploration_disruption_js: 熵增步骤的平均JS散度"
echo "            - compression_loss_js: 熵减步骤的平均JS散度"
echo "            - mean_top10_overlap: 平均Top-10词重叠率"
echo "            - mean_cosine_similarity: 平均余弦相似度"
echo "        - steps[]: 每个推理步骤的详细信息"
echo "          - step_text: 步骤文本"
echo "          - tokens[]: 该步骤的所有token"
echo "          - token_entropies[]: 每个token的完整词表熵"
echo "          - mean_entropy: 该步骤的平均熵"
echo "          - top1_prob: 步骤平均分布的Top-1概率"
echo "          - top5_prob_mass: Top-5概率质量"
echo "          - top10_prob_mass: Top-10概率质量"
echo "          - eff_vocab_size: 有效词表大小"
echo "        - step_transitions[]: 步骤间转换指标"
echo "          - kl_forward/kl_reverse: 正/逆向KL散度"
echo "          - js_divergence: JS散度"
echo "          - top10_overlap: Top-10词重叠率"
echo "          - cosine_sim: 余弦相似度"
echo "          - entropy_delta: 熵变化量"
echo ""
echo "可以使用以下命令查看详细信息："
echo "  jq '.results[0].responses[0].entropy_analysis.overall_stats' $OUTPUT_FILE"
echo "  jq '.results[0].responses[0].entropy_analysis.steps[0]' $OUTPUT_FILE"
echo ""
echo "对比训练时的奖励："
echo "  训练时的奖励函数使用相同的公式："
echo "  entropy_reward = 前7步平均熵 - 后8步平均熵"
echo ""

