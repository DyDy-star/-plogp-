#!/bin/bash
#export VLLM_ATTENTION_BACKEND=XFORMERS
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export CUDA_VISIBLE_DEVICES=4,5,6,7

# ========================================
# 环境配置
# ========================================
# HuggingFace 配置
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACENAME="123YYY123"
export HF_TOKEN="YOUR_HF_TOKEN"
echo "✓ HuggingFace 镜像: $HF_ENDPOINT"
echo "✓ HuggingFace 用户名: $HUGGINGFACENAME"

# WandB 配置
export WANDB_BASE_URL=https://api1.bandw.top
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
echo "✓ WandB 镜像: $WANDB_BASE_URL"
# ========================================

# ------------------------------------------------------------

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

TASK="MATH-TTT"
BACKBONE="Llama-3.2-3B-Oat-Zero"
ADVANTAGE="grpo_confidence"

K=3
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=$((1024 * $K))
if [ "$K" -gt 8 ]; then
  N=4
else
  N=16
fi

EPISODE=10
DATA_TRAIN_BATCH_SIZE=32
# 两阶段TTRL配置
ENABLE_TWO_STAGE_TTRL=True  # 启用两阶段TTRL
TWO_STAGE_MODE="uniform"  # 统一权重模式（权重为1）
NOISE_ETA=0.1  # 噪声公式中的η参数

# 阶段一配置（不带噪声）
STAGE1_N_VOTES=32  # 阶段一：32个干净回答用于计算伪标签
STAGE1_N_SAMPLES=16  # 阶段一：随机采样16个样本用于训练

# 阶段二配置（条件性噪声）
STAGE2_N_VOTES=64  # 阶段二：64个回答（32个干净 + 32个条件性噪声）用于计算伪标签
STAGE2_N_SAMPLES=16  # 阶段二：随机采样16个用于训练（同阶段一）
STAGE2_NOISY_SAMPLES=32  # 阶段二：生成32个回答（置信度<0.5加噪声，>=0.5不加噪声）

DATA_LOCAL_DIR="/data/user5/TTRL/verl/data"
BACKBONE_PATH="/data/user5/models/${BACKBONE}"

MODEL="${TASK}-${BACKBONE}"
EXPERIMENT="TTRL-Len@${K}k"

WANDB_PROJECT="TTRL-verl"
LOG_NAME="${DATE}-${EXPERIMENT}-${MODEL}-${ADVANTAGE}"
OUTPUT_DIR="checkpoints/${WANDB_PROJECT}/${MODEL}/${DATE}/${EXPERIMENT}-${ADVANTAGE}-${TIME_TAG}"

# ------------------------------------------------------------
python -m verl.trainer.main_ppo \
--config-name='ppo_trainer_ttrl.yaml'\
  data.train_files=["$DATA_LOCAL_DIR/$TASK/train.parquet"] \
  data.val_files=["$DATA_LOCAL_DIR/$TASK/test.parquet"] \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.filter_overlong_prompts=True \
  +data.suffix_prompt='"\nPlease reason step by step, and put your final answer within \boxed{}."' \
  data.truncation='error' \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=0.6 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=$N \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  critic.optim.lr=9e-6 \
  critic.model.use_remove_padding=True \
  critic.model.path=$BACKBONE_PATH \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=True \
  algorithm.kl_ctrl.kl_coef=0.03 \
  algorithm.adv_estimator=$ADVANTAGE \
  custom_reward_function.path="./verl/utils/reward_score/ttrl_math/__init__.py" \
  custom_reward_function.name=reward_func \
  ttrl.enable=True \
  ttrl.n_votes_per_prompt=$STAGE2_N_VOTES \
  ttrl.n_samples_per_prompt=$STAGE1_N_SAMPLES \
  +ttrl.two_stage_mode=$ENABLE_TWO_STAGE_TTRL \
  +ttrl.confidence_weighted=$TWO_STAGE_MODE \
  +ttrl.stage1_n_votes=$STAGE1_N_VOTES \
  +ttrl.stage1_n_samples=$STAGE1_N_SAMPLES \
  +ttrl.stage2_n_votes=$STAGE2_N_VOTES \
  +ttrl.stage2_n_samples=$STAGE2_N_SAMPLES \
  +ttrl.stage2_noisy_samples=$STAGE2_NOISY_SAMPLES \
  +ttrl.noise_eta=$NOISE_ETA \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$LOG_NAME \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=2000000 \
  trainer.test_freq=2 \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_epochs=$EPISODE "$@"

echo "Output directory: $OUTPUT_DIR"
echo "Two-Stage TTRL: $ENABLE_TWO_STAGE_TTRL"
echo "Advantage Estimator: $ADVANTAGE (confidence-weighted)"
echo "阶段一配置："
echo "  - 投票样本数: $STAGE1_N_VOTES (干净样本)"
echo "  - 训练样本数: $STAGE1_N_SAMPLES"
echo "阶段二配置："
echo "  - 投票样本数: $STAGE2_N_VOTES (${STAGE1_N_VOTES}个干净 + ${STAGE2_NOISY_SAMPLES}个带噪声)"
echo "  - 带噪声生成数: $STAGE2_NOISY_SAMPLES"
echo "  - 训练样本数: $STAGE2_N_SAMPLES"
echo "总训练样本数: $((STAGE1_N_SAMPLES + STAGE2_N_SAMPLES))"