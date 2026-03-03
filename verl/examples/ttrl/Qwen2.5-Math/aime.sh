#!/bin/bash
#export VLLM_ATTENTION_BACKEND=XFORMERS
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=4,5,6,7

# ========================================
# 确保使用正确的代码目录
# ========================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"
echo "✓ 项目根目录: $PROJECT_ROOT"
echo "✓ PYTHONPATH: $PYTHONPATH"

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
export WANDB_BASE_URL=https://api.bandw.top
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
echo "✓ WandB 镜像: $WANDB_BASE_URL"
# ========================================

# ------------------------------------------------------------

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

TASK="AIME-TTT"
BACKBONE="Qwen2.5-Math-1.5B"
ADVANTAGE="grpo"
REWARD_FUNC_NAME="reward_func_i_grpo"

K=3
MAX_PROMPT_LENGTH=512
MAX_RESPONSE_LENGTH=$((1024 * $K))
if [ "$K" -gt 8 ]; then
  N=4
else
  N=16
fi

EPISODE=80
DATA_TRAIN_BATCH_SIZE=8
N_VOTES_PER_PROMPT=32
N_SAMPLES_PER_PROMPT=32
MAX_TRAIN_SAMPLES_PER_BATCH=$((DATA_TRAIN_BATCH_SIZE * N_SAMPLES_PER_PROMPT))  # 8 * 32 = 256
MINI_BATCH_SIZE=1
MICRO_BATCH_SIZE=2

DATA_LOCAL_DIR="/data/user5/TTRL/verl/data"
BACKBONE_PATH="/data/user5/models/${BACKBONE}"
REWARD_FUNCTION_PATH="/data/user5/TTRL begin/verl/verl/utils/reward_score/ttrl_math/__init__.py"

MODEL="${TASK}-${BACKBONE}"
EXPERIMENT="GT-plogpAdaptive-Len@${K}k"

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
  data.truncation='error' \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
  actor_rollout_ref.rollout.n=$N_VOTES_PER_PROMPT \
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
  critic.model.fsdp_config.optimizer_offload=False \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.adv_estimator=$ADVANTAGE \
  custom_reward_function.path="$REWARD_FUNCTION_PATH" \
  custom_reward_function.name=$REWARD_FUNC_NAME \
  ttrl.enable=False \
  ttrl.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  ttrl.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
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
  ++trainer.max_train_samples_per_batch=$MAX_TRAIN_SAMPLES_PER_BATCH \
  ++trainer.compute_entropy_for_reward=True \
  ++trainer.si_plasticity=False \
  ++actor_rollout_ref.actor.use_push_pull=False \
  ++actor_rollout_ref.actor.use_entropy_coherence=False \
  ++actor_rollout_ref.actor.use_step_advantage=False \
  ++actor_rollout_ref.actor.use_bleu_overlap_advantage=False \
  ++actor_rollout_ref.actor.bleu_overlap_alpha=0.5 \
  ++actor_rollout_ref.actor.use_uniform_advantage=False \
  ++actor_rollout_ref.actor.use_surprisal_redistribution=True \
  trainer.total_epochs=$EPISODE "$@"

echo "Output directory: $OUTPUT_DIR"