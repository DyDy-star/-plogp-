# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    num_repeat=1,
    multi_turn=False,
    norm_adv_by_std_in_grpo=True,
    config=None,
):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.get("pf_ppo_reweight_method", "pow"),
                config.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to "cuda".
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert (
                config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None
                or config.actor_rollout_ref.rollout.multi_turn.interaction_config_path is not None
            ), (
                "tool_config_path or interaction_config_path must be set when enabling multi_turn with tool, "
                "due to no role-playing support"
            )
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], (
                "only GRPO is tested for multi-turn with tool"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # ★ 修复: repeat() 导致 non_tensor_batch 中 dict 共享引用
            for _key in ("extra_info", "reward_model"):
                if _key in test_batch.non_tensor_batch:
                    old_arr = test_batch.non_tensor_batch[_key]
                    new_arr = np.empty(len(old_arr), dtype=object)
                    for _i in range(len(old_arr)):
                        v = old_arr[_i]
                        new_arr[_i] = dict(v) if isinstance(v, dict) else v
                    test_batch.non_tensor_batch[_key] = new_arr

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        repeat_sampling_sglang_grpo = (
            self.config.actor_rollout_ref.rollout.name == "sglang"
            and self.config.actor_rollout_ref.rollout.multi_turn.enable
        )

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]

                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]

                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                if repeat_sampling_sglang_grpo:
                    uids_for_prompts = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    batch.non_tensor_batch["uid"] = uids_for_prompts
                    gen_batch.non_tensor_batch["uid"] = uids_for_prompts
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    assert np.array_equal(batch.non_tensor_batch["uid"], gen_batch.non_tensor_batch["uid"]), (
                        "UIDs must be identical for SGLang rollout"
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if self.config.get("ttrl", {}).get("enable", False):
                            from verl.trainer.ppo.ttrl_utils import select_top_k_per_prompt, apply_ttrl_gt, filter_and_sample_by_reasoning_steps

                            gen_batch.meta_info["kwargs"] = {"n": self.config.ttrl.n_votes_per_prompt}
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                            assert len(gen_batch_output) == len(batch) * self.config.ttrl.n_votes_per_prompt

                            if self.config.trainer.get("i_filter", False):
                                # I-select: vote on ALL N_VOTES responses first (free),
                                # then I-based selection happens after compute_log_prob
                                batch = apply_ttrl_gt(batch, gen_batch_output, self.config.ttrl.n_votes_per_prompt, self.tokenizer)
                                print(f"[I-select] Voted on all {self.config.ttrl.n_votes_per_prompt} responses")
                            elif self.config.trainer.get("i_reward", False):
                                # I-GRPO: skip majority voting, keep original GT for monitoring
                                for i in range(len(batch)):
                                    di = batch[i]
                                    di.non_tensor_batch["reward_model"]["original_gt"] = di.non_tensor_batch["reward_model"]["ground_truth"]
                                print("[I-GRPO] Skipping majority vote — using I_mean as reward")
                            else:
                                batch = apply_ttrl_gt(batch, gen_batch_output, self.config.ttrl.n_votes_per_prompt, self.tokenizer)
                            
                            # 检查是否启用推理步骤过滤
                            if self.config.trainer.get("filter_reasoning_steps", None) is not None:
                                gen_batch_output = filter_and_sample_by_reasoning_steps(
                                    data=gen_batch_output, 
                                    n_votes_per_prompt=self.config.ttrl.n_votes_per_prompt, 
                                    n_samples_per_prompt=self.config.ttrl.n_samples_per_prompt,
                                    target_reasoning_steps=self.config.trainer.filter_reasoning_steps,
                                    tokenizer=self.tokenizer,
                                    random_seed=self.global_steps
                                )
                            elif self.config.trainer.get("i_filter", False):
                                # I-select: keep ALL N_VOTES responses for entropy computation
                                # Selection to N_SAMPLES happens after compute_log_prob
                                print(f"[I-select] Keeping all {len(gen_batch_output)} responses "
                                      f"(will select {self.config.ttrl.n_samples_per_prompt}/prompt after entropy)")
                            else:
                                gen_batch_output = select_top_k_per_prompt(gen_batch_output, self.config.ttrl.n_votes_per_prompt, self.config.ttrl.n_samples_per_prompt)
                                assert len(gen_batch_output) == len(batch) * self.config.ttrl.n_samples_per_prompt
                        else:
                            # 非TTRL模式：检查是否启用独立的筛选逻辑
                            filter_reasoning_steps = self.config.trainer.get("filter_reasoning_steps", None)
                            if filter_reasoning_steps is not None:
                                # 启用筛选模式：生成多个样本，筛选指定步数的，然后随机抽取
                                from verl.trainer.ppo.ttrl_utils import filter_and_sample_by_reasoning_steps
                                n_votes = self.config.actor_rollout_ref.rollout.n  # 使用rollout.n作为每prompt生成的样本数
                                n_samples = self.config.trainer.get("n_samples_per_prompt", 32)  # 默认32
                                
                                gen_batch.meta_info["kwargs"] = {"n": n_votes}
                                if not self.async_rollout_mode:
                                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                                else:
                                    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                                
                                # 筛选和采样
                                gen_batch_output = filter_and_sample_by_reasoning_steps(
                                    data=gen_batch_output,
                                    n_votes_per_prompt=n_votes,
                                    n_samples_per_prompt=n_samples,
                                    target_reasoning_steps=filter_reasoning_steps,
                                    tokenizer=self.tokenizer,
                                    random_seed=self.global_steps
                                )
                            else:
                                # 普通模式：直接生成
                                if not self.async_rollout_mode:
                                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                                else:
                                    # vllm should set async_rollout_mode to enable async rollout
                                    # sglang turns on async_rollout_mode by default
                                    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # 初始化actual_num_repeat变量（用于后续的advantage计算和metrics）
                    actual_num_repeat = self.config.actor_rollout_ref.rollout.n
                    
                    if not repeat_sampling_sglang_grpo:
                        batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                        )
                        # repeat to align with repeated responses in rollout
                        # 根据实际生成的样本数计算repeat次数
                        if self.config.get("ttrl", {}).get("enable", False) or self.config.trainer.get("filter_reasoning_steps", None) is not None:
                            # TTRL模式 或 筛选模式：计算实际的repeat次数
                            actual_num_repeat = len(gen_batch_output) // len(batch.batch)
                            print(f"[Filter] Batch repeat times: {actual_num_repeat} (actual samples: {len(gen_batch_output)}, prompts: {len(batch.batch)})")
                            batch = batch.repeat(repeat_times=actual_num_repeat, interleave=True)
                        else:
                            batch = batch.repeat(repeat_times=actual_num_repeat, interleave=True)

                    batch = batch.union(gen_batch_output)

                    # ★ 关键修复: batch.repeat() 使用 np.repeat 导致 non_tensor_batch 中
                    # object 类型的 numpy 数组元素(dict)变成共享引用。
                    # 同一 prompt 的 32 个样本共享同一个 extra_info dict，后续写入
                    # step_transitions/token_entropies 等会互相覆盖 → 同组所有样本
                    # 得到相同奖励 → GRPO advantage=0 → 无训练信号。
                    # 必须为每个样本创建独立的 dict 副本。
                    for _key in ("extra_info", "reward_model"):
                        if _key in batch.non_tensor_batch:
                            old_arr = batch.non_tensor_batch[_key]
                            new_arr = np.empty(len(old_arr), dtype=object)
                            for _i in range(len(old_arr)):
                                v = old_arr[_i]
                                new_arr[_i] = dict(v) if isinstance(v, dict) else v
                            batch.non_tensor_batch[_key] = new_arr

                    if "response_mask" not in batch.batch:
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # ===== 截断统计（仅记录，不遮罩）=====
                    # 截断响应保留在训练中，由 GRPO 自然赋予负 advantage
                    if hasattr(self, 'tokenizer') and self.tokenizer.eos_token_id is not None:
                        responses = batch.batch["responses"]
                        eos_token_id = self.tokenizer.eos_token_id
                        if isinstance(eos_token_id, (list, tuple)):
                            eos_ids_tensor = torch.tensor(eos_token_id, device=responses.device)
                            is_eos = torch.isin(responses, eos_ids_tensor)
                        else:
                            is_eos = (responses == eos_token_id)
                        has_eos = is_eos.any(dim=1)
                        n_truncated = int((~has_eos).sum().item())
                        n_total = has_eos.numel()

                        if n_truncated > 0:
                            print(f"[TruncInfo] {n_truncated}/{n_total} truncated responses "
                                  f"({100.0 * n_truncated / n_total:.1f}%) — kept for training")

                        rmask = batch.batch["response_mask"]
                        resp_lengths = rmask.sum(dim=-1).float()
                        valid_lengths = resp_lengths[resp_lengths > 0]
                        metrics.update({
                            "training/truncated_ratio": n_truncated / max(n_total, 1),
                            "training/mean_response_length": resp_lengths.mean().item(),
                            "training/mean_valid_response_length": (
                                valid_lengths.mean().item() if len(valid_lengths) > 0 else 0.0
                            ),
                        })

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs and policy entropy
                    # NOTE: Must compute BEFORE reward to provide token_entropies for entropy-based rewards
                    
                    # R_brake token reward 不需要 step boundaries (已精简)
                    
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        
                        # actor/entropy 由 compute_data_metrics → compute_policy_entropy_metrics 统一计算
                        # 与 policy_entropy/token/mean 使用相同数据和公式，保证两者在监控中完全一致
                        
                        # Store token-level entropies in extra_info for entropy-based reward functions
                        compute_entropy_flag = False
                        try:
                            compute_entropy_flag = OmegaConf.select(self.config, "trainer.compute_entropy_for_reward", default=False)
                            if isinstance(compute_entropy_flag, str):
                                compute_entropy_flag = compute_entropy_flag.lower() in ('true', '1', 'yes')
                        except Exception:
                            pass
                        
                        if compute_entropy_flag:
                            try:
                                batch_size = len(batch)
                                
                                # 初始化或验证 extra_info 数组
                                if "extra_info" not in batch.non_tensor_batch:
                                    batch.non_tensor_batch["extra_info"] = np.array(
                                        [{} for _ in range(batch_size)], dtype=object
                                    )
                                elif not isinstance(batch.non_tensor_batch["extra_info"], np.ndarray) or \
                                     len(batch.non_tensor_batch["extra_info"]) != batch_size:
                                    batch.non_tensor_batch["extra_info"] = np.array(
                                        [{} for _ in range(batch_size)], dtype=object
                                    )
                                
                                # 为每个样本设置 token_ids 和 token_entropies
                                for i in range(batch_size):
                                    # 使用 response_mask 获取有效的 response 长度
                                    response_mask = batch.batch["response_mask"][i]
                                    valid_response_length = int(response_mask.sum().item())
                                    
                                    # 获取有效的 response token ids
                                    response_ids = batch.batch["responses"][i]
                                    valid_response_ids = response_ids[:valid_response_length]
                                    
                                    # 获取有效的 token entropies
                                    sample_entropies = entropys[i, :valid_response_length].detach().cpu().numpy()
                                    
                                    # 确保 extra_info[i] 是字典
                                    if not isinstance(batch.non_tensor_batch["extra_info"][i], dict):
                                        batch.non_tensor_batch["extra_info"][i] = {}
                                    
                                    # 设置 token_ids 和 token_entropies
                                    batch.non_tensor_batch["extra_info"][i]["token_ids"] = valid_response_ids.tolist()
                                    batch.non_tensor_batch["extra_info"][i]["token_entropies"] = sample_entropies
                                
                                # Compute h_bar_boxed for entropy coherence loss
                                _use_coherence = self.config.actor_rollout_ref.actor.get(
                                    "use_entropy_coherence", False)
                                if _use_coherence:
                                    import re as _re
                                    h_bar_boxed = torch.zeros(batch_size, device=entropys.device)
                                    for i in range(batch_size):
                                        rmask = batch.batch["response_mask"][i]
                                        vlen = int(rmask.sum().item())
                                        if vlen < 2:
                                            h_bar_boxed[i] = entropys[i, :max(vlen, 1)].mean()
                                            continue
                                        rids = batch.batch["responses"][i][:vlen].tolist()
                                        text, spans = "", []
                                        for ti, tid in enumerate(rids):
                                            s = len(text)
                                            text += self.tokenizer.decode([tid], skip_special_tokens=True)
                                            spans.append((s, len(text)))
                                        bset = set()
                                        for m in _re.finditer(r'\\boxed\{', text):
                                            depth, p = 1, m.end()
                                            while p < len(text) and depth > 0:
                                                if text[p] == '{': depth += 1
                                                elif text[p] == '}': depth -= 1
                                                p += 1
                                            for ti, (ts, te) in enumerate(spans):
                                                if ts < p and te > m.start():
                                                    bset.add(ti)
                                        if len(bset) >= 1:
                                            bidx = list(bset)
                                            h_bar_boxed[i] = entropys[i, bidx].mean()
                                        else:
                                            h_bar_boxed[i] = entropys[i, :vlen].mean()
                                    batch.batch["h_bar_boxed"] = h_bar_boxed
                                    print(f"[Coherence] h_bar_boxed: mean={h_bar_boxed.mean():.4f}, "
                                          f"std={h_bar_boxed.std():.4f}")

                            except Exception as e:
                                import traceback
                                print(f"[EntropyReward] ERROR: {e}")
                                traceback.print_exc()
                        
                        # 将 token_reward (R_brake) 存入 extra_info
                        batch_size_st = len(batch)
                        if hasattr(old_log_prob, 'non_tensor_batch') and \
                           "step_transitions" in old_log_prob.non_tensor_batch:
                            st_list = old_log_prob.non_tensor_batch["step_transitions"]
                            for i in range(min(batch_size_st, len(st_list))):
                                if isinstance(batch.non_tensor_batch["extra_info"][i], dict):
                                    tok_r = None
                                    if st_list[i] is not None and isinstance(st_list[i], dict):
                                        tok_r = st_list[i].get("token_reward", None)
                                    if tok_r is not None:
                                        batch.non_tensor_batch["extra_info"][i]["token_reward"] = tok_r
                            del old_log_prob.non_tensor_batch["step_transitions"]
                        
                        # Keep entropys in batch for detailed policy entropy metrics computation
                        # This will be used by compute_policy_entropy_metrics() in compute_data_metrics()
                        # Do NOT pop entropys here - let it flow through to metrics computation
                        batch = batch.union(old_log_prob)

                    # I-select: now that entropys are available, select best N_SAMPLES by I_mean
                    if self.config.get("ttrl", {}).get("enable", False) and self.config.trainer.get("i_filter", False):
                        from verl.trainer.ppo.ttrl_utils import i_select_by_inertia
                        n_keep = self.config.ttrl.n_samples_per_prompt
                        batch = i_select_by_inertia(batch, n_keep)
                        actual_num_repeat = n_keep
                        entropys = batch.batch["entropys"]

                    # entropys flow through to dp_actor for entropy gradient blocking

                    # Compute reference model log_prob (needed for KL loss in actor update)
                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # ===== CDQ 奖励: R = mean(log P(y_t) + H(t)) =====
                    # CDQ = Confident Decision Quality (不确定中的确定选择)
                    #
                    # log P(y_t): 对选中 token 的信心 (≤0, 越高越好)
                    # H(t):      分布的不确定性 (≥0, 适度即好)
                    # CDQ(t) = log P(y_t) + H(t):
                    #   > 0 → 考虑多选项但有明确首选 → 适应了答案分布 ✓
                    #   < 0 → 不确定且选择不清晰 → 噪声 ✗
                    #   ≈ 0 → 过度锐化 或 完全随机 ✗
                    #
                    # 三部分框架:
                    #   ① 探索时 CDQ>0 → 有方向的探索, 不乱增熵
                    #   ② 压缩时 CDQ>0 → 基于证据的压缩, 不丢信息
                    #   ③ 全程 CDQ>0 → 每步好决策 → 信息自然守恒
                    if "entropys" in batch.batch.keys() and "old_log_probs" in batch.batch.keys():
                        try:
                            cur_ent = batch.batch["entropys"].float()   # (B, seq_len)
                            log_p = batch.batch["old_log_probs"].float()  # (B, seq_len)
                            rmask = batch.batch["response_mask"].float()  # (B, seq_len)
                            lengths = rmask.sum(dim=1).clamp(min=1)  # (B,)
                            batch_size_ent = cur_ent.size(0)

                            # CDQ per token: 在不确定中做出确定选择
                            cdq = log_p + cur_ent  # (B, seq_len)

                            # 奖励 = mean CDQ
                            r_total = (cdq * rmask).sum(dim=1) / lengths  # (B,)

                            # ---- 监控指标 (不参与奖励) ----
                            # mean log P: 序列对数概率 (越高=越可预测)
                            mean_logp = (log_p * rmask).sum(dim=1) / lengths
                            # mean H: 平均熵 (越高=越不确定)
                            mean_h = (cur_ent * rmask).sum(dim=1) / lengths
                            # S3: 路径效率 (方向性)
                            v_theta = cur_ent[:, 1:] - cur_ent[:, :-1]
                            abs_v = torch.abs(v_theta)
                            v_mask = rmask[:, 1:]
                            total_path = (abs_v * v_mask).sum(dim=1).clamp(min=1e-8)
                            cumsum_mask = rmask.cumsum(dim=1)
                            first_mask = (cumsum_mask == 1).float() * rmask
                            rev_cumsum = rmask.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
                            last_mask = (rev_cumsum == 1).float() * rmask
                            h_first = (cur_ent * first_mask).sum(dim=1)
                            h_last = (cur_ent * last_mask).sum(dim=1)
                            r_s3 = (h_first - h_last) / total_path

                            # 批量转 CPU
                            r_total_cpu = r_total.detach().cpu().tolist()
                            mean_logp_cpu = mean_logp.detach().cpu().tolist()
                            mean_h_cpu = mean_h.detach().cpu().tolist()
                            r_s3_cpu = r_s3.detach().cpu().tolist()

                            for i in range(batch_size_ent):
                                if isinstance(batch.non_tensor_batch["extra_info"][i], dict):
                                    ei = batch.non_tensor_batch["extra_info"][i]
                                    ei["entropy_efficiency"] = r_total_cpu[i]
                                    ei["mean_logp"] = mean_logp_cpu[i]
                                    ei["mean_h"] = mean_h_cpu[i]
                                    ei["s3"] = r_s3_cpu[i]

                        except Exception as e:
                            import traceback
                            print(f"[CDQ Reward] ERROR: {e}")
                            traceback.print_exc()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    # Note: ref_log_prob computation has been moved earlier (before reward)
                    # to enable ref_entropy_efficiency reward computation

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = True

                        # 使用之前计算的actual_num_repeat（在TTRL strict模式下可能与配置值不同）
                        print(f"[Advantage] Using num_repeat: {actual_num_repeat}")

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=actual_num_repeat,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                        )

                        _use_uniform_adv = self.config.actor_rollout_ref.actor.get(
                            "use_uniform_advantage", False)
                        if _use_uniform_adv:
                            with torch.no_grad():
                                raw_scores = (batch.batch["token_level_scores"] * batch.batch["response_mask"]).sum(dim=-1)
                                uniform_adv = torch.where(
                                    raw_scores > 0.5,
                                    torch.ones_like(raw_scores),
                                    -torch.ones_like(raw_scores),
                                )
                                batch.batch["advantages"] = uniform_adv.unsqueeze(-1) * batch.batch["response_mask"]
                                n_pos = (raw_scores > 0.5).sum().item()
                                n_neg = (raw_scores <= 0.5).sum().item()
                                print(f"[UniformAdv] Override to ±1: {n_pos} positive, {n_neg} negative")

                        # Token-level W1 additive advantage: w(t) = mean_W1_prompt - mean_j|H_t - H_boxed_j|
                        _use_step_adv = self.config.actor_rollout_ref.actor.get(
                            "use_step_advantage", False)
                        if _use_step_adv and "entropys" in batch.batch.keys():
                            try:
                                import re as _re
                                from collections import defaultdict as _ddict
                                _step_alpha = self.config.actor_rollout_ref.actor.get(
                                    "step_advantage_alpha", 0.1)

                                def _find_boxed_ent(ent_1d, rids, tokenizer):
                                    text = tokenizer.decode(rids, skip_special_tokens=True)
                                    m = list(_re.finditer(r'\\boxed\s*\{', text))
                                    if not m:
                                        return None
                                    start_char = m[-1].start()
                                    depth, end_char = 0, len(text)
                                    for ci in range(m[-1].end() - 1, len(text)):
                                        if text[ci] == '{':
                                            depth += 1
                                        elif text[ci] == '}':
                                            depth -= 1
                                            if depth == 0:
                                                end_char = ci + 1
                                                break
                                    cum_len, st, en = 0, 0, len(rids)
                                    for ti, tid in enumerate(rids):
                                        tok_str = tokenizer.decode([tid], skip_special_tokens=True)
                                        prev = cum_len
                                        cum_len += len(tok_str)
                                        if prev <= start_char < cum_len:
                                            st = ti
                                        if prev < end_char <= cum_len:
                                            en = ti + 1
                                            break
                                    return ent_1d[st:en] if en > st else None

                                with torch.no_grad():
                                    B = len(batch)
                                    ent_all = batch.batch["entropys"]
                                    resp_all = batch.batch["responses"]
                                    rmask_all = batch.batch["response_mask"]
                                    uids = batch.non_tensor_batch["uid"]

                                    w1_per_token = torch.zeros_like(ent_all)

                                    for i in range(B):
                                        vlen = int(rmask_all[i].sum().item())
                                        if vlen < 5:
                                            continue
                                        ent = ent_all[i, :vlen]
                                        rids = resp_all[i, :vlen].tolist()

                                        boxed_ent = _find_boxed_ent(ent, rids, self.tokenizer)
                                        if boxed_ent is None or len(boxed_ent) < 2:
                                            continue

                                        # W1_token(t) = mean_j |H_t - H_boxed_j|
                                        w1_per_token[i, :vlen] = torch.mean(
                                            torch.abs(ent.unsqueeze(-1) - boxed_ent.unsqueeze(0)),
                                            dim=-1)

                                    uid_to_idx = _ddict(list)
                                    for i in range(B):
                                        uid_to_idx[uids[i]].append(i)

                                    token_adv = torch.zeros_like(ent_all)
                                    last_mean_w1 = 0.0
                                    for uid, indices in uid_to_idx.items():
                                        vals = []
                                        for idx in indices:
                                            active = w1_per_token[idx][rmask_all[idx] > 0]
                                            if len(active) > 0:
                                                vals.append(active)
                                        all_v = torch.cat(vals) if vals else torch.tensor([0.0])
                                        mean_w1 = all_v.mean().item() if len(all_v) > 0 else 0.0
                                        last_mean_w1 = mean_w1
                                        for idx in indices:
                                            token_adv[idx] = (mean_w1 - w1_per_token[idx]) * rmask_all[idx]

                                    batch.batch["advantages"] = (
                                        batch.batch["advantages"] + _step_alpha * token_adv)

                                    ta_active = token_adv[rmask_all > 0]
                                    metrics.update({
                                        "token_adv/mean": ta_active.mean().item(),
                                        "token_adv/std": ta_active.std().item(),
                                        "token_adv/alpha": _step_alpha,
                                        "token_adv/mean_w1": last_mean_w1,
                                    })
                                    print(f"[TokenW1Adv] mean={ta_active.mean():.4f} "
                                          f"std={ta_active.std():.4f} alpha={_step_alpha}")
                            except Exception as e:
                                import traceback
                                print(f"[TokenW1Adv] ERROR: {e}")
                                traceback.print_exc()

                        # Token-level n-gram overlap advantage shaping (group-centered)
                        # Pos: adj[t] = alpha * (overlap_t - group_mean)  高重叠增大优势
                        # Neg: adj[t] = alpha * (overlap_t - group_mean)  高重叠减惩罚
                        # 0-var groups: skipped
                        _use_bleu_overlap = self.config.actor_rollout_ref.actor.get(
                            "use_bleu_overlap_advantage", False)
                        if _use_bleu_overlap:
                            try:
                                import numpy as _np
                                from collections import defaultdict as _ddict

                                _bleu_alpha = self.config.actor_rollout_ref.actor.get(
                                    "bleu_overlap_alpha", 0.5)
                                _bleu_max_n = self.config.actor_rollout_ref.actor.get(
                                    "bleu_overlap_max_n", 4)

                                def _token_ngram_overlap(token_ids, other_ids_list, max_n):
                                    T = len(token_ids)
                                    if T == 0 or not other_ids_list:
                                        return _np.zeros(T)
                                    ngram_sets = []
                                    for n in range(1, max_n + 1):
                                        combined = set()
                                        for o_ids in other_ids_list:
                                            for k in range(len(o_ids) - n + 1):
                                                combined.add(tuple(o_ids[k:k + n]))
                                        ngram_sets.append(combined)
                                    scores = _np.zeros(T)
                                    for t in range(T):
                                        total, matched = 0, 0
                                        for ni, n in enumerate(range(1, max_n + 1)):
                                            lo = max(0, t - n + 1)
                                            hi = min(t + 1, T - n + 1)
                                            for s in range(lo, hi):
                                                total += 1
                                                if tuple(token_ids[s:s + n]) in ngram_sets[ni]:
                                                    matched += 1
                                        scores[t] = matched / total if total > 0 else 0.0
                                    return scores

                                with torch.no_grad():
                                    B = len(batch)
                                    resp_all = batch.batch["responses"]
                                    rmask_all = batch.batch["response_mask"]
                                    scores_all = batch.batch["token_level_scores"].sum(dim=-1)
                                    uids = batch.non_tensor_batch["uid"]

                                    uid_to_idx = _ddict(list)
                                    for i in range(B):
                                        uid_to_idx[uids[i]].append(i)

                                    bleu_adv = torch.zeros_like(batch.batch["advantages"])
                                    n_cross, n_zv = 0, 0
                                    all_resp_adj = []

                                    for uid, g_indices in uid_to_idx.items():
                                        g_tids, g_correct, g_vlens = [], [], []
                                        for idx in g_indices:
                                            vl = int(rmask_all[idx].sum().item())
                                            g_tids.append(resp_all[idx, :vl].tolist())
                                            g_correct.append(scores_all[idx].item() > 0)
                                            g_vlens.append(vl)

                                        pos_loc = [j for j, c in enumerate(g_correct) if c]
                                        neg_loc = [j for j, c in enumerate(g_correct) if not c]

                                        if pos_loc and neg_loc:
                                            n_cross += 1
                                            pos_tids = [g_tids[j] for j in pos_loc]
                                            neg_tids = [g_tids[j] for j in neg_loc]

                                            all_ovs = []
                                            ov_per_resp = {}
                                            for li in range(len(g_indices)):
                                                vl = g_vlens[li]
                                                if vl < 3:
                                                    continue
                                                other = neg_tids if g_correct[li] else pos_tids
                                                ov = _token_ngram_overlap(
                                                    g_tids[li], other, _bleu_max_n)
                                                ov_per_resp[li] = ov
                                                all_ovs.append(ov)

                                            if all_ovs:
                                                gmean = float(_np.concatenate(all_ovs).mean())
                                            else:
                                                gmean = 0.0

                                            for li, gi in enumerate(g_indices):
                                                if li not in ov_per_resp:
                                                    continue
                                                ov = ov_per_resp[li]
                                                vl = g_vlens[li]
                                                if g_correct[li]:
                                                    adj = _bleu_alpha * (ov - gmean)
                                                else:
                                                    adj = _bleu_alpha * (ov - gmean)
                                                all_resp_adj.append(float(adj.sum() / vl))
                                                bleu_adv[gi, :vl] = torch.tensor(
                                                    adj, dtype=bleu_adv.dtype,
                                                    device=bleu_adv.device)
                                        else:
                                            n_zv += 1

                                    batch.batch["advantages"] = batch.batch["advantages"] + bleu_adv
                                    ba_active = bleu_adv[rmask_all > 0]
                                    ra_std = float(_np.std(all_resp_adj)) if all_resp_adj else 0.0
                                    metrics.update({
                                        "token_gc_overlap/mean": ba_active.mean().item(),
                                        "token_gc_overlap/std": ba_active.std().item(),
                                        "token_gc_overlap/alpha": _bleu_alpha,
                                        "token_gc_overlap/n_cross": n_cross,
                                        "token_gc_overlap/n_zero_var": n_zv,
                                        "token_gc_overlap/resp_adj_std": ra_std,
                                    })
                                    print(f"[TokenGC] mean={ba_active.mean():.6f} "
                                          f"std={ba_active.std():.4f} a={_bleu_alpha} "
                                          f"cross={n_cross} zv={n_zv} "
                                          f"resp_adj_std={ra_std:.6f}")
                            except Exception as e:
                                import traceback
                                print(f"[TokenGC] ERROR: {e}")
                                traceback.print_exc()

                        # Two-pass adversarial training with entropy-weighted advantages
                        # w = H (per-token entropy), w_norm = H / H̄ (mean=1 per response)
                        # Phase 1 (push): A₁ = -push * A * w_norm → explore away
                        # Phase 2 (pull): A₂ = +pull * A * w_norm → exploit back
                        # NOTE: kl_direction_flip / SAM in dp_actor replaces the two-pass mechanism
                        _kl_flip = self.config.actor_rollout_ref.actor.get("kl_direction_flip", False)
                        _use_sam = self.config.actor_rollout_ref.actor.get("use_sam", False)
                        _use_push_pull = self.config.actor_rollout_ref.actor.get("use_push_pull", False)
                        si_two_pass = (self.config.trainer.get("si_plasticity", False)
                                       and "entropys" in batch.batch.keys()
                                       and not _kl_flip
                                       and not _use_sam
                                       and not _use_push_pull)

                        if si_two_pass:
                            with torch.no_grad():
                                rmask_f = batch.batch["response_mask"].float()
                                advantages_std = batch.batch["advantages"].clone()
                                push = self.config.trainer.get("si_push", 0.5)
                                pull = self.config.trainer.get("si_pull", 1.5)

                                H = batch.batch["entropys"].float()
                                w = (H / (H + 1.0)) * rmask_f

                                n_repeat = actual_num_repeat
                                B, S = w.shape
                                n_prompts = B // n_repeat
                                w_3d = w.view(n_prompts, n_repeat, S)
                                rmask_3d = rmask_f.view(n_prompts, n_repeat, S)
                                w_bar_3d = (w_3d.sum(dim=(1, 2), keepdim=True) /
                                            rmask_3d.sum(dim=(1, 2), keepdim=True).clamp(min=1))
                                w_norm = ((w_3d / (w_bar_3d + 1e-8)) * rmask_3d).view(B, S)

                                w_active = w_norm[rmask_f > 0]
                                metrics.update({
                                    "si_plasticity/H_mean":      H[rmask_f > 0].mean().item(),
                                    "si_plasticity/H_std":       H[rmask_f > 0].std().item(),
                                    "si_plasticity/w_norm_mean": w_active.mean().item(),
                                    "si_plasticity/w_norm_std":  w_active.std().item(),
                                    "si_plasticity/w_norm_max":  w_active.max().item(),
                                    "si_plasticity/push":        push,
                                    "si_plasticity/pull":        pull,
                                })
                                print(f"[SI] H={H[rmask_f > 0].mean():.3f}±{H[rmask_f > 0].std():.3f}, "
                                      f"w_norm={w_active.mean():.2f}±{w_active.std():.2f}, "
                                      f"push={push}, pull={pull}")

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable

                        if si_two_pass:
                            # Phase 1 (Push): explore away from current policy
                            with marked_timer("update_actor_push", timing_raw, color="orange"):
                                batch.batch["advantages"] = (-push * advantages_std * w_norm).to(advantages_std.device)
                                batch.meta_info["step_lr"] = False
                                actor_output_push = self.actor_rollout_wg.update_actor(batch)
                            push_metrics = reduce_metrics(actor_output_push.meta_info["metrics"])
                            metrics.update({f"actor_push/{k.split('/')[-1]}": v
                                            for k, v in push_metrics.items() if "pg_loss" in k or "kl" in k or "clipfrac" in k})

                            # Phase 2 (Pull): exploit back toward correct direction
                            with marked_timer("update_actor_pull", timing_raw, color="red"):
                                batch.batch["advantages"] = (pull * advantages_std * w_norm).to(advantages_std.device)
                                batch.meta_info["step_lr"] = True
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)
                        elif _use_push_pull:
                            _pp_alpha = self.config.actor_rollout_ref.actor.get("push_pull_alpha", 0.3)
                            with torch.no_grad():
                                rmask_f = batch.batch["response_mask"].float()
                                A_orig = batch.batch["advantages"].clone()

                                if "entropys" in batch.batch.keys():
                                    H = batch.batch["entropys"].float()
                                else:
                                    H = torch.ones_like(rmask_f)

                                H_sum = (H * rmask_f).sum(-1, keepdim=True)
                                H_count = rmask_f.sum(-1, keepdim=True).clamp(min=1)
                                H_mean = H_sum / H_count
                                w = (H / H_mean.clamp(min=1e-6)).clamp(0.2, 5.0) * rmask_f
                                w_mean = (w * rmask_f).sum(-1, keepdim=True) / H_count
                                w = w / w_mean.clamp(min=1e-6)

                                A_push = (-_pp_alpha * w * A_orig).to(A_orig.dtype)
                                A_pull = ((1.0 + _pp_alpha) * w * A_orig).to(A_orig.dtype)

                            pp_batch = batch.repeat(repeat_times=2, interleave=True)
                            pp_batch.batch["advantages"][0::2] = A_push
                            pp_batch.batch["advantages"][1::2] = A_pull

                            w_active = w[rmask_f > 0]
                            metrics.update({
                                "push_pull/w_mean": w_active.mean().item(),
                                "push_pull/w_std": w_active.std().item(),
                                "push_pull/H_mean": H_mean.mean().item(),
                                "push_pull/alpha": _pp_alpha,
                            })

                            with marked_timer("update_actor", timing_raw, color="red"):
                                actor_output = self.actor_rollout_wg.update_actor(pp_batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)
                        else:
                            # Standard single-pass actor update
                            with marked_timer("update_actor", timing_raw, color="red"):
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                    if self.config.get("ttrl", {}).get("enable", False):
                        from verl.trainer.ppo.ttrl_utils import apply_original_gt, compute_ttrl_metrics
                        batch = apply_original_gt(batch)
                        reward_tensor_original, reward_extra_infos_dict_original = compute_reward(batch, self.reward_fn)
                        batch.batch["token_level_scores_original"] = reward_tensor_original
                        # Compute ttrl metrics (使用实际的repeat次数)
                        # 在strict模式下，actual_num_repeat可能与n_samples_per_prompt不同
                        ttrl_metrics = compute_ttrl_metrics(batch, actual_num_repeat)
                        for key, value in ttrl_metrics.items():
                                metrics.update({f"train/{key}": value})

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                
                # Collect reward extra info metrics
                reward_extra_metrics = {}
                reward_metric_keys = [
                    "entropy_reward",
                    "mean_entropy",
                    "high_entropy_avg",
                    "low_entropy_avg",
                    "high_diff_entropy_avg",
                    "low_diff_entropy_avg",
                    "high_relative_diff",
                    "low_relative_diff",
                    "n_high_steps",
                    "n_low_steps",
                    "n_steps",
                    "acc",
                    "correctness_score",
                    # 正确性奖励 + Phi(z) 熵门控 per-token 加权 metrics
                    "gate_mean",         # mean(Phi) 门控通过率 (监控)
                    "H_bar",             # 自适应熵阈值 (监控)
                    "mean_H",            # 平均熵 (检测坍缩)
                ]
                # Log eta reward metrics for monitoring
                try:
                    ei_arr = batch.non_tensor_batch.get("extra_info", None)
                    if ei_arr is not None and len(ei_arr) > 0 and isinstance(ei_arr[0], dict):
                        for eta_key in ["eta_up", "eta_down", "w_linear", "prompt_accuracy", "entropy_eta_reward"]:
                            if eta_key in ei_arr[0]:
                                vals = [ei.get(eta_key, 0.0) for ei in ei_arr if isinstance(ei, dict) and eta_key in ei]
                                if vals:
                                    reward_extra_metrics[f"train/reward/{eta_key}"] = float(np.mean(vals))
                except Exception:
                    pass
                for key in reward_metric_keys:
                    if key in batch.non_tensor_batch:
                        values = batch.non_tensor_batch[key]
                        if len(values) > 0:
                            reward_extra_metrics[f"train/reward/{key}"] = float(np.mean(values))
                
                if reward_extra_metrics:
                    metrics.update(reward_extra_metrics)
                elif OmegaConf.select(self.config, "trainer.compute_entropy_for_reward", default=False):
                    print(f"[EntropyReward] ⚠️ 警告: 启用了熵奖励但未找到额外指标，batch.non_tensor_batch中的键: {list(batch.non_tensor_batch.keys())}")

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
