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
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, get_policy_loss_fn, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SurprisalRedistribution(torch.autograd.Function):
    """Parameter-free entropy-adaptive gradient redistribution (positive only).

    Replaces the standard per-token gradient weight (∝ p) with:

        w_v = p_v · surp_v / (surp_v + H)

    where surp_v = -log(p_v) and H = Σ p·(-log p) is the distribution entropy.

    Each token's relative surprisal surp_v / (surp_v + H) acts as a
    smooth, self-normalizing correction factor:

        surp_v ≪ H  (runners, more probable than average) → w ≈ p·surp/H
        surp_v ≫ H  (tail, less probable than average)   → w ≈ p  (standard)
        surp_v = H  (average)                             → w = p/2

    The distribution's own entropy H serves as the natural transition scale.
    No constants, no hyperparameters, no blending.

    Requires inplace_backward=False in downstream logprobs_from_logits.
    """

    _CHUNK = 128

    @staticmethod
    def forward(ctx, logits, target_ids, advantages):
        active = (advantages > 0).nonzero(as_tuple=True)[0]
        n_active = len(active)

        if n_active > 0:
            ctx.save_for_backward(logits)
            ctx.active_indices = active.detach()
            ctx.active_targets = target_ids[active].detach()
        else:
            ctx.save_for_backward(logits[:0])

        ctx.n_active = n_active
        return logits

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.n_active == 0:
            return grad_output, None, None

        (logits,) = ctx.saved_tensors
        chunk = SurprisalRedistribution._CHUNK

        for s in range(0, ctx.n_active, chunk):
            e = min(s + chunk, ctx.n_active)
            idx = ctx.active_indices[s:e]
            tgt = ctx.active_targets[s:e]

            with torch.no_grad():
                z = logits[idx].float()
                lse = torch.logsumexp(z, dim=-1, keepdim=True)
                surp = lse - z                                      # -log(p)
                p_local = torch.exp(z - lse)
                H = (p_local * surp).sum(-1, keepdim=True)          # entropy
                w = p_local * surp / (surp + H + 1e-8)             # self-adaptive

                w_at_tgt = w.gather(-1, tgt.unsqueeze(-1))
                w_total = w.sum(-1, keepdim=True) - w_at_tgt + 1e-10

                g = grad_output[idx].float()
                g_tgt = g.gather(-1, tgt.unsqueeze(-1))
                g.scatter_(-1, tgt.unsqueeze(-1), 0.0)
                G_total = g.sum(-1, keepdim=True)

                g_new = G_total * w / w_total
                g_new.scatter_(-1, tgt.unsqueeze(-1), g_tgt)

                grad_output[idx] = g_new.to(grad_output.dtype)
                del z, lse, surp, p_local, H, w, g_new

        return grad_output, None, None


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, return_logits=False,
        surprisal_config=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            response_logits (optional): # (bs, response_len, vocab_size) when return_logits=True
        """
        response_length = micro_batch["responses"].size(-1)
        response_logits_out = None  # Will be set if return_logits=True and logits are available
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    
                    # Save original logits for entropy/JS calculation (before temperature scaling)
                    # According to policy entropy formula: H(π_θ, D) = -E[log π_θ(y_t|y_{<t})]
                    # Entropy should be computed using original policy (temperature=1), not scaled policy
                    if calculate_entropy or return_logits:
                        logits_rmpad_original = logits_rmpad.clone()
                    
                    logits_rmpad.div_(temperature)

                    if surprisal_config is not None and not self.use_ulysses_sp:
                        adv = surprisal_config["advantages"]
                        adv_full = torch.zeros(batch_size, seqlen, device=logits_rmpad.device,
                                               dtype=logits_rmpad.dtype)
                        adv_full[:, -response_length - 1 : -1] = adv.to(logits_rmpad.dtype)
                        adv_rmpad = adv_full.reshape(-1)[indices]
                        logits_rmpad = SurprisalRedistribution.apply(
                            logits_rmpad, input_ids_rmpad_rolled,
                            adv_rmpad,
                        )

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy or surprisal_config is not None:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy using original logits (temperature=1)
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad_original)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad_original
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

                # Return response logits for JS computation (rmpad branch)
                if return_logits and not self.use_fused_kernels:
                    logits_for_pad = logits_rmpad_original
                    if self.use_ulysses_sp:
                        logits_for_pad = gather_outpus_and_unpad(
                            logits_for_pad, gather_dim=0, unpad_dim=0, padding_size=pad_size,
                        )
                    full_logits = pad_input(
                        hidden_states=logits_for_pad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    response_logits_out = full_logits[:, -response_length - 1 : -1, :]
                    del full_logits, logits_for_pad

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits
                    
                    # Save original logits for entropy/JS calculation (before temperature scaling)
                    # According to policy entropy formula: H(π_θ, D) = -E[log π_θ(y_t|y_{<t})]
                    # Entropy should be computed using original policy (temperature=1), not scaled policy
                    if calculate_entropy or return_logits:
                        logits_original = logits.clone()

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)

                    if surprisal_config is not None:
                        B, T, V = logits.shape
                        logits_flat = logits.reshape(B * T, V)
                        target_flat = micro_batch["responses"].reshape(B * T)
                        adv_flat = surprisal_config["advantages"].reshape(B * T).to(logits_flat.dtype)
                        logits_flat = SurprisalRedistribution.apply(
                            logits_flat, target_flat, adv_flat,
                        )
                        logits = logits_flat.reshape(B, T, V)

                    _ib = surprisal_config is None
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"],
                                                     inplace_backward=_ib)
                    if calculate_entropy:
                        # Use original logits (temperature=1) for entropy calculation
                        logits_original_sliced = logits_original[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                        entropy = verl_F.entropy_from_logits(logits_original_sliced)  # (bsz, response_length)

                    # Return response logits for JS computation (non-rmpad branch)
                    if return_logits:
                        response_logits_out = logits_original[:, -response_length - 1 : -1, :]

            if return_logits:
                return entropy, log_probs, response_logits_out
            return entropy, log_probs

    @staticmethod
    def _compute_peak_ratio(token_entropies, boundaries):
        """
        Peak Ratio (尖峰比): 熵变的结构性信息度量。

        核心思想（由 Epiplexity 理论和实证验证支持）：
          - 正确推理 = 结构化的熵变 = 大部分步骤平稳 + 少数关键跳变（重尾/稀疏）
          - 错误推理 = 随机噪声的熵变 = 所有步骤均匀波动（高斯/稠密）

        计算方法：
          1. 按步骤边界计算每步的平均熵 H_0, H_1, ..., H_T
          2. 计算步间速度 v_t = H_{t+1} - H_t
          3. R = max|v_t| / mean|v_t|  （尖峰比，越大 = 越稀疏 = 越结构化）

        信息论解释：
          - 高尖峰比 = 少数步骤有大幅熵变（关键突破/精准收敛）+ 多数步骤平稳
          - 低尖峰比 = 所有步骤熵变幅度相近 = 随机噪声（趋近高斯/均匀）
          - 本质是度量熵变向量的稀疏性（L∞/L1 的变体）

        优点：
          - 无超参数
          - 不需要 logits（只需 entropy），省 GPU 内存
          - 两个模型上都有正向 GRPO gap（+0.10, +0.13）
          - 抗 Goodhart：压平所有熵时尖峰比 = 1（最低），无法作弊

        Args:
            token_entropies: (response_len,) tensor，每个 token 位置的熵
            boundaries: list of (start, end) tuples，步骤 token 范围

        Returns:
            float: max|v| / mean|v|，越高越好（越结构化）
        """
        if boundaries is None or len(boundaries) < 3:
            return 0.0

        # 计算每个步骤的平均熵
        step_entropies = []
        for start, end in boundaries:
            if start >= end or end > token_entropies.size(0):
                continue
            step_entropy = token_entropies[start:end].float().mean().item()
            step_entropies.append(step_entropy)

        if len(step_entropies) < 3:
            return 0.0

        # 计算步间速度
        velocities = []
        for i in range(len(step_entropies) - 1):
            velocities.append(step_entropies[i + 1] - step_entropies[i])

        # 计算 |v| 的均值和最大值
        abs_velocities = [abs(v) for v in velocities]
        mean_abs_v = sum(abs_velocities) / len(abs_velocities)
        max_abs_v = max(abs_velocities)

        # 防止除零：如果 mean ≈ 0，说明几乎没有熵变，返回 0
        if mean_abs_v < 1e-10:
            return 0.0

        # Peak ratio = max|v| / mean|v|
        # 范围: [1, len(velocities)]
        # 1 = 完全均匀（最差），len(v) = 完全集中于一步（最好）
        return max_abs_v / mean_abs_v

    @staticmethod
    def _compute_step_transitions(logits, token_entropies, boundaries):
        """
        计算步间转移指标和 per-step 特征。
        
        步间转移指标: JSD, KL_forward, KL_reverse, entropy_delta,
                     gauss_distance, jsd_from_p0, jsd_from_p0_next
        
        Per-step 特征 (用于关键步骤识别):
          mean_H:    步骤内 token 熵的均值
          std_H:     步骤内 token 熵的标准差
          eff_vocab:  有效词表大小 = mean(2^h_i)
        
        Args:
            logits: (seq_len, vocab_size) - 单样本的原始 logits (temperature=1)
            token_entropies: (seq_len,) - 每个 token 位置的熵
            boundaries: list of (start, end) - 步骤 token 范围
        
        Returns:
            dict: {
                "transitions": list of dicts (步间转移指标),
                "step_features": list of dicts (per-step 特征: mean_H, std_H, eff_vocab)
            }
        """
        import torch
        import torch.nn.functional as F
        import math
        
        _empty = {"transitions": [], "step_features": []}
        
        if boundaries is None or len(boundaries) < 2:
            return _empty
        
        # 计算每个步骤的平均概率分布和 per-step 特征
        step_distributions = []
        step_mean_entropies = []
        step_std_entropies = []
        step_eff_vocabs = []
        
        for start, end in boundaries:
            if start >= end or end > logits.size(0):
                continue
            # 步骤内每个 token 的 softmax 概率的均值 → 该步骤的 "典型分布"
            step_logits = logits[start:end, :]  # (n_tokens, vocab_size)
            step_probs = F.softmax(step_logits.float(), dim=-1).mean(dim=0)  # (vocab_size,)
            step_distributions.append(step_probs)
            
            # per-step 特征
            step_h = token_entropies[start:end].float()
            n_tokens_step = end - start
            step_mean_entropies.append(step_h.mean().item())
            step_std_entropies.append(step_h.std().item() if n_tokens_step > 1 else 0.0)
            step_eff_vocabs.append((2.0 ** step_h).mean().item())
        
        # 构建 step_features (所有有效步骤)
        step_features = []
        for i in range(len(step_mean_entropies)):
            # dH_in: 从上一步到本步的熵变 (>0 = 注入/探索, <0 = 利用/压缩)
            dH_in = (step_mean_entropies[i] - step_mean_entropies[i - 1]) if i > 0 else 0.0
            step_features.append({
                "mean_H": step_mean_entropies[i],
                "std_H": step_std_entropies[i],
                "eff_vocab": step_eff_vocabs[i],
                "dH_in": dH_in,
            })
        
        if len(step_distributions) < 2:
            return {"transitions": [], "step_features": step_features}
        
        # 计算相邻步骤间的转移指标
        transitions = []
        eps = 1e-10
        sqrt2 = math.sqrt(2.0)
        
        # ============================================================
        # p_0 锚点: 缓存初始分布，用于计算信息积累轨迹
        # ============================================================
        p_0 = step_distributions[0].clamp(min=eps)
        log_p0 = p_0.log()
        
        for i in range(len(step_distributions) - 1):
            p = step_distributions[i].clamp(min=eps)
            q = step_distributions[i + 1].clamp(min=eps)
            
            # 熵变
            dH = step_mean_entropies[i + 1] - step_mean_entropies[i]
            
            # KL_forward = KL(p_{t+1} || p_t): 新分布相对旧分布的 "创新"
            kl_forward = (q * (q.log() - p.log())).sum().item()
            
            # KL_reverse = KL(p_t || p_{t+1}): 旧分布相对新分布的 "遗忘"
            kl_reverse = (p * (p.log() - q.log())).sum().item()
            
            # JSD = (KL(p||m) + KL(q||m)) / 2, m=(p+q)/2: 对称分布距离
            m = (p + q) / 2
            jsd = 0.5 * (p * (p.log() - m.log())).sum().item() + \
                  0.5 * (q * (q.log() - m.log())).sum().item()
            
            # ============================================================
            # gauss_distance (D_t): KS 统计量 — 步间变化的非高斯性
            # ============================================================
            log_p = p.log()
            log_q = q.log()
            delta_l = log_q - log_p  # (vocab_size,)
            
            vocab_size = p.size(0)
            threshold = 1.0 / vocab_size
            active_mask = (p > threshold) | (q > threshold)
            delta_active = delta_l[active_mask]
            
            n_active = delta_active.numel()
            if n_active > 10:
                mu_da = delta_active.mean()
                sigma_da = delta_active.std()
                if sigma_da > eps:
                    z_std = (delta_active - mu_da) / sigma_da
                    z_sorted, _ = z_std.sort()
                    empirical_cdf = torch.arange(
                        1, n_active + 1,
                        dtype=torch.float32,
                        device=z_sorted.device
                    ) / n_active
                    normal_cdf = 0.5 * (1.0 + torch.erf(z_sorted / sqrt2))
                    ks_stat = (empirical_cdf - normal_cdf).abs().max().item()
                else:
                    ks_stat = 0.0
            else:
                ks_stat = 0.0
            
            # ============================================================
            # p_0 信息轨迹: JSD(p_t, p_0) 和 JSD(p_{t+1}, p_0)
            # d_t = JSD(p_t, p_0) 衡量步骤 t 相对初始状态的偏离程度
            # ============================================================
            # JSD(p, p_0)
            m_p0 = (p + p_0) / 2
            jsd_p0 = 0.5 * (p * (p.log() - m_p0.log())).sum().item() + \
                     0.5 * (p_0 * (log_p0 - m_p0.log())).sum().item()
            
            # JSD(q, p_0)
            m_q0 = (q + p_0) / 2
            jsd_q0 = 0.5 * (q * (q.log() - m_q0.log())).sum().item() + \
                     0.5 * (p_0 * (log_p0 - m_q0.log())).sum().item()
            
            # ============================================================
            # KL-谱集中度 (KL effective dimension → 归一化集中度)
            #
            # delta_i = q_i * log(q_i / p_i) — token i 对 KL 散度的贡献
            # w_i = |delta_i| / Σ|delta_j| — 归一化 KL 贡献权重
            #
            # KL 有效维度 kl_eff_dim = exp(H(w)) = exp(-Σ w_i log(w_i))
            #   小 → 变化集中在少数 token (有用)
            #   大 → 变化分散在长尾 (噪声)
            #
            # 归一化: c_t = 1 - log(kl_eff_dim)/log(V) ∈ [0,1]
            #   值1 → 全集中在 1 个 token, 值0 → 均匀分散到 V 个 token
            #
            # 相比 top-k 集中度的优势: top-100 在 150k 词表下几乎恒=1,
            # 而有效维度能精确刻画分散程度 (Cohen's d = -0.84 vs +0.34)
            # ============================================================
            kl_contrib = q * (q.log() - p.log())  # (vocab_size,)
            abs_kl_contrib = kl_contrib.abs()
            total_kl_contrib = abs_kl_contrib.sum().item()
            if total_kl_contrib > eps:
                import math
                w = abs_kl_contrib / total_kl_contrib  # 归一化权重
                w_pos = w[w > 1e-30]  # 只对非零权重计算
                kl_entropy = -float(torch.sum(w_pos * w_pos.log()))
                kl_eff_dim = math.exp(kl_entropy)
                vocab_size = p.size(0)
                kl_concentration = max(0.0, 1.0 - kl_entropy / math.log(vocab_size))
            else:
                kl_eff_dim = 0.0
                kl_concentration = 0.0
            
            # ============================================================
            # CC_t: Content Change (温度不变的内容变化率)
            #
            # log_ratio_i = log(P_{t+1,i} / P_t,i) = log(q_i) - log(p_i)
            # R^2 = Corr(log_ratio, log_P_t)^2
            # CC_t = 1 - R^2
            #
            # 温度缩放 logit' = c·logit 时:
            #   log_ratio = (c-1)·logit + const → 与 log_P 完美线性
            #   → R^2 = 1 → CC_t = 0 (零奖励, 数学免疫)
            #
            # 真实推理 logit' = c·logit + delta 时:
            #   delta 破坏线性 → R^2 < 1 → CC_t > 0
            #
            # CC_t ∈ [0, 1]:
            #   0 → 纯温度效应 (无内容变化)
            #   1 → 纯内容变化 (与温度完全无关)
            # ============================================================
            log_p_active = log_p[active_mask]
            if n_active > 10 and delta_active.var() > eps and log_p_active.var() > eps:
                cov_rl = ((delta_active - delta_active.mean()) * (log_p_active - log_p_active.mean())).mean()
                r_squared = (cov_rl ** 2) / (delta_active.var() * log_p_active.var() + eps)
                content_change = 1.0 - min(1.0, float(r_squared))
            else:
                content_change = 0.0  # 无足够数据 → 保守认为无内容变化
            
            transitions.append({
                'entropy_delta': dH,
                'step_entropy': step_mean_entropies[i],  # H_t: 步骤 i 的平均 token 熵 (EWD 权重)
                'js_divergence': max(0.0, jsd),  # 数值稳定
                'kl_forward': max(0.0, kl_forward),
                'kl_reverse': max(0.0, kl_reverse),
                'gauss_distance': ks_stat,  # D_t: 非高斯性 ∈ [0, 1]
                'jsd_from_p0': max(0.0, jsd_p0),       # d_t: JSD(p_t, p_0)
                'jsd_from_p0_next': max(0.0, jsd_q0),  # d_{t+1}: JSD(p_{t+1}, p_0)
                'kl_concentration': kl_concentration,   # c_t: 归一化 KL 集中度 ∈ [0, 1]
                'kl_eff_dim': kl_eff_dim,               # KL 有效维度 (越小越集中)
                'content_change': content_change,        # CC_t: 内容变化率 ∈ [0, 1] (温度不变)
            })
            
            # 及时释放中间变量
            del p, q, m, log_p, log_q, delta_l, delta_active, m_p0, m_q0
            del kl_contrib, abs_kl_contrib
        
        # 释放步骤分布和 p_0
        del step_distributions, p_0, log_p0
        
        return {"transitions": transitions, "step_features": step_features}

    @staticmethod
    def _compute_token_level_reward(token_entropies, valid_len, group_H_bar, group_H_std):
        """
        Phi(z_t) 熵门控 per-token 权重 — 用于加权正确性奖励的优势 A

        公式:
            z_t = (H_t - H_bar) / (sigma_H + eps)
            w_t = Phi(z_t) = 0.5 * (1 + erf(z_t / sqrt(2)))

        Per-Group 归一化:
            H_bar, sigma_H 由同一问题 (prompt) 的所有回答的 token 熵共同计算。
            效果: 高熵回答 (深度思考) → 大部分 w > 0.5 → 梯度放大
                  低熵回答 (机械化)   → 大部分 w < 0.5 → 梯度衰减
            与 GRPO group-based advantage 逻辑一致。

        Args:
            token_entropies: (seq_len,)    预计算的 token 熵
            valid_len: 有效 response 长度
            group_H_bar: float tensor — 组级平均熵 (来自同一 prompt 的所有回答, 必须提供)
            group_H_std: float tensor — 组级熵标准差 (必须提供)

        Returns:
            dict with:
                'token_weights': (T,) tensor — Phi(z_t) per-token 权重
                'gate_mean': float — mean(Phi) 门控通过率
                'H_bar': float — 组级熵阈值
                'mean_H': float — 本条回答的平均熵 (检测坍缩)
        """
        import torch, math
        eps = 1e-8

        T = int(valid_len)
        T = min(T, token_entropies.size(0))

        _empty = {'token_weights': None,
                  'gate_mean': 0.0, 'H_bar': 0.0,
                  'mean_H': 0.0}
        if T < 3:
            return _empty

        H = token_entropies[:T].float()
        if H.sum() < eps:
            return _empty

        # ── Phi(z_t): 高斯 CDF 熵门控 (Per-Group 归一化) ──
        z = (H - group_H_bar) / (group_H_std + eps)
        token_weights = 0.5 * (1.0 + torch.erf(z / math.sqrt(2)))

        return {
            'token_weights': token_weights.detach().cpu(),
            'gate_mean':    float(token_weights.mean().item()),
            'H_bar':        float(group_H_bar.item()) if isinstance(group_H_bar, torch.Tensor) else float(group_H_bar),
            'mean_H':       float(H.mean().item()),
        }

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        def _get_micro_batches(data: DataProto) -> Tuple[list, list | None]:
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            batch = data.select(batch_keys=select_keys).batch
            has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch

            if has_multi_modal_inputs:
                all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                if use_dynamic_bsz:
                    max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                    rearranged_text_micro_batches, textual_indices = rearrange_micro_batches(
                        batch=batch, max_token_len=max_token_len
                    )

                    final_micro_batches_list = []
                    for i, text_mb_td in enumerate(rearranged_text_micro_batches):
                        current_original_indices = textual_indices[i]
                        current_mm_inputs_list = [all_multi_modal_inputs_list[idx] for idx in current_original_indices]

                        mb_dict = {k: v for k, v in text_mb_td.items()}
                        mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                        final_micro_batches_list.append(mb_dict)
                    return final_micro_batches_list, textual_indices
                else:
                    num_micro_batches = batch.batch_size[0] // micro_batch_size
                    micro_batches_dp = data.chunk(num_micro_batches)
                    return micro_batches_dp, None
            elif use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
                return micro_batches, indices
            else:
                micro_batches = batch.split(micro_batch_size)
                return micro_batches, None

        micro_batches, indices = _get_micro_batches(data)

        log_probs_lst = []
        entropy_lst = []

        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                entropy, log_probs_mb = self._forward_micro_batch(
                    micro_batch, temperature=temperature,
                    calculate_entropy=calculate_entropy
                )

            log_probs_lst.append(log_probs_mb)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        step_efficiency_tensor = None

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]

        # NOTE: Phi(z) 熵门控在 ray_trainer.py 中计算（全量 batch），
        # 而非此处（DP worker 只看到部分数据，同一 prompt 的回答被拆散到不同 GPU）。
        return log_probs, entropys, step_efficiency_tensor, None, None

    def _micro_batch_forward_backward(self, data, temperature):
        """Forward pass + loss computation + backward for a single micro-batch.
        Used by update_policy; extracted to support SAM (two-pass gradient).
        """
        micro_batch_metrics = {}

        if isinstance(data, DataProto):
            data = {**data.batch.to(get_device_id()), **data.non_tensor_batch}
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(get_device_id())
                elif k == "multi_modal_inputs" and v is not None:
                    data[k] = [
                        {kk: vv.to(get_device_id()) for kk, vv in item_dict.items()} for item_dict in v
                    ]
                else:
                    data[k] = v
        else:
            data = data.to(get_device_id())

        response_mask = data["response_mask"]
        old_log_prob = data["old_log_probs"]
        advantages = data["advantages"]

        clip_ratio = self.config.clip_ratio
        clip_ratio_low = (
            self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
        )
        clip_ratio_high = (
            self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
        )
        clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
        entropy_coeff = self.config.entropy_coeff
        loss_agg_mode = self.config.loss_agg_mode

        _use_coherence = self.config.get("use_entropy_coherence", False)
        _coherence_lambda = self.config.get("entropy_coherence_lambda", 0.1)
        calculate_entropy = entropy_coeff != 0 or _use_coherence

        _use_surprisal = self.config.get("use_surprisal_redistribution", False)
        surprisal_config = None
        if _use_surprisal:
            surprisal_config = {
                "advantages": advantages,
            }

        entropy, log_prob = self._forward_micro_batch(
            micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy,
            surprisal_config=surprisal_config,
        )

        if self.config.policy_loss.loss_mode == "vanilla":
            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=advantages,
                response_mask=response_mask,
                cliprange=clip_ratio,
                cliprange_low=clip_ratio_low,
                cliprange_high=clip_ratio_high,
                clip_ratio_c=clip_ratio_c,
                loss_agg_mode=loss_agg_mode,
            )
        else:
            loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
            policy_loss_fn = get_policy_loss_fn(loss_mode)
            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                old_log_prob, log_prob, advantages, response_mask, loss_agg_mode, self.config
            )

        if entropy_coeff != 0:
            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
            policy_loss = pg_loss - entropy_loss * entropy_coeff
        else:
            policy_loss = pg_loss

        if self.config.use_kl_loss:
            ref_log_prob = data["ref_log_prob"]
            kld = kl_penalty(
                logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
            )

            kl_direction_flip = self.config.get("kl_direction_flip", False)
            if kl_direction_flip:
                sign_A = torch.sign(advantages)
                kl_fwd_frac = (sign_A > 0).float().mean().item()
                kl_rev_frac = (sign_A < 0).float().mean().item()
                kld_flipped = -sign_A * kld
                kl_loss = agg_loss(loss_mat=kld_flipped, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                micro_batch_metrics["actor/kl_fwd_frac"] = kl_fwd_frac
                micro_batch_metrics["actor/kl_rev_frac"] = kl_rev_frac
            else:
                kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
            micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
            micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

        if _use_coherence and entropy is not None and "h_bar_boxed" in data:
            h_target = data["h_bar_boxed"].unsqueeze(-1)  # (bs, 1)
            coherence_sq = (entropy - h_target) ** 2  # (bs, response_len)
            coherence_loss = agg_loss(
                loss_mat=coherence_sq, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
            )
            policy_loss = policy_loss + coherence_loss * _coherence_lambda
            micro_batch_metrics["actor/coherence_loss"] = coherence_loss.detach().item()
            micro_batch_metrics["actor/coherence_lambda"] = _coherence_lambda

        if self.config.use_dynamic_bsz:
            loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
        else:
            loss = policy_loss / self.gradient_accumulation
        loss.backward()

        micro_batch_metrics.update(
            {
                "actor/pg_loss": pg_loss.detach().item(),
                "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                "actor/ppo_kl": ppo_kl.detach().item(),
                "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
            }
        )

        return micro_batch_metrics, advantages.mean().item()

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if "entropys" in data.batch.keys():
            select_keys.append("entropys")
        has_st_weights = "st_token_weights" in data.batch.keys()
        if has_st_weights:
            select_keys.append("st_token_weights")
        if "h_bar_boxed" in data.batch.keys():
            select_keys.append("h_bar_boxed")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    micro_batches = []
                    if self.config.use_dynamic_bsz:
                        all_multi_modal_inputs_list = data.non_tensor_batch["multi_modal_inputs"]
                        batch_tensordict_for_rearrange = data.batch

                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        rearranged_text_micro_batches_tds, textual_indices = rearrange_micro_batches(
                            batch=batch_tensordict_for_rearrange, max_token_len=max_token_len
                        )

                        for current_original_indices, text_mb_td in zip(
                            textual_indices, rearranged_text_micro_batches_tds
                        ):
                            current_mm_inputs_list = [
                                all_multi_modal_inputs_list[idx] for idx in current_original_indices
                            ]
                            mb_dict = {k: v for k, v in text_mb_td.items()}
                            mb_dict["multi_modal_inputs"] = current_mm_inputs_list
                            micro_batches.append(mb_dict)
                    else:
                        self.gradient_accumulation = (
                            self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        )
                        num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                        micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                use_sam = self.config.get("use_sam", False)
                sam_rho = self.config.get("sam_rho", 0.05)

                _sam_adv_accum = []
                for data in micro_batches:
                    mb_metrics, adv_mean = self._micro_batch_forward_backward(data, temperature)
                    _sam_adv_accum.append(adv_mean)
                    if not use_sam:
                        append_to_dict(metrics, mb_metrics)

                if use_sam:
                    # --- Sign-Aware SAM ---
                    _mean_adv = sum(_sam_adv_accum) / max(len(_sam_adv_accum), 1)
                    sam_direction = 1.0

                    _sam_saved_grads = {}
                    _local_nsq = torch.tensor(0.0, device=get_device_id())
                    for _n, _p in self.actor_module.named_parameters():
                        if _p.grad is not None:
                            _sam_saved_grads[_n] = _p.grad.data.clone()
                            _local_nsq += _p.grad.data.norm(2) ** 2
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(_local_nsq, op=torch.distributed.ReduceOp.SUM)
                    _sam_gnorm = _local_nsq.sqrt().item() + 1e-12

                    with torch.no_grad():
                        for _n, _p in self.actor_module.named_parameters():
                            if _n in _sam_saved_grads:
                                _p.data.add_(sam_direction * sam_rho * _sam_saved_grads[_n] / _sam_gnorm)

                    # Phase 2: forward/backward at perturbed theta'
                    self.actor_optimizer.zero_grad()
                    for data in micro_batches:
                        mb_metrics, _ = self._micro_batch_forward_backward(data, temperature)
                        append_to_dict(metrics, mb_metrics)

                    with torch.no_grad():
                        for _n, _p in self.actor_module.named_parameters():
                            if _n in _sam_saved_grads:
                                _p.data.sub_(sam_direction * sam_rho * _sam_saved_grads[_n] / _sam_gnorm)

                    del _sam_saved_grads
                    append_to_dict(metrics, {
                        "actor/sam_direction": sam_direction,
                        "actor/sam_grad_norm_phase1": _sam_gnorm,
                        "actor/sam_rho": sam_rho,
                        "actor/sam_adv_mean": _mean_adv,
                    })

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
