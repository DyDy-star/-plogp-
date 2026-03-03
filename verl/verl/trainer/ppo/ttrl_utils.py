# Copyright 2025 TTRL Team (https://arxiv.org/abs/2504.16084)
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
from typing import List
from collections import Counter
import torch
import numpy as np
import random
from verl.utils.reward_score.ttrl_math import extract_answer, simplify_expression_string, grade, process_thoughts

def select_top_k_per_prompt(data, n_votes_per_prompt, n_samples_per_prompt):
    """
    Select the first k rollouts per prompt, used for TTRL downsampling.
    """
    assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
    num_prompts = len(data) // n_votes_per_prompt

    selected_indices = []
    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        selected_indices.extend(range(start, start + n_samples_per_prompt))

    return data[selected_indices]


def filter_and_sample_by_reasoning_steps(
    data, 
    n_votes_per_prompt, 
    n_samples_per_prompt, 
    target_reasoning_steps, 
    tokenizer,
    random_seed=42
):
    """
    过滤并采样：只选择恰好包含指定推理步骤数的样本
    
    工作流程：
    1. 对每个prompt的所有样本（64个）进行推理步骤计数
    2. 筛选出恰好有target_reasoning_steps个推理步骤的样本
    3. 如果符合条件的样本≥32个，随机选择32个
    4. 如果符合条件的样本<32个，只使用这些符合条件的样本（不补充其他样本）
    
    Args:
        data: DataProto，包含所有生成的样本
        n_votes_per_prompt: 每个prompt生成的总样本数（例如：64）
        n_samples_per_prompt: 期望选择的样本数（例如：32）
        target_reasoning_steps: 目标推理步骤数（例如：15）
        tokenizer: 用于解码response的tokenizer
        random_seed: 随机种子，用于可复现的随机采样
    
    Returns:
        DataProto: 过滤并采样后的数据（注意：总样本数可能小于 num_prompts * n_samples_per_prompt）
    """
    assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
    num_prompts = len(data) // n_votes_per_prompt
    
    selected_indices = []
    filter_stats = {
        "total_samples": 0,
        "filtered_samples": 0,
        "prompts_with_insufficient_samples": 0,
        "prompts_with_zero_samples": 0,
        "all_reasoning_steps": [],
        "selected_per_prompt": []
    }
    
    # 设置随机种子以确保可复现性
    rng = random.Random(random_seed)
    
    for prompt_idx in range(num_prompts):
        start_idx = prompt_idx * n_votes_per_prompt
        end_idx = start_idx + n_votes_per_prompt
        
        # 收集这个prompt的所有有效样本索引
        valid_indices = []
        
        for sample_idx in range(start_idx, end_idx):
            data_item = data[sample_idx]
            
            # 解码response
            response_ids = data_item.batch["responses"]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            
            # 根据attention_mask获取有效的response长度
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            # 使用process_thoughts计算推理步骤数
            reasoning_steps = process_thoughts(response_str)
            num_steps = len(reasoning_steps)
            
            filter_stats["all_reasoning_steps"].append(num_steps)
            filter_stats["total_samples"] += 1
            
            # 筛选：只保留恰好有target_reasoning_steps个步骤的样本
            if num_steps == target_reasoning_steps:
                valid_indices.append(sample_idx)
                filter_stats["filtered_samples"] += 1
        
        # 从有效样本中随机采样（严格模式：只用符合条件的样本）
        if len(valid_indices) >= n_samples_per_prompt:
            # 情况1：过滤后的样本充足，随机选择n_samples_per_prompt个
            sampled_indices = rng.sample(valid_indices, n_samples_per_prompt)
            filter_stats["selected_per_prompt"].append(n_samples_per_prompt)
            
        elif len(valid_indices) > 0:
            # 情况2：过滤后样本不足，只使用所有符合条件的样本
            sampled_indices = valid_indices.copy()
            filter_stats["selected_per_prompt"].append(len(valid_indices))
            filter_stats["prompts_with_insufficient_samples"] += 1
            
        else:
            # 情况3：没有符合条件的样本，跳过这个prompt
            sampled_indices = []
            filter_stats["selected_per_prompt"].append(0)
            filter_stats["prompts_with_zero_samples"] += 1
        
        selected_indices.extend(sampled_indices)
    
    # 详细的过滤统计信息
    total_expected = num_prompts * n_samples_per_prompt
    print(f"[FILTER] ========== 推理步骤过滤统计 ==========")
    print(f"[FILTER] 配置: {num_prompts} prompts × {n_votes_per_prompt} samples/prompt = {len(data)} 总样本")
    print(f"[FILTER] 目标: 每个prompt筛选{target_reasoning_steps}步的样本, 然后选择最多{n_samples_per_prompt}个")
    print(f"[FILTER] 每个prompt筛选结果 (符合条件/选择): {list(zip([sum(1 for s in filter_stats['all_reasoning_steps'][i*n_votes_per_prompt:(i+1)*n_votes_per_prompt] if s == target_reasoning_steps) for i in range(num_prompts)], filter_stats['selected_per_prompt']))}")
    print(f"[FILTER] 结果: {len(selected_indices)}/{total_expected} 样本被选择 ({len(selected_indices)/total_expected*100:.1f}%)")
    print(f"[FILTER] 不足样本的prompt数: {filter_stats['prompts_with_insufficient_samples']}, 零样本的prompt数: {filter_stats['prompts_with_zero_samples']}")
    print(f"[FILTER] ==========================================")
    
    # 返回过滤和采样后的数据
    return data[selected_indices]


# === I-Filtered Voting ===


def apply_i_filtered_voting(batch, entropys, response_masks, tokenizer, i_filter_ratio=0.5):
    """Legacy: I-filter for voting only. Kept for backward compatibility."""
    uids = batch.non_tensor_batch["uid"]
    unique_uids = []
    seen = set()
    for uid in uids:
        if uid not in seen:
            seen.add(uid)
            unique_uids.append(uid)

    prompt_length = batch.batch["prompts"].shape[1]
    total_kept, total_all = 0, 0
    per_entry_ratio = np.zeros(len(batch), dtype=float)

    for uid in unique_uids:
        indices = np.where(uids == uid)[0]
        n = len(indices)

        i_means = np.full(n, float('inf'))
        for j, idx in enumerate(indices):
            H = entropys[idx].float()
            rmask = response_masks[idx].float()
            alive = rmask > 0
            if alive.sum().item() < 3:
                continue
            Ha = H[alive]
            nt = len(Ha)
            I_ema = torch.empty(nt, device=Ha.device)
            I_ema[0] = Ha[0]
            for t in range(1, nt):
                I_ema[t] = 0.3 * Ha[t] + 0.7 * I_ema[t - 1]
            i_means[j] = float(I_ema.mean())

        k = max(1, int(n * i_filter_ratio))
        keep_order = np.argsort(i_means)[:k]
        keep_indices = indices[keep_order]
        total_kept += len(keep_indices)
        total_all += n

        model_outputs = []
        for idx in keep_indices:
            resp_ids = batch.batch["responses"][idx]
            attn = batch.batch["attention_mask"][idx]
            vlen = int(attn[prompt_length:].sum().item())
            text = tokenizer.decode(resp_ids[:vlen], skip_special_tokens=True)
            model_outputs.append(text)

        if model_outputs:
            majority_gt, majority_ratio = _majority_vote(model_outputs)
        else:
            majority_gt, majority_ratio = "None", 0.0

        for idx in indices:
            batch.non_tensor_batch["reward_model"][idx]["ground_truth"] = majority_gt
            batch.non_tensor_batch["reward_model"][idx]["majority_gt"] = majority_gt
            per_entry_ratio[idx] = majority_ratio

    batch.non_tensor_batch["majority_ratio_list"] = per_entry_ratio
    print(f"[I-filter] Kept {total_kept}/{total_all} ({100*total_kept/max(total_all,1):.0f}%) "
          f"responses for voting across {len(unique_uids)} prompts")
    return batch


def i_select_by_inertia(batch, n_keep):
    """
    I-based sample selection: for each prompt group (by uid),
    select n_keep responses with lowest I_mean (most confident).

    Flow: generate N → vote on all N → compute_log_prob on all N → I-select K here.
    Replaces select_top_k_per_prompt with entropy-informed selection.

    Returns filtered batch (only selected entries).
    """
    uids = batch.non_tensor_batch["uid"]
    entropys = batch.batch["entropys"]
    response_masks = batch.batch["response_mask"]

    unique_uids = []
    seen = set()
    for uid in uids:
        if uid not in seen:
            seen.add(uid)
            unique_uids.append(uid)

    selected_indices = []
    i_sel_vals, i_rej_vals = [], []

    for uid in unique_uids:
        indices = np.where(uids == uid)[0]
        n = len(indices)

        i_means = np.full(n, float('inf'))
        for j, idx in enumerate(indices):
            H = entropys[idx].float()
            rmask = response_masks[idx].float()
            alive = rmask > 0
            if alive.sum().item() < 3:
                continue
            Ha = H[alive]
            nt = len(Ha)
            I_ema = torch.empty(nt, device=Ha.device)
            I_ema[0] = Ha[0]
            for t in range(1, nt):
                I_ema[t] = 0.3 * Ha[t] + 0.7 * I_ema[t - 1]
            i_means[j] = float(I_ema.mean())

        k = min(n_keep, n)
        order = np.argsort(i_means)
        keep = order[:k]
        reject = order[k:]

        selected_indices.extend(indices[keep].tolist())

        valid_keep = i_means[keep][i_means[keep] < float('inf')]
        valid_rej = i_means[reject][i_means[reject] < float('inf')] if len(reject) > 0 else np.array([])
        if len(valid_keep) > 0:
            i_sel_vals.append(float(valid_keep.mean()))
        if len(valid_rej) > 0:
            i_rej_vals.append(float(valid_rej.mean()))

    selected_indices.sort()
    batch = batch[selected_indices]

    mean_sel = np.mean(i_sel_vals) if i_sel_vals else 0
    mean_rej = np.mean(i_rej_vals) if i_rej_vals else 0
    print(f"[I-select] {len(selected_indices)} responses kept "
          f"({n_keep}/prompt × {len(unique_uids)} prompts) "
          f"I_mean: kept={mean_sel:.3f} dropped={mean_rej:.3f} gap={mean_rej-mean_sel:.3f}")

    return batch


# === Ground Truth Manipulation ===


def apply_original_gt(batch):
    """
    Apply the original ground truth to the batch.
    """
    for i in range(len(batch)):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["original_gt"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt

    return batch


def apply_ttrl_gt(batch, gen_batch_output, n, tokenizer):
    """
    Apply the majority vote ground truth to the batch.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"

    model_outputs = []  
    for i in range(num_prompts):
        start = i * n
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            model_outputs.append(response_str)

    majority_gt_list, majority_ratio_list = _batch_majority_vote(model_outputs, n)
    
    assert len(batch) == len(majority_gt_list), "batch length must be equal to the number of model outputs"
    
    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = majority_gt_list[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

    batch.non_tensor_batch["majority_ratio_list"] = np.array(majority_ratio_list, dtype=float)
    return batch


def _batch_majority_vote(model_outputs: List[str], n: int) -> tuple[List[str], List[float]]:
    """
    Used to generate the ground truth for TTRL.
    Input:
        model_outputs: list of str
        n: int
    Output:
        majority_gt_list: list of str
        majority_ratio_list: list of float
    """
    majority_gt_list = []
    majority_ratio_list = []
    assert len(model_outputs) % n == 0
    n_prompts = len(model_outputs) // n
    for i in range(n_prompts):
        prompt_outputs = model_outputs[i * n:(i + 1) * n]
        prompt_majority_gt, prompt_majority_ratio = _majority_vote(prompt_outputs)
        majority_gt_list.append(prompt_majority_gt)
        majority_ratio_list.append(prompt_majority_ratio)
        
    return majority_gt_list, majority_ratio_list


def _majority_vote(model_outputs: List[str]) -> tuple[str, float]:
    assert len(model_outputs) > 0
    model_answers = [extract_answer(generated_text) for generated_text in model_outputs]
    model_answers = [answer for answer in model_answers if answer is not None]
    model_answers = [simplify_expression_string(answer) for answer in model_answers]
    if len(model_answers) == 0:
        return "None", 0.0
    
    counter = Counter(model_answers)
    
    majority_answer, majority_count = counter.most_common(1)[0]
    majority_ratio = majority_count / len(model_outputs)
    
    return majority_answer, majority_ratio


# === Metrics Computation ===


def compute_ttrl_metrics(batch, n):
    """
    Compute the TTRL metrics.
    """
    assert len(batch) % n == 0, "batch length must be divisible by n"
    num_prompts = len(batch) // n

    # Sort the batch by the ID
    idx = sorted(range(len(batch)), key=lambda x: batch[x].non_tensor_batch["extra_info"]["index"])

    majority_reward = []
    gt_reward = []
    majority_label = []
    gt_label = []

    for i in range(len(batch)):
        data_item = batch[idx[i]]
        majority_reward.append(data_item.batch["token_level_scores"].sum().item())
        gt_reward.append(data_item.batch["token_level_scores_original"].sum().item())
        majority_label.append(data_item.non_tensor_batch["reward_model"]["majority_gt"])
        gt_label.append(data_item.non_tensor_batch["reward_model"]["original_gt"]) 

    ttrl_metrics = _batch_compute_ttrl_metrics(majority_reward, gt_reward, majority_label, gt_label, n=n)
    majority_ratio_list = batch.non_tensor_batch["majority_ratio_list"]
    majority_ratio = sum(majority_ratio_list) / len(majority_ratio_list)
    ttrl_metrics["majority_ratio"] = majority_ratio

    return ttrl_metrics


def _batch_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: List[str],
    gt_label: List[str],
    n: int,
):
    """
    Compute the TTRL metrics for batch inputs.
    """
    assert len(majority_reward) == len(gt_reward) == len(majority_label) == len(gt_label)
    assert len(majority_reward) % n == 0
    n_prompts = len(majority_reward) // n
    ttrl_metrics = []
    for i in range(n_prompts):
        prompt_majority_reward = majority_reward[i * n:(i + 1) * n]
        prompt_gt_reward = gt_reward[i * n:(i + 1) * n]
        prompt_majority_label = majority_label[i * n:(i + 1) * n]
        prompt_gt_label = gt_label[i * n:(i + 1) * n]

        assert Counter(prompt_majority_label).most_common(1)[0][1] == n
        assert Counter(prompt_gt_label).most_common(1)[0][1] == n

        prompt_majority_label = prompt_majority_label[0]
        prompt_gt_label = prompt_gt_label[0]

        ttrl_metric = _prompt_compute_ttrl_metrics(prompt_majority_reward, prompt_gt_reward, prompt_majority_label, prompt_gt_label)
        ttrl_metrics.append(ttrl_metric)

    # Compute the average metrics
    ttrl_metrics = {k: sum(d[k] for d in ttrl_metrics) / len(ttrl_metrics) for k in ttrl_metrics[0]}

    return ttrl_metrics

def _prompt_compute_ttrl_metrics(
    majority_reward: List[float],
    gt_reward: List[float],
    majority_label: str,
    gt_label: str,
    ):    
    assert len(majority_reward) == len(gt_reward)

    hit_rate = 1.0 if grade(majority_label, gt_label) else 0.0    
    rewards_hit_rate = 0
    for estimate_reward, true_reward in zip(majority_reward, gt_reward):
        if estimate_reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(majority_reward)
    
    ttrl_metric = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "majority_voting_reward": sum(majority_reward) / len(majority_reward),
        "ground_truth_reward": sum(gt_reward) / len(gt_reward),
        f"pass@{len(majority_reward)}": 1.0 if sum(gt_reward) >= 1 else 0.0,
    }
    return ttrl_metric