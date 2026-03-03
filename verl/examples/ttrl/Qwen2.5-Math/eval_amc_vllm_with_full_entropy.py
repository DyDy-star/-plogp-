#!/usr/bin/env python3
"""
评估脚本（vLLM生成 + PyTorch完整词表熵）

功能：
1. 使用vLLM进行快速生成
2. 使用PyTorch模型计算完整词表的熵（与训练时一致）
3. 集成CoVo的process_thoughts进行步骤划分
4. 输出详细的步骤和熵信息到JSON

关键改进：
- 熵计算基于完整词表（~100k维），与训练时一致
- 使用与训练相同的 entropy_from_logits 函数
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import argparse
import math
import torch
import torch.nn.functional as F

# 显式禁用vLLM V1（在导入vLLM之前）
os.environ['VLLM_USE_V1'] = '0'

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from verl.utils.reward_score.ttrl_math import compute_score, extract_answer
from verl.utils.torch_functional import entropy_from_logits  # 与训练时相同的熵计算函数

# 导入vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("警告: vLLM未安装，请使用: pip install vllm")

from transformers import AutoTokenizer, AutoModelForCausalLM


# ========================================
# CoVo的步骤划分函数（process_thoughts）
# ========================================

def merge_colon_ended_elements(lst):
    """
    合并以冒号结尾的元素与下一个元素
    来源: CoVo/covo/openrlhf/trainer/ppo_utils/score.py (lines 33-52)
    """
    result1 = []
    i = 0
    while i < len(lst):
        # 如果当前元素以冒号结尾
        if lst[i].endswith(':'):
            # 将当前元素与下一个元素合并
            if i + 1 < len(lst):  # 确保有下一个元素
                merged_element = lst[i] + "\n" + lst[i + 1]
                result1.append(merged_element)
                i += 2  # 跳过当前元素和下一个元素
            else:
                # 如果没有下一个元素，直接添加当前元素
                result1.append(lst[i])
                i += 1
        else:
            # 如果当前元素不以冒号结尾，直接添加到结果列表
            result1.append(lst[i])
            i += 1
    return result1


def merge_steps(lst):
    """
    将步骤列表合并为固定的15步
    来源: CoVo/covo/openrlhf/trainer/ppo_utils/score.py (lines 54-65)
    """
    thred = len(lst) // 15
    result = []
    x = (thred + 1) * 15 - len(lst)
    y = len(lst) - 15 * thred
    split_list = [thred] * x + [thred+1] * y

    merge_idx = 0
    for item in split_list:
        result.append("\n".join(lst[merge_idx: merge_idx + item]))
        merge_idx += item
    return result


def process_thoughts(resp):
    """
    自然步骤划分函数 (取消 15 步上限限制)
    
    功能:
    1. 按行分割响应, 去除空行
    2. 合并包含 LaTeX 公式的多行 (检测 \\[ 和 \\])
    3. 保留自然步数, 不做强制合并
    """
    thoughts = [line.strip() for line in resp.split('\n') if line.strip()]
    result = []
    merge_mode = False
    temp_merge = []

    for item in thoughts:
        if "\\[" in item and "\\]" not in item:
            merge_mode = True
            temp_merge.append(item)
        elif "\\]" in item and "\\[" not in item:
            merge_mode = False
            temp_merge.append(item)
            if temp_merge:
                result.append("\n".join(temp_merge))
                temp_merge = []
        elif merge_mode:
            temp_merge.append(item)
        else:
            result.append(item)
    
    if len(temp_merge) > 0:
        result += temp_merge

    return result


# process_thoughts_natural 现在与 process_thoughts 完全一致 (向后兼容别名)
process_thoughts_natural = process_thoughts


# ========================================
# 完整词表熵计算（与训练时一致）
# ========================================

def compute_full_vocab_entropy_for_sequence(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str = "cuda",
    return_logits: bool = False
):
    """
    使用PyTorch模型计算完整词表的token级熵
    
    参数:
        model: PyTorch语言模型
        tokenizer: tokenizer
        input_ids: (seq_len,) 输入token IDs
        attention_mask: (seq_len,) attention mask
        device: 设备
        return_logits: 是否同时返回logits（用于分布指标计算）
    
    返回:
        如果return_logits=False: List[float] 每个生成token的熵值
        如果return_logits=True: (List[float], torch.Tensor) 熵值 + logits (seq_len, vocab_size)
    """
    model.eval()
    
    with torch.no_grad():
        # 确保在正确的设备上
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # 前向传播获取logits
        outputs = model(
            input_ids=input_ids.unsqueeze(0),  # (1, seq_len)
            attention_mask=attention_mask.unsqueeze(0),
        )
        logits = outputs.logits.squeeze(0)  # (seq_len, vocab_size)
        
        # 计算每个位置的熵（使用与训练时相同的函数）
        # entropy_from_logits: H = logsumexp(logits) - Σ(p * logits)
        entropies = entropy_from_logits(logits)  # (seq_len,)
        
        if return_logits:
            return entropies.cpu().tolist(), logits.cpu()
        return entropies.cpu().tolist()


# ========================================
# 步骤级分布指标计算
# ========================================

def compute_step_distribution_metrics(
    logits: torch.Tensor,
    step_analyses: List[Dict],
    prompt_len: int,
    response_token_ids_list: List[int],
) -> Dict:
    """
    基于logits计算步骤级分布指标和步骤间转换指标
    
    参数:
        logits: (seq_len, vocab_size) 完整序列的logits
        step_analyses: 已有的步骤分析列表（包含token偏移信息）
        prompt_len: prompt的token长度
        response_token_ids_list: response的token IDs
    
    返回:
        包含步骤级分布指标、步骤间转换和IB摘要的字典
    """
    n_steps = len(step_analyses)
    if n_steps == 0:
        return {'step_metrics': [], 'step_transitions': [], 'ib_summary': {}}
    
    # 提取response部分的logits: response第i个token的预测logits在位置 prompt_len-1+i
    resp_logits = logits[prompt_len-1:prompt_len-1+len(response_token_ids_list)]  # (resp_len, vocab_size)
    
    # 计算每个步骤的平均概率分布和分布指标
    step_avg_probs = []  # 每个步骤的平均概率分布
    step_metrics = []
    
    token_offset = 0
    for step_idx, step_info in enumerate(step_analyses):
        step_len = step_info['n_tokens']
        end_offset = min(token_offset + step_len, resp_logits.size(0))
        
        if end_offset <= token_offset:
            # 空步骤
            step_metrics.append({
                'top1_prob': 0.0,
                'top5_prob_mass': 0.0,
                'top10_prob_mass': 0.0,
                'eff_vocab_size': 1.0,
            })
            step_avg_probs.append(None)
            token_offset = end_offset
            continue
        
        # 该步骤所有token的logits -> softmax概率
        step_logits = resp_logits[token_offset:end_offset]  # (step_tokens, vocab_size)
        step_probs = F.softmax(step_logits, dim=-1)  # (step_tokens, vocab_size)
        
        # 步骤平均概率分布
        avg_prob = step_probs.mean(dim=0)  # (vocab_size,)
        step_avg_probs.append(avg_prob)
        
        # Top-K 概率质量
        sorted_probs, _ = avg_prob.sort(descending=True)
        top1_prob = float(sorted_probs[0])
        top5_prob_mass = float(sorted_probs[:5].sum())
        top10_prob_mass = float(sorted_probs[:10].sum())
        
        # 有效词表大小 (exponential of entropy of the average distribution)
        avg_entropy = -torch.sum(avg_prob * torch.log(avg_prob + 1e-10))
        eff_vocab_size = float(torch.exp(avg_entropy))
        
        step_metrics.append({
            'top1_prob': top1_prob,
            'top5_prob_mass': top5_prob_mass,
            'top10_prob_mass': top10_prob_mass,
            'eff_vocab_size': eff_vocab_size,
        })
        
        token_offset = end_offset
    
    # 计算步骤间转换指标
    step_transitions = []
    eps = 1e-10
    
    # ============================================================
    # p_0 锚点: 缓存初始分布，用于计算信息积累轨迹
    # p_0 = 第一个非 None 步骤的分布 (模型推理前的信念)
    # d_t = JSD(p_t, p_0) 衡量步骤 t 相对初始状态的偏离程度
    # ============================================================
    p_0 = None
    for sp in step_avg_probs:
        if sp is not None:
            p_0 = sp + eps
            break
    log_p0 = p_0.log() if p_0 is not None else None
    
    for i in range(n_steps - 1):
        p = step_avg_probs[i]
        q = step_avg_probs[i + 1]
        
        if p is None or q is None:
            null_kl = {f'kl_conc_top{k}': 0.0 for k in [5, 10, 20, 50]}
            null_kl['kl_eff_dim'] = 0.0
            null_kl['kl_concentration'] = 0.0
            step_transitions.append({
                'step_from': i,
                'step_to': i + 1,
                'kl_forward': 0.0,
                'kl_reverse': 0.0,
                'js_divergence': 0.0,
                'top10_overlap': 0.0,
                'cosine_sim': 0.0,
                'entropy_delta': 0.0,
                'jsd_from_p0': 0.0,
                'jsd_from_p0_next': 0.0,
                'content_change': 0.0,
                **null_kl,
            })
            continue
        
        p_safe = p + eps
        q_safe = q + eps
        
        # KL(P||Q) - forward KL
        kl_forward = float(torch.sum(p_safe * torch.log(p_safe / q_safe)))
        # KL(Q||P) - reverse KL
        kl_reverse = float(torch.sum(q_safe * torch.log(q_safe / p_safe)))
        
        # JS divergence = 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = (P+Q)/2
        m = 0.5 * (p_safe + q_safe)
        js_div = float(0.5 * torch.sum(p_safe * torch.log(p_safe / m)) + 
                       0.5 * torch.sum(q_safe * torch.log(q_safe / m)))
        
        # Top-10 overlap
        _, top10_p = p.topk(10)
        _, top10_q = q.topk(10)
        top10_p_set = set(top10_p.tolist())
        top10_q_set = set(top10_q.tolist())
        top10_overlap = len(top10_p_set & top10_q_set) / 10.0
        
        # Cosine similarity
        cosine_sim = float(F.cosine_similarity(p.unsqueeze(0), q.unsqueeze(0)))
        
        # Entropy delta
        entropy_delta = step_analyses[i+1]['mean_entropy'] - step_analyses[i]['mean_entropy']
        
        # ============================================================
        # p_0 信息轨迹: JSD(p_t, p_0) 和 JSD(p_{t+1}, p_0)
        # ============================================================
        jsd_p0 = 0.0
        jsd_q0 = 0.0
        if p_0 is not None:
            # JSD(p, p_0)
            m_p0 = 0.5 * (p_safe + p_0)
            jsd_p0 = float(0.5 * torch.sum(p_safe * torch.log(p_safe / m_p0)) +
                           0.5 * torch.sum(p_0 * (log_p0 - m_p0.log())))
            jsd_p0 = max(0.0, jsd_p0)
            
            # JSD(q, p_0)
            m_q0 = 0.5 * (q_safe + p_0)
            jsd_q0 = float(0.5 * torch.sum(q_safe * torch.log(q_safe / m_q0)) +
                           0.5 * torch.sum(p_0 * (log_p0 - m_q0.log())))
            jsd_q0 = max(0.0, jsd_q0)
        
        # ============================================================
        # KL-谱集中度 (多 k 值 + 有效维度)
        # delta_i = q_i * log(q_i / p_i) — token i 对 KL 散度的贡献
        # w_i = |delta_i| / Σ|delta_j| — 归一化 KL 贡献权重
        #
        # 1) top-k 集中度: c_t(k) = Σ_{top-k} |delta_i| / Σ_all |delta_j|
        #    k 越小, 区分度越高 (k=100 在 150k 词表下几乎恒=1)
        #
        # 2) KL 有效维度: kl_eff_dim = exp(-Σ w_i * log(w_i))
        #    类似 effective vocab size, 但衡量 KL 变化的分散度
        #    kl_eff_dim 小 → 变化集中 (有用), 大 → 变化分散 (噪声)
        #    归一化: kl_concentration = 1 - log(kl_eff_dim) / log(V)
        # ============================================================
        kl_contrib = q_safe * (q_safe.log() - p_safe.log())  # (vocab_size,)
        abs_kl_contrib = kl_contrib.abs()
        total_kl_c = abs_kl_contrib.sum().item()
        
        kl_conc_dict = {}
        if total_kl_c > eps:
            vocab_size = p_safe.size(0)
            
            # top-k 集中度 (多个 k 值)
            for k_val in [5, 10, 20, 50]:
                k_actual = min(k_val, vocab_size)
                topk_c = abs_kl_contrib.topk(k_actual).values.sum().item()
                kl_conc_dict[f'kl_conc_top{k_val}'] = topk_c / total_kl_c
            
            # KL 有效维度: exp(entropy of normalized |kl_contrib|)
            w = abs_kl_contrib / total_kl_c  # 归一化权重
            # 只对非零权重计算熵 (避免 0*log(0))
            w_pos = w[w > 1e-30]
            kl_entropy = -float(torch.sum(w_pos * w_pos.log()))
            kl_eff_dim = float(np.exp(kl_entropy))
            
            # 归一化集中度: 1 - log(eff_dim)/log(V) ∈ [0,1]
            # kl_eff_dim=1 → 全集中=1, kl_eff_dim=V → 全分散=0
            kl_conc_dict['kl_eff_dim'] = kl_eff_dim
            kl_conc_dict['kl_concentration'] = max(0.0, 1.0 - kl_entropy / np.log(vocab_size))
        else:
            for k_val in [5, 10, 20, 50]:
                kl_conc_dict[f'kl_conc_top{k_val}'] = 0.0
            kl_conc_dict['kl_eff_dim'] = 0.0
            kl_conc_dict['kl_concentration'] = 0.0
        
        # CC_t: Content Change (温度不变的内容变化率)
        # log_ratio = log(q/p), R^2 = Corr(log_ratio, log_p)^2, CC = 1 - R^2
        # 温度缩放 → R^2=1 → CC=0 (数学免疫)
        log_p_t = p_safe.log()
        log_q_t = q_safe.log()
        log_ratio = log_q_t - log_p_t
        vocab_sz = p_safe.size(0)
        threshold_cc = 1.0 / vocab_sz
        active_cc = (p_safe > threshold_cc) | (q_safe > threshold_cc)
        lr_active = log_ratio[active_cc]
        lp_active = log_p_t[active_cc]
        n_act = lr_active.numel()
        if n_act > 10 and lr_active.var() > eps and lp_active.var() > eps:
            cov_rl = ((lr_active - lr_active.mean()) * (lp_active - lp_active.mean())).mean()
            r_sq = (cov_rl ** 2) / (lr_active.var() * lp_active.var() + eps)
            content_change = 1.0 - min(1.0, float(r_sq))
        else:
            content_change = 0.0
        
        step_transitions.append({
            'step_from': i,
            'step_to': i + 1,
            'kl_forward': kl_forward,
            'kl_reverse': kl_reverse,
            'js_divergence': js_div,
            'top10_overlap': top10_overlap,
            'cosine_sim': cosine_sim,
            'entropy_delta': entropy_delta,
            'jsd_from_p0': jsd_p0,
            'jsd_from_p0_next': jsd_q0,
            'content_change': content_change,
            **kl_conc_dict,
        })
    
    # 计算IB摘要
    # 根据熵变化趋势划分exploration（熵增/探索）和compression（熵减/压缩）步骤
    exploration_transitions = [t for t in step_transitions if t['entropy_delta'] > 0]
    compression_transitions = [t for t in step_transitions if t['entropy_delta'] <= 0]
    
    ib_summary = {}
    
    # Exploration (熵增) 统计
    if exploration_transitions:
        ib_summary['exploration_disruption_js'] = float(np.mean([t['js_divergence'] for t in exploration_transitions]))
        ib_summary['exploration_disruption_kl'] = float(np.mean([t['kl_forward'] for t in exploration_transitions]))
        ib_summary['exploration_top10_overlap'] = float(np.mean([t['top10_overlap'] for t in exploration_transitions]))
        ib_summary['exploration_kl_concentration'] = float(np.mean([t['kl_concentration'] for t in exploration_transitions]))
        ib_summary['exploration_kl_eff_dim'] = float(np.mean([t['kl_eff_dim'] for t in exploration_transitions]))
        for k_val in [5, 10, 20, 50]:
            ib_summary[f'exploration_kl_conc_top{k_val}'] = float(np.mean([t[f'kl_conc_top{k_val}'] for t in exploration_transitions]))
        ib_summary['n_exploration_steps'] = len(exploration_transitions)
    else:
        ib_summary['exploration_disruption_js'] = 0.0
        ib_summary['exploration_disruption_kl'] = 0.0
        ib_summary['exploration_top10_overlap'] = 0.0
        ib_summary['exploration_kl_concentration'] = 0.0
        ib_summary['exploration_kl_eff_dim'] = 0.0
        for k_val in [5, 10, 20, 50]:
            ib_summary[f'exploration_kl_conc_top{k_val}'] = 0.0
        ib_summary['n_exploration_steps'] = 0
    
    # Compression (熵减) 统计
    if compression_transitions:
        ib_summary['compression_loss_js'] = float(np.mean([t['js_divergence'] for t in compression_transitions]))
        ib_summary['compression_loss_kl'] = float(np.mean([t['kl_forward'] for t in compression_transitions]))
        ib_summary['compression_top10_overlap'] = float(np.mean([t['top10_overlap'] for t in compression_transitions]))
        ib_summary['compression_kl_concentration'] = float(np.mean([t['kl_concentration'] for t in compression_transitions]))
        ib_summary['compression_kl_eff_dim'] = float(np.mean([t['kl_eff_dim'] for t in compression_transitions]))
        for k_val in [5, 10, 20, 50]:
            ib_summary[f'compression_kl_conc_top{k_val}'] = float(np.mean([t[f'kl_conc_top{k_val}'] for t in compression_transitions]))
        ib_summary['n_compression_steps'] = len(compression_transitions)
    else:
        ib_summary['compression_loss_js'] = 0.0
        ib_summary['compression_loss_kl'] = 0.0
        ib_summary['compression_top10_overlap'] = 0.0
        ib_summary['compression_kl_concentration'] = 0.0
        ib_summary['compression_kl_eff_dim'] = 0.0
        for k_val in [5, 10, 20, 50]:
            ib_summary[f'compression_kl_conc_top{k_val}'] = 0.0
        ib_summary['n_compression_steps'] = 0
    
    # 全局统计
    if step_transitions:
        ib_summary['mean_js_divergence'] = float(np.mean([t['js_divergence'] for t in step_transitions]))
        ib_summary['total_js_path_length'] = float(np.sum([t['js_divergence'] for t in step_transitions]))
        ib_summary['mean_top10_overlap'] = float(np.mean([t['top10_overlap'] for t in step_transitions]))
        ib_summary['mean_cosine_similarity'] = float(np.mean([t['cosine_sim'] for t in step_transitions]))
        ib_summary['mean_kl_concentration'] = float(np.mean([t['kl_concentration'] for t in step_transitions]))
        ib_summary['mean_kl_eff_dim'] = float(np.mean([t['kl_eff_dim'] for t in step_transitions]))
        for k_val in [5, 10, 20, 50]:
            ib_summary[f'mean_kl_conc_top{k_val}'] = float(np.mean([t[f'kl_conc_top{k_val}'] for t in step_transitions]))
    else:
        ib_summary['mean_js_divergence'] = 0.0
        ib_summary['total_js_path_length'] = 0.0
        ib_summary['mean_top10_overlap'] = 0.0
        ib_summary['mean_cosine_similarity'] = 0.0
        ib_summary['mean_kl_concentration'] = 0.0
        ib_summary['mean_kl_eff_dim'] = 0.0
        for k_val in [5, 10, 20, 50]:
            ib_summary[f'mean_kl_conc_top{k_val}'] = 0.0
    
    # A+B 结合指标: compute_combined_reward
    # 使用 step_transitions 中的 kl_concentration + jsd_from_p0 计算
    try:
        from verl.utils.reward_score.ttrl_math import (
            compute_combined_reward,
            compute_entropy_efficiency_from_transitions,
            compute_trajectory_progress_score,
        )
        combined = compute_combined_reward(step_transitions)
        ib_summary['combined_reward'] = combined.get('reward', 0.0)
        ib_summary['combined_eta_explore'] = combined.get('eta_explore', 0.0)
        ib_summary['combined_eta_compress'] = combined.get('eta_compress', 0.0)
        ib_summary['combined_mode'] = combined.get('mode', 'unknown')
        
        eff = compute_entropy_efficiency_from_transitions(step_transitions)
        ib_summary['entropy_efficiency_reward'] = eff.get('reward', 0.0)
        ib_summary['entropy_efficiency_eta_up'] = eff.get('eta_up', 0.0)
        ib_summary['entropy_efficiency_eta_down'] = eff.get('eta_down', 0.0)
        
        traj = compute_trajectory_progress_score(step_transitions)
        ib_summary['trajectory_progress_reward'] = traj.get('reward', 0.0)
        ib_summary['trajectory_eta_diverge'] = traj.get('eta_diverge', 0.0)
        ib_summary['trajectory_eta_converge'] = traj.get('eta_converge', 0.0)
    except ImportError:
        pass
    
    return {
        'step_metrics': step_metrics,
        'step_transitions': step_transitions,
        'ib_summary': ib_summary,
    }


# ========================================
# vLLM模型加载
# ========================================

def load_vllm_model(model_path: str, tensor_parallel_size: int = 1):
    """使用vLLM加载模型（仅用于生成）"""
    if not VLLM_AVAILABLE:
        raise RuntimeError("vLLM未安装，请使用: pip install vllm")
    
    print(f"正在使用vLLM加载模型（用于生成）: {model_path}")
    print(f"配置: tensor_parallel={tensor_parallel_size}, gpu_memory_utilization=0.5")
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.5,  # 降低以为PyTorch模型留空间
    )
    
    return llm


def load_pytorch_model(model_path: str, device: str = "cuda"):
    """加载PyTorch模型（用于计算完整词表熵）"""
    print(f"\n正在加载PyTorch模型（用于熵计算）: {model_path}")
    print(f"设备: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    
    print("✅ PyTorch模型加载完成")
    return model, tokenizer


# ========================================
# 生成响应（使用vLLM）
# ========================================

def generate_responses_vllm(
    llm: LLM,
    tokenizer,
    prompts: List[str],
    n_samples: int = 16,
    temperature: float = 1,
    top_p: float = 0.95,
    max_tokens: int = 3072,
) -> List[List[Dict]]:
    """
    使用vLLM生成响应（仅获取token_ids）
    
    返回:
        List[List[Dict]]: 每个prompt的多个响应，每个响应包含text和token_ids
    """
    # 应用chat template
    full_prompts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_prompts.append(formatted)
    
    # 配置采样参数（不请求logprobs以节省内存）
    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    
    # 批量生成
    print(f"正在为 {len(prompts)} 个问题生成响应...")
    outputs = llm.generate(full_prompts, sampling_params)
    
    # 组织结果
    all_responses = []
    for output in outputs:
        responses = []
        for o in output.outputs:
            responses.append({
                'text': o.text,
                'token_ids': o.token_ids,
            })
        all_responses.append(responses)
    
    return all_responses


# ========================================
# 分析响应的步骤和熵（使用完整词表）
# ========================================

def analyze_response_entropy_full_vocab(
    response_text: str,
    prompt_text: str,
    response_token_ids: List[int],
    pytorch_model,
    tokenizer,
    device: str = "cuda",
    natural_steps: bool = False
) -> Dict:
    """
    分析响应的步骤划分和token熵（使用完整词表）
    
    参数:
        response_text: 完整的响应文本
        prompt_text: 完整的prompt文本（包含chat template）
        response_token_ids: 响应的token IDs
        pytorch_model: PyTorch模型
        tokenizer: tokenizer
        device: 设备
        natural_steps: 如果为True，使用自然步骤划分（不强制15步）
    
    返回:
        包含步骤划分和熵信息的字典
    """
    # 1. 构建完整的输入序列（prompt + response）
    # 注意：vLLM返回的token_ids只是response部分，我们需要加上prompt
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    # 确保response_token_ids是列表（vLLM可能返回元组）
    response_token_ids_list = list(response_token_ids)
    full_input_ids = prompt_ids + response_token_ids_list
    attention_mask = [1] * len(full_input_ids)
    
    # 2. 使用PyTorch模型计算完整词表的熵（同时获取logits用于分布指标）
    input_ids_tensor = torch.tensor(full_input_ids, dtype=torch.long)
    attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
    
    full_entropies, full_logits = compute_full_vocab_entropy_for_sequence(
        pytorch_model,
        tokenizer,
        input_ids_tensor,
        attention_mask_tensor,
        device=device,
        return_logits=True
    )
    
    # 3. 提取response部分的熵（跳过prompt部分和最后一个位置）
    # 熵的索引对应的是"预测下一个token时的熵"
    # 所以response的第一个token对应的熵在 prompt_len-1 位置
    prompt_len = len(prompt_ids)
    response_entropies = full_entropies[prompt_len-1:prompt_len-1+len(response_token_ids_list)]
    
    # 4. 使用CoVo的process_thoughts进行步骤划分
    if natural_steps:
        steps = process_thoughts_natural(response_text)
    else:
        steps = process_thoughts(response_text)
    
    # 5. 为每个步骤分配token和熵
    step_analyses = []
    token_offset = 0
    
    for step_idx, step_text in enumerate(steps):
        # 对步骤文本进行tokenization以确定token数量
        step_tokens = tokenizer.encode(step_text, add_special_tokens=False)
        step_length = len(step_tokens)
        
        # 获取该步骤对应的token和熵
        end_offset = min(token_offset + step_length, len(response_token_ids_list))
        step_token_ids = response_token_ids_list[token_offset:end_offset]
        step_token_entropies = response_entropies[token_offset:end_offset]
        
        # 解码tokens
        step_token_texts = [tokenizer.decode([tid]) for tid in step_token_ids]
        
        # 计算步骤级别的统计
        if step_token_entropies:
            mean_entropy = np.mean(step_token_entropies)
            std_entropy = np.std(step_token_entropies)
            min_entropy = np.min(step_token_entropies)
            max_entropy = np.max(step_token_entropies)
        else:
            mean_entropy = std_entropy = min_entropy = max_entropy = 0.0
        
        step_analyses.append({
            'step_index': step_idx,
            'step_text': step_text,
            'n_tokens': len(step_token_ids),
            'tokens': step_token_texts,
            'token_entropies': step_token_entropies,
            'mean_entropy': float(mean_entropy),
            'std_entropy': float(std_entropy),
            'min_entropy': float(min_entropy),
            'max_entropy': float(max_entropy),
        })
        
        token_offset = end_offset
    
    # 6. 计算整体统计
    all_entropies = [e for e in response_entropies if e is not None]
    
    # 计算前12个步骤和后3个步骤的平均熵（与奖励函数一致）
    if len(step_analyses) > 0:
        n_early_steps = min(12, len(step_analyses))
        early_step_entropies = [s['mean_entropy'] for s in step_analyses[:n_early_steps]]
        early_avg_entropy = np.mean(early_step_entropies) if early_step_entropies else 0.0
        
        n_late_steps = min(3, len(step_analyses))
        late_step_entropies = [s['mean_entropy'] for s in step_analyses[-n_late_steps:]]
        late_avg_entropy = np.mean(late_step_entropies) if late_step_entropies else 0.0
        
        entropy_reward = float(early_avg_entropy - late_avg_entropy)
    else:
        early_avg_entropy = late_avg_entropy = entropy_reward = 0.0
    
    overall_stats = {
        'n_steps': len(steps),
        'total_tokens': len(all_entropies),
        'overall_mean_entropy': float(np.mean(all_entropies)) if all_entropies else 0.0,
        'overall_std_entropy': float(np.std(all_entropies)) if all_entropies else 0.0,
        'overall_min_entropy': float(np.min(all_entropies)) if all_entropies else 0.0,
        'overall_max_entropy': float(np.max(all_entropies)) if all_entropies else 0.0,
        # 与奖励函数一致的指标
        'early_avg_entropy': float(early_avg_entropy),
        'late_avg_entropy': float(late_avg_entropy),
        'entropy_reward': entropy_reward,
    }
    
    # 7. 计算步骤级分布指标（KL, JS, Top-K overlap, cosine similarity等）
    try:
        dist_metrics = compute_step_distribution_metrics(
            full_logits, step_analyses, prompt_len, response_token_ids_list
        )
        
        # 将分布指标合并到每个步骤
        for i, sm in enumerate(dist_metrics['step_metrics']):
            if i < len(step_analyses):
                step_analyses[i].update(sm)
        
        # 添加IB摘要到overall_stats
        overall_stats['ib_summary'] = dist_metrics['ib_summary']
        
        step_transitions = dist_metrics['step_transitions']
    except Exception as e:
        print(f"  警告: 分布指标计算失败: {e}")
        step_transitions = []
        overall_stats['ib_summary'] = {}
    
    # 释放logits内存
    del full_logits
    
    return {
        'overall_stats': overall_stats,
        'steps': step_analyses,
        'step_transitions': step_transitions,
    }


# ========================================
# 评估函数（带完整词表熵分析）
# ========================================

def evaluate_pass_at_1_with_full_entropy(
    vllm_model: LLM,
    pytorch_model,
    tokenizer,
    test_data: List[Dict],
    n_samples: int = 16,
    temperature: float = 1,
    top_p: float = 0.95,
    max_tokens: int = 3072,
    batch_size: int = 4,
    output_file: str = None,
    device: str = "cuda",
    natural_steps: bool = False,
) -> Dict:
    """评估pass@1并进行步骤划分和完整词表熵分析"""
    
    results = []
    pass_at_1_scores = []
    
    print(f"\n开始评估 {len(test_data)} 个测试样本...")
    print(f"每个样本生成 {n_samples} 个响应")
    print(f"批次大小: {batch_size}")
    print(f"参数: temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}")
    print(f"步骤划分: {'自然步数（不限15步）' if natural_steps else '强制15步'}")
    print(f"熵计算: 完整词表（与训练时一致）\n")
    
    # 批量处理
    for batch_start in tqdm(range(0, len(test_data), batch_size), desc="批次进度"):
        batch_end = min(batch_start + batch_size, len(test_data))
        batch_items = test_data[batch_start:batch_end]
        
        # 准备批次的prompts
        batch_prompts = [item["prompt"] for item in batch_items]
        
        # 批量生成响应（使用vLLM）
        batch_responses = generate_responses_vllm(
            vllm_model, tokenizer, batch_prompts, n_samples, temperature, top_p, max_tokens
        )
        
        # 评估每个样本
        for item, responses in zip(batch_items, batch_responses):
            ground_truth = item["answer"]
            item_id = item.get("id", "")
            
            # 获取完整的prompt（带chat template）
            messages = [{"role": "user", "content": item["prompt"]}]
            full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # 评估每个响应
            correct_count = 0
            response_results = []
            
            for i, resp_data in enumerate(tqdm(responses, desc=f"样本 {item_id} - 熵计算", leave=False)):
                response_text = resp_data['text']
                response_token_ids = resp_data['token_ids']
                
                # 计算正确性
                score_dict = compute_score(response_text, ground_truth, fast=True)
                is_correct = score_dict.get("acc", False)
                
                if is_correct:
                    correct_count += 1
                
                # 分析步骤和熵（使用完整词表）
                try:
                    entropy_analysis = analyze_response_entropy_full_vocab(
                        response_text,
                        full_prompt,
                        response_token_ids,
                        pytorch_model,
                        tokenizer,
                        device=device,
                        natural_steps=natural_steps
                    )
                except Exception as e:
                    print(f"\n警告: 熵计算失败 (样本 {item_id}, 响应 {i}): {e}")
                    entropy_analysis = {
                        'overall_stats': {},
                        'steps': [],
                        'error': str(e)
                    }
                
                response_results.append({
                    "response_id": i,
                    "response": response_text,
                    "extracted_answer": score_dict.get("pred", ""),
                    "is_correct": is_correct,
                    "entropy_analysis": entropy_analysis,
                })
            
            # 计算pass@1
            pass_at_1 = correct_count / n_samples
            pass_at_1_scores.append(pass_at_1)
            
            results.append({
                "id": item_id,
                "prompt": item["prompt"],
                "ground_truth": ground_truth,
                "n_samples": n_samples,
                "correct_count": correct_count,
                "pass@1": pass_at_1,
                "responses": response_results,
            })
    
    # 计算整体pass@1
    overall_pass_at_1 = np.mean(pass_at_1_scores)
    pass_at_1_std = np.std(pass_at_1_scores)
    total_correct = sum(r["correct_count"] for r in results)
    total_responses = len(results) * n_samples
    
    # 保存详细结果
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "overall_pass@1": overall_pass_at_1,
                "pass@1_std": pass_at_1_std,
                "n_test_samples": len(test_data),
                "n_responses_per_sample": n_samples,
                "total_correct_responses": total_correct,
                "total_responses": total_responses,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "entropy_type": "full_vocab",  # 标记使用完整词表
                "step_mode": "natural" if natural_steps else "forced_15",
                "results": results,
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n详细结果已保存到: {output_file}")
    
    # 打印统计信息
    print("\n" + "=" * 70)
    print("评估结果")
    print("=" * 70)
    print(f"测试样本数: {len(test_data)}")
    print(f"每个样本的响应数: {n_samples}")
    print(f"总响应数: {total_responses}")
    print(f"正确响应数: {total_correct}")
    print(f"正确率: {total_correct/total_responses:.4f}")
    print(f"\nOverall Pass@1: {overall_pass_at_1:.4f}")
    print(f"Pass@1 标准差: {pass_at_1_std:.4f}")
    print(f"\n熵计算方式: 完整词表（与训练时一致）")
    print("=" * 70)
    
    return {
        "overall_pass@1": overall_pass_at_1,
        "results": results,
    }


# ========================================
# 主函数
# ========================================

def main():
    parser = argparse.ArgumentParser(description="评估AIME/AMC（带完整词表熵分析）")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--test_data_path", type=str, required=True, help="测试数据路径")
    parser.add_argument("--output_file", type=str, required=True, help="输出文件路径")
    parser.add_argument("--n_samples", type=int, default=16, help="每个问题生成的响应数量")
    parser.add_argument("--temperature", type=float, default=1, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p采样参数")
    parser.add_argument("--max_tokens", type=int, default=3072, help="最大生成token数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="vLLM张量并行大小")
    parser.add_argument("--max_samples", type=int, default=None, help="限制评估样本数")
    parser.add_argument("--device", type=str, default="cuda", help="PyTorch模型设备")
    parser.add_argument("--natural_steps", action="store_true",
                        help="使用自然步骤划分（不强制合并到15步），保留原始推理步骤结构")
    
    args = parser.parse_args()
    
    # 加载测试数据
    print(f"正在加载测试数据: {args.test_data_path}")
    
    if args.test_data_path.endswith('.parquet'):
        import pandas as pd
        df = pd.read_parquet(args.test_data_path)
        
        test_data = []
        for idx, row in df.iterrows():
            if isinstance(row['prompt'], (list, np.ndarray)):
                prompt = row['prompt'][0]['content'] if len(row['prompt']) > 0 else ""
            else:
                prompt = row['prompt']
            
            ground_truth = row['reward_model']['ground_truth']
            
            test_data.append({
                "prompt": prompt,
                "answer": ground_truth,
                "id": row['id']
            })
    else:
        with open(args.test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    
    # 限制样本数
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    
    print(f"加载了 {len(test_data)} 个测试样本")
    
    # 加载vLLM模型（用于生成）
    print("\n" + "="*70)
    print("步骤 1/2: 加载vLLM模型（用于快速生成）")
    print("="*70)
    vllm_model = load_vllm_model(args.model_path, args.tensor_parallel_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 加载PyTorch模型（用于计算完整词表熵）
    print("\n" + "="*70)
    print("步骤 2/2: 加载PyTorch模型（用于完整词表熵计算）")
    print("="*70)
    pytorch_model, _ = load_pytorch_model(args.model_path, device=args.device)
    
    # 评估
    print("\n" + "="*70)
    print("开始评估")
    print("="*70)
    if args.natural_steps:
        print("\n⚠ 使用自然步骤划分（不强制15步）")
        print("  步骤数将取决于模型生成的自然段落结构\n")
    
    evaluate_pass_at_1_with_full_entropy(
        vllm_model,
        pytorch_model,
        tokenizer,
        test_data,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        output_file=args.output_file,
        device=args.device,
        natural_steps=args.natural_steps,
    )


if __name__ == "__main__":
    main()

