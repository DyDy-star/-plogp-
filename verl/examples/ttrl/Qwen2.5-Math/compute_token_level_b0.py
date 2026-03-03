#!/usr/bin/env python3
"""
Token 级 B0 计算 & 评估

在整个序列上逐 token 计算双向对称 KL 效率：
  对每对 (P_t, P_{t+1})：
    - KL_fwd = KL(P_t || P_{t+1})
    - KL_rev = KL(P_{t+1} || P_t)
    - H_t, H_{t+1} = 对应位置的 entropy
    - 若 H_{t+1} > H_t (熵增): constructive = KL_rev, destructive = KL_fwd
    - 若 H_{t+1} <= H_t (熵减): constructive = KL_fwd, destructive = KL_rev
    - q_t = (constructive - destructive) / (constructive + destructive)
  
  R = mean(q_t) over all token transitions

同时计算多种聚合变体，与共识度对比。
"""

import json, os, sys, time
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE_DIR, "eval_results_aime_full_entropy")
EVAL_JSON = os.path.join(EVAL_DIR, "aime_eval_full_entropy_20260206_131034.json")

MODELS = {
    "Trained": "/data/user5/TTRL begin/verl/examples/ttrl/Qwen2.5-Math/checkpoints/TTRL-verl/AIME-TTT-Qwen2.5-Math-1.5B/0130/TTRL-Len@3k-grpo-175656/global_step_240/actor/huggingface_merged",
    "Base": "/data/user5/models/Qwen2.5-Math-1.5B",
}

DEVICE = "cuda:0"  # Set CUDA_VISIBLE_DEVICES=3 externally


def compute_token_level_b0(logits_seq):
    """
    给定一个序列的 logits [seq_len, vocab_size]，
    计算每对连续 token 的对称 B0。
    
    返回:
        q_values: list of q_t values
        entropies: list of H_t values  
        kl_fwds: list of KL(P_t || P_{t+1})
        kl_revs: list of KL(P_{t+1} || P_t)
        entropy_deltas: list of H_{t+1} - H_t
    """
    eps = 1e-10
    seq_len = logits_seq.size(0)
    
    if seq_len < 2:
        return [], [], [], [], []
    
    # 一次性计算所有位置的概率分布和 log 概率
    log_probs = F.log_softmax(logits_seq.float(), dim=-1)  # [seq_len, vocab]
    probs = log_probs.exp()  # [seq_len, vocab]
    
    # 计算每个位置的 entropy
    entropies = -(probs * log_probs).sum(dim=-1)  # [seq_len]
    
    # 计算相邻位置的 KL 散度 (向量化)
    # KL(P_t || P_{t+1}) = sum(P_t * (log P_t - log P_{t+1}))
    p = probs[:-1]       # [seq_len-1, vocab]
    q = probs[1:]        # [seq_len-1, vocab]
    log_p = log_probs[:-1]
    log_q = log_probs[1:]
    
    kl_fwd = (p * (log_p - log_q)).sum(dim=-1)  # [seq_len-1]
    kl_rev = (q * (log_q - log_p)).sum(dim=-1)  # [seq_len-1]
    
    # 数值安全：clamp to >= 0
    kl_fwd = kl_fwd.clamp(min=0)
    kl_rev = kl_rev.clamp(min=0)
    
    # entropy delta
    entropy_deltas = entropies[1:] - entropies[:-1]  # [seq_len-1]
    
    # 计算 q_t
    # 熵增: constructive = kl_rev, destructive = kl_fwd
    # 熵减: constructive = kl_fwd, destructive = kl_rev
    entropy_increasing = entropy_deltas > 0
    
    constructive = torch.where(entropy_increasing, kl_rev, kl_fwd)
    destructive = torch.where(entropy_increasing, kl_fwd, kl_rev)
    
    denom = constructive + destructive
    q_values = torch.where(
        denom > eps,
        (constructive - destructive) / denom,
        torch.zeros_like(denom)
    )
    
    return (
        q_values.cpu().tolist(),
        entropies.cpu().tolist(),
        kl_fwd.cpu().tolist(),
        kl_rev.cpu().tolist(),
        entropy_deltas.cpu().tolist(),
    )


def process_model(model_path, model_name, eval_data):
    """对一个模型处理所有样本"""
    print(f"\n{'='*70}")
    print(f"  Loading {model_name}: {model_path}")
    print(f"{'='*70}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(DEVICE).eval()
    
    samples = []
    total = sum(len(r['responses']) for r in eval_data['results'])
    count = 0
    t0 = time.time()
    
    for result in eval_data['results']:
        qid = result['id']
        prompt_text = result['prompt']
        gt = result['ground_truth']
        
        for resp in result['responses']:
            count += 1
            response_text = resp['response']
            is_correct = resp.get('is_correct', False)
            extracted_answer = resp.get('extracted_answer')
            
            # 构造完整输入
            messages = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": response_text},
            ]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(DEVICE)
            input_ids = inputs['input_ids']
            
            # 获取 prompt 长度（不含 response 的部分）
            messages_prompt = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": prompt_text},
            ]
            prompt_only = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer(prompt_only, return_tensors="pt", truncation=True, max_length=4096)['input_ids']
            prompt_len = prompt_ids.size(1)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[0]  # [seq_len, vocab_size]
            
            # 只取 response 部分的 logits
            # logits[t] 预测的是 position t+1 的 token
            # response tokens 从 prompt_len 开始
            # 所以预测 response 的 logits 从 prompt_len-1 开始
            response_logits = logits[prompt_len - 1: input_ids.size(1) - 1]  # 预测 response token 的分布
            
            if response_logits.size(0) < 2:
                samples.append({
                    'qid': qid, 'is_correct': is_correct,
                    'extracted_answer': extracted_answer,
                    'token_b0_mean': 0.0, 'n_tokens': 0,
                })
                continue
            
            # 计算 token 级 B0
            q_values, entropies, kl_fwds, kl_revs, entropy_deltas = \
                compute_token_level_b0(response_logits)
            
            n_tokens = len(q_values)
            q_arr = np.array(q_values)
            ent_arr = np.array(entropies[:len(q_values)])  # entropies has one more
            kl_fwd_arr = np.array(kl_fwds)
            kl_rev_arr = np.array(kl_revs)
            ed_arr = np.array(entropy_deltas)
            
            # 各种聚合方式
            sample = {
                'qid': qid,
                'is_correct': is_correct,
                'extracted_answer': extracted_answer,
                'n_tokens': n_tokens,
                # Token-level B0 聚合
                'token_b0_mean': float(np.mean(q_arr)),
                'token_b0_median': float(np.median(q_arr)),
                'token_b0_std': float(np.std(q_arr)),
                'token_b0_frac_pos': float(np.mean(q_arr > 0)),
                'token_b0_max': float(np.max(q_arr)),
                'token_b0_min': float(np.min(q_arr)),
                'token_b0_skewness': float(np.mean(((q_arr - np.mean(q_arr)) / (np.std(q_arr) + 1e-10)) ** 3)),
                # 只看前 1/3 和后 1/3
                'token_b0_first_third': float(np.mean(q_arr[:n_tokens//3])) if n_tokens >= 3 else 0,
                'token_b0_last_third': float(np.mean(q_arr[-n_tokens//3:])) if n_tokens >= 3 else 0,
                # 只看熵增 token 和熵减 token
                'token_b0_entropy_up': float(np.mean(q_arr[ed_arr > 0])) if np.sum(ed_arr > 0) > 0 else 0,
                'token_b0_entropy_down': float(np.mean(q_arr[ed_arr <= 0])) if np.sum(ed_arr <= 0) > 0 else 0,
                'frac_entropy_up': float(np.mean(ed_arr > 0)),
                # KL 统计
                'mean_kl_fwd': float(np.mean(kl_fwd_arr)),
                'mean_kl_rev': float(np.mean(kl_rev_arr)),
                'mean_entropy': float(np.mean(entropies)),
                # 高不确定性 token 处的 B0
                'token_b0_high_entropy': float(np.mean(q_arr[ent_arr > np.median(ent_arr)])) if len(ent_arr) > 0 else 0,
                'token_b0_low_entropy': float(np.mean(q_arr[ent_arr <= np.median(ent_arr)])) if len(ent_arr) > 0 else 0,
                # 不求均值，用percentile
                'token_b0_p10': float(np.percentile(q_arr, 10)),
                'token_b0_p90': float(np.percentile(q_arr, 90)),
            }
            
            samples.append(sample)
            
            if count % 50 == 0 or count == total:
                elapsed = time.time() - t0
                rate = count / elapsed
                eta = (total - count) / rate if rate > 0 else 0
                print(f"  [{count}/{total}] {rate:.1f} samples/s, ETA {eta:.0f}s | "
                      f"b0_mean={sample['token_b0_mean']:.4f}, n_tok={n_tokens}")
    
    # 清理 GPU
    del model
    torch.cuda.empty_cache()
    
    return samples


def evaluate_all(samples, model_name):
    """评估所有 token-level B0 变体"""
    print(f"\n{'#'*70}")
    print(f"  {model_name} — Token-Level B0 Evaluation")
    print(f"{'#'*70}")
    
    # 添加 consensus
    by_qid = defaultdict(list)
    for s in samples:
        by_qid[s['qid']].append(s)
    
    for qid, group in by_qid.items():
        answers = [s['extracted_answer'] for s in group]
        counter = Counter(a for a in answers if a is not None and a != '')
        gs = len(group)
        for s in group:
            a = s['extracted_answer']
            if a and a in counter:
                s['consensus'] = counter[a] / gs
            else:
                s['consensus'] = 1.0 / gs
    
    # 全局统计
    print(f"\n  === 全局统计 ===")
    for key in ['token_b0_mean', 'token_b0_std', 'token_b0_frac_pos', 'mean_kl_fwd', 'mean_kl_rev', 'mean_entropy']:
        vals = [s[key] for s in samples]
        vc = [s[key] for s in samples if s['is_correct']]
        vi = [s[key] for s in samples if not s['is_correct']]
        mc, mi = np.mean(vc) if vc else 0, np.mean(vi) if vi else 0
        print(f"  {key:<25} all={np.mean(vals):.4f}±{np.std(vals):.4f}  "
              f"C={mc:.4f} I={mi:.4f} Δ={mc-mi:+.4f}")
    
    # 所有变体
    variant_keys = [
        'token_b0_mean', 'token_b0_median', 'token_b0_std', 'token_b0_frac_pos',
        'token_b0_max', 'token_b0_min', 'token_b0_skewness',
        'token_b0_first_third', 'token_b0_last_third',
        'token_b0_entropy_up', 'token_b0_entropy_down',
        'token_b0_high_entropy', 'token_b0_low_entropy',
        'token_b0_p10', 'token_b0_p90',
        'frac_entropy_up', 'n_tokens',
        'consensus',
    ]
    
    # GRPO 组级评估
    print(f"\n  === GRPO 组级评估 ===")
    results = {}
    
    for v in variant_keys:
        stds, enrichments, gaps = [], [], []
        for qid, group in by_qid.items():
            if len(group) < 4:
                continue
            vals = np.array([s[v] for s in group])
            corr = np.array([s['is_correct'] for s in group], dtype=bool)
            
            std_v = float(np.std(vals))
            stds.append(std_v)
            
            nc = int(np.sum(corr))
            if 0 < nc < len(corr) and std_v > 1e-8:
                med = np.median(vals)
                top = vals >= med
                overall = np.mean(corr.astype(float))
                top_rate = np.mean(corr[top].astype(float))
                if overall > 0:
                    enrichments.append(top_rate / overall)
                
                advs = (vals - np.mean(vals)) / std_v
                ac = float(np.mean(advs[corr]))
                ai = float(np.mean(advs[~corr]))
                gaps.append(ac - ai)
        
        results[v] = {
            'std': np.mean(stds) if stds else 0,
            'enrich': np.mean(enrichments) if enrichments else 1.0,
            'gap': np.mean(gaps) if gaps else 0,
        }
    
    sorted_variants = sorted(variant_keys, key=lambda v: -results[v]['gap'])
    
    print(f"\n  {'Rank':<5} {'变体':<27} {'组内std':>9} {'enrichment':>11} {'adv_gap':>10}")
    print(f"  {'-'*65}")
    for rank, v in enumerate(sorted_variants, 1):
        r = results[v]
        marker = ' ★' if v == 'consensus' else ''
        print(f"  {rank:<5} {v:<27} {r['std']:>9.4f} {r['enrich']:>10.3f}x {r['gap']:>+10.4f}{marker}")
    
    return results


def plot_results(all_results, all_samples, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Token-Level B0 Analysis (Per-Token KL Efficiency)', fontsize=14, fontweight='bold')
    
    for idx, (model_name, (samples, results)) in enumerate(all_results.items()):
        # Plot 1: Advantage Gap ranking
        ax = axes[idx, 0]
        variant_keys = sorted(results.keys(), key=lambda v: -results[v]['gap'])
        names = [v.replace('token_b0_', 'tb0_') for v in variant_keys]
        gaps = [results[v]['gap'] for v in variant_keys]
        colors = ['#f39c12' if v == 'consensus' else '#2ecc71' if results[v]['gap'] > 0 else '#e74c3c'
                  for v in variant_keys]
        
        bars = ax.barh(range(len(variant_keys)), gaps, color=colors, alpha=0.8)
        ax.set_yticks(range(len(variant_keys)))
        ax.set_yticklabels(names, fontsize=7)
        for b, g in zip(bars, gaps):
            ax.text(max(b.get_width(), 0) + 0.01, b.get_y() + b.get_height()/2,
                    f'{g:+.4f}', va='center', fontsize=7)
        ax.set_xlabel('Advantage Gap')
        ax.set_title(f'{model_name}: Advantage Gap Ranking\n(orange=consensus, green=positive, red=negative)')
        ax.axvline(0, color='gray', ls='--', alpha=0.5)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        
        # Plot 2: B0 distribution (correct vs incorrect)
        ax = axes[idx, 1]
        b0_correct = [s['token_b0_mean'] for s in samples if s['is_correct']]
        b0_incorrect = [s['token_b0_mean'] for s in samples if not s['is_correct']]
        
        bins = np.linspace(
            min(s['token_b0_mean'] for s in samples) - 0.01,
            max(s['token_b0_mean'] for s in samples) + 0.01,
            40
        )
        ax.hist(b0_correct, bins, alpha=0.6, color='#2ecc71', label=f'Correct (n={len(b0_correct)})', density=True)
        ax.hist(b0_incorrect, bins, alpha=0.6, color='#e74c3c', label=f'Incorrect (n={len(b0_incorrect)})', density=True)
        ax.axvline(np.mean(b0_correct), color='#27ae60', ls='--', lw=2,
                   label=f'μ_c={np.mean(b0_correct):.4f}')
        ax.axvline(np.mean(b0_incorrect), color='#c0392b', ls='--', lw=2,
                   label=f'μ_i={np.mean(b0_incorrect):.4f}')
        ax.set_xlabel('Token-Level B0 (mean)')
        ax.set_ylabel('Density')
        ax.set_title(f'{model_name}: Token-Level B0 Distribution')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved: {output_path}")
    plt.close()


def main():
    with open(EVAL_JSON) as f:
        eval_data = json.load(f)
    
    all_results = {}
    all_samples_dict = {}
    
    for model_name, model_path in MODELS.items():
        samples = process_model(model_path, model_name, eval_data)
        results = evaluate_all(samples, model_name)
        all_results[model_name] = (samples, results)
        all_samples_dict[model_name] = samples
    
    # 对比输出
    print(f"\n{'#'*70}")
    print(f"  Step-Level B0 vs Token-Level B0 对比")
    print(f"{'#'*70}")
    print(f"\n  Step-Level B0 (from previous analysis):")
    print(f"    Trained: gap = +0.2107 (强制 15 步)")
    print(f"\n  Token-Level B0 (this analysis):")
    for model_name, (samples, results) in all_results.items():
        tb0 = results.get('token_b0_mean', {})
        cons = results.get('consensus', {})
        print(f"    {model_name}:")
        print(f"      token_b0_mean:  gap = {tb0.get('gap', 0):+.4f}, std = {tb0.get('std', 0):.4f}")
        print(f"      consensus:      gap = {cons.get('gap', 0):+.4f}, std = {cons.get('std', 0):.4f}")
    
    output_path = os.path.join(EVAL_DIR, "token_level_b0_computed.png")
    plot_results(all_results, all_samples_dict, output_path)
    
    # 保存中间结果
    save_path = os.path.join(EVAL_DIR, "token_level_b0_results.json")
    save_data = {}
    for model_name, (samples, results) in all_results.items():
        save_data[model_name] = {
            'samples': samples,
            'results': {k: v for k, v in results.items()},
        }
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"  Results saved: {save_path}")


if __name__ == '__main__':
    main()
