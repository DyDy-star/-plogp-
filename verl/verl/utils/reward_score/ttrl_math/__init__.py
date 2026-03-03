# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except Exception in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides a math answer grading function with high recall.
Based on HF math_verify, verl, open reasoner zero, etc.
"""

from latex2sympy2_extended import latex2sympy
from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr
import traceback

from .math_utils import extract_boxed_answer, is_latex_equal, grade_answer_mathd, grade_answer_sympy, timeout_ours

import signal

"""
This code is adapted from Entropy Machanism Recipe (https://github.com/volcengine/verl/tree/main/recipe/entropy/).
"""

# ========================================
# 安全的 compute_score 包装器
# 防止 math_verify / SIGALRM 超时信号导致崩溃
# ========================================
_SAFE_SCORE_DEFAULT = {"score": 0.0, "format_score": 0.0, "acc": False, "extracted_gt": "", "pred": ""}

def safe_compute_score(model_response, gt_answer, fast=False):
    """
    对 compute_score 的安全包装:
    捕获 TimeoutError, math_verify.TimeoutException, 以及任何未预期的异常。
    失败时返回默认的零分结果，不会导致训练崩溃。
    
    额外保护: 清理残留的 SIGALRM 信号，防止信号在垃圾回收等上下文中误触发。
    """
    try:
        result = compute_score(model_response, gt_answer, fast=fast)
        return result
    except TimeoutError:
        # 来自 timeout_ours 装饰器
        pass
    except Exception as e:
        # 捕获 math_verify.errors.TimeoutException 及其他所有异常
        # 不直接 import math_verify.errors 以避免引入额外依赖问题
        e_name = type(e).__name__
        if e_name not in ("TimeoutException",):
            # 非超时异常，打印警告以便排查
            print(f"[safe_compute_score WARNING] Unexpected error ({e_name}): {e}")
    finally:
        # 关键: 清理可能残留的 SIGALRM，防止信号在 GC 等上下文误触发
        try:
            signal.alarm(0)
        except Exception:
            pass
    
    return dict(_SAFE_SCORE_DEFAULT, extracted_gt=str(gt_answer))

def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None


def grade(model_answer: str, gt_answer: str, fast: bool = True):
    if "\\boxed" in gt_answer:
        gt_answer = extract_answer(gt_answer)
    correct = grade_answer_mathd(model_answer, gt_answer) or grade_answer_sympy(model_answer, gt_answer)
    if not fast:
        # This mode further uses math_verify to recall originally false positives.
        # Will be a bit slower, and sensitive to bad inputs.
        correct = correct or is_latex_equal(
            model_answer,
            gt_answer,
        )
    return correct

@timeout_ours(timeout_seconds=10)
def simplify_expression_string(expression_string: str) -> str:
    try:
        sympy_expr = parse_expr(expression_string, transformations="all", evaluate=False)
        simplified_expr = simplify(sympy_expr)
        return str(simplified_expr)
    except TimeoutError:
        return expression_string
    except Exception as e:
        try:
            sympy_expr = latex2sympy(expression_string)
            simplified_expr = simplify(sympy_expr)
            return str(simplified_expr)
        except TimeoutError:
            return expression_string
        except Exception as e:
            return expression_string

def compute_score(model_response, gt_answer, fast=False):
    model_answer = extract_answer(model_response)

    if model_answer is None:
        return {
            "score": 0.0,
            "format_score": 0.0,
            "acc": False,
            "extracted_gt": gt_answer,
            "pred": "",
        }
        # return 0.0, 0.0  # Cannot even parse anything.
    is_correct = False
    if isinstance(gt_answer, float) or isinstance(gt_answer, int):
        gt_answer = str(gt_answer)
    if isinstance(gt_answer, str):
        is_correct = grade(model_answer, gt_answer, fast)
    elif isinstance(gt_answer, list):
        is_correct = False
        for gt in gt_answer:
            is_correct |= grade(model_answer, gt, fast)
    if is_correct:
        return {
            "score": 1.0,
            "format_score": 1.0,
            "acc": True,
            "extracted_gt": gt_answer,
            "pred": model_answer,
        }
    else:
        return {
            "score": 0.0,
            "format_score": 1.0,
            "acc": False,
            "extracted_gt": gt_answer,
            "pred": model_answer,
        }

def compute_step_entropies_aligned_with_process_thoughts(solution_str, token_ids, token_entropies, tokenizer, return_eff_vocab=False):
    """
    精准对齐版本：计算每个推理步骤的平均熵（与 process_thoughts 步骤划分完全一致）。
    
    核心方法（CoVo 风格的字符-token 映射）：
    1. 使用 process_thoughts(solution_str) 划分步骤（与筛选逻辑一致）
    2. 逐 token 累积解码，建立字符位置到 token 索引的精确映射
    3. 在解码后的文本中搜索每个步骤，映射到对应的 token 范围
    4. 确保所有 token 都被覆盖（第一步从 0 开始，最后一步到 n_tokens 结束）
    
    这种方法解决了换行符 token 检测不准确的问题（某些 tokenizer 如 Qwen 
    会将换行符与前面的字符合并成一个 token）。
    
    Args:
        solution_str: 模型响应文本，用于步骤划分
        token_ids: 生成的 token 序列
        token_entropies: 每个 token 位置的熵值
        tokenizer: tokenizer
        return_eff_vocab: 是否同时返回每步的有效词表 V_t = mean(2^h_i)
    
    Returns:
        return_eff_vocab=False: 每个推理步骤的平均熵列表
        return_eff_vocab=True:  (step_entropies, step_eff_vocabs) 元组
    
    Raises:
        AssertionError: 如果 token_ids 和 token_entropies 长度不一致
    """
    import numpy as np
    import torch
    
    # 转换为 list 和 numpy array
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    if isinstance(token_entropies, torch.Tensor):
        token_entropies = token_entropies.cpu().numpy()
    elif not isinstance(token_entropies, np.ndarray):
        token_entropies = np.array(token_entropies)
    
    # 严格检查长度一致
    assert len(token_ids) == len(token_entropies), \
        f"token_ids 长度 ({len(token_ids)}) 与 token_entropies 长度 ({len(token_entropies)}) 不一致"
    
    n_tokens = len(token_ids)
    _empty = ([], []) if return_eff_vocab else []
    
    if n_tokens == 0:
        return _empty
    
    # ========================================
    # Step 1: 使用 CoVo 方法精确对齐 process_thoughts 步骤与 token
    # ========================================
    # 核心思路：
    # 1. 使用 process_thoughts 划分步骤（与筛选逻辑一致）
    # 2. 通过逐 token 累积解码，建立字符位置到 token 的精确映射
    # 3. 在原始文本中找到每个步骤的位置，映射到 token 范围
    
    # 使用 process_thoughts 划分步骤
    steps = process_thoughts(solution_str)
    n_steps = len(steps)
    
    if n_steps == 0:
        return _empty
    
    # 逐 token 累积解码，建立字符到 token 的映射
    # char_to_token[i] = 字符 i 对应的 token 索引
    cumulative_text = ""
    char_to_token = []
    
    for token_idx, tid in enumerate(token_ids):
        decoded = tokenizer.decode([tid], skip_special_tokens=True)
        for _ in decoded:
            char_to_token.append(token_idx)
        cumulative_text += decoded
    
    # ========================================
    # Step 2: 找到每个步骤结束位置对应的 token 索引
    # ========================================
    # 策略：连续不重叠的 token 范围分配
    # - 找到每个步骤结束时的字符位置，映射到 token 索引
    # - 每个步骤范围 = [prev_token_end, current_token_end)
    # - 确保不重叠、不遗漏
    
    step_token_boundaries = [0]  # 每个步骤的结束 token 索引，从 0 开始
    search_start = 0
    
    for step_idx, step_text in enumerate(steps):
        step_text_stripped = step_text.strip()
        
        # 尝试找到该步骤在累积文本中的结束位置
        pos = cumulative_text.find(step_text_stripped, search_start)
        
        if pos == -1:
            # 找不到完整匹配，尝试找第一行
            first_line = step_text_stripped.split('\n')[0].strip()
            pos = cumulative_text.find(first_line, search_start)
        
        if pos != -1:
            # 找到了，计算该步骤结束位置对应的 token
            end_char_pos = pos + len(step_text_stripped)
            end_char_pos = min(end_char_pos, len(char_to_token))
            
            if end_char_pos > 0 and end_char_pos - 1 < len(char_to_token):
                token_end = char_to_token[end_char_pos - 1] + 1
            else:
                token_end = n_tokens
            
            # 确保 token_end 递增（避免因搜索失败导致的非单调性）
            token_end = max(token_end, step_token_boundaries[-1] + 1)
            token_end = min(token_end, n_tokens)
            
            step_token_boundaries.append(token_end)
            search_start = end_char_pos
        else:
            # 找不到，使用比例估算
            estimated_end = int((step_idx + 1) * n_tokens / n_steps)
            estimated_end = max(estimated_end, step_token_boundaries[-1] + 1)
            estimated_end = min(estimated_end, n_tokens)
            step_token_boundaries.append(estimated_end)
    
    # 确保最后一个边界是 n_tokens
    step_token_boundaries[-1] = n_tokens
    
    # ========================================
    # Step 3: 构建连续不重叠的 token 范围
    # ========================================
    result = []
    for step_idx, step_text in enumerate(steps):
        token_start = step_token_boundaries[step_idx]
        token_end = step_token_boundaries[step_idx + 1]
        
        # 确保每个步骤至少有 1 个 token
        if token_start >= token_end:
            token_end = min(token_start + 1, n_tokens)
        
        result.append({
            'text': step_text.strip(),
            'token_ranges': [(token_start, token_end)]
        })
    
    # ========================================
    # Step 4: 计算每个步骤的平均熵
    # ========================================
    step_entropies = []
    token_ranges_summary = []
    
    for step in result:
        token_ranges = step['token_ranges']
        
        if not token_ranges:
            step_entropies.append(0.0)
            token_ranges_summary.append((0, 0))
            continue
        
        # 获取该步骤的 token 范围统计
        step_start = min(s for s, _ in token_ranges)
        step_end = max(e for _, e in token_ranges)
        token_ranges_summary.append((step_start, step_end))
        
        # 计算平均熵（优化版本）
        if len(token_ranges) == 1:
            start, end = token_ranges[0]
            if start < end:
                step_entropies.append(float(np.mean(token_entropies[start:end])))
            else:
                step_entropies.append(0.0)
        else:
            # 多个范围：使用 np.concatenate
            entropy_arrays = [token_entropies[s:e] for s, e in token_ranges if s < e]
            if entropy_arrays:
                step_entropies.append(float(np.mean(np.concatenate(entropy_arrays))))
            else:
                step_entropies.append(0.0)
    
    # ========================================
    # 调试输出（简洁版）
    # ========================================
    total_tokens_covered = sum(
        sum(e - s for s, e in step['token_ranges']) 
        for step in result
    )
    first_range = token_ranges_summary[0] if token_ranges_summary else (0, 0)
    last_range = token_ranges_summary[-1] if token_ranges_summary else (0, 0)
    
    print(f"[StepEntropy] n_steps={n_steps}, n_tokens={n_tokens}, "
          f"token_ranges: [{first_range[0]},{first_range[1]})....[{last_range[0]},{last_range[1]}), "
          f"covered={total_tokens_covered}/{n_tokens}, "
          f"entropy: {step_entropies[0]:.3f}→{step_entropies[-1]:.3f}")
    
    return step_entropies


def compute_step_token_boundaries(solution_str, token_ids, tokenizer):
    """
    计算每个推理步骤对应的 token 范围（仅返回边界，不计算熵）。
    
    与 compute_step_entropies_aligned_with_process_thoughts 使用相同的对齐逻辑，
    但只返回步骤边界列表，用于后续 JS 散度计算。
    
    Args:
        solution_str: 模型响应文本
        token_ids: 生成的 token 序列 (list of int)
        tokenizer: tokenizer
    
    Returns:
        list of (start, end) tuples: 每个步骤的 token 范围
        如果无法计算，返回空列表
    """
    import torch
    
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    
    n_tokens = len(token_ids)
    if n_tokens == 0:
        return []
    
    # Step 1: 使用 process_thoughts 划分步骤
    steps = process_thoughts(solution_str)
    n_steps = len(steps)
    if n_steps == 0:
        return []
    
    # Step 2: 逐 token 累积解码，建立字符到 token 的映射
    cumulative_text = ""
    char_to_token = []
    for token_idx, tid in enumerate(token_ids):
        decoded = tokenizer.decode([tid], skip_special_tokens=True)
        for _ in decoded:
            char_to_token.append(token_idx)
        cumulative_text += decoded
    
    # Step 3: 找到每个步骤结束位置对应的 token 索引
    step_token_ends = [0]
    search_start = 0
    
    for step_idx, step_text in enumerate(steps):
        step_text_stripped = step_text.strip()
        pos = cumulative_text.find(step_text_stripped, search_start)
        
        if pos == -1:
            first_line = step_text_stripped.split('\n')[0].strip()
            pos = cumulative_text.find(first_line, search_start)
        
        if pos != -1:
            end_char_pos = pos + len(step_text_stripped)
            end_char_pos = min(end_char_pos, len(char_to_token))
            if end_char_pos > 0 and end_char_pos - 1 < len(char_to_token):
                token_end = char_to_token[end_char_pos - 1] + 1
            else:
                token_end = n_tokens
            token_end = max(token_end, step_token_ends[-1] + 1)
            token_end = min(token_end, n_tokens)
            step_token_ends.append(token_end)
            search_start = end_char_pos
        else:
            estimated_end = int((step_idx + 1) * n_tokens / n_steps)
            estimated_end = max(estimated_end, step_token_ends[-1] + 1)
            estimated_end = min(estimated_end, n_tokens)
            step_token_ends.append(estimated_end)
    
    step_token_ends[-1] = n_tokens
    
    # Step 4: 构建 (start, end) 列表
    boundaries = []
    for step_idx in range(n_steps):
        start = step_token_ends[step_idx]
        end = step_token_ends[step_idx + 1]
        if start >= end:
            end = min(start + 1, n_tokens)
        boundaries.append((start, end))
    
    return boundaries


def compute_depe(step_features):
    """
    DEPE (Directed Entropy Path Efficiency) — 有向熵路径效率

    DEPE = (H_first - H_last) / (sum(|ΔH_t|) + eps)

    等价形式:
      injection  = sum(ΔH_t for ΔH_t > 0)  → 总熵增
      utilization = sum(|ΔH_t| for ΔH_t < 0) → 总熵减
      DEPE = (utilization - injection) / (utilization + injection + eps)

    范围: [-1, +1]
      +1: 完美单调收敛 (所有步骤熵递减)
       0: 净进展为零 (注入 = 利用) 或坍缩 (无变化)
      -1: 完美单调发散

    抗坍缩: 当 H 均匀降低时 DEPE 不变 (比值不受绝对尺度影响)
    
    验证结果 (离线):
      Cohen's d = 0.100, d×Std = 0.0048 (全步骤) → 信号较弱
      作为监控指标保留, 不直接用于奖励

    Args:
        step_features: list of dict, 每个 dict 至少包含 'mean_H'
    Returns:
        float: DEPE value, 0.0 if cannot compute
    """
    if not step_features or len(step_features) < 2:
        return 0.0
    H = [s['mean_H'] for s in step_features]
    total_var = sum(abs(H[t + 1] - H[t]) for t in range(len(H) - 1))
    if total_var < 1e-10:
        return 0.0
    return (H[0] - H[-1]) / total_var


def compute_critical_injection_quality(step_features, critical_ratio=0.2):
    """
    关键注入步的质量: q_inj = -mean(mean_H of 关键注入步)

    分解思路:
      关键步骤 = eff_vocab top-K% (模型最不确定的决策点)
      关键注入步 = 关键步骤中 dH_in > 0 的 (在不确定点进一步探索)

      正确推理: 注入步虽然分布分散，但 mean_H 更低 → 有方向的探索
      错误推理: 注入步分布分散且 mean_H 高 → 无目的的混乱

    验证结果:
      - Cohen's d = 0.865 (注入步区分力极强)
      - d×Std = 0.4325 (比简单版 mean_H 高 27%)
      - LenR = 0.003 (几乎零长度偏差)

    Args:
        step_features: list of dict, 每个 dict 包含:
            - mean_H: 步骤平均熵
            - std_H: 步骤熵标准差
            - eff_vocab: 有效词表大小 = mean(2^h_i)
            - dH_in: 从上一步到本步的熵变 (>0 = 注入, <0 = 利用)
        critical_ratio: 关键步骤比例 (默认 0.2 = top 20%)

    Returns:
        float or None: q_inj = -mean(mean_H of injection steps), higher = better
                       None if cannot compute
    """
    if not step_features:
        return None

    n = len(step_features)
    if n == 0:
        return None

    # 获取 eff_vocab 值
    eff_vocabs = [s['eff_vocab'] for s in step_features]

    # 确定阈值: top critical_ratio%
    sorted_eff = sorted(eff_vocabs, reverse=True)
    n_critical = max(1, int(n * critical_ratio))
    threshold = sorted_eff[min(n_critical - 1, n - 1)]

    # 识别关键步骤 (eff_vocab >= threshold)
    critical_indices = [i for i, ev in enumerate(eff_vocabs) if ev >= threshold]

    if not critical_indices:
        critical_indices = list(range(n))

    # 从关键步骤中筛选注入步 (dH_in > 0)
    injection_indices = [i for i in critical_indices
                         if step_features[i].get('dH_in', 0) > 0.001]

    if not injection_indices:
        # Fallback: 如果没有注入步, 使用所有关键步骤
        injection_indices = critical_indices

    # q_inj = -mean(mean_H), higher = better (更低的 mean_H = 更有方向的探索)
    q_inj = -sum(step_features[i]['mean_H'] for i in injection_indices) / len(injection_indices)

    return q_inj


def compute_relative_injection_quality(step_features, critical_ratio=0.2):
    """
    自归一化注入质量 (对比度奖励):
      q_relative = mean_H(非关键步) - mean_H(关键注入步)

    核心思想:
      不直接测量绝对熵水平 (q_inj = -mean(H) → 坍缩), 而是测量关键注入步
      相对于回答自身基线 (非关键步) 的聚焦度对比。

    抗坍缩机制:
      - 模型全局降低 H → 两项同降 → q_relative 不变 (GRPO 内相对排序不变)
      - 只有关键注入步比非关键步更聚焦时才获得更高奖励
      - 非关键步作为内部锚点, 随模型变化而自适应

    验证结果 (离线, 30 prompts × 16 responses):
      - Cohen's d = 0.828 (区分力强)
      - d×Std = 0.0983 (连续), WinR% = 62.4%
      - LenR = -0.013 (几乎零长度偏差)

    Args:
        step_features: list of dict, 每个 dict 包含:
            - mean_H: 步骤平均熵
            - eff_vocab: 有效词表大小 = mean(2^h_i)
            - dH_in: 从上一步到本步的熵变 (>0 = 注入, <0 = 利用)
        critical_ratio: 关键步骤比例 (默认 0.2 = top 20%)

    Returns:
        float or None: q_relative = H_noncrit - H_crit_inj, higher = better
                       None if cannot compute
    """
    if not step_features or len(step_features) < 3:
        return None

    n = len(step_features)

    # 获取 eff_vocab 值
    eff_vocabs = [s['eff_vocab'] for s in step_features]

    # 确定阈值: top critical_ratio%
    sorted_eff = sorted(eff_vocabs, reverse=True)
    n_critical = max(1, int(n * critical_ratio))
    threshold = sorted_eff[min(n_critical - 1, n - 1)]

    # 识别关键步骤 (eff_vocab >= threshold)
    critical_set = set(i for i, ev in enumerate(eff_vocabs) if ev >= threshold)

    # 非关键步骤
    non_critical = [i for i in range(n) if i not in critical_set]

    if not non_critical:
        return None

    # 关键注入步: 关键步骤 + dH_in > 0 (在不确定点进一步探索)
    crit_injection = [i for i in critical_set
                      if step_features[i].get('dH_in', 0) > 0.001]

    if not crit_injection:
        # Fallback: 如果没有注入步, 使用所有关键步骤
        crit_injection = list(critical_set)

    # q_relative = H_noncrit - H_crit_inj
    H_noncrit = sum(step_features[i]['mean_H'] for i in non_critical) / len(non_critical)
    H_crit_inj = sum(step_features[i]['mean_H'] for i in crit_injection) / len(crit_injection)

    return H_noncrit - H_crit_inj


def compute_eta_reward_from_transitions(step_transitions, prompt_accuracy):
    """
    [旧版, 保留兼容] 计算 η↑ × η↓ × w_linear 奖励
    """
    eps = 1e-10

    if not step_transitions or len(step_transitions) < 2:
        w = 1.0 - 2.0 * abs(0.5 - prompt_accuracy) if prompt_accuracy is not None else 0.0
        return {
            'reward': 0.0,
            'eta_up': 0.0, 'eta_down': 0.0,
            'w_linear': w, 'prompt_accuracy': prompt_accuracy or 0.0,
        }

    def anti_struct(t):
        kl_sum = t['kl_forward'] + t['kl_reverse']
        if kl_sum < eps:
            return 0.0
        return max(0.0, 1.0 - 2.0 * t['js_divergence'] / kl_sum)

    up_as = [anti_struct(t) for t in step_transitions if t['entropy_delta'] > 0.001]
    dn_as = [anti_struct(t) for t in step_transitions if t['entropy_delta'] < -0.001]

    if not up_as or not dn_as:
        w = 1.0 - 2.0 * abs(0.5 - prompt_accuracy) if prompt_accuracy is not None else 0.0
        return {
            'reward': 0.0,
            'eta_up': 0.0 if not up_as else float(sum(up_as) / len(up_as)),
            'eta_down': 0.0 if not dn_as else float(sum(dn_as) / len(dn_as)),
            'w_linear': w, 'prompt_accuracy': prompt_accuracy or 0.0,
        }

    import numpy as np
    eta_up = float(np.mean(up_as))
    eta_dn = float(np.mean(dn_as))
    w = 1.0 - 2.0 * abs(0.5 - prompt_accuracy) if prompt_accuracy is not None else 0.0
    reward = eta_up * eta_dn * max(0.0, w)

    return {
        'reward': reward,
        'eta_up': eta_up,
        'eta_down': eta_dn,
        'w_linear': w,
        'prompt_accuracy': prompt_accuracy or 0.0,
    }


def compute_structure_reward(step_transitions):
    """
    p_0 信息轨迹奖励 (面积比): R = mean(d_t) / d_max  ∈ [0, 1]

    核心思想:
      有用信息 = 推理过程中积累并存活到最终答案的、相对初始状态的偏离。
      p_0 = 第一步的分布 (模型推理前的信念)
      d_t = JSD(p_t, p_0) = 步骤 t 相对初始状态的信息积累量

    奖励公式:
      R = mean(d_0, d_1, ..., d_T) / max(d_0, d_1, ..., d_T)
      = 轨迹在峰值信息水平上的平均利用率 (面积填充率)

    几何直觉:
      d_t 轨迹 vs 时间 → 曲线下面积 / 峰值矩形面积
      - 快速上升并保持 → mean ≈ d_max → R ≈ 1 (好)
      - 上升后崩溃 → mean << d_max → R 低 (坏: 探索后遗忘)
      - 仅末尾突变 → mean << d_max → R 低 (坏: 无渐进积累)

    性质:
      - 有界 [0, 1], 无常数, 无缩放系数
      - 长度基本无关 (比值)
      - 无分相, 无阈值, 全序列整合

    兼容性: 若 step_transitions 中无 jsd_from_p0 字段, 回退到旧方案

    Args:
        step_transitions: list of dict, 每个 dict 包含:
            - jsd_from_p0: JSD(p_t, p_0)
            - jsd_from_p0_next: JSD(p_{t+1}, p_0)
            (兼容旧格式: 若无上述字段, 回退到 gauss_distance 或 s_t)

    Returns:
        dict: reward, eta, n_steps, d_max, d_final, conservation (供分析)
    """
    eps = 1e-10

    if not step_transitions or len(step_transitions) < 2:
        return {'reward': 0.0, 'eta': 0.0, 'n_steps': 0}

    n = len(step_transitions)

    # ============================================================
    # 检查是否有 p_0 轨迹数据
    # ============================================================
    has_p0 = 'jsd_from_p0' in step_transitions[0]

    if has_p0:
        # ========================================================
        # p_0 信息轨迹奖励: R = mean(d_t) / d_max
        # ========================================================
        # 提取 d_t 轨迹: d_0 = jsd_from_p0[0], ..., d_T = jsd_from_p0_next[-1]
        d_trajectory = [step_transitions[0].get('jsd_from_p0', 0.0)]
        for t in step_transitions:
            d_trajectory.append(t.get('jsd_from_p0_next', 0.0))

        d_max = max(d_trajectory)
        d_final = d_trajectory[-1]
        d_mean = sum(d_trajectory) / len(d_trajectory)

        # 核心奖励: 面积填充率
        reward = d_mean / d_max if d_max > eps else 0.0

        # 供分析的辅助指标
        conservation = d_final / d_max if d_max > eps else 0.0

        return {
            'reward': reward,
            'eta': reward,
            'n_steps': n,
            'd_max': d_max,
            'd_final': d_final,
            'd_mean': d_mean,
            'conservation': conservation,
        }

    # ============================================================
    # 回退: 旧的 D_t / s_t 分相瓶颈守恒奖励 (兼容无 p_0 数据)
    # ============================================================
    d_values = []
    abs_dh_values = []
    delta_values = []

    for t in step_transitions:
        delta = t['entropy_delta']
        abs_dh = abs(delta)

        if 'gauss_distance' in t:
            d = t['gauss_distance']
        else:
            kl_sum = t.get('kl_forward', 0) + t.get('kl_reverse', 0)
            if kl_sum > eps:
                d = max(0.0, 1.0 - 2.0 * t.get('js_divergence', 0) / kl_sum)
            else:
                d = 0.0

        d_values.append(d)
        abs_dh_values.append(abs_dh)
        delta_values.append(delta)

    up_idx = [i for i in range(n) if delta_values[i] > eps]
    dn_idx = [i for i in range(n) if delta_values[i] < -eps]

    if not up_idx or not dn_idx:
        eta = min(d_values)
        return {'reward': eta, 'eta': eta, 'n_steps': n}

    up_dhs = [abs_dh_values[i] for i in up_idx]
    up_median = sorted(up_dhs)[len(up_dhs) // 2]
    up_big = [i for i in up_idx if abs_dh_values[i] >= up_median]
    if not up_big:
        up_big = up_idx
    eta_up_min = min(d_values[i] for i in up_big)

    dn_dhs = [abs_dh_values[i] for i in dn_idx]
    dn_median = sorted(dn_dhs)[len(dn_dhs) // 2]
    dn_big = [i for i in dn_idx if abs_dh_values[i] >= dn_median]
    if not dn_big:
        dn_big = dn_idx
    eta_dn_min = min(d_values[i] for i in dn_big)

    I_val = sum(abs_dh_values[i] * d_values[i] for i in up_idx)
    U_val = sum(abs_dh_values[i] * d_values[i] for i in dn_idx)
    conservation = min(1.0, U_val / I_val) if I_val > eps else 1.0

    eta = min(eta_up_min, eta_dn_min)
    reward = eta * conservation

    return {
        'reward': reward,
        'eta': eta,
        'n_steps': n,
    }


def compute_trajectory_progress_score(step_transitions):
    """
    方案 B: 轨迹锚定进展分 (Trajectory-Anchored Progress Score)

    核心思想: 有用信息 = 对全局推理轨迹的累积进展有贡献的信息。
    不是看"局部变化有多集中/对称", 而是看"局部变化有多少转化为全局进展"。

    利用 jsd_from_p0 (步骤 t 相对初始分布 p_0 的 JSD 距离) 构建信息轨迹:
      d_t = JSD(p_t, p_0) — 步骤 t 的累积信息量

    进展效率:
      e_t = |Δd_t| / JSD(p_t, p_{t+1})
      度量: "局部变化中有多少比例转化为了相对 p_0 的累积进展?"
      - e_t → 1: 几乎所有局部变化都转化为全局进展 (高效)
      - e_t → 0: 局部变化大但全局进展小 (低效, 原地打转)

    更优雅的相分类 (用轨迹方向, 非熵变方向):
      - 发散相 (Δd > 0): 模型远离 p_0 → 探索
      - 收敛相 (Δd < 0): 模型回归 p_0 → 压缩

    Part 1 (探索效率):
      η_diverge = Σ_{Δd>0}(e_t × Δd_t) / Σ_{Δd>0}(Δd_t)  ∈ [0, 1]
      高 → 探索步骤高效地增加了远离 p_0 的信息
      低 → 探索步骤大量无效变化, 只有少量转化为进展

    Part 2 (压缩保留率):
      η_converge = d_final / d_max  ∈ [0, 1]
      高 → 压缩后保留了大部分峰值信息 (最终结论稳固)
      低 → 压缩后信息大量丢失 (探索后遗忘或崩溃)

    守恒天然内蕴: d_final ≤ d_max 恒成立, 无需额外约束因子。

    总分: R = η_diverge × η_converge ∈ [0, 1]
    = 探索效率 × 信息保留率

    与 compute_structure_reward 的区别:
      - structure_reward = mean(d_t) / d_max (面积填充率, 无相分类)
      - trajectory_progress = η_diverge × η_converge (有探索/压缩分相, 有效率度量)

    Args:
        step_transitions: list of dict, 每个 dict 包含:
            - jsd_from_p0: JSD(p_t, p_0) — 步骤 t 的累积信息量
            - jsd_from_p0_next: JSD(p_{t+1}, p_0) — 步骤 t+1 的累积信息量
            - js_divergence: JSD(p_t, p_{t+1}) — 局部变化量

    Returns:
        dict: 奖励值和分项指标; 若无 jsd_from_p0 数据则返回空结果
    """
    eps = 1e-10

    if not step_transitions or len(step_transitions) < 2:
        return {
            'reward': 0.0,
            'eta_diverge': 0.0, 'eta_converge': 0.0,
            'n_diverge': 0, 'n_converge': 0,
            'd_max': 0.0, 'd_final': 0.0,
        }

    # 检查是否有 p_0 轨迹数据
    if 'jsd_from_p0' not in step_transitions[0]:
        return {
            'reward': 0.0,
            'eta_diverge': 0.0, 'eta_converge': 0.0,
            'n_diverge': 0, 'n_converge': 0,
            'd_max': 0.0, 'd_final': 0.0,
            'error': 'no_jsd_from_p0',
        }

    # 构建 d_t 轨迹: d_0, d_1, ..., d_T
    d_trajectory = [step_transitions[0].get('jsd_from_p0', 0.0)]
    for t in step_transitions:
        d_trajectory.append(t.get('jsd_from_p0_next', 0.0))

    d_max = max(d_trajectory)
    d_final = d_trajectory[-1]

    if d_max < eps:
        return {
            'reward': 0.0,
            'eta_diverge': 0.0, 'eta_converge': 0.0,
            'n_diverge': 0, 'n_converge': 0,
            'd_max': d_max, 'd_final': d_final,
        }

    # 计算每个转移的轨迹进展和效率
    diverge_weighted_e = 0.0   # Σ(e_t × Δd_t) for Δd > 0
    diverge_total_dd = 0.0     # Σ(Δd_t) for Δd > 0
    converge_count = 0
    diverge_count = 0

    for i, t in enumerate(step_transitions):
        d_t = d_trajectory[i]
        d_t1 = d_trajectory[i + 1]
        delta_d = d_t1 - d_t  # 轨迹进展
        jsd_local = t.get('js_divergence', 0.0)  # 局部变化量

        # 进展效率: e_t = |Δd| / JSD_local
        # 上界截断到 1.0 (数值稳定, Δd ≤ JSD_local 通常成立但不绝对保证)
        e_t = min(1.0, abs(delta_d) / jsd_local) if jsd_local > eps else 0.0

        if delta_d > eps:
            # 发散相: 远离 p_0 → 探索
            diverge_weighted_e += e_t * delta_d
            diverge_total_dd += delta_d
            diverge_count += 1
        elif delta_d < -eps:
            converge_count += 1

    # Part 1: 探索效率
    eta_diverge = diverge_weighted_e / diverge_total_dd if diverge_total_dd > eps else 0.0

    # Part 2: 压缩保留率 (最终信息 / 峰值信息)
    eta_converge = d_final / d_max if d_max > eps else 0.0

    # 总分: 探索效率 × 压缩保留率
    reward = eta_diverge * eta_converge

    return {
        'reward': reward,
        'eta_diverge': eta_diverge,
        'eta_converge': eta_converge,
        'n_diverge': diverge_count,
        'n_converge': converge_count,
        'd_max': d_max,
        'd_final': d_final,
        'd_trajectory': d_trajectory,
    }


def compute_combined_reward(step_transitions):
    """
    R = EWD_cc × conservation   (内容变化守恒加权奖励)

    三层防线:
      CC_t = 1 - R^2(log_ratio, log_P)    温度不变的内容变化率
      d_t  = 1 - √σ_t                     方向性 (KL 非对称性)
      u_t  = CC_t × d_t                   有用率 (内容变化 × 方向性)

    温度缩放 logit' = c·logit 时:
      log_ratio = (c-1)·logit + const → 与 log_P 完美线性
      → R^2 = 1 → CC_t = 0 → u_t = 0 → 零奖励 (数学免疫)

    信息分解 (u_t 作为连续软阈值):
      每步熵变 ΔH_t 被 u_t 分解为:
        有用部分 = u_t × |ΔH_t|    (真实内容变化 + 方向性)
        噪声部分 = (1-u_t) × |ΔH_t| (温度效应 + 随机扰动)

    按熵变方向聚合:
      U_inj = Σ_{ΔH>0} u_t × ΔH_t      熵增步注入的有用信息
      U_use = Σ_{ΔH<0} u_t × |ΔH_t|    熵减步消耗的有用信息
      N_inj = Σ_{ΔH>0} (1-u_t) × ΔH_t  熵增步注入的噪声

    公式:
      EWD_cc = Σ(H_t · u_t) / ΣH_t              熵加权有用率
      conservation = min(1, U_use / (U_inj + ε))  守恒因子
      R = EWD_cc × conservation

    ① 熵增 (ΔH>0): CC 高 + σ 低 → 真实方向性探索 (奖励)
    ② 熵减 (ΔH<0): CC 高 + σ 低 → 真实方向性收敛 (奖励)
    ③ 守恒: U_inj ≤ U_use (注入 ≤ 消耗)
    ④ 阈值: u_t = CC × (1-√σ), 0=hack/噪声, 高=有用
    ⑤ 抗 hack: 温度缩放 → CC=0 → 数学免疫

    回退: 若 content_change 字段不存在, 回退到 Conservation-EWD (σ only).

    Args:
        step_transitions: list of dict, 每个 dict 包含:
            - js_divergence, kl_forward, kl_reverse
            - step_entropy: H_t (可选)
            - entropy_delta: ΔH_t (可选)
            - content_change: CC_t (可选)
    """
    eps = 1e-10
    import math

    sigma_values = []
    entropy_values = []
    cc_values = []
    has_entropy = len(step_transitions) > 0 and 'step_entropy' in step_transitions[0]
    has_delta = len(step_transitions) > 0 and 'entropy_delta' in step_transitions[0]
    has_cc = len(step_transitions) > 0 and 'content_change' in step_transitions[0]

    for t in step_transitions:
        kl_sum = t['kl_forward'] + t['kl_reverse']
        if kl_sum > eps:
            sigma_t = 2.0 * t['js_divergence'] / kl_sum
            sigma_t = max(0.0, min(1.0, sigma_t))
        else:
            sigma_t = 1.0
        sigma_values.append(sigma_t)
        if has_entropy:
            entropy_values.append(t['step_entropy'])
        cc_values.append(t.get('content_change', 1.0) if has_cc else 1.0)

    mean_sigma = sum(sigma_values) / len(sigma_values)
    mean_sigma_sq = sum(s ** 2 for s in sigma_values) / len(sigma_values)
    mean_cc = sum(cc_values) / len(cc_values)

    # ---- 有用率 u_t = CC_t × (1-√σ_t) ----
    u_values = [cc_values[i] * (1.0 - math.sqrt(sigma_values[i]))
                for i in range(len(sigma_values))]

    # ---- EWD_cc = Σ(H_t · u_t) / ΣH_t ----
    if has_entropy and sum(entropy_values) > eps:
        sum_H = sum(entropy_values)
        ewd_cc = sum(entropy_values[i] * u_values[i]
                     for i in range(min(len(u_values), len(entropy_values)))) / sum_H
        # 也保留纯 EWD 用于对比
        ewd = sum(entropy_values[i] * (1.0 - math.sqrt(sigma_values[i]))
                   for i in range(len(sigma_values))) / sum_H
    else:
        ewd = 1.0 - mean_sigma_sq
        ewd_cc = ewd * mean_cc

    # ---- 信息流分解 + 守恒因子 ----
    U_inj = 0.0
    U_use = 0.0
    N_inj = 0.0

    if has_delta:
        for i, t in enumerate(step_transitions):
            dH = t['entropy_delta']
            abs_dH = abs(dH)
            u_t = u_values[i] if i < len(u_values) else 0.0
            if dH > 0:
                U_inj += u_t * abs_dH
                N_inj += (1.0 - u_t) * abs_dH
            elif dH < 0:
                U_use += u_t * abs_dH

    # 守恒因子
    if has_delta and U_inj > eps:
        conservation = min(1.0, U_use / (U_inj + eps))
    else:
        conservation = 1.0

    reward = ewd_cc * conservation

    return {
        'reward': reward,
        'mean_sigma': mean_sigma,
        'mean_sigma_sq': mean_sigma_sq,
        'ewd': ewd,
        'ewd_cc': ewd_cc,
        'mean_cc': mean_cc,
        'conservation': conservation,
        'U_inj': U_inj,
        'U_use': U_use,
        'N_inj': N_inj,
        'mean_directionality': reward,
        'n_steps': len(step_transitions),
    }


def compute_noise_weight(step_transitions):
    """
    计算响应的噪声权重 w = 1 - noise%

    noise% = Σ(σ_t × |ΔH_t|) / Σ|ΔH_t|   (N2, |ΔH|加权σ, 无常数)

    σ_t = 2·JSD(p_t, p_{t+1}) / (KL(p_{t+1}||p_t) + KL(p_t||p_{t+1}))

    σ_t ∈ [0, 1]:
      → 0: 信息流不对称 (有方向, 是信号)
      → 1: 信息流对称   (无方向, 是噪声)

    w ∈ [0, 1]:
      → 1: 全是信号, 保留完整梯度
      → 0: 全是噪声, 压制梯度

    用途: A_shaped = A_norm × w (优势加权)

    Args:
        step_transitions: list of dict, 每个 dict 包含:
            - entropy_delta, js_divergence, kl_forward, kl_reverse

    Returns:
        float: noise weight w = 1 - noise%
    """
    eps = 1e-10

    if not step_transitions or len(step_transitions) < 2:
        return 0.95  # fallback: 中等偏高的权重

    total_sigma_dH = 0.0
    total_dH = 0.0

    for t in step_transitions:
        kl_sum = t['kl_forward'] + t['kl_reverse']
        if kl_sum > eps:
            sigma_t = 2.0 * t['js_divergence'] / kl_sum
        else:
            sigma_t = 1.0
        abs_dH = abs(t['entropy_delta'])
        total_sigma_dH += sigma_t * abs_dH
        total_dH += abs_dH

    noise_pct = total_sigma_dH / total_dH if total_dH > eps else 0.05
    return max(0.0, min(1.0, 1.0 - noise_pct))


def compute_entropy_efficiency_from_transitions(step_transitions):
    """
    A3 熵效率奖励: R = η_up × η_down × ρ ∈ [0, 1] (无常数、无缩放系数)

    有用信息定义 (KL-谱集中度):
      c_t = Σ_{top-k by |δ_i|} |δ_i| / Σ_i |δ_i|  ∈ [0, 1]
      其中 δ_i = p_{t+1,i} × log(p_{t+1,i} / p_{t,i}) 为 token i 对 KL 散度的贡献

      物理含义:
      - c_t → 1: KL 变化集中在少数关键 token (有结构的、有用的变化)
      - c_t → 0: KL 变化均匀分散在长尾 (弥散噪声)

      优于旧 u_t = 1 - 2·JSD/(KL_f + KL_r) 的原因:
      - 旧 u_t 度量 KL 非对称性: 好的步骤同时加减概率反而 u_t 低
      - 新 c_t 度量变化集中度: 变化聚焦在少数 token 则 c_t 高, 无论方向

      兼容性: 若 step_transitions 中有 'kl_concentration' 字段则直接使用,
              否则回退到旧的非对称性度量 u_t

    将熵效率拆成两个核心部分 + 守恒约束:

    Part 1 (探索质量):
      η_up = Σ_{ΔH>0}(c_t × ΔH_t) / Σ_{ΔH>0}(ΔH_t)  ∈ [0, 1]
      - 熵增阶段集中度的加权平均 (权重 = 熵增量)
      - η_up 高 → 探索时概率变化集中在特定 token → 有目标的假设展开
      - η_up 低 → 探索时变化分散在长尾 → 漫无目的的噪声扩散

    Part 2 (压缩质量):
      η_down = Σ_{ΔH<0}(c_t × |ΔH_t|) / Σ_{ΔH<0}(|ΔH_t|)  ∈ [0, 1]
      - 熵减阶段集中度的加权平均 (权重 = 熵减量)
      - η_down 高 → 压缩时变化集中在特定 token → 精准的信息聚焦
      - η_down 低 → 压缩时变化分散 → 无差别截断, 丢了重要信息

    守恒约束:
      ρ = min(1, U_down / U_up)  ∈ [0, 1]
      - U_up = Σ_{ΔH>0}(c_t × ΔH_t): 探索阶段产出的集中信息总量
      - U_down = Σ_{ΔH<0}(c_t × |ΔH_t|): 压缩阶段消耗的集中信息总量
      - ρ ≈ 1 → 压缩完全消耗了探索产出的集中信息 → 预算平衡
      - ρ << 1 → 过度探索, 压缩容量不足

    总分: R = η_up × η_down × ρ ∈ [0, 1]
    确保: 高质量探索 × 高质量压缩 × 预算平衡 = 全局最优

    Args:
        step_transitions: list of dict, 每个 dict 包含:
            - entropy_delta: ΔH = H_{t+1} - H_t
            - kl_concentration: c_t (KL-谱集中度, 可选; 无则回退到旧 u_t)
            - js_divergence: JSD(p_t, p_{t+1}) (回退时使用)
            - kl_forward: KL(p_{t+1} || p_t) (回退时使用)
            - kl_reverse: KL(p_t || p_{t+1}) (回退时使用)

    Returns:
        dict: 奖励值和各分项指标
    """
    eps = 1e-10

    if not step_transitions or len(step_transitions) < 2:
        return {
            'reward': 0.0,
            'eta_up': 0.0, 'eta_down': 0.0, 'conservation': 0.0,
            'n_up': 0, 'n_down': 0,
        }

    # 检查是否有新的 kl_concentration 字段
    has_kl_conc = 'kl_concentration' in step_transitions[0]

    # 按熵变方向分组
    up_tr = [t for t in step_transitions if t['entropy_delta'] > 0]
    down_tr = [t for t in step_transitions if t['entropy_delta'] < 0]

    if not up_tr or not down_tr:
        return {
            'reward': 0.0,
            'eta_up': 0.0, 'eta_down': 0.0, 'conservation': 0.0,
            'n_up': len(up_tr), 'n_down': len(down_tr),
        }

    def useful_info(t):
        """
        有用信息度量:
          - 优先使用 kl_concentration (KL-谱集中度): 变化越集中越有用
          - 回退: u_t = 1 - 2·JSD/(KL_f + KL_r) (旧非对称性度量)
        """
        if has_kl_conc:
            return t.get('kl_concentration', 0.0)
        # 回退到旧度量
        kl_sum = t['kl_forward'] + t['kl_reverse']
        if kl_sum < eps:
            return 0.0
        return max(0.0, 1.0 - 2.0 * t['js_divergence'] / kl_sum)

    # Part 1 探索质量: η_up = Σ(c_t × ΔH_t) / Σ(ΔH_t)
    U_up = sum(useful_info(t) * t['entropy_delta'] for t in up_tr)
    T_up = sum(t['entropy_delta'] for t in up_tr)
    eta_up = U_up / T_up if T_up > eps else 0.0

    # Part 2 压缩质量: η_down = Σ(c_t × |ΔH_t|) / Σ(|ΔH_t|)
    U_down = sum(useful_info(t) * abs(t['entropy_delta']) for t in down_tr)
    T_down = sum(abs(t['entropy_delta']) for t in down_tr)
    eta_down = U_down / T_down if T_down > eps else 0.0

    # 守恒约束: ρ = min(1, U_down / U_up) — 压缩消耗 >= 探索产出 → ρ = 1
    conservation = min(1.0, U_down / U_up) if U_up > eps else (1.0 if U_down > eps else 0.0)

    # 总奖励: 两部分 × 守恒, R ∈ [0, 1]
    reward = eta_up * eta_down * conservation

    return {
        'reward': reward,
        'eta_up': eta_up,
        'eta_down': eta_down,
        'conservation': conservation,
        'U_up': U_up,
        'U_down': U_down,
        'T_up': T_up,
        'T_down': T_down,
        'n_up': len(up_tr),
        'n_down': len(down_tr),
        'metric_type': 'kl_concentration' if has_kl_conc else 'asymmetry_legacy',
    }


def reward_func_entropy_efficiency(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    """
    A3 熵效率奖励函数: R = η_up × η_down × ρ ∈ [0, 1] (无常数、无缩放系数)
    
    有用信息 (KL-谱集中度):
      c_t = top-k fraction of |δ_i| ∈ [0, 1]
      (δ_i = token i 对 KL 散度的贡献, 集中度越高越有用)
    
    Part 1  η_up = Σ(c_t×ΔH) / Σ(ΔH)     探索质量: 集中的探索 vs 弥散噪声
    Part 2  η_down = Σ(c_t×|ΔH|) / Σ(|ΔH|)  压缩质量: 精准聚焦 vs 无差别截断
    守恒    ρ = min(1, U_down / U_up)       信息预算: 压缩消耗 ≥ 探索产出
    
    训练阶段: 使用 extra_info 中预计算的步间转移指标
    验证阶段: 使用 ground_truth 计算正确性奖励
    """
    # 训练阶段: 使用预计算的熵效率奖励
    if extra_info and "entropy_efficiency_reward" in extra_info:
        reward = float(extra_info["entropy_efficiency_reward"])
        
        # 计算正确性分数 (仅监控, 不影响训练奖励)
        correctness = safe_compute_score(solution_str, str(ground_truth))
        is_correct = correctness.get('acc', False) if isinstance(correctness, dict) else bool(correctness)
        
        result = {
            "score": reward,
            "acc": is_correct,
            "correctness_score": correctness.get('score', 0.0) if isinstance(correctness, dict) else float(correctness),
        }
        
        # 记录分项 (用于 wandb 监控)
        for key in ["eta_up", "eta_down", "conservation"]:
            if key in extra_info:
                result[key] = float(extra_info[key])
        
        return result
    
    # 训练阶段备选: 如果有步间转移数据, 直接计算
    if extra_info and "step_transitions" in extra_info:
        transitions = extra_info["step_transitions"]
        eff = compute_entropy_efficiency_from_transitions(transitions)
        
        correctness = safe_compute_score(solution_str, str(ground_truth))
        is_correct = correctness.get('acc', False) if isinstance(correctness, dict) else bool(correctness)
        
        return {
            "score": eff['reward'],
            "acc": is_correct,
            "correctness_score": correctness.get('score', 0.0) if isinstance(correctness, dict) else float(correctness),
            "eta_up": eff['eta_up'],
            "eta_down": eff['eta_down'],
            "conservation": eff['conservation'],
        }
    
    # 验证阶段: 使用 ground_truth 计算正确性奖励
    correctness = safe_compute_score(solution_str, str(ground_truth))
    if isinstance(correctness, dict):
        return correctness
    else:
        return {"score": float(correctness), "acc": bool(correctness)}


def reward_func_ttrl_mv(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    """
    正确性奖励函数 + Token 级 s_t 加权 Advantage

    奖励: 0/1 正确性 (TTRL majority voting 提供 ground_truth)
    梯度聚焦: s_t token 权重在 actor update 中应用 (不在此函数)

    架构:
      - 正确性奖励 → GRPO advantage (学什么方向)
      - s_t token 权重 → advantage 加权 (在哪聚焦梯度)
      - 两者分离: 奖励不可被游戏, s_t 仅控制梯度分配

    训练/验证阶段统一: 使用 ground_truth 计算正确性奖励
    (训练时 ground_truth = majority vote; 验证时 = original gt)
    """
    correctness = safe_compute_score(solution_str, str(ground_truth))
    if isinstance(correctness, dict):
        return correctness
    else:
        return {"score": float(correctness), "acc": bool(correctness)}


def _find_boxed_token_indices(token_ids, tokenizer):
    """
    从 token_ids 中定位 \\boxed{...} 区域对应的 token 索引。

    Returns:
        (boxed_indices, reason_indices): 分别是 boxed 区域和 reasoning 区域的 token 下标列表
    """
    import re
    text = ""
    spans = []
    for i, tid in enumerate(token_ids):
        s = len(text)
        text += tokenizer.decode([tid], skip_special_tokens=True)
        spans.append((s, len(text)))

    boxed_set = set()
    for m in re.finditer(r'\\boxed\{', text):
        depth, p = 1, m.end()
        while p < len(text) and depth > 0:
            if text[p] == '{':
                depth += 1
            elif text[p] == '}':
                depth -= 1
            p += 1
        for i, (ts, te) in enumerate(spans):
            if ts < p and te > m.start():
                boxed_set.add(i)

    n = len(token_ids)
    boxed_indices = sorted(boxed_set)
    reason_indices = [i for i in range(n) if i not in boxed_set]
    return boxed_indices, reason_indices


def reward_func_wasserstein(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    """
    无监督奖励: R = -W1(reason_entropies, boxed_entropies)

    W1 = Wasserstein-1 距离 (Earth Mover's Distance)
    正确回答的 reasoning 熵分布与 boxed 熵分布更接近 → W1 更小 → 奖励更高

    完全无参数, O(n log n), scipy.stats.wasserstein_distance 一行即可算。
    """
    import numpy as np
    from scipy.stats import wasserstein_distance

    correctness = safe_compute_score(solution_str, str(ground_truth))
    is_correct = correctness.get('acc', False) if isinstance(correctness, dict) else bool(correctness)

    token_entropies = extra_info.get("token_entropies") if extra_info else None

    if token_entropies is None or len(token_entropies) == 0:
        return {
            "score": 1.0 if is_correct else 0.0,
            "wasserstein": 0.0,
            "n_boxed_tokens": 0,
            "n_reason_tokens": 0,
            "acc": is_correct,
            "pred": correctness.get('pred', '') if isinstance(correctness, dict) else '',
            "extracted_gt": correctness.get('extracted_gt', '') if isinstance(correctness, dict) else '',
        }

    token_ids = extra_info["token_ids"]
    tokenizer = extra_info["tokenizer"]

    if isinstance(token_entropies, np.ndarray):
        ent = token_entropies
    else:
        ent = np.array(token_entropies, dtype=np.float64)

    boxed_idx, reason_idx = _find_boxed_token_indices(token_ids, tokenizer)

    if len(boxed_idx) < 2 or len(reason_idx) < 2:
        reward = 0.0
        w_dist = 0.0
    else:
        reason_ent = ent[reason_idx]
        boxed_ent = ent[boxed_idx]
        w_dist = float(wasserstein_distance(reason_ent, boxed_ent))
        reward = -w_dist

    return {
        "score": reward,
        "wasserstein": w_dist,
        "n_boxed_tokens": len(boxed_idx),
        "n_reason_tokens": len(reason_idx),
        "acc": is_correct,
        "pred": correctness.get('pred', '') if isinstance(correctness, dict) else '',
        "extracted_gt": correctness.get('extracted_gt', '') if isinstance(correctness, dict) else '',
    }


def reward_func_i_grpo(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    """
    I-filter + 投票: 纯正确性奖励

    配合 ray_trainer.py 中的 I-filter 机制使用:
      1. 生成 N 个响应
      2. 计算每个响应的 I_mean (EMA of entropy)
      3. 过滤: 保留 I_mean 最低的 K 个响应 (高置信度)
      4. 对过滤后的响应进行多数投票 → 伪标签
      5. 奖励 = 响应是否匹配伪标签 (0/1)

    token_entropies 仅用于 H_mean 监控, 不参与奖励计算
    """
    correctness = safe_compute_score(solution_str, str(ground_truth))
    is_correct = correctness.get('acc', False) if isinstance(correctness, dict) else bool(correctness)

    H_mean = 0.0
    token_entropies = extra_info.get("token_entropies", None) if extra_info else None
    if token_entropies is not None and len(token_entropies) >= 5:
        import numpy as _np
        H_mean = float(_np.mean(token_entropies))

    return {
        "score": 1.0 if is_correct else 0.0,
        "acc": is_correct,
        "pred": correctness.get('pred', '') if isinstance(correctness, dict) else '',
        "extracted_gt": correctness.get('extracted_gt', '') if isinstance(correctness, dict) else '',
        "H_mean": H_mean,
        "H_mean": 0.0,
    }


def reward_func_combined_unsupervised(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    """
    正确性奖励 + Phi(z) 熵门控优势加权

    ===== 奖励逻辑 =====
    reward = 1.0 if correct else 0.0  (正确性二值奖励)

    ===== 优势加权 (由 dp_actor 计算) =====
    w_t = Phi(z_t) = 0.5 * (1 + erf(z_t / sqrt(2)))
    z_t = (H_t - H_bar) / (sigma_H + eps)

    · 高熵 token (思考) → w > 0.5 → 梯度正常通过
    · 低熵 token (公式) → w < 0.5 → 梯度衰减
    · 门控宽度由 sigma_H 自适应决定, 零超参数

    extra_info["token_reward"] 由 dp_actor._compute_token_level_reward 计算.
    """
    correctness = safe_compute_score(solution_str, str(ground_truth))

    # ===== 主路径: 正确性奖励 + Phi(z) 熵门控 =====
    # 奖励: correctness (1/0), 由 PPO 归一化为优势 A
    # 优势加权: A_t = A × w_t, w_t = Phi(z_t) (由 dp_actor 计算)
    is_correct = correctness.get('acc', False)
    reward = 1.0 if is_correct else 0.0

    # 提取 token_reward 中的监控指标 (Phi(z) 熵门控)
    token_reward = extra_info.get("token_reward", None) if extra_info else None
    gate_mean = 0.0
    H_bar = 0.0
    mean_H = 0.0
    if token_reward and isinstance(token_reward, dict):
        gate_mean = token_reward.get('gate_mean', 0.0)
        H_bar = token_reward.get('H_bar', 0.0)
        mean_H = token_reward.get('mean_H', 0.0)

    return {
        "score": reward,
        "format_score": correctness.get('format_score', 0.0),
        "acc": correctness.get('acc', False),
        "extracted_gt": correctness.get('extracted_gt', ''),
        "pred": correctness.get('pred', ''),
        "correctness_score": correctness.get('score', 0.0),
        # 监控指标 (Phi(z) 熵门控 per-token 权重)
        "gate_mean": gate_mean,             # mean(Phi) — 门控通过率
        "H_bar": H_bar,                     # 自适应熵阈值
        "mean_H": mean_H,                   # 平均熵 (检测坍缩)
    }


def reward_func_mean_js(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    """
    CDQ 奖励函数: R = mean(log P(y_t) + H(t))

    CDQ = Confident Decision Quality (不确定中的确定选择)
      log P(y_t): 对选中 token 的信心
      H(t):      分布的不确定性 (探索广度)
      CDQ(t) = log P + H:
        > 0 → 考虑多选项但有明确首选 → 适应答案分布
        < 0 → 不确定且无方向 → 噪声
        ≈ 0 → 过度锐化或完全随机

    三部分:
      ① 探索时 CDQ>0 → 有方向的探索
      ② 压缩时 CDQ>0 → 基于证据的压缩
      ③ 全程 CDQ>0 → 信息自然守恒

    训练阶段: score = entropy_efficiency (ray_trainer 预计算)
    验证阶段: 使用 ground_truth 计算正确性奖励
    """
    # 训练阶段: 使用预计算的CDQ奖励
    if extra_info and "entropy_efficiency" in extra_info:
        reward = float(extra_info["entropy_efficiency"])

        # 计算正确性分数 (仅监控, 不影响训练奖励)
        correctness = safe_compute_score(solution_str, str(ground_truth))
        is_correct = correctness.get('acc', False) if isinstance(correctness, dict) else bool(correctness)

        result = {
            "score": reward,
            "acc": is_correct,
            "correctness_score": correctness.get('score', 0.0) if isinstance(correctness, dict) else float(correctness),
        }

        # 记录分项 (用于监控)
        for key in ["mean_logp", "mean_h", "s3"]:
            if key in extra_info:
                result[key] = float(extra_info[key])

        return result

    # 验证阶段: 使用 ground_truth 计算正确性奖励
    correctness = safe_compute_score(solution_str, str(ground_truth))
    if isinstance(correctness, dict):
        return correctness
    else:
        return {"score": float(correctness), "acc": bool(correctness)}


def calculate_differential_entropy(step_entropies, threshold):
    """
    计算高熵和低熵步骤的差分熵
    
    差分熵计算规则：
    - 高熵步骤：第一个高熵步骤差分熵为0，后续为 |当前熵 - 前面所有高熵步骤平均熵|
    - 低熵步骤：如果前一步是高熵，则重置（差分熵为0）；否则为 |当前熵 - 当前连续低熵序列平均熵|
    
    Args:
        step_entropies: 每个步骤的熵值列表
        threshold: 划分高低熵的阈值（通常为所有步骤的平均熵）
    
    Returns:
        dict: 包含高熵和低熵步骤的统计信息
    """
    import numpy as np
    
    n_steps = len(step_entropies)
    if n_steps == 0:
        return {
            'high_entropy_avg': 0.0,
            'low_entropy_avg': 0.0,
            'high_diff_entropy_avg': 0.0,
            'low_diff_entropy_avg': 0.0,
            'high_relative_diff': 0.0,
            'low_relative_diff': 0.0,
            'n_high_steps': 0,
            'n_low_steps': 0,
        }
    
    # 分类高熵和低熵步骤
    high_entropy_steps = []  # (index, entropy)
    low_entropy_steps = []   # (index, entropy)
    
    for i, entropy in enumerate(step_entropies):
        if entropy > threshold:
            high_entropy_steps.append((i, entropy))
        else:
            low_entropy_steps.append((i, entropy))
    
    # 计算高熵步骤的差分熵
    high_diff_entropies = []
    high_entropies_so_far = []
    for idx, entropy in high_entropy_steps:
        if len(high_entropies_so_far) == 0:
            diff = 0.0  # 第一个高熵步骤差分熵为0
        else:
            avg_prev = np.mean(high_entropies_so_far)
            diff = abs(entropy - avg_prev)
        high_diff_entropies.append(diff)
        high_entropies_so_far.append(entropy)
    
    # 计算低熵步骤的差分熵（有阻断规则）
    low_diff_entropies = []
    current_low_sequence = []  # 当前连续低熵序列
    prev_was_high = True  # 初始假设前一步是高熵，第一个低熵步骤差分熵为0
    
    for idx, entropy in low_entropy_steps:
        # 检查前一步是否为高熵
        if idx > 0:
            prev_entropy = step_entropies[idx - 1]
            prev_was_high = prev_entropy > threshold
        
        if prev_was_high:
            # 阻断：重置序列
            diff = 0.0
            current_low_sequence = [entropy]
        else:
            if len(current_low_sequence) == 0:
                diff = 0.0
                current_low_sequence = [entropy]
            else:
                avg_prev = np.mean(current_low_sequence)
                diff = abs(entropy - avg_prev)
                current_low_sequence.append(entropy)
        
        low_diff_entropies.append(diff)
    
    # 计算统计量
    high_entropy_avg = np.mean([e for _, e in high_entropy_steps]) if high_entropy_steps else 0.0
    low_entropy_avg = np.mean([e for _, e in low_entropy_steps]) if low_entropy_steps else 0.0
    high_diff_entropy_avg = np.mean(high_diff_entropies) if high_diff_entropies else 0.0
    low_diff_entropy_avg = np.mean(low_diff_entropies) if low_diff_entropies else 0.0
    
    # 计算相对差分熵 = 差分熵 / 平均熵
    high_relative_diff = high_diff_entropy_avg / high_entropy_avg if high_entropy_avg > 0 else 0.0
    low_relative_diff = low_diff_entropy_avg / low_entropy_avg if low_entropy_avg > 0 else 0.0
    
    return {
        'high_entropy_avg': float(high_entropy_avg),
        'low_entropy_avg': float(low_entropy_avg),
        'high_diff_entropy_avg': float(high_diff_entropy_avg),
        'low_diff_entropy_avg': float(low_diff_entropy_avg),
        'high_relative_diff': float(high_relative_diff),
        'low_relative_diff': float(low_relative_diff),
        'n_high_steps': len(high_entropy_steps),
        'n_low_steps': len(low_entropy_steps),
    }


def reward_func(
    data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None
):
    """
    方案1：相对差分熵比值奖励函数
    
    训练阶段：使用相对差分熵比值作为奖励
    验证阶段：使用ground_truth计算正确性奖励
    
    公式（训练阶段）：
    1. 以所有步骤平均熵为阈值，划分高熵/低熵步骤
    2. 高熵相对差分熵 = 高熵差分熵平均值 / 高熵平均熵
    3. 低熵相对差分熵 = 低熵差分熵平均值 / 低熵平均熵
    4. 奖励 = 低熵相对差分熵 / 高熵相对差分熵
    
    差分熵计算规则：
    - 高熵步骤：第一个差分熵为0，后续为 |当前熵 - 前面所有高熵步骤平均熵|
    - 低熵步骤：若前一步是高熵则重置（差分熵为0），否则为 |当前熵 - 当前连续低熵序列平均熵|
    
    设计意图：
    - 无缩放参数、无常数，纯比值形式
    - 训练后模型：低熵相对差分熵更高，高熵相对差分熵更低
    - 因此比值 low/high 训练后更高，奖励值更大
    - 鼓励模型在低熵区域保持稳定探索，高熵区域快速收敛
    """
    import numpy as np
    
    # 检查是否有token_entropies（验证阶段没有）
    token_entropies = extra_info.get("token_entropies") if extra_info else None
    
    # 调试：打印 extra_info 的键（只打印前几次）
    global _debug_print_count
    if '_debug_print_count' not in globals():
        _debug_print_count = 0
    if _debug_print_count < 3:
        print(f"[RewardFunc Debug] extra_info keys: {list(extra_info.keys()) if extra_info else 'None'}")
        if token_entropies is not None:
            print(f"[RewardFunc Debug] token_entropies length: {len(token_entropies)}")
        else:
            print(f"[RewardFunc Debug] token_entropies is None!")
        _debug_print_count += 1
    
    # 验证阶段：使用ground_truth计算正确性奖励
    if token_entropies is None or len(token_entropies) == 0:
        correctness = safe_compute_score(solution_str, str(ground_truth))
        if isinstance(correctness, dict):
            return correctness
        else:
            return {"score": float(correctness), "acc": bool(correctness)}
    
    # 训练阶段：使用相对差分熵比值奖励
    tokenizer = extra_info["tokenizer"]
    token_ids = extra_info["token_ids"]
    
    # 使用 process_thoughts 划分步骤（与筛选逻辑完全一致）
    # 然后基于这些步骤计算每个步骤的平均熵（token-步骤对齐）
    step_entropies = compute_step_entropies_aligned_with_process_thoughts(
        solution_str, token_ids, token_entropies, tokenizer
    )
    n_steps = len(step_entropies)
    
    # 处理边界情况：如果没有有效步骤
    if n_steps == 0:
        correctness = safe_compute_score(solution_str, str(ground_truth))
        is_correct = correctness.get('acc', False) if isinstance(correctness, dict) else bool(correctness)
        return {
            "score": 0.0,
            "entropy_reward": 0.0,
            "mean_entropy": 0.0,
            "high_entropy_avg": 0.0,
            "low_entropy_avg": 0.0,
            "high_diff_entropy_avg": 0.0,
            "low_diff_entropy_avg": 0.0,
            "high_relative_diff": 0.0,
            "low_relative_diff": 0.0,
            "n_high_steps": 0,
            "n_low_steps": 0,
            "n_steps": 0,
            "acc": is_correct,
            "correctness_score": correctness.get('score', 0.0) if isinstance(correctness, dict) else float(correctness),
        }
    
    step_entropies_arr = np.array(step_entropies)
    
    # 计算所有步骤的平均熵作为阈值
    mean_entropy = float(np.mean(step_entropies_arr))
    
    # 计算差分熵统计
    diff_stats = calculate_differential_entropy(step_entropies, mean_entropy)
    
    # 计算奖励：低熵相对差分熵 / 高熵相对差分熵
    high_relative_diff = diff_stats['high_relative_diff']
    low_relative_diff = diff_stats['low_relative_diff']
    
    if high_relative_diff > 0:
        entropy_reward = low_relative_diff / high_relative_diff
    else:
        # 如果高熵相对差分熵为0，使用低熵相对差分熵作为奖励
        entropy_reward = low_relative_diff if low_relative_diff > 0 else 0.0
    
    # 也计算正确性分数作为参考（仅用于监控，不影响训练奖励）
    correctness = safe_compute_score(solution_str, str(ground_truth))
    is_correct = correctness.get('acc', False) if isinstance(correctness, dict) else bool(correctness)
    
    return {
        "score": entropy_reward,
        "entropy_reward": entropy_reward,
        "mean_entropy": mean_entropy,
        "high_entropy_avg": diff_stats['high_entropy_avg'],
        "low_entropy_avg": diff_stats['low_entropy_avg'],
        "high_diff_entropy_avg": diff_stats['high_diff_entropy_avg'],
        "low_diff_entropy_avg": diff_stats['low_diff_entropy_avg'],
        "high_relative_diff": high_relative_diff,
        "low_relative_diff": low_relative_diff,
        "n_high_steps": diff_stats['n_high_steps'],
        "n_low_steps": diff_stats['n_low_steps'],
        "n_steps": n_steps,
        "acc": is_correct,
        "correctness_score": correctness.get('score', 0.0) if isinstance(correctness, dict) else float(correctness),
    }


# ========================================
# CoVo Style Reasoning Step Segmentation
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
    将步骤列表合并到最多15个步骤
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
    
    功能：
    1. 按行分割响应，去除空行
    2. 合并包含LaTeX公式的多行（检测 \\[ 和 \\]）
    3. 保留自然步数，不做强制合并
    
    Args:
        resp: 模型响应文本
    
    Returns:
        推理步骤列表
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