#!/usr/bin/env python3
"""
评估脚本（vLLM版本）：根据pass@k评估协议评估AMC-TTT测试集

论文评估设置（基于论文Section "Evaluation Setup"）：

**主实验（Main Experiments）**：
- 采用pass@k评估协议（Chen et al., 2021）
- 使用非零温度采样报告pass@1
- 生成16个响应（对于32k context生成4个）
- temperature=0.6, top_p=0.95
- pass@1 = (1/k) * Σ(p_i)，其中p_i表示第i个响应是否正确

**Qwen2.5-MATH的分析和额外实验**：
- 使用贪婪解码报告pass@1（temperature=0, n=1）
- 为了与之前工作的公平比较

使用vLLM进行高效推理
"""

# 必须在导入torch之前设置multiprocessing启动方法
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import json
import os
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import argparse


# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from verl.utils.reward_score.ttrl_math import compute_score, extract_answer

# 尝试导入vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("警告: vLLM未安装，将使用HuggingFace Transformers（速度较慢）")


def load_vllm_model(model_path: str, tensor_parallel_size: int = 1):
    """
    使用vLLM加载模型
    
    参数说明：
    - tensor_parallel_size: 张量并行大小
      * =1: 模型完整加载到单GPU，通过batch_size利用多GPU（推荐）
      * >1: 模型分片到多GPU（更快，但与训练配置不同）
    - max_model_len: 4096 (训练时: MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)
    - gpu_memory_utilization: 0.85 (训练时为0.7，这里用0.85加速评估)
    """
    if not VLLM_AVAILABLE:
        raise RuntimeError("vLLM未安装，请使用: pip install vllm")
    
    print(f"正在使用vLLM加载模型: {model_path}")
    print(f"配置: tensor_parallel={tensor_parallel_size}, gpu_memory_utilization=0.85")
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
    )
    
    # 同时加载tokenizer用于应用chat template
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    return llm, tokenizer


def generate_responses_vllm(
    llm: LLM,
    tokenizer,
    prompts: List[str],
    n_samples: int = 16,
    temperature: float = 1,
    top_p: float = 0.95,
    max_tokens: int = 3072,
) -> List[List[str]]:
    """使用vLLM为多个问题批量生成响应
    
    注意：使用与训练验证一致的采样参数（temperature=0.6, n=16, top_p=0.95）
    并应用chat template（Qwen2.5-Math的chat template中内置了system prompt）
    """
    
    # 应用chat template（与训练时一致）
    # Qwen2.5-Math的tokenizer会自动添加system prompt
    full_prompts = []
    for p in prompts:
        # 构造聊天格式
        messages = [{"role": "user", "content": p}]
        # 应用chat template
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_prompts.append(formatted)
    
    # 配置采样参数
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
        responses = [o.text for o in output.outputs]
        all_responses.append(responses)
    
    return all_responses


def evaluate_pass_at_1_vllm(
    llm: LLM,
    tokenizer,
    test_data: List[Dict],
    n_samples: int = 16,
    temperature: float = 1,
    top_p: float = 0.95,
    max_tokens: int = 3072,
    batch_size: int = 8,
    output_file: str = None,
) -> Dict:
    """
    评估pass@1（vLLM版本）
    
    基于pass@k评估协议（Chen et al., 2021）：
    - 主实验：n_samples=16, temperature=0.6, top_p=0.95
    - 分析实验：n_samples=1, temperature=0.0 (贪婪解码)
    - pass@1 = (1/k) * Σ(p_i)，其中p_i表示第i个响应是否正确
    
    参考文献：
    - Chen et al., 2021: Evaluating Large Language Models Trained on Code
    - Guo et al., 2025: DeepSeek-R1 (TTRL论文遵循此评估协议)
    """
    
    results = []
    pass_at_1_scores = []
    
    print(f"\n开始评估 {len(test_data)} 个测试样本...")
    print(f"每个样本生成 {n_samples} 个响应")
    print(f"批次大小: {batch_size}")
    print(f"参数: temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}\n")
    
    # 批量处理
    for batch_start in tqdm(range(0, len(test_data), batch_size), desc="批次进度"):
        batch_end = min(batch_start + batch_size, len(test_data))
        batch_items = test_data[batch_start:batch_end]
        
        # 准备批次的prompts
        batch_prompts = [item["prompt"] for item in batch_items]
        
        # 批量生成响应
        batch_responses = generate_responses_vllm(
            llm, tokenizer, batch_prompts, n_samples, temperature, top_p, max_tokens
        )
        
        # 评估每个样本
        for item, responses in zip(batch_items, batch_responses):
            ground_truth = item["answer"]
            item_id = item.get("id", "")
            
            # 评估每个响应
            correct_count = 0
            response_results = []
            
            for i, response in enumerate(responses):
                score_dict = compute_score(response, ground_truth, fast=True)
                is_correct = score_dict.get("acc", False)
                
                if is_correct:
                    correct_count += 1
                
                response_results.append({
                    "response_id": i,
                    "response": response,
                    "extracted_answer": score_dict.get("pred", ""),
                    "is_correct": is_correct,
                })
            
            # 计算该样本的pass@1: (1/k) * Σ(p_i)
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
    
    # 计算其他统计指标
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
    print(f"正确率: {total_correct/total_responses:.4f} ({total_correct/total_responses*100:.2f}%)")
    print(f"\nOverall Pass@1: {overall_pass_at_1:.4f} ({overall_pass_at_1*100:.2f}%)")
    print(f"Pass@1 标准差: {pass_at_1_std:.4f}")
    print("=" * 70)
    
    # 打印一些样本的详细信息
    print("\n样本详情（前3个）:")
    for i, result in enumerate(results[:3]):
        print(f"\n样本 {i+1} (ID: {result['id']}):")
        print(f"  问题: {result['prompt'][:100]}...")
        print(f"  正确答案: {result['ground_truth']}")
        print(f"  正确响应数: {result['correct_count']}/{n_samples}")
        print(f"  Pass@1: {result['pass@1']:.4f}")
    
    return {
        "overall_pass@1": overall_pass_at_1,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="评估AMC-TTT测试集的Pass@1（vLLM版本）")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/user5/models/Qwen2.5-Math-1.5B",
        help="模型路径（可以是基础模型或检查点）"
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="/data/user5/TTRL/verl/data/AMC-TTT/test.parquet",
        help="测试数据路径（支持.json或.parquet格式）"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="eval_results_pass@1_vllm.json",
        help="输出结果文件路径"
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="main",
        choices=["main", "analysis"],
        help="评估模式：'main'=主实验(采样), 'analysis'=分析实验(贪婪解码)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="每个问题生成的响应数量（默认：main=16, analysis=1）"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="采样温度（默认：main=0.6, analysis=0.0）"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p采样参数（默认：main=0.95, analysis=1.0）"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=3072,
        help="最大生成token数"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="批次大小（采样模式推荐使用较小的batch，如8）"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="张量并行大小（GPU数量）"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="限制评估的样本数（用于快速测试）"
    )
    
    args = parser.parse_args()
    
    # 根据评估模式设置默认参数
    if args.eval_mode == "main":
        # 主实验：采样模式（论文主实验设置）
        n_samples = args.n_samples if args.n_samples is not None else 16
        temperature = args.temperature if args.temperature is not None else 0.6
        top_p = args.top_p if args.top_p is not None else 0.95
        batch_size = args.batch_size if args.batch_size != 8 else 8
        print("\n=== 主实验评估模式 ===")
        print("采样参数（论文标准）：temperature=0.6, n=16, top_p=0.95")
    else:
        # 分析实验：贪婪解码（用于与之前工作公平比较）
        n_samples = args.n_samples if args.n_samples is not None else 1
        temperature = args.temperature if args.temperature is not None else 0.0
        top_p = args.top_p if args.top_p is not None else 1.0
        batch_size = args.batch_size if args.batch_size != 8 else 64
        print("\n=== 分析实验评估模式（Qwen2.5-MATH） ===")
        print("贪婪解码参数：temperature=0.0, n=1 (与之前工作公平比较)")
    
    # 更新args
    args.n_samples = n_samples
    args.temperature = temperature
    args.top_p = top_p
    args.batch_size = batch_size
    
    print(f"最终参数：n_samples={n_samples}, temperature={temperature}, top_p={top_p}, batch_size={batch_size}\n")
    
    # 加载测试数据
    print(f"正在加载测试数据: {args.test_data_path}")
    
    if args.test_data_path.endswith('.parquet'):
        # 加载parquet格式（与训练一致）
        import pandas as pd
        df = pd.read_parquet(args.test_data_path)
        
        # 转换为统一格式
        test_data = []
        for idx, row in df.iterrows():
            # Parquet中prompt是聊天格式的数组，提取content
            if isinstance(row['prompt'], (list, np.ndarray)):
                prompt = row['prompt'][0]['content'] if len(row['prompt']) > 0 else ""
            else:
                prompt = row['prompt']
            
            # 从reward_model字段获取ground_truth
            ground_truth = row['reward_model']['ground_truth']
            
            test_data.append({
                "prompt": prompt,
                "answer": ground_truth,
                "id": row['id']
            })
    else:
        # 加载JSON格式
        with open(args.test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    
    # 限制样本数（如果指定）
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    
    print(f"加载了 {len(test_data)} 个测试样本")
    
    # 加载模型和tokenizer
    llm, tokenizer = load_vllm_model(args.model_path, args.tensor_parallel_size)
    
    # 评估
    evaluate_pass_at_1_vllm(
        llm,
        tokenizer,
        test_data,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()

