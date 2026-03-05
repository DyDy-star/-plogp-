#!/usr/bin/env python3
"""
AIME Pass@K 评估: Standard GRPO vs SurprisalRedistribution-GRPO

功能:
  1. 使用 vLLM 批量生成 (高效)
  2. Pass@1 (贪心) + Pass@32 (温度采样)
  3. 自动评判答案正确性
  4. 绘制对比图表 (analyze_step_entropy.py 风格)

Usage:
    python eval_pass_k.py \\
        --model_std /path/to/std_hf_merged \\
        --model_sr  /path/to/sr_hf_merged  \\
        --device cuda:2
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Style: consistent with analyze_step_entropy.py ----
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['font.sans-serif'] = [
    'Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

STD_COLOR = "#3498DB"
SR_COLOR = "#E74C3C"
BASE_COLOR = "#9B59B6"
CORRECT_COLOR = "#2ECC71"
WRONG_COLOR = "#E74C3C"

# ---- Default paths ----
CKPT_BASE = (
    "/data/user5/TTRL begin/verl/checkpoints/TTRL-verl/"
    "AIME-TTT-Qwen2.5-Math-1.5B/0303"
)
DEFAULT_STD_MODEL = os.path.join(
    CKPT_BASE, "GT-baseline-Len@3k-grpo-151743",
    "global_step_240/actor/huggingface_merged")
DEFAULT_SR_MODEL = os.path.join(
    CKPT_BASE, "GT-plogpPos-Len@3k-grpo-064004",
    "global_step_240/actor/huggingface_merged")
DEFAULT_BASE_MODEL = "/data/user5/models/Qwen2.5-Math-1.5B"
DEFAULT_DATA_PATH = "/data/user5/TTRL/verl/data/AIME-TTT/test.parquet"


# =====================================================================
# Answer checking
# =====================================================================

def extract_boxed(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} content, handling nested braces."""
    results = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        start = idx + len("\\boxed{")
        depth = 1
        pos = start
        while pos < len(text) and depth > 0:
            if text[pos] == '{':
                depth += 1
            elif text[pos] == '}':
                depth -= 1
            pos += 1
        results.append(text[start:pos - 1])
        i = pos
    return results[-1].strip() if results else None


def normalize_answer(answer: str) -> str:
    """Normalize for numeric comparison (AIME answers are integers 0-999)."""
    if answer is None:
        return ""
    s = answer.strip()
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\text{", "").replace("}", "")
    s = s.replace("\\", "").replace(",", "").replace(" ", "")
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return s
    except ValueError:
        return s


def check_answer(response: str, ground_truth: str) -> bool:
    extracted = extract_boxed(response)
    if extracted is None:
        return False
    return normalize_answer(extracted) == normalize_answer(ground_truth)


def check_answer_with_project_grader(response: str, ground_truth: str) -> bool:
    """Try the project's grader first, fallback to simple comparison."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from verl.utils.reward_score.ttrl_math import extract_answer, grade
        model_answer = extract_answer(response)
        if model_answer is None:
            return False
        return grade(model_answer, str(ground_truth), fast=True)
    except Exception:
        return check_answer(response, ground_truth)


# =====================================================================
# Data loading
# =====================================================================

def load_aime_data(data_path: str) -> List[Dict]:
    """Load AIME test data from parquet."""
    df = pd.read_parquet(data_path)
    samples = []
    for _, row in df.iterrows():
        prompt_field = row.get("prompt", None)
        if isinstance(prompt_field, list):
            messages = prompt_field
        elif isinstance(prompt_field, str):
            messages = [{"role": "user", "content": prompt_field}]
        else:
            messages = [{"role": "user", "content": str(prompt_field)}]

        gt = ""
        rm = row.get("reward_model", None)
        if rm is not None:
            if isinstance(rm, dict):
                gt = rm.get("ground_truth", "")
            elif isinstance(rm, str):
                try:
                    gt = json.loads(rm).get("ground_truth", "")
                except (json.JSONDecodeError, TypeError):
                    gt = rm
        if not gt:
            gt = str(row.get("answer", ""))

        samples.append({
            "messages": messages,
            "ground_truth": str(gt),
            "index": row.get("id", _),
        })
    return samples


# =====================================================================
# Model evaluation (vLLM)
# =====================================================================

def evaluate_model(
    model_path: str,
    samples: List[Dict],
    n_samples: int = 32,
    max_tokens: int = 3072,
    device: str = "cuda:0",
    gpu_memory: float = 0.85,
) -> Dict:
    """Evaluate a model on AIME with Pass@1 and Pass@N."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    gpu_id = int(device.split(":")[-1]) if ":" in device else 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"\n  Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    formatted_prompts = []
    for s in samples:
        text = tokenizer.apply_chat_template(
            s["messages"], tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(text)

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=max_tokens + 1536,
        gpu_memory_utilization=gpu_memory,
    )

    # ---- Pass@1: greedy ----
    print("  Evaluating Pass@1 (greedy) ...")
    t0 = time.time()
    params_greedy = SamplingParams(
        temperature=0, max_tokens=max_tokens, n=1)
    outputs_1 = llm.generate(formatted_prompts, params_greedy)
    t1 = time.time()
    print(f"  Pass@1 generation done in {t1 - t0:.1f}s")

    pass1_results = []
    for i, output in enumerate(outputs_1):
        resp = output.outputs[0].text
        correct = check_answer_with_project_grader(
            resp, samples[i]["ground_truth"])
        pass1_results.append({
            "problem_idx": i,
            "response": resp[:500],
            "correct": correct,
            "ground_truth": samples[i]["ground_truth"],
            "extracted": extract_boxed(resp),
        })

    # ---- Pass@N: temperature sampling ----
    print(f"  Evaluating Pass@{n_samples} (T=0.6, top_p=0.95) ...")
    t0 = time.time()
    params_sample = SamplingParams(
        temperature=0.6, top_p=0.95, max_tokens=max_tokens, n=n_samples)
    outputs_n = llm.generate(formatted_prompts, params_sample)
    t1 = time.time()
    print(f"  Pass@{n_samples} generation done in {t1 - t0:.1f}s")

    pass_n_results = []
    for i, output in enumerate(outputs_n):
        n_correct = 0
        for resp_out in output.outputs:
            if check_answer_with_project_grader(
                    resp_out.text, samples[i]["ground_truth"]):
                n_correct += 1
        pass_n_results.append({
            "problem_idx": i,
            "n_correct": n_correct,
            "n_total": n_samples,
            "any_correct": n_correct > 0,
            "ground_truth": samples[i]["ground_truth"],
        })

    # cleanup
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pass1_acc = sum(r["correct"] for r in pass1_results) / len(pass1_results)
    pass_n_acc = sum(r["any_correct"] for r in pass_n_results) / len(pass_n_results)
    print(f"  Pass@1  = {pass1_acc:.1%}  ({sum(r['correct'] for r in pass1_results)}/{len(pass1_results)})")
    print(f"  Pass@{n_samples} = {pass_n_acc:.1%}  ({sum(r['any_correct'] for r in pass_n_results)}/{len(pass_n_results)})")

    return {
        "pass1": pass1_results,
        "pass_n": pass_n_results,
        "pass1_acc": pass1_acc,
        "pass_n_acc": pass_n_acc,
        "n_samples": n_samples,
    }


# =====================================================================
# Plotting
# =====================================================================

def plot_pass_k_bar(results: Dict, output_dir: str, n_samples: int = 32):
    """Plot 1: Main Pass@K comparison bar chart."""
    models = list(results.keys())
    pass1_accs = [results[m]["pass1_acc"] * 100 for m in models]
    pass_n_accs = [results[m]["pass_n_acc"] * 100 for m in models]

    colors = {"Standard GRPO": STD_COLOR, "SR-GRPO": SR_COLOR,
              "Base Model": BASE_COLOR}

    fig, ax = plt.subplots(figsize=(10, 7))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width / 2, pass1_accs, width,
                   label="Pass@1 (Greedy)",
                   color=[colors.get(m, "#888") for m in models],
                   alpha=0.85, edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width / 2, pass_n_accs, width,
                   label=f"Pass@{n_samples} (T=0.6)",
                   color=[colors.get(m, "#888") for m in models],
                   alpha=0.50, edgecolor='black', linewidth=0.8,
                   hatch='///')

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f'{h:.1f}%', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    ax.set_title(f"AIME: Pass@1 vs Pass@{n_samples} Comparison",
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(max(pass1_accs), max(pass_n_accs)) * 1.18)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    n_problems = len(results[models[0]]["pass1"])
    info = f"AIME Test Set: {n_problems} problems"
    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    fig.tight_layout()
    path = os.path.join(output_dir, "pass_k_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_problem(results: Dict, output_dir: str, n_samples: int = 32):
    """Plot 2: Per-problem correctness heatmap."""
    models = list(results.keys())
    n_problems = len(results[models[0]]["pass1"])

    col_labels = []
    data_matrix = []
    for m in models:
        p1 = [1 if r["correct"] else 0 for r in results[m]["pass1"]]
        pn = [1 if r["any_correct"] else 0 for r in results[m]["pass_n"]]
        data_matrix.append(p1)
        data_matrix.append(pn)
        col_labels.append(f"{m}\nPass@1")
        col_labels.append(f"{m}\nPass@{n_samples}")

    matrix = np.array(data_matrix).T  # (n_problems, n_columns)

    fig, ax = plt.subplots(figsize=(max(8, len(col_labels) * 2.2), max(8, n_problems * 0.35)))
    cmap = matplotlib.colors.ListedColormap([WRONG_COLOR, CORRECT_COLOR])

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1,
                   interpolation='nearest')
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10, fontweight='bold')
    ax.set_yticks(range(n_problems))
    ax.set_yticklabels([f"Q{i + 1}" for i in range(n_problems)], fontsize=9)
    ax.set_xlabel("Model × Metric", fontsize=12, fontweight='bold')
    ax.set_ylabel("Problem", fontsize=12, fontweight='bold')
    ax.set_title("Per-Problem Correctness",
                 fontsize=14, fontweight='bold', pad=15)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "✓" if matrix[i, j] else "✗",
                    ha='center', va='center', fontsize=10,
                    color='white', fontweight='bold')

    col_accs = matrix.sum(axis=0)
    for j in range(len(col_labels)):
        ax.text(j, n_problems + 0.3,
                f"{col_accs[j]}/{n_problems}",
                ha='center', va='top', fontsize=10, fontweight='bold',
                color='black')

    fig.tight_layout()
    path = os.path.join(output_dir, "per_problem_heatmap.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_pass_n_distribution(results: Dict, output_dir: str, n_samples: int = 32):
    """Plot 3: Pass@N solve rate per problem (n_correct / n_total)."""
    models = list(results.keys())
    n_problems = len(results[models[0]]["pass_n"])
    colors_map = {"Standard GRPO": STD_COLOR, "SR-GRPO": SR_COLOR,
                  "Base Model": BASE_COLOR}

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(n_problems)
    width = 0.35
    n_models = len(models)

    for mi, m in enumerate(models):
        solve_rates = [r["n_correct"] / r["n_total"] * 100
                       for r in results[m]["pass_n"]]
        offset = (mi - (n_models - 1) / 2) * width
        ax.bar(x + offset, solve_rates, width,
               label=m, color=colors_map.get(m, "#888"),
               alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xlabel("Problem Index", fontsize=12, fontweight='bold')
    ax.set_ylabel(f"Solve Rate in {n_samples} Samples (%)",
                  fontsize=12, fontweight='bold')
    ax.set_title(f"AIME: Per-Problem Solve Rate (Pass@{n_samples})",
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i + 1}" for i in range(n_problems)],
                       fontsize=7, rotation=45, ha='right')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    path = os.path.join(output_dir, f"pass{n_samples}_distribution.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_improvement_delta(results: Dict, output_dir: str, n_samples: int = 32):
    """Plot 4: SR-GRPO improvement over Standard GRPO per problem."""
    models = list(results.keys())
    if len(models) < 2:
        return

    m_std, m_sr = models[0], models[1]
    n_problems = len(results[m_std]["pass_n"])

    std_rates = [r["n_correct"] / r["n_total"] for r in results[m_std]["pass_n"]]
    sr_rates = [r["n_correct"] / r["n_total"] for r in results[m_sr]["pass_n"]]
    deltas = [(sr - std) * 100 for std, sr in zip(std_rates, sr_rates)]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = [CORRECT_COLOR if d >= 0 else WRONG_COLOR for d in deltas]
    ax.bar(range(n_problems), deltas, color=colors, alpha=0.7,
           edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel("Problem Index", fontsize=12, fontweight='bold')
    ax.set_ylabel("SR-GRPO − Standard GRPO (%)", fontsize=12, fontweight='bold')
    ax.set_title(f"Per-Problem Improvement: SR-GRPO vs Standard GRPO (Pass@{n_samples})",
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(range(n_problems))
    ax.set_xticklabels([f"Q{i + 1}" for i in range(n_problems)],
                       fontsize=7, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    n_better = sum(1 for d in deltas if d > 0)
    n_worse = sum(1 for d in deltas if d < 0)
    n_same = sum(1 for d in deltas if d == 0)
    avg_delta = np.mean(deltas)
    info = (f"SR better: {n_better}  |  Same: {n_same}  |  STD better: {n_worse}\n"
            f"Avg Δ = {avg_delta:+.1f}%")
    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    fig.tight_layout()
    path = os.path.join(output_dir, f"improvement_delta_pass{n_samples}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AIME Pass@K 评估: Standard GRPO vs SR-GRPO")
    parser.add_argument("--model_std", type=str, default=DEFAULT_STD_MODEL,
                        help="Standard GRPO HF merged model path")
    parser.add_argument("--model_sr", type=str, default=DEFAULT_SR_MODEL,
                        help="SR-GRPO HF merged model path")
    parser.add_argument("--model_base", type=str, default=None,
                        help="Base model path (optional)")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--n_samples", type=int, default=32,
                        help="Number of samples for Pass@N")
    parser.add_argument("--max_tokens", type=int, default=3072)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gpu_memory", type=float, default=0.85)
    parser.add_argument("--output_dir", type=str,
                        default="experiments/results_pass_k")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load data ----
    print("=" * 60)
    print("Loading AIME data ...")
    print("=" * 60)
    samples = load_aime_data(args.data_path)
    print(f"Loaded {len(samples)} problems")

    # ---- Evaluate models ----
    all_results = {}
    model_configs = [
        ("Standard GRPO", args.model_std),
        ("SR-GRPO", args.model_sr),
    ]
    if args.model_base:
        model_configs.append(("Base Model", args.model_base))

    for name, path in model_configs:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {name}")
        print(f"  Path: {path}")
        print(f"{'=' * 60}")

        if not os.path.exists(path):
            print(f"  ⚠ Model path not found: {path}")
            print(f"  Skipping {name}")
            continue

        has_weights = (
            any(f.endswith('.safetensors') for f in os.listdir(path))
            or any(f.endswith('.bin') for f in os.listdir(path))
        ) if os.path.isdir(path) else False

        if not has_weights:
            print(f"  ⚠ No model weights found in {path}")
            print(f"  Please run merge first: python -m verl.model_merger merge ...")
            continue

        result = evaluate_model(
            model_path=path,
            samples=samples,
            n_samples=args.n_samples,
            max_tokens=args.max_tokens,
            device=args.device,
            gpu_memory=args.gpu_memory,
        )
        all_results[name] = result

    if len(all_results) < 1:
        print("\n⚠ No models evaluated. Please check model paths / merge status.")
        return

    # ---- Save raw results ----
    summary = {}
    for name, r in all_results.items():
        summary[name] = {
            "pass1_acc": r["pass1_acc"],
            f"pass{args.n_samples}_acc": r["pass_n_acc"],
            "pass1_correct": sum(x["correct"] for x in r["pass1"]),
            f"pass{args.n_samples}_correct": sum(x["any_correct"] for x in r["pass_n"]),
            "n_problems": len(r["pass1"]),
            "pass1_detail": r["pass1"],
            f"pass{args.n_samples}_detail": [
                {"problem_idx": x["problem_idx"],
                 "n_correct": x["n_correct"],
                 "n_total": x["n_total"],
                 "ground_truth": x["ground_truth"]}
                for x in r["pass_n"]
            ],
        }

    json_path = os.path.join(args.output_dir, "eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {json_path}")

    # ---- Print summary table ----
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    header = f"{'Model':<20} {'Pass@1':>10} {'Pass@' + str(args.n_samples):>10}"
    print(header)
    print("-" * len(header))
    for name, r in all_results.items():
        n = len(r["pass1"])
        p1 = sum(x["correct"] for x in r["pass1"])
        pn = sum(x["any_correct"] for x in r["pass_n"])
        print(f"{name:<20} {p1:>3}/{n} ({r['pass1_acc']:.1%}) "
              f"{pn:>3}/{n} ({r['pass_n_acc']:.1%})")
    print(f"{'=' * 60}")

    # ---- Plot ----
    print("\nGenerating plots ...")
    plot_pass_k_bar(all_results, args.output_dir, args.n_samples)
    plot_per_problem(all_results, args.output_dir, args.n_samples)
    plot_pass_n_distribution(all_results, args.output_dir, args.n_samples)
    if len(all_results) >= 2:
        plot_improvement_delta(all_results, args.output_dir, args.n_samples)

    print(f"\n✅ All done! Results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
