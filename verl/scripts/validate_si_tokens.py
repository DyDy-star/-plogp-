"""
验证 S-type / I-high / I-low token 分类在真实数学推理文本上是否合理。

运行方式:
  conda run -n ttrl python scripts/validate_si_tokens.py

输出:
  - 对每个样本, 逐 token 打印分类 (颜色区分)
  - 汇总统计: 哪类 token 对应哪类文本特征
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

# ── 颜色常量 ──────────────────────────────────────────────────────────────────
RED    = "\033[91m"   # S-type (惊喜/决策)
GREEN  = "\033[92m"   # I-high (持续推理)
YELLOW = "\033[93m"   # I-low  (机械计算)
GRAY   = "\033[90m"   # padding / ignored
BOLD   = "\033[1m"
RESET  = "\033[0m"

MODEL_PATH  = "/data/user5/models/Qwen2.5-Math-1.5B"
DATA_PATH   = "/data/user5/TTRL/verl/data/AIME-TTT/train.parquet"
N_SAMPLES   = 3       # 验证几道题
MAX_NEW_TOKENS = 512  # 生成长度 (足够看到完整推理)
EMA_ALPHA_BASE = 0.3  # I_EMA 的基础 alpha (与训练代码一致)


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """标准 Shannon 熵, 与训练代码一致"""
    pd_ = torch.softmax(logits.float(), dim=-1)
    entropy = torch.logsumexp(logits.float(), dim=-1) - (pd_ * logits.float()).sum(dim=-1)
    return entropy


def compute_ema(H: torch.Tensor) -> torch.Tensor:
    """自适应 EMA (与训练代码 si_plasticity 逻辑完全一致)"""
    T = H.shape[0]
    ema = torch.zeros_like(H)
    ema[0] = H[0]
    for t in range(1, T):
        err = (H[t-1] - ema[t-1]).abs()
        alpha_t = err / (err + ema[t-1].clamp(min=1e-4))
        ema[t] = alpha_t * H[t-1] + (1 - alpha_t) * ema[t-1]
    return ema


def classify_tokens(H: torch.Tensor, I: torch.Tensor, mean_I: float):
    """
    返回每个 token 的类别:
      's'      : S-type  (H > I)
      'i_high' : I-high  (H ≤ I, I ≥ mean_I)
      'i_low'  : I-low   (H ≤ I, I < mean_I)
    """
    classes = []
    for h, i in zip(H.tolist(), I.tolist()):
        if h > i:
            classes.append('s')
        elif i >= mean_I:
            classes.append('i_high')
        else:
            classes.append('i_low')
    return classes


def colorize(token_str: str, cls: str) -> str:
    if cls == 's':
        return f"{RED}{BOLD}{token_str}{RESET}"
    elif cls == 'i_high':
        return f"{GREEN}{token_str}{RESET}"
    else:
        return f"{YELLOW}{token_str}{RESET}"


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}加载数据...{RESET}")
    df = pd.read_parquet(DATA_PATH)
    problems = df['prompt'].tolist()[:N_SAMPLES]
    gts      = df['reward_model'].tolist()[:N_SAMPLES] if 'reward_model' in df.columns else ['?'] * N_SAMPLES

    print(f"{BOLD}加载模型: {MODEL_PATH}{RESET}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    # 汇总统计: 每类 token 的平均熵值、典型文本片段
    stat = defaultdict(list)    # cls → list of token strings
    stat_h = defaultdict(list)  # cls → list of entropy values

    for idx, (prompt_data, gt) in enumerate(zip(problems, gts)):
        # 取 prompt 文本
        if isinstance(prompt_data, list):
            prompt_text = prompt_data[-1]['content'] if isinstance(prompt_data[-1], dict) else str(prompt_data[-1])
        elif isinstance(prompt_data, str):
            prompt_text = prompt_data
        else:
            prompt_text = str(prompt_data)

        print(f"\n{'='*80}")
        print(f"{BOLD}[题目 {idx+1}]{RESET} {prompt_text[:200]}...")
        if isinstance(gt, dict):
            print(f"{BOLD}答案:{RESET} {gt.get('ground_truth', '?')}")

        # 构造输入
        chat = [{"role": "user", "content": prompt_text +
                 "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
        input_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        prompt_len = inputs['input_ids'].shape[1]

        # 生成
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=1.0,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response_ids = output_ids[0, prompt_len:]
        resp_len = response_ids.shape[0]
        if resp_len < 5:
            print("(生成太短, 跳过)")
            continue

        # 重新跑 forward 拿到每 token logits (用于精确熵计算)
        full_ids = output_ids[0:1]  # (1, full_len)
        with torch.no_grad():
            logits_all = model(full_ids).logits[0]  # (full_len, vocab)

        # 只取 response 部分的 logits (每个位置预测下一个 token, 所以用 [prompt_len-1 : prompt_len+resp_len-1])
        resp_logits = logits_all[prompt_len - 1 : prompt_len + resp_len - 1]  # (resp_len, vocab)
        H = entropy_from_logits(resp_logits)  # (resp_len,)
        I = compute_ema(H)                    # (resp_len,)
        mean_I = H.mean().item()              # 用批均值作为阈值 (与训练代码一致)

        classes = classify_tokens(H, I, mean_I)
        tokens = tokenizer.convert_ids_to_tokens(response_ids.tolist())

        # ── 打印分类结果 ─────────────────────────────────────────────────────
        print(f"\n{BOLD}颜色图例:{RESET} "
              f"{RED}{BOLD}S-type(决策){RESET}  "
              f"{GREEN}I-high(推理){RESET}  "
              f"{YELLOW}I-low(机械){RESET}\n")

        colored = ""
        for tok, cls in zip(tokens, classes):
            tok_str = tok.replace("▁", " ").replace("Ċ", "\n").replace("Ġ", " ")
            colored += colorize(tok_str, cls)
            # 累计统计
            stat[cls].append(tok_str.strip())
            stat_h[cls].append(H[len(stat[cls])-1 + len(stat['s']) + len(stat['i_high']) + len(stat['i_low'])
                                  - len(stat[cls])].item()
                               if False else H[classes.index(cls, classes.index(cls))].item()
                               )

        # 简单统计
        for tok_str, cls, h_val, i_val in zip(tokens, classes, H.tolist(), I.tolist()):
            tok_clean = tok_str.replace("▁", " ").replace("Ċ", "⏎").replace("Ġ", " ").strip()
            stat[cls].append(tok_clean)
            stat_h[cls].append(h_val)

        print(colored[:3000])  # 最多打印 3000 字符

        # ── 本题 token 统计 ──────────────────────────────────────────────────
        from collections import Counter
        cnt = Counter(classes)
        total = len(classes)
        print(f"\n{BOLD}本题统计:{RESET} "
              f"{RED}S={cnt['s']}({cnt['s']/total:.0%}){RESET}  "
              f"{GREEN}I-high={cnt['i_high']}({cnt['i_high']/total:.0%}){RESET}  "
              f"{YELLOW}I-low={cnt['i_low']}({cnt['i_low']/total:.0%}){RESET}")
        print(f"  mean_H={H.mean():.3f}  mean_I={mean_I:.3f}  "
              f"H_std={H.std():.3f}  I_std={I.std():.3f}")

        # ── 典型 token 举例 ──────────────────────────────────────────────────
        for cls_name, color in [('s', RED+BOLD), ('i_high', GREEN), ('i_low', YELLOW)]:
            paired = [(tok.replace("▁"," ").replace("Ċ","⏎").replace("Ġ"," ").strip(), h)
                      for tok, cls_t, h in zip(tokens, classes, H.tolist())
                      if cls_t == cls_name and tok.strip()]
            top5_high = sorted(paired, key=lambda x: x[1], reverse=True)[:8]
            top5_low  = sorted(paired, key=lambda x: x[1])[:8]
            label = {'s': 'S-type(惊喜/决策)', 'i_high': 'I-high(持续推理)', 'i_low': 'I-low(机械计算)'}[cls_name]
            print(f"\n  {color}{label}{RESET} 熵最高: "
                  + "  ".join(f"'{t}'({h:.2f})" for t, h in top5_high))
            print(f"  {color}{label}{RESET} 熵最低: "
                  + "  ".join(f"'{t}'({h:.2f})" for t, h in top5_low))

    # ── 全局汇总 ──────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{BOLD}全局汇总{RESET}")
    for cls_name, color in [('s', RED+BOLD), ('i_high', GREEN), ('i_low', YELLOW)]:
        toks = stat[cls_name]
        hs   = stat_h[cls_name]
        if not toks:
            continue
        # 最常见 token
        from collections import Counter
        top_toks = Counter(t for t in toks if t).most_common(15)
        avg_h = np.mean(hs) if hs else 0
        label = {'s': 'S-type (决策/惊喜)', 'i_high': 'I-high (持续推理)', 'i_low': 'I-low (机械)'}[cls_name]
        print(f"\n{color}{label}{RESET}  avg_H={avg_h:.3f}  count={len(toks)}")
        print(f"  最常见 token: {', '.join(repr(t) for t, _ in top_toks)}")


if __name__ == "__main__":
    main()
